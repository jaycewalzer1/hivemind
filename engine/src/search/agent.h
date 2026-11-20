#pragma once

#include "environment/adjudication.h"
#include <optional>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "search/node.h"
#include "search/mate_candidate.h"
#include "nn/engine.h"
#include "search/search_params.h"
#include "search/transposition_table.h"
#include "search/gc_thread.h"
#include "common/globals.h"
#include "environment/joint_action.h"

class SearchThread;
struct SearchInfo;

/**
 * @brief Search options to configure Agent::run_search behavior.
 */
struct SearchOptions {
    // Stopping conditions (one must be set)
    size_t targetNodes = 0;      // Stop after this many nodes (0 = use time)
    int moveTimeMs = 0;          // Stop after this many milliseconds (0 = use nodes)
    // How far time extensions may carry a moveTimeMs search in a critical
    // position. At or below moveTimeMs the move time is the whole allocation.
    int maxMoveTimeMs = 0;
    uint64_t mateProbeNodes = 0; // Override the Fairy-Stockfish root probe budget
    bool completeMateProbe = false; // Let a bounded probe finish after MCGS stops
    
    // UCI mode options
    bool verbose = false;        // Output UCI info strings (info, bestmove)
    int multiPV = 1;             // Number of principal variations to output
    bool isPonder = false;       // Ponder mode: think continuously until ponderhit or stop
    // While searching, print the whole MultiPV block this often
    // (0 = only the rank-1 line at each new depth).
    int reportIntervalMs = 0;
    bool enablePonder = true;    // Output ponder move with bestmove
    // Permanent-brain search: prints nothing, skips the root scans, runs until
    // stopped or until its own node/time cap, and retains its entire root
    // subtree for reuse rather than a subtree below a move it is about to play.
    bool background = false;
    // The match clock this search is rooted at. Unset, the TimeAdvantage flag
    // passed to run_search stands for a permanent lead or deficit, as it always
    // has. Set, it decides who may sit at every node, and the flag passed to
    // run_search is ignored in favour of what it implies.
    std::optional<TimeState> clock;
    // Index the subtree below the move this search chooses for the next
    // search to reuse. A caller that plays some other move (self-play samples
    // one) turns this off and keeps that move's subtree with
    // Agent::retain_played_line instead.
    bool retainForReuse = true;
    // A node target counts the visits a reused root already holds: the search
    // tops the root up to targetNodes, adding at least minNewNodes of its own.
    // Off, targetNodes is all new nodes, as for a root searched from scratch.
    bool reusedVisitsCount = false;
    size_t minNewNodes = 1;
    // The team the tree's terminal verdicts are judged for (the waiting-board
    // mate's race, draw contempt). Unset, the team to move at the root. A tree
    // is only reusable by searches that judge for the same team: the
    // permanent brain searches the opponents' position to prepare our next
    // search, so it judges for us.
    std::optional<Stockfish::Color> verdictTeam;
    // Wave 2c, fix 1 (UCI TeamTreeReuse, docs/custom_engine_spec_v5_continuity.md):
    // under a durative clock, keep the whole tree between this team's
    // searches and root the next one at the node the game reached, re-timed
    // to the real clocks, instead of the exact-hash candidates above (which
    // never match a seat clock). Its seat clocks and the other team's ready
    // times may differ by at most teamTreeReuseToleranceMs.
    bool teamTreeReuse = false;
    int teamTreeReuseToleranceMs = 250;
    // Wave 2c, fix 3 (UCI PlanCarry): under a durative clock, remember the
    // halves of the pair this search chooses for boards our team is on turn
    // on (Agent::plans_), and at this team's next search mix the root prior
    // of a remembered move that is still valid by planCarryPermille / 1000
    // and expand the pair holding it first, so the partner continues the
    // plan unless the new search finds better.
    bool planCarry = false;
    int planCarryPermille = 300;

    SearchParams::RuntimeConfig search;
    
    // Convenience constructors
    static SearchOptions uci(int moveTimeMs, int multiPV = 1, bool isPonder = false) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = true;
        opts.multiPV = multiPV;
        opts.isPonder = isPonder;
        return opts;
    }
    
};

/**
 * @brief What fix 1 (TeamTreeReuse) did at the start of one search: whether
 * it adopted a node of the retained tree, and if not why. Reasons: "no node"
 * (no node of the tree has the live boards), "clock" (the closest has
 * clocks beyond the tolerance), "shape" (it would act on other boards, or
 * sits or holds pieces back differently), "committed" (fix 2), "solved"
 * (unexpanded or proven), "no tree" (nothing retained).
 */
struct TeamTreeReuseReport {
    bool attempted = false;
    bool adopted = false;
    std::string reason;
    // The adopted root's visits after re-timing; edges cleared by it.
    int visits = 0;
    int pruned = 0;
    // Largest clock difference at the adopted node, or at the closest node
    // with the live boards (-1: none).
    int clockDiffMs = -1;
    // Nodes of the old tree with the live boards and our team to play.
    int boardMatches = 0;
    // Fix 2: when the refusal was `committed`, that node's commitment per
    // seat (fix 3 reads our seats' as the tree's plan for them).
    std::array<Stockfish::Move, 4> committed{Stockfish::MOVE_NONE, Stockfish::MOVE_NONE,
                                             Stockfish::MOVE_NONE, Stockfish::MOVE_NONE};
};

/**
 * @brief Wave 2c, fix 3 (PlanCarry): the remembered half of a chosen pair on
 * one board, for the seat there whose own decision is still to come.
 */
struct CarriedPlan {
    int board = -1;
    Stockfish::Move move = Stockfish::MOVE_NONE;
    std::string fen;               // that board's FEN when the plan was made
    Stockfish::Color team = Stockfish::WHITE;
    float edgeQ = 0.0f;            // telemetry only
    float edgeVisitShare = 0.0f;   // telemetry only
};

/**
 * @brief What fix 3 did at the start of one search, per board: the slot's
 * move (MOVE_NONE: no plan), whether it was mixed into the root's priors,
 * and if not why: "other team", "not on turn", "board changed", "pocket"
 * (our seat's pocket lost a piece), "illegal", "not a root action" (legal
 * but held back, InstantTransfer off). `fromTree`: the slot was filled from
 * fix 1's `committed` refusal (fix 2's plan for that seat).
 */
struct PlanCarryReport {
    bool attempted = false;
    std::array<Stockfish::Move, 2> move{Stockfish::MOVE_NONE, Stockfish::MOVE_NONE};
    std::array<bool, 2> applied{false, false};
    std::array<bool, 2> fromTree{false, false};
    std::array<std::string, 2> reason;
    float weight = 0.0f;
    bool any_applied() const { return applied[0] || applied[1]; }
};

struct RootEdgeStats {
    JointActionCandidate action;
    int visits = 0;
    float q = 0.0f;  // edge Q from the root team's view
    NodeType childType = NodeType::UNSOLVED;
};

struct InternalMateProbeStats {
    uint64_t childProbes = 0;
    uint64_t grandchildProbes = 0;
    uint64_t childHits = 0;
    uint64_t grandchildHits = 0;
    uint64_t childActionableHits = 0;
    uint64_t grandchildActionableHits = 0;
    uint64_t alreadyGeneratedHits = 0;
    uint64_t promotionsSkipped = 0;
    uint64_t alreadySolvedHits = 0;
    uint64_t staleHits = 0;
    uint64_t checksOnlyCertificates = 0;
    uint64_t reducedPartnerCertificates = 0;
    uint64_t jointCertificates = 0;
    uint64_t nodes = 0;
    uint64_t certificateNodes = 0;
};

/** What certifying the selected move cost and decided, for one search. */
struct SelectedMoveCertStats {
    uint64_t candidates = 0;
    uint64_t probeHits = 0;
    uint64_t certificates = 0;
    uint64_t nodes = 0;
    uint64_t certificateNodes = 0;
    bool vetoed = false;
    bool replaced = false;
};

namespace joint_mate {
struct JointMateCache;
}

/**
 * The verifier's standing on one root action. A proof or an exhaustive
 * refutation within the bound is final for the search; anything cut short is
 * UNKNOWN and stays eligible for another slice. The proof cache stays with
 * the action so a later slice resumes from settled subtrees.
 */
struct RootActionVerdict {
    enum class State { UNKNOWN, PROVEN_LOSS, REFUTED_AT_BOUND };
    State state = State::UNKNOWN;
    bool fairyProbed = false;
    int slices = 0;
    int plyToMate = 0;
    uint64_t nodes = 0;
    std::shared_ptr<joint_mate::JointMateCache> cache;
};

/** What the concurrent verifier did and decided during one search. */
struct ConcurrentVerifierStats {
    uint64_t slices = 0;
    uint64_t probes = 0;
    uint64_t probeHits = 0;
    uint64_t proven = 0;
    uint64_t refuted = 0;
    uint64_t nodes = 0;
    /// Nodes spent on the actions that ended proven, in total.
    uint64_t provenNodes = 0;
    uint64_t actions = 0;
    /// Verdict on the action finally played: "proven", "refuted", "unknown",
    /// or "unseen" when the verifier never reached it.
    const char* leaderVerdict = "unseen";
};

enum class MateCertificateTier : uint8_t {
    NONE,
    CHECKS_ONLY,
    REDUCED_PARTNER,
    FULL_JOINT,
};

/**
 * Why a mate candidate was not certified, from the most final reason to the
 * one where more search could still succeed. Of several attempts, the
 * latest in this order is kept.
 */
enum class MateCertificateFailure : uint8_t {
    NONE,
    ILLEGAL,   // No legal joint action carries the candidate.
    RACE,      // The candidate loses the mate race.
    REFUTED,   // A defence answered every forcing line searched.
    DEPTH,     // Refuted only once the attacker moves ran out.
    BUDGET,    // Nodes or time ran out first.
};

/** One joint ply retained from an exact mate proof. */
struct MateProofPly {
    Stockfish::Move moveA = Stockfish::MOVE_NONE;
    Stockfish::Move moveB = Stockfish::MOVE_NONE;
};

/** Exact opponent-mate proof for one legal action at the current root. */
struct RootLossProof {
    JointActionCandidate action;
    int plyToMate = 0;
    std::vector<MateProofPly> principalVariation;
};

/**
 * Work several threads share in one loss scan: each takes the next defense
 * from `nextDefense` and sizes its share from the nodes the threads have
 * left between them.
 */
struct LossScanShare {
    std::atomic<size_t> nextDefense{0};
    std::atomic<uint64_t> remainingNodes{0};
};

/** What one loss scan did with the opponents' direct-threat certificates. */
struct LossScanStats {
    /** Certificate searches run, at most one per distinct untouched board. */
    uint64_t certificateSearches = 0;
    /** Defenses proven lost by a certificate instead of their own search. */
    uint64_t certificateProofs = 0;
};

/**
 * @brief Manages multi-threaded MCGS (Monte Carlo Graph Search) for Bughouse.
 *
 * Runs multiple search threads in parallel, each with its own engine instance.
 * All threads share the same search graph with thread-safe node operations.
 * Uses a transposition table to detect when different move sequences reach
 * the same position, enabling more efficient value estimation.
 */
class Agent {
private:
    GameStatus searchStatus_ = GameStatus::ONGOING;
    bool certifiedChoice_ = false;
    struct RetainedRootCandidate {
        std::shared_ptr<Node> node;
        uint64_t positionHash = 0;
        std::string signature;
    };

    struct MateContinuation {
        uint64_t positionHash = 0;
        std::string signature;
        Stockfish::Color teamSide = Stockfish::WHITE;
        bool teamHasTimeAdvantage = true;
        JointActionCandidate action;
        int plyToMate = 0;
    };

    std::vector<SearchThread*> searchThreads;
    std::vector<std::thread> workerPool_;
    std::atomic<bool> running;                            
    std::mutex searchMutex_;
    std::mutex workerMutex_;
    std::condition_variable workerCv_;
    std::condition_variable workersDoneCv_;
    uint64_t workerGeneration_ = 0;
    size_t activeWorkerCount_ = 0;
    size_t completedWorkerCount_ = 0;
    bool shutdownWorkers_ = false;
    const Board* workerBoard_ = nullptr;
    std::vector<Engine*> workerEngines_;
    SearchInfo* workerSearchInfo_ = nullptr;
    TimeState workerTime_;
    // Clock state the current search is rooted at; set by run_search.
    TimeState searchTime_;
    size_t workerTargetNodes_ = 0;
    int workerMoveTimeMs_ = 0;
    std::exception_ptr workerException_;
    std::shared_ptr<Node> rootNode;
    std::unique_ptr<TranspositionTable> transpositionTable;  // MCGS transposition table
    MateCandidateTable mateCandidateHints_;
    int numThreads;                                          // Search threads per engine
    SearchParams::RuntimeConfig lastRuntimeConfig_;
    std::atomic<bool> isPondering_{false};                   // Whether current search is in ponder mode
    // Latched by set_is_running(false). The permanent brain has no clock of its
    // own, so it needs a stop signal that survives dispatch_workers setting
    // `running` back to true.
    std::atomic<bool> stopRequested_{false};
    // Stop signals for the two root scans. Both latch on a stop request. A
    // node-limited search has no wall-clock deadline; its workers cancel both
    // scans when the tree finishes. Timed workers do the same, and the winning
    // scan also has the tree's deadline. A proven mate cancels the loss scan.
    std::atomic<bool> rootWinScanStop_{false};
    std::atomic<bool> rootLossScanStop_{false};
    std::atomic<SearchInfo*> currentSearchInfo_{nullptr};    // Active search info pointer
    
    // Tree reuse support (CrazyAra-style). Retain every generated opponent
    // response below the selected move, not only the predicted response.
    // Keyed by position hash so any retained position is a single lookup, not
    // a scan of the two or three predicted ones.
    std::unordered_map<uint64_t, RetainedRootCandidate> nextRootCandidates_;
    // Exact positions and winning moves emitted by the bounded mate solver.
    // Unlike the synthetic one-edge search tree used for UCI reporting, these
    // retain every defender branch that the proof recursively verified.
    std::vector<MateContinuation> mateContinuations_;
    uint64_t lastSearchHash_ = 0;            // Hash of last search position

    // Wave 2c, fix 1 (TeamTreeReuse): the last durative search's whole tree,
    // kept for this team's next search to adopt the node the game reached.
    struct TeamTree {
        std::shared_ptr<Node> root;      // the last search's root; owns the tree
        std::shared_ptr<Board> board;    // its position, to replay edges from
        std::string boardA, boardB;      // its FENs (pockets included)
        std::array<int, 2> plies{};      // game_ply per board
        TimeState rootTime;              // the durative clock it was rooted at
        Stockfish::Color team = Stockfish::WHITE;  // only this team reuses it
    };
    std::optional<TeamTree> teamTree_;
    TeamTreeReuseReport lastTeamTreeReuse_;
    // The adoption's re-timed nodes, for the fresh transposition table.
    std::vector<std::shared_ptr<Node>> teamTreeWalked_;

    // Wave 2c, fix 3 (PlanCarry): one slot per board, written at the end of
    // a durative search (or by record_plans) and read at the start of the
    // team's next one. Outside the tree: reset_search_state() leaves them,
    // clear_plans() (ucinewgame, game start) empties them.
    std::array<CarriedPlan, 2> plans_;
    PlanCarryReport lastPlanCarry_;
    void record_plans_locked(Board& board, Stockfish::Color team,
                             const JointActionCandidate& action, std::array<bool, 2> boards);
    /** The root's prior mix for this search, validated; fills lastPlanCarry_. */
    void resolve_plans(Board& board, Stockfish::Color team, int permille,
                       SearchParams::RuntimeConfig& config);

    /** Where find_team_tree_node got to. */
    struct TeamTreeMatch {
        std::shared_ptr<Node> node;
        int clockDiffMs = -1;
        std::string reason = "no node";
        int boardMatches = 0;
        // Fix 2: when the furthest refusal was `committed`, that node's
        // commitments per seat (fix 3 reads our seats' as the tree's plan).
        std::array<Stockfish::Move, 4> committed{Stockfish::MOVE_NONE, Stockfish::MOVE_NONE,
                                                 Stockfish::MOVE_NONE, Stockfish::MOVE_NONE};
    };
    /** What retime_team_tree did. */
    struct TeamTreeRetime {
        int pruned = 0;
        std::vector<std::shared_ptr<Node>> walked;
    };
    TeamTreeMatch find_team_tree_node(Board& board, Stockfish::Color team,
                                      const TimeState& live, int toleranceMs);
    TeamTreeRetime retime_team_tree(const std::shared_ptr<Node>& root, Board& board,
                                    const TimeState& rootTime, bool background,
                                    size_t maxRetained = SearchParams::TEAM_TREE_MAX_RETAINED);
    std::shared_ptr<Node> adopt_team_tree(Board& board, Stockfish::Color team,
                                          const SearchOptions& options);
    void store_team_tree(Board& board, Stockfish::Color team);
    void drop_team_tree(bool background);

    /**
     * Root pre-pass accounting.
     *
     * The winning-mate and reverse-loss scans run inside the move time.
     * Reverse proofs are also attached to
     * individual root actions, so a proof can prevent a blunder even when the
     * whole position is not proven lost. The cost and whole-root hit rate are
     * accumulated here so the trade can be read off real games.
     */
    struct RootScanStats {
        uint64_t searches = 0;
        uint64_t proofs = 0;
        uint64_t scanNanos = 0;
        uint64_t thinkNanos = 0;
    };
    RootScanStats rootScanStats_;
    mutable std::mutex internalMateProbeStatsMutex_;
    InternalMateProbeStats lastInternalMateProbeStats_;
    std::string root_scan_summary() const;
    
    // Garbage collection thread for async tree cleanup
    GCThread gcThread_;

    friend class AgentTreeReuseTestPeer;
    friend class AgentDurativeTestPeer;
    friend class AgentDurativePart2Peer;
    friend class AgentContinuityPeer;
    friend class AgentContinuityFix2Peer;
    friend class AgentContinuityFix3Peer;

    void ensure_worker_pool(size_t workerCount);
    void worker_loop(size_t workerIndex, uint64_t observedGeneration);
    void dispatch_workers(const Board& board,
                          const std::vector<Engine*>& engines,
                          SearchInfo& searchInfo,
                          size_t targetNodes,
                          int moveTimeMs,
                          size_t workerCount);
    void wait_for_workers();
    void reindex_reused_subtree(const std::shared_ptr<Node>& reusedRoot,
                                bool background);
    /**
     * @brief Hand a discarded tree or table to the GC threads.
     *
     * A foreground search is on the path to a move, so it never waits for a
     * backlog short of the hard cap. A background search waits for room, but
     * its stop flag ends the wait: the position it is stopped for is waiting
     * on this thread.
     */
    void discard(std::shared_ptr<void> item, bool background);
    void retain_reuse_candidates(const std::shared_ptr<Node>& subtreeRoot,
                                 Board& board,
                                 const TimeState& subtreeTime);
    void retain_reuse_level(const std::shared_ptr<Node>& node,
                            Board& board,
                            const TimeState& nodeTime,
                            int depth,
                            int level);
    bool try_reuse_mate_continuation(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        bool avoidRepetition = false) const;
    /** Replays only a line whose remaining moves the lead in @p time covers. */
    bool try_reuse_mate_continuation(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        JointActionCandidate& outAction, int& outPlyToMate,
        bool avoidRepetition = false) const;
    static std::string format_root_aware_uci_score(
        const std::shared_ptr<Node>& root,
        const std::shared_ptr<Node>& pvChild,
        float childQ,
        int childVisits,
        float centipawnScale = 180.0f,
        float tangentScale = 1.56f);

public:
    /**
     * @brief Constructs a multi-threaded Agent with MCGS support.
    * @param numThreads Search threads per engine (0 = use SearchParams::NUM_SEARCH_THREADS)
     */
    Agent(int numThreads = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Runs a UCI search.
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param side The side to move.
     * @param teamHasTimeAdvantage If true, team is ahead on time and can double-sit.
     * @param options Search options (stopping conditions, verbosity, noise).
     * @return The best joint action found.
     */
    JointActionCandidate run_search(Board& board, const std::vector<Engine*>& engines, 
                                    Stockfish::Color side, bool teamHasTimeAdvantage,
                                    const SearchOptions& options);

    /** Returns an immutable snapshot of the expanded root edges after search. */
    std::vector<RootEdgeStats> root_edge_stats() const;

    /**
     * @brief Keep the subtree a line of play from the last search's root leads to.
     *
     * Makes the node reached by `line` the only reuse candidate for the next
     * search, which is rooted where the line leads with the clock charged as
     * the search charged it. For a caller that chooses its own moves and ran
     * the search with SearchOptions::retainForReuse off. Self-play keeps one
     * agent per team and retains the agent's own move and the reply: the
     * tree's terminal verdicts (mate races, waiting-board mates) are judged
     * for its root team, so a tree is only ever reused by the same team.
     * @param board The position the last search was rooted at.
     * @return Visits the retained node holds; 0 when there is none to keep.
     */
    int retain_played_line(const Board& board,
                           const std::vector<JointActionCandidate>& line);

    /** Returns the evaluated Q-value of the root node after search. */
    float root_q() const;
    GameStatus search_status() const { return searchStatus_; }
    NodeType root_type() const { return rootNode ? rootNode->get_node_type() : NodeType::UNSOLVED; }
    bool certified_choice() const { return certifiedChoice_; }
    /** Fix 1: what TeamTreeReuse did at the start of the last search. */
    const TeamTreeReuseReport& team_tree_reuse() const { return lastTeamTreeReuse_; }
    /** Fix 3: what PlanCarry did at the start of the last search. */
    const PlanCarryReport& plan_carry() const { return lastPlanCarry_; }
    /** Fix 3: the remembered halves, one slot per board. */
    const std::array<CarriedPlan, 2>& carried_plans() const { return plans_; }
    /** Fix 3: forget both remembered halves (ucinewgame, a new game). */
    void clear_plans();
    /**
     * @brief Fix 3: replace both slots with the halves of @p action on the
     * boards in @p boards where @p team is on turn and the half is a move;
     * every other slot is cleared. For a caller that plays another action
     * than the search chose (self-play samples one) or only some of its
     * halves: it records the halves of seats whose decision is still to
     * come. Call after run_search, from @p board, the position it searched.
     */
    void record_plans(Board& board, Stockfish::Color team, const JointActionCandidate& action,
                      std::array<bool, 2> boards);

    /** Returns telemetry from the most recent internal mate-probe experiment. */
    InternalMateProbeStats internal_mate_probe_stats() const;

    /**
     * @brief Node and time budget for the root forced-mate search.
     *
     * Proof scans run alongside MCTS workers, with node caps and cancellation.
     * Running out is reported as "not proven", never as a proof.
     *
     * A node count alone cannot bound it in absolute time: a joint proof node
     * enumerates board-move combinations and costs two orders of magnitude
     * more than a node of the single-board check scan. Timed searches
     * therefore also set a deadline, sampled every few probes so the clock
     * read stays negligible next to the work it guards. Node-limited searches
     * have no deadline to give and use `cancelled` instead, raised from the
     * MCTS workers when the tree finishes its node target.
     */
    struct MateSearchBudget {
        using Clock = std::chrono::steady_clock;

        uint64_t remainingNodes = SearchParams::MATE_SEARCH_NODE_BUDGET;
        bool exhausted = false;
        Clock::time_point deadline{};
        uint32_t probesSinceTimeCheck = 0;
        const std::atomic<bool>* cancelled = nullptr;
        // The concurrent MCTS root stays alive until the scans have unwound.
        // A solved root makes further winning/reverse proofs unnecessary.
        const Node* stopOnSolvedRoot = nullptr;

        bool consume() {
            if (exhausted) {
                return false;
            }
            if (remainingNodes == 0) {
                exhausted = true;
                return false;
            }
            if (++probesSinceTimeCheck
                    >= SearchParams::MATE_SEARCH_TIME_CHECK_INTERVAL) {
                probesSinceTimeCheck = 0;
                if (out_of_time()) {
                    exhausted = true;
                    return false;
                }
            }
            --remainingNodes;
            return true;
        }

        /// Charge a whole sub-search to this budget in one go.
        ///
        /// A Fairy-Stockfish probe answers in its own node count rather than
        /// one probe at a time, so it is billed after the fact. The budget can
        /// go to zero but not below, and a probe that emptied it exhausts it.
        void consume_bulk(uint64_t nodes) {
            if (nodes >= remainingNodes) {
                remainingNodes = 0;
                exhausted = true;
                return;
            }
            remainingNodes -= nodes;
        }

        /// Stop condition for loops that do not consume node probes.
        bool out_of_time() const {
            return (cancelled
                    && cancelled->load(std::memory_order_relaxed))
                || (SearchParams::ENABLE_MATE_EARLY_EXIT && stopOnSolvedRoot
                    && stopOnSolvedRoot->get_node_type() != NodeType::UNSOLVED)
                || (deadline != Clock::time_point{}
                    && Clock::now() >= deadline);
        }
    };

    /**
     * @brief Searches for a forced checkmate on a single board where all attacker moves are checks.
     *
     * @param budget Optional node budget; the search aborts and returns false once it is spent.
     */
    static bool search_single_board_forced_mate(
        Board& board, int boardNum, Stockfish::Color attackerColor,
        int currentPly, int maxAttackerMoves,
        Stockfish::Move& outMove, int& outPlyToMate,
        MateSearchBudget* budget = nullptr,
        bool partnerBoardAgnostic = false);

    /**
     * @brief Performs checkmate detection at the root before starting MCTS.
     */
    static bool find_root_mate(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});
    /**
     * As above under a clock. Every proof charges the actions it plays, and
     * the shortcuts that sit the partner board for a whole line run only
     * while the lead covers every move of it.
     */
    static bool find_root_mate(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});

    /**
     * @brief Whether a joint action hands the opponents a mate they play first.
     *
     * Each board keeps its own clock, so a mate that is on the board is played
     * whenever its owner wants it - there is no turn to wait for. A team that
     * does not mate with this action, and leaves the opponents one that does,
     * never gets to make the second move its own mate needs: theirs lands
     * first. That race has no place in an alternating turn model, so a win
     * whose first move is such an action - a capture that feeds the mating
     * piece across is the usual shape - is reported as a mate in two when it
     * is a loss.
     *
     * The test is asked of the action, so a claimed win from a retained tree,
     * a cached continuation, a root scan or the probe is all held to it.
     * Answers false once @p budget is spent, so a veto only ever rests on a
     * mate this scan actually saw. Leaves @p board as it found it.
     */
    static bool action_loses_mate_race(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        MateSearchBudget* budget = nullptr);
    /** As above; the opponents answer under the clock @p action leaves. */
    static bool action_loses_mate_race(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, const TimeState& time,
        MateSearchBudget* budget = nullptr);

    /**
     * @brief Whether @p teamSide, on the boards it is on turn, has an action
     *        that ends the game at once under @p time - a mate, or a win by
     *        leaving the opponents no legal action.
     */
    static bool has_immediate_mate(Board& board, Stockfish::Color teamSide,
                                   const TimeState& time);

    enum class RaceVerdict { PROVEN, REFUTED, UNKNOWN };
    struct FrozenRaceStats {
        int targets = 0;
        int races = 0;
        int proven = 0;
        int unplayedWitnesses = 0;
        uint64_t nodes = 0;
    };
    /**
     * @brief The probe thread's frozen-race stage: the leading root actions
     *        and their leading replies, each checked for a frozen race and,
     *        where there is one, proven within the short-race bound.
     *
     * A proof marks that node lost for its team to play, or - when that team
     * wins - marks lost the child its proof plays, once the tree has generated
     * it, so the node is won by propagation and keeps its move. Nothing else
     * in the tree changes. Stops at @p budgetMs, when @p settled answers true,
     * or when the stage's node budget is spent.
     */
    static void run_frozen_race_stage(
        Board& rootBoard, const std::shared_ptr<Node>& root,
        Stockfish::Color rootTeam, const TimeState& rootTime, int budgetMs,
        const std::function<bool()>& settled,
        const std::atomic<bool>* cancelled, FrozenRaceStats& stats);
    /**
     * @brief Prove that @p attackingTeam wins the race of a frozen board.
     *
     * @p raceBoard is frozen against the defenders, or the attackers can
     * freeze it (after a feed, if need be). Every defender action is searched;
     * the attackers pass or check on @p raceBoard and play anything on the
     * other. REFUTED means only that this restricted strategy fails within
     * @p maxAttackerMoves actions.
     */
    static RaceVerdict prove_frozen_race(
        Board& board, Stockfish::Color attackingTeam, const TimeState& time,
        Stockfish::Color teamToPlay, int raceBoard, int maxAttackerMoves,
        MateSearchBudget& budget, int* outPlies = nullptr,
        std::string* outLine = nullptr,
        JointActionCandidate* outAction = nullptr);

    /**
     * @brief The action a solved node stands behind, empty when it claims no
     *        forced win.
     *
     * Only a claimed win is worth a race scan. An ordinary move was chosen
     * against the whole position already, and a losing race is exactly the
     * kind of thing an unproven score is free to notice for itself.
     */
    static std::optional<JointActionCandidate> claimed_win_action(
        const Node& node, float qVetoDelta, float qValueWeight,
        bool avoidSolvedDraw);

    /**
     * @brief A replacement for a claimed win that loses the mate race.
     *
     * Empty when @p node claims no win, when the win it claims is race-safe,
     * or when none of the alternatives the search visited is - the last of
     * those meaning the race is lost whatever this team plays, where the
     * claimed win is as good a try as any. Alternatives are ranked by the
     * visits the search spent on them, with children the solver already proved
     * lost ranked last, and only the first few are scanned.
     */
    static std::optional<JointActionCandidate> race_safe_alternative(
        Board& board, const Node& node, Stockfish::Color teamSide,
        bool teamHasTimeAdvantage, float qVetoDelta, float qValueWeight,
        bool avoidSolvedDraw,
        const std::atomic<bool>* cancelled = nullptr);
    static std::optional<JointActionCandidate> race_safe_alternative(
        Board& board, const Node& node, Stockfish::Color teamSide,
        const TimeState& time, float qVetoDelta, float qValueWeight,
        bool avoidSolvedDraw,
        const std::atomic<bool>* cancelled = nullptr);

    /**
     * @brief Exact joint forced-mate proof for @p attackingTeam to move.
     *
     * The full two-board AND/OR solver: checking actions on either board,
     * every defender reply, hands updated by each capture, sits by the clock
     * rules. Proves at most @p maxAttackerMoves attacker moves. With
     * @p shortestFirst it deepens from two and answers with the shortest
     * mate it holds; without, it searches the full depth at once, which is
     * far cheaper when any proof will do, as for a veto. Spent budget is
     * UNKNOWN, never a claim. Leaves @p board as it found it; the proof's
     * first action and length come back on success. A caller that keeps
     * @p cache between calls resumes from the subtrees earlier calls
     * settled; @p outRefutedAtBound reports that every depth up to the bound
     * was refuted outright, with budget and time to spare. @p treeNode, the
     * search tree's node for this position when it has one, supplies the
     * network priors that order the proof's moves; the proof follows the
     * tree down for as long as it has the positions.
     */
    static bool prove_joint_forced_mate(
        Board& board, Stockfish::Color attackingTeam,
        bool attackingTeamHasTimeAdvantage, int maxAttackerMoves,
        MateSearchBudget& budget, JointActionCandidate& outAction,
        int& outPlyToMate, std::vector<MateProofPly>* outLine = nullptr,
        bool shortestFirst = true,
        joint_mate::JointMateCache* cache = nullptr,
        bool* outRefutedAtBound = nullptr,
        const Node* treeNode = nullptr);
    /** As above under a clock, charged action by action down every line. */
    static bool prove_joint_forced_mate(
        Board& board, Stockfish::Color attackingTeam,
        const TimeState& time, int maxAttackerMoves,
        MateSearchBudget& budget, JointActionCandidate& outAction,
        int& outPlyToMate, std::vector<MateProofPly>* outLine = nullptr,
        bool shortestFirst = true,
        joint_mate::JointMateCache* cache = nullptr,
        bool* outRefutedAtBound = nullptr,
        const Node* treeNode = nullptr);

    /**
     * @brief One slice of verification for a root action.
     *
     * Plays @p action on @p board, then: a first slice asks Fairy for the
     * opponents' single-board mate and certifies a hit exactly; later slices
     * run the joint solver for up to @p sliceNodes with the verdict's own
     * cache. A proof or a refutation with budget to spare settles the
     * verdict; a slice cut short by nodes, @p deadline or @p cancelled leaves
     * it UNKNOWN. Nodes spent are added to @p nodesSpent. Leaves @p board as
     * it found it. Only meaningful when this team lacks the time advantage.
     * @p actionNode is the tree's child for the action, whose priors order
     * the joint proof.
     */
    static void verify_root_action_slice(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, RootActionVerdict& verdict,
        uint64_t sliceNodes, int probeMs,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled, const Node* stopOnSolvedRoot,
        uint64_t& nodesSpent, ConcurrentVerifierStats* stats = nullptr,
        const Node* actionNode = nullptr,
        int maxAttackerMoves =
            SearchParams::SELECTED_MOVE_JOINT_MAX_ATTACKER_MOVES);
    /** As above, with this team's clock @p time before the action. */
    static void verify_root_action_slice(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, const TimeState& time,
        RootActionVerdict& verdict,
        uint64_t sliceNodes, int probeMs,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled, const Node* stopOnSolvedRoot,
        uint64_t& nodesSpent, ConcurrentVerifierStats* stats = nullptr,
        const Node* actionNode = nullptr,
        int maxAttackerMoves =
            SearchParams::SELECTED_MOVE_JOINT_MAX_ATTACKER_MOVES);

    /**
     * @brief Whether @p action hands the opponents a proven forced mate.
     *
     * Plays the action on @p board, asks Fairy-Stockfish for the opponents'
     * mate from the position it leaves, and holds that line to the exact
     * two-board certifier. Answers true only on a certificate, with the
     * mate's length in @p outPlyToMate: a hit the certifier could not prove
     * before @p deadline, or no hit at all, is a no. Without the time
     * advantage the opponents cannot sit the other board through a line, so
     * ahead on time this team is never vetoed. Leaves @p board as it found it.
     */
    static bool action_walks_into_certified_mate(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled,
        int& outPlyToMate, SelectedMoveCertStats* stats = nullptr);
    /**
     * As above under a clock: skipped only while this team's lead outlasts
     * any line the opponents could be asked to prove, and their proofs run
     * under the clock @p action leaves them.
     */
    static bool action_walks_into_certified_mate(
        Board& board, const JointActionCandidate& action,
        Stockfish::Color teamSide, const TimeState& time,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled,
        int& outPlyToMate, SelectedMoveCertStats* stats = nullptr);

    /**
     * @brief A replacement for @p chosen when it walks into a certified mate.
     *
     * Empty when the chosen action survives the test, or when every
     * alternative tried fails it too, where the chosen action is as good a
     * try as any. Alternatives are the actions @p node visited, ranked by
     * visits with children the solver already proved lost ranked last, and
     * only the first few are tried before @p deadline. A child proven lost
     * here is marked so in the tree, where tree reuse keeps the proof.
     */
    static std::optional<JointActionCandidate> certified_mate_free_alternative(
        Board& board, const Node& node, const JointActionCandidate& chosen,
        Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled = nullptr,
        SelectedMoveCertStats* stats = nullptr);
    static std::optional<JointActionCandidate> certified_mate_free_alternative(
        Board& board, const Node& node, const JointActionCandidate& chosen,
        Stockfish::Color teamSide, const TimeState& time,
        MateSearchBudget::Clock::time_point deadline,
        const std::atomic<bool>* cancelled = nullptr,
        SelectedMoveCertStats* stats = nullptr);

    /**
     * @brief Whether playing @p move on @p boardNum reaches that board's
     *        position for the third time, ending the game as a draw.
     *
     * Used to keep an unproven mate score from being spent on a shuffle.
     * Leaves @p board as it found it.
     */
    static bool move_completes_repetition(
        Board& board, int boardNum, Stockfish::Move move);

    /**
     * @brief Bounded Fairy-Stockfish mate search for the root, board by board.
     *
     * Answers the question the checking-move scans cannot ask - a mate that
     * needs a quiet preparing move - and reports it as a strong candidate
     * rather than a proof: the move it names is checked against this board,
     * and the line replayed against the partner board.
     *
     * Only a team ahead on time may sit one board while the other plays a
     * mate out, so without @p teamHasTimeAdvantage this reports nothing.
     *
     * The boards are searched one after the other out of one budget, so a
     * caller running this beside its own search is told through @p onMate as
     * each mate lands rather than only when the last board's share is spent.
     * The out parameters hold the shortest mate found so far whenever it runs.
     */
    static bool probe_position_mate(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        uint64_t nodeBudget, int budgetMs,
        const std::function<bool()>& abort,
        JointActionCandidate& outAction, int& outPlyToMate,
        std::string& outPrincipalVariation,
        const std::function<void()>& onMate = {},
        bool avoidRepetition = false,
        uint64_t* outNodes = nullptr);
    /**
     * As above under a clock: the probe sits the partner board for the whole
     * line, so a mate is reported only when this team's lead covers all of
     * its moves.
     */
    static bool probe_position_mate(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        uint64_t nodeBudget, int budgetMs,
        const std::function<bool()>& abort,
        JointActionCandidate& outAction, int& outPlyToMate,
        std::string& outPrincipalVariation,
        const std::function<void()>& onMate = {},
        bool avoidRepetition = false,
        uint64_t* outNodes = nullptr);

    static bool probe_root_mate(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        uint64_t nodeBudget, int budgetMs,
        const std::function<bool()>& abort,
        JointActionCandidate& outAction, int& outPlyToMate,
        std::string& outPrincipalVariation,
        const std::function<void()>& onMate = {},
        bool avoidRepetition = false);

    /** The verdict of prove_joint_mate_exhaustive(). */
    struct ExhaustiveMateResult {
        enum class Status : uint8_t { PROVEN, NO_WIN, UNKNOWN };
        Status status = Status::UNKNOWN;
        /// Attacker actions against the longest defence, when proven.
        int attackerMoves = 0;
        /// Search nodes created.
        uint64_t nodes = 0;
        /// The first action, then the longest defence's line, when proven.
        std::vector<MateProofPly> line;
        std::string pv;
        /// The proof was walked again: every legal defence is in it and
        /// every leaf is mate.
        bool verified = false;
    };
    /**
     * @brief Offline mate oracle for labelling positions: a proof-number
     * search over every legal joint action of both teams under the clock.
     *
     * The in-search provers restrict the attackers to forcing actions; this
     * restricts neither team, so it also plays the attackers' quiet moves and
     * sits. Its rules are theirs - the clock, the joint-action rules, a
     * defender capture that feeds an immediate mate, the root's mate race -
     * except that repetitions are counted per board. PROVEN is a mate within
     * @p maxAttackerMoves attacker actions whatever the defenders do, NO_WIN
     * that there is none within them, UNKNOWN that @p maxNodes ran out first.
     */
    static ExhaustiveMateResult prove_joint_mate_exhaustive(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        int maxAttackerMoves, uint64_t maxNodes);

    /**
     * Proves that a specific Fairy candidate forces mate before publication.
     * @p treeNode, the tree's node for @p board when it has one, lets the
     * proof follow the tree and take its priors from the candidate's child.
     */
    static bool certify_mate_candidate(
        Board& board, Stockfish::Color teamSide,
        bool teamHasTimeAdvantage,
        const JointActionCandidate& candidate,
        int candidatePlyToMate,
        MateSearchBudget& budget,
        int& outPlyToMate,
        MateCertificateTier& outTier,
        const Node* treeNode = nullptr);
    /**
     * As above under a clock. The single-board and sat-board tiers need the
     * lead to cover every move of the line; the joint tier follows the clock,
     * so it may certify a team that is level or behind.
     *
     * A probe's candidate sits the board it does not move on. Where this
     * clock does not allow that sit, the candidate's move is tried with each
     * legal move on the other board instead, and @p outAction (when given)
     * receives the joint action that was certified.
     */
    static bool certify_mate_candidate(
        Board& board, Stockfish::Color teamSide,
        const TimeState& time,
        const JointActionCandidate& candidate,
        int candidatePlyToMate,
        MateSearchBudget& budget,
        int& outPlyToMate,
        MateCertificateTier& outTier,
        const Node* treeNode = nullptr,
        JointActionCandidate* outAction = nullptr,
        MateCertificateFailure* outFailure = nullptr);

    /**
     * @brief Proves that every legal root action permits a forced opponent mate.
     *
     * Returns the action that delays terminal loss longest. Exhausting the
     * bounded probe budget is reported conservatively as "not proven".
     */
    static bool find_root_forced_loss(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});
    static bool find_root_forced_loss(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});

    /**
     * @brief Proves losing root actions individually.
     *
     * Every proven action is stored in outProofs even when another legal
     * action refutes a whole-root loss. Returns true only when every legal root
     * action was proven losing; budget exhaustion remains conservative.
     */
    static bool find_root_loss_proofs(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        std::vector<RootLossProof>& outProofs,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {},
        const std::atomic<bool>* cancelled = nullptr,
        const std::vector<JointActionCandidate>* preferredActions = nullptr,
        Node* liveRoot = nullptr,
        bool scanWaitingChecks = true);
    /** As above under a clock; each defense is charged before it is answered. */
    static bool find_root_loss_proofs(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        std::vector<RootLossProof>& outProofs,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {},
        const std::atomic<bool>* cancelled = nullptr,
        const std::vector<JointActionCandidate>* preferredActions = nullptr,
        Node* liveRoot = nullptr,
        bool scanWaitingChecks = true,
        // Set when every defense was either proven lost or settled safe
        // within the budget, so a larger one would only repeat the answer.
        bool* outComplete = nullptr,
        // Called with each proof as soon as it is found.
        const std::function<void(const RootLossProof&)>& onProof = {},
        // Proofs an earlier call already found in this position. Their
        // defenses are not searched again and count toward a proven loss.
        const std::vector<RootLossProof>* knownProofs = nullptr,
        // Set when several threads search the defenses together, each
        // calling with its own board: they take defenses from the share's
        // queue and nodes from its pool (nodeBudget still sizes the focused
        // defenses). Each call returns the known proofs and those it found;
        // whether every defense loses is for the caller to tell from their
        // union.
        LossScanShare* share = nullptr,
        // Counts this call's certificate searches and the proofs they gave.
        LossScanStats* stats = nullptr);
    
    /**
     * @brief Extracts PV line starting from a specific child index.
     * @param board The current board position.
     * @param childIdx The child index to start the PV from.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv_from_child(
        Board& board, int childIdx, int maxDepth,
        Stockfish::Color rootTeam, const TimeState& rootTime);

    /**
     * @brief Extracts the best move from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the best joint move.
     */
    std::string extract_best_move(Board& board);

    /**
     * @brief Extracts the predicted opponent reply from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the predicted opponent joint move, or empty string.
     */
    std::string extract_ponder_move(Board& board);

    /**
     * @brief Extracts the principal variation (PV) by following most-visited children.
     * @param board The current board position.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv(Board& board, int maxDepth);

    /**
     * @brief Signals ponderhit from the UCI interface when the expected ponder move was played.
     * Switches search from ponder mode to active timed search, resetting search time.
     */
    void ponderhit();

    /**
     * @brief Checks if the agent is currently in ponder mode.
     * @return true if pondering, false otherwise.
     */
    bool is_pondering() const;

    /**
     * @brief Sets the running state of the agent.
     * @param value Boolean indicating whether the agent should be running.
     */
    void set_is_running(bool value);

    /**
     * @brief Checks if the agent is currently running.
     * @return true if running, false otherwise.
     */
    bool is_running();
    
    /**
     * @brief Set the hash table size in MB.
     * 
     * Resizes the transposition table used for MCGS.
     * @param sizeMB Size in megabytes (1 - 33554432)
     */
    void setHashSize(size_t sizeMB);

    /**
     * @brief Discard all search state retained between moves.
     */
    void reset_search_state();
    
    /**
     * @brief Exact positional identity of a board (both FENs, pockets included).
     *
     * Used alongside the search hash when adopting a retained subtree,
     * protecting against collisions and stale pocket contents.
     */
    static std::string board_signature(Board& board);

    /**
     * @brief Try to reuse the search tree from a previous search.
     *
     * Checks whether the current position matches a saved next-root candidate.
     * Under a clock the search hash includes the exact lead, so a subtree is
     * only adopted by a position whose clock is the one it was searched
     * under: every node's legal actions and proofs hold as they stand.
     * @param positionHash Search hash of the current position.
     * @param teamSide Team to play at the current position.
     * @param signature Exact board signature of the current position.
     * @return Shared pointer to reusable root, or nullptr if no reuse possible
     */
    std::shared_ptr<Node> try_reuse_tree(uint64_t positionHash,
                                         Stockfish::Color teamSide,
                                         const std::string& signature,
                                         bool background);
    
    /**
     * @brief Store next-root candidates for tree reuse.
     * 
     * Called after search completes to retain the selected child and every
     * generated opponent response beneath it.
     */
    void store_next_root_candidates(Board& board, bool teamHasTimeAdvantage);
    /** As above, from the clock state the root was searched under. */
    void store_next_root_candidates(Board& board, const TimeState& rootTime);

    /**
     * @brief Keeps searching between moves, from the position our own move creates.
     *
     * Runs a silent, unbounded search rooted at the position that follows
     * playedAction, so that work accrues under every opponent reply instead of
     * only the one ponder would have guessed. Returns when the next position,
     * go, or stop arrives, or when the background caps are reached. Exceptions
     * are contained here: a bestmove has already been emitted for this move.
     *
     * @param board The position the just-finished search was rooted at.
     * @param playedAction The joint action that search chose.
     */
    void run_permanent_brain(Board& board, const std::vector<Engine*>& engines,
                             Stockfish::Color teamSide,
                             bool teamHasTimeAdvantage,
                             const JointActionCandidate& playedAction,
                             const SearchOptions& options);

    /**
     * @brief Clear a stop latched by earlier commands.
     *
     * Every setoption and position stops the running search, and the stop
     * stays latched so a background search that has not yet dispatched still
     * sees it. A background search the front end starts deliberately must not
     * inherit that stop; a stop arriving after this latches again as usual.
     */
    void clear_stop_request();

    /**
     * @brief Search the opponents' replies from a position they are to move in.
     *
     * The permanent brain, rooted where the caller says rather than where this
     * engine's own move would lead. A front end that played a different move -
     * an opening book, a safety substitution - restarts it from the position
     * actually on the board; one that played this engine's move restarts it on
     * the same root, which the stopped search retained, so its tree carries
     * over. Returns when the next position, go, or stop arrives, or when the
     * background caps are reached. Emits no bestmove: no move is played here.
     *
     * @param board The position after our team's move, the opponents to play.
     * @param teamSide Our team, which is not on move at board.
     * @param teamHasTimeAdvantage Our team's time advantage.
     */
    void run_background_search(Board& board,
                               const std::vector<Engine*>& engines,
                               Stockfish::Color teamSide,
                               bool teamHasTimeAdvantage,
                               const SearchOptions& options);

    /** The options run_background_search searches the opponents' position with. */
    static SearchOptions background_search_options(const SearchOptions& options,
                                                   Stockfish::Color teamSide);

private:
    struct SingleBoardMateCacheEntry {
        Stockfish::Move move = Stockfish::MOVE_NONE;
        int plyToMate = 0;
        std::vector<MateProofPly> principalVariation;
    };
    using SingleBoardMateCache =
        std::unordered_map<std::string, SingleBoardMateCacheEntry>;

    static bool search_single_board_forced_mate_impl(
        Board& board, int boardNum, Stockfish::Color attackerColor,
        int currentPly, int maxAttackerMoves,
        Stockfish::Move& outMove, int& outPlyToMate,
        MateSearchBudget* budget,
        std::vector<MateContinuation>* continuations,
        bool partnerBoardAgnostic = false,
        std::vector<MateProofPly>* outPrincipalVariation = nullptr);
    static bool find_root_mate_impl(
        Board& board, Stockfish::Color teamSide, const TimeState& time,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget,
        std::vector<MateContinuation>* continuations,
        MateSearchBudget* hardBudget = nullptr,
        bool includeCaptureFeeds = true,
        MateSearchBudget::Clock::time_point deadline = {},
        bool includeImmediateMate = true,
        bool attackerWinsMateRace = false,
        std::vector<MateProofPly>* outPrincipalVariation = nullptr,
        const std::atomic<bool>* cancelled = nullptr,
        SingleBoardMateCache* singleBoardMateCache = nullptr,
        const Node* stopOnSolvedRoot = nullptr,
        bool stopAtFirstProof = false, // Background search publishes immediately.
        // Set when the scan settled every line it searched within its
        // budget, so a larger one would only repeat the same answer.
        bool* outComplete = nullptr,
        // Set when its deepest joint search was refuted only for want of
        // attacker moves, so a longer horizon could still find a mate.
        bool* outDepthLimited = nullptr);
    
};
