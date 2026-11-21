#pragma once

#include <array>
#include <functional>
#include <optional>
#include <unordered_set>

#include "environment/board.h"
#include "environment/joint_action.h"
#include "environment/planes.h"
#include "nn/engine.h"
#include "search/node.h"
#include "search/mate_candidate.h"
#include "search/searchinfo.h"
#include "search/transposition_table.h"

using namespace std;

class Node;

enum class TerminalOutcome : uint8_t {
    NONE,
    WIN,
    LOSS,
    DRAW,
};

/** One representative reply-and-mate suffix for a waiting-board mate proof. */
struct WaitingMateContinuation {
    int activeBoard = -1;
    Stockfish::Move reply = Stockfish::MOVE_NONE;
    int waitingBoard = -1;
    Stockfish::Move matingMove = Stockfish::MOVE_NONE;
};

/**
 * @param partnerBoardAgnostic Classify without reading the waiting board:
 *        assume every interposable check can be blocked with a piece from the
 *        partner, and skip the waiting-board mate. Both only withdraw terminal
 *        verdicts, so a proof built on this model also holds for any other
 *        state of the waiting board - which lets a caller reuse one proof
 *        across positions that differ only there.
 * @param allowMatedTeamToMove At a live down-time root, defer a LOSS when the
 *        nominally mated team still has a legal move on its partner board. This
 *        permits one neural expansion to rank that seat's moves; descendants
 *        use normal terminal adjudication.
 */
TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             const std::array<int, 2>& boardSearchPlies,
                                             int* endInPly = nullptr,
                                             bool partnerBoardAgnostic = false,
                                             bool allowMatedTeamToMove = false,
                                             WaitingMateContinuation* waitingMate = nullptr);

TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             int searchPly,
                                             int* endInPly = nullptr,
                                             bool partnerBoardAgnostic = false,
                                             bool allowMatedTeamToMove = false,
                                             WaitingMateContinuation* waitingMate = nullptr);

/**
 * @brief Classify a position reached in search under its own clock state.
 *
 * @param time The clock state at this position, after every action that led
 *        to it has been charged. Who may sit is read from it for both teams,
 *        so neither team's right is derived by negating the other's.
 */
TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             const TimeState& time,
                                             const std::array<int, 2>& boardSearchPlies,
                                             int* endInPly = nullptr,
                                             bool partnerBoardAgnostic = false,
                                             bool allowMatedTeamToMove = false,
                                             WaitingMateContinuation* waitingMate = nullptr);

/**
 * @brief Whether @p team is frozen on @p boardNum: on turn there, in a check
 *        no dropped piece can block, and every legal move it has there is
 *        answered by an immediate mate that holds whatever the other board
 *        supplies. Nothing will be captured on that board again.
 */
bool board_is_frozen(Board& board, int boardNum, Stockfish::Color team);

/**
 * @brief A move by @p team, on turn on @p boardNum, that freezes its opponents
 *        there, or MOVE_NONE.
 */
Stockfish::Move freezing_move(Board& board, int boardNum, Stockfish::Color team);

/**
 * @brief The board a frozen race would be decided on, or -1.
 *
 * A board where the team on turn is frozen, or can freeze its opponents - at
 * once, or after a capture on the other board hands it the piece. Sets
 * @p outAttackers to the team the race would be won by. Detection only: the
 * race itself is for Agent::prove_frozen_race(). Answers -1 once @p abort
 * does.
 */
int frozen_race_board(Board& board, Stockfish::Color& outAttackers,
                      const std::function<bool()>& abort = {});

/**
 * @brief Whether @p teamToPlay is lost to a frozen board: frozen there, unable
 *        to sit (no lead, and no capture on the other board to pass with), so
 *        every action it has moves on the frozen board and is mated at once -
 *        unless the action's other move mates or draws first, which is checked.
 *
 * Exact under @p time. The tree marks such a leaf lost; the mate provers,
 * which report shortest lines, do not use it.
 */
bool frozen_board_loss(Board& board, Stockfish::Color teamToPlay, const TimeState& time);

/**
 * @brief Which boards a leaf of @p team acts on, as collect_batch expands it:
 * the boards it is on turn on, under a durative clock only those whose seat
 * lands now; neither is a forced wait (the single action pass, pass).
 */
struct LeafTurn {
    bool boardAOnTurn = false;
    bool boardBOnTurn = false;
    bool forcedWait = false;
};
LeafTurn leaf_turn(Board& board, Stockfish::Color team, const TimeState& time);

/**
 * @brief Diagnosis fix D3 (WaitNodeActingEval): the team whose planes a leaf
 * of @p team is evaluated from. The network's value says "the team to play
 * acts next", but at a forced wait the other team does; with the option a
 * wait leaf is evaluated from that team (~@p team) and the value negated
 * (process_batch). Otherwise, and off a durative clock, @p team.
 */
inline Stockfish::Color wait_node_planes_team(Stockfish::Color team, bool forcedWait,
                                              const TimeState& time) {
    return forcedWait && time.wait_node_acting_eval() ? ~team : team;
}

/**
 * @brief The actions a leaf gets on one board, as collect_batch expands it:
 * the representable legal moves when @p onTurn, without the drops of pieces
 * that landed during the seat's current decision (InstantTransfer off), and
 * always a trailing pass (MOVE_NONE).
 */
vector<Stockfish::Move> leaf_board_actions(Board& board, int boardNumber, bool onTurn,
                                           const TimeState& time);

/**
 * @brief Wave 2c, fix 1 (TeamTreeReuse): a hash of everything besides the
 * board that fixed a durative node's expansion, so a retained node re-timed
 * to new clocks can tell whether it would still be expanded the same way:
 * the acting boards after the durative filter, a forced wait, whether the
 * team may sit (the generator's pass rule), the drops held back on the
 * acting boards (InstantTransfer off), and the time inputs of the terminal
 * check (the other team's sit right, a clock result, and with
 * MateRaceByLanding the landing order and early landings). Never 0 for a
 * durative state, so a node expanded without a key (0) never matches; 0
 * outside a durative clock.
 */
uint32_t durative_expansion_key(Board& board, Stockfish::Color team, const TimeState& time);

/**
 * @brief The durative clock after @p team plays @p action from @p time, as
 * the search charges it: the applied halves move, the pieces they capture
 * reach the partners' pockets, and a planned half (fix 2) becomes that
 * seat's committed move on the board planned_boards() names. @p board is
 * the position before the action. Every walk that recomputes path times
 * (selection, fix 1's descent and re-timing, tests) goes through this.
 */
TimeState durative_time_after(const TimeState& time, Board& board, Stockfish::Color team,
                              const JointActionCandidate& action);

/**
 * @brief Wave 2c, fix 2 (JointPlans): the planned half of a planning leaf.
 *
 * @p actions are the planned seat's representable legal moves on @p board
 * (held-back drops removed as on an acting board), then drops of each type
 * one of @p immediate's captures would send it that it cannot drop now
 * (tagged in @p neededTypes; offered only where the transfer rule lets the
 * seat drop it at its landing: with InstantTransfer, or for a sitting seat),
 * then "decide later" (MOVE_NONE). @p mates flags each move that mates on
 * its board as it lands (checked only for checking moves, with the needed
 * piece in the pocket). @p suppliedTypes is, per entry of @p immediate, the
 * type its capture sends (0 = none).
 */
struct PlannedHalf {
    vector<Stockfish::Move> actions;
    vector<uint8_t> neededTypes;
    vector<uint8_t> mates;
    vector<uint8_t> suppliedTypes;
};
PlannedHalf leaf_planned_half(Board& board, int plannedBoard, Stockfish::Color team,
                              const TimeState& time, int immediateBoard,
                              const vector<Stockfish::Move>& immediate);

/**
 * @brief Fix 2: the planned half's priors, in place. @p priors is the
 * network's policy over @p actions (MOVE_NONE = its pass logit). "Decide
 * later" gets max(its pass prior, @p freeShare) and the moves share the rest
 * in proportion; each move flagged in @p mates is then raised to at least
 * SearchParams::JOINT_PLAN_MATE_PRIOR and the half renormalised.
 */
void shape_planned_priors(vector<float>& priors, const vector<Stockfish::Move>& actions,
                          const vector<uint8_t>& mates, float freeShare);

/**
 * @brief Entry in an MCTS selection trajectory.
 * 
 * Stores the node, action taken, and selected child index for backup.
 */
struct TrajectoryEntry {
    Node* node;
    std::shared_ptr<Node> nodeOwner;
    JointActionCandidate action;
    int selectedChildIdx;  // Index of the child that was selected (-1 for root/leaf)
    
    TrajectoryEntry(Node* n, const JointActionCandidate& a, int idx = -1)
        : node(n), action(a), selectedChildIdx(idx) {}

    TrajectoryEntry(std::shared_ptr<Node> n, const JointActionCandidate& a, int idx = -1)
        : node(n.get()), nodeOwner(std::move(n)), action(a), selectedChildIdx(idx) {}
};

/**
 * @brief Stores context for a single leaf node in a minibatch.
 * 
 * Used to track the trajectory, board state, and leaf node for each
 * position in the inference batch so we can properly process results.
 */
struct LeafContext {
    std::shared_ptr<Node> leaf;
    vector<TrajectoryEntry> trajectory;
    vector<Stockfish::Move> actionsA;
    vector<Stockfish::Move> actionsB;
    vector<uint8_t> capturesA;
    vector<uint8_t> capturesB;
    vector<int> policyIndicesA;
    vector<int> policyIndicesB;
    Stockfish::Color teamToPlay;
    TimeState time;      // Clock state at the leaf, every action on the path charged
    bool boardAOnTurn;
    bool boardBOnTurn;
    // Durative clock: the other team's seat lands first, so this team only
    // waits; its node gets the single action (pass, pass).
    bool forcedWait = false;
    // Wave 2c, fix 1: durative_expansion_key() of this leaf (0 outside a
    // durative clock), stored in the node at its expansion.
    uint32_t expansionKey = 0;
    // Wave 2c, fix 2 (JointPlans): the board whose list (actionsA or
    // actionsB) is a planned half, -1 = none; per action the needed type
    // (planned board) or supplied type (other board); and which planned
    // moves mate as they land. A board whose seat lands a committed move has
    // that move as its only action (forced half).
    int plannedBoard = -1;
    vector<uint8_t> handTypesA;
    vector<uint8_t> handTypesB;
    vector<uint8_t> plannedMates;
    bool isTerminal;     // True if this is a terminal node (draw/checkmate)
    bool hasEvaluationReservation;
    float terminalValue; // Value to use for terminal nodes
    bool isTransposition; // True if this leaf was found via transposition table (MCGS)
    uint64_t leafHash;    // Hash of the leaf position for transposition lookup
    
    LeafContext() : leaf(nullptr), teamToPlay(Stockfish::WHITE),
                    boardAOnTurn(false), boardBOnTurn(false),
                    isTerminal(false), hasEvaluationReservation(false),
                    terminalValue(0.0f),
                    isTransposition(false), leafHash(0) {}
    
    // Move constructor
    LeafContext(LeafContext&& other) noexcept = default;
    LeafContext& operator=(LeafContext&& other) noexcept = default;
    
    // Delete copy operations
    LeafContext(const LeafContext&) = delete;
    LeafContext& operator=(const LeafContext&) = delete;
};

struct LeafSelection {
    std::shared_ptr<Node> leaf;
    bool hasEvaluationReservation = false;
    std::shared_ptr<Node> pendingEvaluation;
    std::shared_ptr<Node> exhaustedSubtree;
    // Clock state at the selected leaf, every action on the path charged.
    TimeState time;
};

class SearchThread {
public:
    /**
     * Fix 2's invariant: at every leaf of a durative search the board and
     * the clock agree on who is to move (board.side_to_move(b) ==
     * time.side_on(b)); a planned move made early, or a committed one never
     * made, breaks it at once. Asserted in debug builds; with
     * checkDurativeSides on (tests) every leaf counts its violations.
     */
    static std::atomic<bool> checkDurativeSides;
    static std::atomic<int> durativeSideMismatches;

private: 
    struct SearchBatch {
        __half* observations = nullptr;
        // Wave 2b: the extra planes' values per slot, for a wide network.
        vector<float> wideExtras;
        vector<LeafContext> contexts;
        int validInferenceCount = 0;
        int sameBatchCollisions = 0;
        int reservationCollisions = 0;
    };

    struct CanonicalChildResult {
        bool expanded = false;
        std::shared_ptr<Node> pendingEvaluation;
    };

    Node* root;
    std::weak_ptr<Node> rootOwner;
    SearchInfo* searchInfo;
    TranspositionTable* transpositionTable;  // Shared across all search threads (MCGS)
    MateCandidateTable* mateCandidateTable;
    SearchParams::RuntimeConfig runtimeConfig;
    // The team this search's terminal verdicts are judged for: the
    // waiting-board mate is held to its race only against its opponents, and
    // a draw is scored with contempt for it. set_root_node makes it the
    // root's team; see SearchOptions::verdictTeam. Unset, a path is judged
    // for the team at its first node.
    std::optional<Stockfish::Color> verdictTeam;
    
    // Trajectory stores entries for backup and move undoing
    vector<TrajectoryEntry> trajectoryBuffer;
    
    // Double-buffered batches allow leaf collection to overlap GPU inference.
    array<SearchBatch, 2> batches;
    int pendingBatchIndex = -1;
    
    // Current batch size (initialized on first run_iteration call)
    int currentBatchSize = 0;
    // Whether this thread's engine runs a wave 2b wide network: collect_batch
    // then fills each batch's wideExtras.
    bool wideInputs = false;
    size_t inferenceWorkerIndex = 0;
    
    // Allocate/reallocate buffers for given batch size
    void ensureBufferSize(int batchSize);
    /** Waits on a pending evaluation, bounded by the move deadline. */
    bool wait_for_pending_evaluation(Node& node);

    void collect_batch(SearchBatch& batch, Board& board,
                       const TimeState& rootTime, bool allowReservationWait);
    void process_batch(SearchBatch& batch,
                       const Engine::HalfInferenceOutputs& inferenceOutputs);
    void abort_batch(SearchBatch& batch);
    CanonicalChildResult canonicalize_child(
        Board& board,
        Node* parent,
        int childIdx,
        const JointActionCandidate& action,
        shared_ptr<Node>& child,
        bool& hasEvaluationReservation,
        const TimeState& childTime);

public: 
    SearchThread();
    ~SearchThread(); 

    void set_search_info(SearchInfo* info);
    /**
     * @brief Drop this worker's references into the tree it just searched.
     *
     * The trajectory buffer and batch contexts hold shared_ptrs to the last
     * iteration's nodes, the root among them. Left in place they outlive the
     * agent's own reference, so the GC thread's release is not the last one
     * and the next search frees the old tree here instead, synchronously,
     * when it first clears them: half a second or more on a large tree, and
     * a stop waits for all of it.
     */
    void release_tree_references();
    void set_root_node(const std::shared_ptr<Node>& node);
    void set_transposition_table(TranspositionTable* table);
    void set_mate_candidate_table(MateCandidateTable* table);
    void set_runtime_config(const SearchParams::RuntimeConfig& config);
    void set_verdict_team(Stockfish::Color team);
    void set_inference_worker_index(size_t workerIndex);
    
    // MCGS (Monte Carlo Graph Search) with prior-ordered joint action expansion.
    // @param rootTime Clock state at the root; each action on the selected
    //        path is charged to it on the way down.
    LeafSelection select_and_expand(
        Board& board,
        const TimeState& rootTime,
        const std::unordered_set<const Node*>* blockedNodes = nullptr);
    /** Legacy form: the root team's permanent TimeAdvantage flag. */
    LeafSelection select_and_expand(
        Board& board,
        bool teamHasTimeAdvantage,
        const std::unordered_set<const Node*>* blockedNodes = nullptr);
    void expand_leaf_node(Node* leaf, 
                          const vector<Stockfish::Move>& actionsA,
                          const vector<Stockfish::Move>& actionsB,
                          const vector<float>& priorsA,
                          const vector<float>& priorsB,
                          bool teamHasTimeAdvantage,
                          bool boardAOnTurn,
                          bool boardBOnTurn,
                          const vector<uint8_t>& capturesA,
                          const vector<uint8_t>& capturesB,
                          const vector<float>& jointFactorsA = {},
                          const vector<float>& jointFactorsB = {},
                          size_t jointFactorRank = 0,
                          uint64_t positionHash = 0,
                          uint32_t expansionKey = 0,
                          int plannedBoard = -1,
                          const vector<uint8_t>& handTypesA = {},
                          const vector<uint8_t>& handTypesB = {});
    void backup(vector<TrajectoryEntry>& trajectory, float value);
    void cancel_virtual_losses(const vector<TrajectoryEntry>& trajectory);
    
    // Minibatch MCGS - collects SearchParams::BATCH_SIZE leaves, runs batched inference, processes results
    void run_iteration(Board& board, Engine* engine, const TimeState& rootTime,
                       const std::atomic<bool>* keepRunning = nullptr);
    void finish_pending_iteration(Board& board, Engine* engine,
                                  const TimeState& rootTime);
    void discard_pending_iteration(Board& board, Engine* engine);
};
