#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/time_state.h"
#include "interface/uci.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"
#include "search/searchthread.h"
#include "search/transposition_table.h"

// Wave 2c, fix 1: team tree reuse (docs/custom_engine_spec_v5_continuity.md,
// acceptance tests 0.1 and 1.x). A durative search keeps its whole tree; the
// team's next search finds the node the game reached, re-times it to the real
// clocks and searches on from there.

class AgentContinuityPeer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
    static std::shared_ptr<Node> team_tree_root(Agent& agent) {
        return agent.teamTree_ ? agent.teamTree_->root : nullptr;
    }
    static size_t root_candidates(Agent& agent) { return agent.nextRootCandidates_.size(); }
    static size_t mate_continuations(Agent& agent) { return agent.mateContinuations_.size(); }
    static void plant_mate_continuation(Agent& agent) {
        agent.mateContinuations_.push_back({});
    }
    static auto retime(Agent& agent, const std::shared_ptr<Node>& root, Board& board,
                       const TimeState& time,
                       size_t maxRetained = SearchParams::TEAM_TREE_MAX_RETAINED) {
        return agent.retime_team_tree(root, board, time, false, maxRetained);
    }
    /** Drop every reference the agent holds to its last tree. */
    static void release_tree(Agent& agent) {
        agent.reset_search_state();
    }
    static size_t gc_pending(Agent& agent) { return agent.gcThread_.pending_count(); }
};

class UCIContinuityPeer {
public:
    static Agent& agent(UCI& uci) { return *uci.agent; }
    static std::optional<TimeState> clock(const UCI& uci) { return uci.current_clock(); }
    static bool searching(const UCI& uci) { return uci.ongoingSearch.load(); }
};

namespace {

using Stockfish::BLACK;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

// Few legal moves on each board, so a few hundred nodes reach several plies.
constexpr const char* kSmallWhite = "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1";
constexpr const char* kSmallBlack = "4k3/p7/8/8/8/8/P7/4K3[] b - - 0 1";

struct Clocks {
    std::array<int32_t, 4> clocks{60000, 60000, 60000, 60000};
    std::array<int32_t, 4> paces{100, 100, 100, 100};
    std::array<int32_t, 4> elapsed{0, 0, 0, 0};
    std::array<bool, 4> sitting{false, false, false, false};
    int32_t rootMoveMs = 100;
    bool flagLoss = true;
    bool instantTransfer = false;
    bool sitGate = false;
};

/** A live root as UCI builds it (current_clock): @p team decides now. */
TimeState live_clock(Board& board, Stockfish::Color team, const Clocks& c) {
    DurativeConfig config;
    config.flagLoss = c.flagLoss;
    config.instantTransfer = c.instantTransfer;
    config.rootMoveTimeMs = c.rootMoveMs;
    config.seatClocksMs = c.clocks;
    config.seatPaceMs = c.paces;
    config.seatElapsedMs = c.elapsed;
    config.seatSitting = c.sitting;
    const TimeState base = TimeState::clocked(team, 0, 100, 100).with_sit_gate(c.sitGate);
    return TimeState::durative(base, team, board.side_to_move(BOARD_A),
                               board.side_to_move(BOARD_B), config);
}

/**
 * The live root a front end would send at a tree node whose path time is
 * @p path: the same clocks plus @p offsets, the same paces, and each of the
 * other team's seats on turn with the elapsed time that gives it the path's
 * ready time.
 */
TimeState live_at(Board& board, Stockfish::Color team, const TimeState& path,
                  const Clocks& base, std::array<int32_t, 4> offsets = {0, 0, 0, 0},
                  int32_t readyShiftMs = 0) {
    Clocks c = base;
    for (int seat = 0; seat < 4; ++seat) {
        c.clocks[seat] = path.seat_clock_ms(seat) + offsets[seat];
        c.elapsed[seat] = 0;
        c.sitting[seat] = false;
    }
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        const int seat = TimeState::seat_of(boardNumber, board.side_to_move(boardNumber));
        if (TimeState::team_of_seat(seat) != team) {
            c.elapsed[seat] = c.paces[seat] - path.seat_ready_ms(seat) - readyShiftMs;
            c.sitting[seat] = path.seat_sitting(seat);
        }
    }
    return live_clock(board, team, c);
}

TimeState after(const TimeState& time, Board& board, Stockfish::Color team,
                const JointActionCandidate& action) {
    return time.after_action(team, action.moveA != Stockfish::MOVE_NONE,
                             action.moveB != Stockfish::MOVE_NONE,
                             board.hand_piece_type(BOARD_A, action.moveA),
                             board.hand_piece_type(BOARD_B, action.moveB));
}

/** One step down a searched tree: its most visited child whose action passes @p accept. */
struct Step {
    std::shared_ptr<Node> child;
    JointActionCandidate action;
};
Step most_visited(Node& node, const std::function<bool(const JointActionCandidate&)>& accept) {
    Step best;
    const std::vector<int> visits = node.get_child_visits();
    const std::vector<std::shared_ptr<Node>> children = node.get_children();
    int most = -1;
    for (size_t i = 0; i < children.size() && i < visits.size(); ++i) {
        const JointActionCandidate action = node.get_joint_action(static_cast<int>(i));
        if (children[i] && children[i]->is_expanded()
            && children[i]->get_node_type() == NodeType::UNSOLVED
            && accept(action) && visits[i] > most) {
            most = visits[i];
            best = {children[i], action};
        }
    }
    return best;
}

bool moves_on(const JointActionCandidate& action, int boardNumber) {
    return (boardNumber == BOARD_A ? action.moveA : action.moveB) != Stockfish::MOVE_NONE;
}

/** A walk down a searched tree that keeps the board and the path time. */
struct Line {
    Board board;
    TimeState time;
    std::shared_ptr<Node> node;

    /** Follow the most visited child moving exactly on @p boards (none: a wait). */
    bool step(std::array<bool, 2> boards) {
        const Step next = most_visited(*node, [&](const JointActionCandidate& action) {
            return moves_on(action, BOARD_A) == boards[0]
                && moves_on(action, BOARD_B) == boards[1];
        });
        if (!next.child) {
            return false;
        }
        time = after(time, board, node->get_team_to_play(), next.action);
        board.make_moves(next.action.moveA, next.action.moveB);
        node = next.child;
        return true;
    }
};

/** The position a FEN-only front end (the arena) sends for @p board. */
Board from_fens(const Board& board) {
    Board live;
    live.set(board.fen(BOARD_A) + " | " + board.fen(BOARD_B));
    return live;
}

/** Every node reachable from @p root, each once. */
std::vector<Node*> reachable(Node* root) {
    std::vector<Node*> nodes;
    std::unordered_set<Node*> seen;
    std::vector<Node*> stack = {root};
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        if (!node || !seen.insert(node).second) {
            continue;
        }
        nodes.push_back(node);
        for (const std::shared_ptr<Node>& child : node->get_children()) {
            stack.push_back(child.get());
        }
    }
    return nodes;
}

/**
 * Walks the tree from @p root at @p time as the search charges it and calls
 * @p onNode for every node, each once (the first path that reaches it).
 */
void walk_once(Node* root, Board& board, const TimeState& time,
               const std::function<void(Node*, Board&, const TimeState&)>& onNode) {
    std::unordered_set<const Node*> seen;
    std::function<void(Node*, const TimeState&)> visit = [&](Node* node, const TimeState& at) {
        if (!node || !seen.insert(node).second) {
            return;
        }
        onNode(node, board, at);
        if (!node->is_expanded()) {
            return;
        }
        const std::vector<std::shared_ptr<Node>> children = node->get_children();
        for (size_t i = 0; i < children.size(); ++i) {
            const JointActionCandidate action = node->get_joint_action(static_cast<int>(i));
            const TimeState next = after(at, board, node->get_team_to_play(), action);
            board.make_moves(action.moveA, action.moveB);
            visit(children[i].get(), next);
            board.unmake_moves(action.moveA, action.moveB);
        }
    };
    visit(root, time);
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class TeamTreeReuse : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        engine = std::make_unique<Engine>(0, 1);
        const auto model = std::filesystem::path(__FILE__).parent_path()
            / "data" / "uniform.onnx";
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    }

    static SearchOptions options(const TimeState& clock, size_t nodes, bool teamTree,
                                 bool verbose = false) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.verbose = verbose;
        opts.clock = clock;
        opts.teamTreeReuse = teamTree;
        // The time-boxed root scans depend on wall time, not on the tree.
        opts.search.enableRootMateSearch = false;
        opts.search.certifySelectedMove = false;
        opts.search.enableMateProbe = false;
        return opts;
    }

    /** Runs one search, returning what it printed (verbose) or "". */
    std::string search(Agent& agent, Board& board, Stockfish::Color team,
                       const SearchOptions& opts) {
        testing::internal::CaptureStdout();
        agent.run_search(board, {engine.get()}, team, false, opts);
        return testing::internal::GetCapturedStdout();
    }

    std::unique_ptr<Engine> engine;
};

/**
 * The common sequence of 1.1-1.3 and 1.17: white decides on board A (Bw on B
 * lands late), its move lands, black's Ab replies on A; white's next search
 * comes at that position. Returns the tree's node for it (the most visited
 * line), the path board and the path time.
 */
struct Reached {
    std::optional<Line> line;
    Clocks clocks;
    Board root;
};
Reached white_then_black_reply(TeamTreeReuse& fixture, Agent& agent, bool teamTree,
                               size_t nodes = 800) {
    Reached reached;
    reached.root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    reached.clocks.paces = {100, 100, 5000, 100};
    const TimeState t0 = live_clock(reached.root, WHITE, reached.clocks);
    fixture.search(agent, reached.root, WHITE,
                   TeamTreeReuse::options(t0, nodes, teamTree));
    reached.line.emplace(Line{reached.root, t0, AgentContinuityPeer::root(agent)});
    return reached;
}

// ---- 0.1 -------------------------------------------------------------------

TEST(ContinuityDefaults, TeamTreeReuseOptionsAdvertisedAndOff) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);
    EXPECT_NE(output.str().find("option name TeamTreeReuse type check default false"),
              std::string::npos);
    EXPECT_NE(output.str().find(
                  "option name TeamTreeReuseToleranceMs type spin default 250 min 0 max 5000"),
              std::string::npos);
    const SearchOptions defaults;
    EXPECT_FALSE(defaults.teamTreeReuse);
    EXPECT_EQ(defaults.teamTreeReuseToleranceMs, 250);
    // The option does not enter the clock: current_clock() is the same either way.
    std::istringstream durative("name DurativeTree value true");
    uci.setoption(durative);
    std::istringstream model("name TimeModel value clock");
    uci.setoption(model);
    const std::optional<TimeState> before = UCIContinuityPeer::clock(uci);
    std::istringstream on("name TeamTreeReuse value true");
    uci.setoption(on);
    const std::optional<TimeState> afterOn = UCIContinuityPeer::clock(uci);
    ASSERT_TRUE(before && afterOn);
    EXPECT_EQ(*before, *afterOn);
    EXPECT_EQ(before->exact_key(), afterOn->exact_key());
}

TEST(ContinuityDefaults, ExpansionKeyIsZeroOutsideADurativeClock) {
    init_fairy_stockfish();
    Board board;
    EXPECT_EQ(durative_expansion_key(board, WHITE, TimeState::legacy(WHITE, true)), 0u);
    EXPECT_EQ(durative_expansion_key(board, WHITE, TimeState::clocked(WHITE, 500)), 0u);
    const TimeState durative = live_clock(board, WHITE, Clocks{});
    EXPECT_NE(durative_expansion_key(board, WHITE, durative), 0u);
}

// ---- 1.1 - 1.4 ----------------------------------------------------------------

TEST_F(TeamTreeReuse, ExactPathStillMissesUnderSeatClock) {
    Agent agent(1);
    Reached reached = white_then_black_reply(*this, agent, false);
    Line& line = *reached.line;
    ASSERT_TRUE(line.step({true, false}));   // white's root action on A
    ASSERT_TRUE(line.step({true, false}));   // Ab's reply
    const std::shared_ptr<Node> expected = line.node;
    Board live = from_fens(line.board);
    const TimeState clock = live_at(live, WHITE, line.time, reached.clocks, {3, 5, 5, 2});
    const std::string out = search(agent, live, WHITE, options(clock, 64, false, true));
    EXPECT_NE(AgentContinuityPeer::root(agent), expected);
    EXPECT_EQ(out.find("info string Tree reuse:"), std::string::npos) << out;
    EXPECT_EQ(out.find("Team tree reuse"), std::string::npos);
    EXPECT_FALSE(agent.team_tree_reuse().attempted);
}

TEST_F(TeamTreeReuse, AdoptsAfterLandingWithClocksAFewMsOff) {
    Agent agent(1);
    Reached reached = white_then_black_reply(*this, agent, true);
    Line& line = *reached.line;
    ASSERT_TRUE(line.step({true, false}));
    ASSERT_TRUE(line.step({true, false}));
    const std::shared_ptr<Node> expected = line.node;
    ASSERT_EQ(expected->get_team_to_play(), WHITE);
    Board live = from_fens(line.board);
    const TimeState clock = live_at(live, WHITE, line.time, reached.clocks, {3, 5, 7, 2});
    const std::string out = search(agent, live, WHITE, options(clock, 300, true, true));
    const TeamTreeReuseReport& reuse = agent.team_tree_reuse();
    ASSERT_TRUE(reuse.adopted) << reuse.reason << "\n" << out;
    EXPECT_EQ(AgentContinuityPeer::root(agent), expected);
    EXPECT_EQ(reuse.clockDiffMs, 7);
    EXPECT_GT(reuse.visits, 0);
    EXPECT_NE(out.find("info string Team tree reuse: " + std::to_string(reuse.visits)
                       + " visits recovered, clock diff 7 ms"),
              std::string::npos) << out;
    EXPECT_EQ(out.find("info string Tree reuse:"), std::string::npos);
    // 300 new nodes on top of what was recovered (one batch of slack).
    const int visits = AgentContinuityPeer::root(agent)->get_visits();
    EXPECT_GE(visits, reuse.visits + 300);
    EXPECT_LE(visits, reuse.visits + 302);
}

TEST_F(TeamTreeReuse, RefusesBeyondTolerance) {
    Agent agent(1);
    Reached reached = white_then_black_reply(*this, agent, true);
    Line& line = *reached.line;
    ASSERT_TRUE(line.step({true, false}));
    ASSERT_TRUE(line.step({true, false}));
    const std::shared_ptr<Node> expected = line.node;
    Board live = from_fens(line.board);
    SearchOptions opts;
    const int tolerance = opts.teamTreeReuseToleranceMs;
    const TimeState clock = live_at(live, WHITE, line.time, reached.clocks,
                                    {0, tolerance + 1, 0, 0});
    const std::string out = search(agent, live, WHITE, options(clock, 64, true, true));
    EXPECT_FALSE(agent.team_tree_reuse().adopted);
    EXPECT_EQ(agent.team_tree_reuse().reason, "clock");
    EXPECT_EQ(agent.team_tree_reuse().clockDiffMs, tolerance + 1);
    EXPECT_NE(AgentContinuityPeer::root(agent), expected);
    EXPECT_NE(out.find("info string Team tree reuse: none (clock)"), std::string::npos) << out;

    // The same distance within a wider tolerance is adopted.
    Agent wide(1);
    Reached again = white_then_black_reply(*this, wide, true);
    ASSERT_TRUE(again.line->step({true, false}));
    ASSERT_TRUE(again.line->step({true, false}));
    Board liveAgain = from_fens(again.line->board);
    SearchOptions wider = options(live_at(liveAgain, WHITE, again.line->time, again.clocks,
                                          {0, tolerance + 1, 0, 0}),
                                  64, true);
    wider.teamTreeReuseToleranceMs = tolerance + 1;
    search(wide, liveAgain, WHITE, wider);
    EXPECT_TRUE(wide.team_tree_reuse().adopted) << wide.team_tree_reuse().reason;
}

TEST_F(TeamTreeReuse, RootShapeMismatchNotAdopted) {
    // White decides on A (Aw); Bw lands on B at 150, Ab on A at 200. At the
    // white node after both, Aw is on turn but not ready (300) while Bb is
    // (250): the tree acted on B alone there. The live root acts on both
    // boards, so that node has the wrong shape.
    Agent agent(1);
    Board root;
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    Clocks clocks;
    clocks.paces = {100, 100, 150, 100};
    const TimeState t0 = live_clock(root, WHITE, clocks);
    search(agent, root, WHITE, options(t0, 2000, true));
    Line line{root, t0, AgentContinuityPeer::root(agent)};
    ASSERT_TRUE(line.step({true, false}));    // Aw at the root
    ASSERT_TRUE(line.step({false, true}));    // Bw lands first
    ASSERT_TRUE(line.step({false, false}));   // white waits: Ab lands next
    ASSERT_TRUE(line.step({true, false}));    // Ab
    ASSERT_EQ(line.node->get_team_to_play(), WHITE);
    const std::array<bool, 2> acting = line.time.acting_boards(WHITE);
    ASSERT_FALSE(acting[BOARD_A]);
    ASSERT_TRUE(acting[BOARD_B]);
    Board live = from_fens(line.board);
    const TimeState clock = live_at(live, WHITE, line.time, clocks);
    const std::string out = search(agent, live, WHITE, options(clock, 64, true, true));
    EXPECT_FALSE(agent.team_tree_reuse().adopted);
    EXPECT_EQ(agent.team_tree_reuse().reason, "shape");
    EXPECT_GT(agent.team_tree_reuse().boardMatches, 0);
    EXPECT_NE(out.find("info string Team tree reuse: none (shape)"), std::string::npos) << out;
}

// ---- 1.5, 1.6, 1.14: re-timing ------------------------------------------------

/**
 * White decides on A; black's next landing is a near tie: Ab at 200 (100
 * after white's root landing) against Bw at @p bwReady.
 */
std::shared_ptr<Node> near_tie_tree(TeamTreeReuse& fixture, Agent& agent, Board& root,
                                    Clocks& clocks, int32_t bwReady, size_t nodes) {
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    clocks.paces = {100, 100, 1000, 100};
    clocks.elapsed = {0, 0, 1000 - bwReady, 0};
    fixture.search(agent, root, WHITE,
                   TeamTreeReuse::options(live_clock(root, WHITE, clocks), nodes, false));
    return AgentContinuityPeer::root(agent);
}

TEST_F(TeamTreeReuse, RetimeRecomputesHashesAndPrunesShapeChanges) {
    Agent agent(1);
    Board root;
    Clocks clocks;
    const std::shared_ptr<Node> tree = near_tie_tree(*this, agent, root, clocks, 201, 600);
    // Under the old clock Ab acted at every black node below the root.
    size_t expandedChildren = 0;
    const std::vector<std::shared_ptr<Node>> before = tree->get_children();
    for (size_t i = 0; i < before.size(); ++i) {
        expandedChildren += before[i] && before[i]->is_expanded()
            && tree->get_joint_action(static_cast<int>(i)).moveA != Stockfish::MOVE_NONE ? 1 : 0;
    }
    ASSERT_GT(expandedChildren, 0u);
    // 3 ms later for Bw: it lands at 198, before Ab, below every root edge.
    Clocks shifted = clocks;
    shifted.elapsed[BW] += 3;
    const TimeState moved = live_clock(root, WHITE, shifted);
    Board walkBoard(root);
    const auto retimed = AgentContinuityPeer::retime(agent, tree, walkBoard, moved);
    EXPECT_GE(retimed.pruned, static_cast<int>(expandedChildren));
    EXPECT_EQ(walkBoard.fen(BOARD_A), root.fen(BOARD_A));
    // Below every root edge that moved on A, Ab no longer lands first. (After
    // a pass on A only Bw is on turn for black, either way.)
    const std::vector<std::shared_ptr<Node>> children = tree->get_children();
    for (size_t i = 0; i < children.size(); ++i) {
        ASSERT_TRUE(children[i]);
        if (tree->get_joint_action(static_cast<int>(i)).moveA != Stockfish::MOVE_NONE) {
            EXPECT_FALSE(children[i]->is_expanded());
        }
    }
    // Every node still in the tree carries the hash of its path under the new clock.
    std::unordered_set<const Node*> walked;
    for (const std::shared_ptr<Node>& node : retimed.walked) {
        walked.insert(node.get());
    }
    int checked = 0;
    Board check(root);
    walk_once(tree.get(), check, moved, [&](Node* node, Board& at, const TimeState& time) {
        EXPECT_TRUE(walked.contains(node));
        EXPECT_EQ(node->get_hash(), at.search_hash_key(node->get_team_to_play(), time));
        ++checked;
    });
    EXPECT_EQ(static_cast<size_t>(checked), walked.size());
    // The table built from the walk holds exactly those nodes.
    TranspositionTable table;
    std::set<uint64_t> hashes;
    for (const std::shared_ptr<Node>& node : retimed.walked) {
        table.insertOrGet(node->get_hash(), node);
        hashes.insert(node->get_hash());
    }
    EXPECT_EQ(table.size(), hashes.size());
    EXPECT_EQ(hashes.size(), retimed.walked.size());

    // Without the shift nothing changes shape and nothing is pruned.
    Agent same(1);
    Board sameRoot;
    Clocks sameClocks;
    const std::shared_ptr<Node> again = near_tie_tree(*this, same, sameRoot, sameClocks, 201, 600);
    Board sameBoard(sameRoot);
    const auto unchanged = AgentContinuityPeer::retime(
        same, again, sameBoard, live_clock(sameRoot, WHITE, sameClocks));
    EXPECT_EQ(unchanged.pruned, 0);
}

/**
 * Bw (black) has 350 ms on a 1000 ms pace: every line flags it during
 * black's second reply on A (at 400), so the solver proves the tree before
 * the search ends, two plies below each root edge.
 */
std::shared_ptr<Node> flagged_tree(TeamTreeReuse& fixture, Agent& agent, Board& root,
                                   Clocks& clocks) {
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    clocks.paces = {100, 100, 1000, 100};
    clocks.clocks[BW] = 350;
    fixture.search(agent, root, WHITE,
                   TeamTreeReuse::options(live_clock(root, WHITE, clocks), 1500, false));
    return AgentContinuityPeer::root(agent);
}

int solved_below(Node* root) {
    int solved = 0;
    for (Node* node : reachable(root)) {
        solved += node != root && node->get_node_type() != NodeType::UNSOLVED ? 1 : 0;
    }
    return solved;
}

TEST_F(TeamTreeReuse, ClockVerdictsDroppedWhenClocksMove) {
    const auto run = [&](const std::function<void(Clocks&)>& change) {
        Agent agent(1);
        Board root;
        Clocks clocks;
        const std::shared_ptr<Node> tree = flagged_tree(*this, agent, root, clocks);
        const int before = solved_below(tree.get());
        Clocks next = clocks;
        change(next);
        Board walkBoard(root);
        AgentContinuityPeer::retime(agent, tree, walkBoard, live_clock(root, WHITE, next));
        return std::pair<int, int>{before, solved_below(tree.get())};
    };
    const auto kept = run([](Clocks&) {});
    ASSERT_GT(kept.first, 0);
    EXPECT_EQ(kept.second, kept.first);
    EXPECT_EQ(run([](Clocks& c) { c.clocks[BW] += 1; }).second, 0);
    EXPECT_EQ(run([](Clocks& c) { c.clocks[AW] += 1; }).second, 0);
    // The clocks equal, but the budget or a pace differs: every ready time of
    // our team below the root moves, so no verdict is kept.
    EXPECT_EQ(run([](Clocks& c) { c.rootMoveMs += 1; }).second, 0);
    EXPECT_EQ(run([](Clocks& c) { c.paces[AB] += 1; }).second, 0);
}

TEST_F(TeamTreeReuse, StaleFlagEdgeCarriesNoValue) {
    // The old tree proved root edges that flag Bw: they hold visits at Q
    // near +1. With 5 s on Bw's clock nothing flags, and the edges forget.
    Agent agent(1);
    Board root;
    Clocks clocks;
    const std::shared_ptr<Node> tree = flagged_tree(*this, agent, root, clocks);
    const std::vector<int> visitsBefore = tree->get_child_visits();
    const std::vector<float> qBefore = tree->get_q_values();
    // The root edges into children proven lost for black (Bw flags).
    std::vector<size_t> proven;
    int busiest = -1;
    for (size_t i = 0; i < visitsBefore.size(); ++i) {
        const std::shared_ptr<Node> child = tree->get_child(static_cast<int>(i));
        if (child && child->get_node_type() == NodeType::LOSS) {
            proven.push_back(i);
            if (busiest < 0 || visitsBefore[i] > visitsBefore[busiest]) {
                busiest = static_cast<int>(i);
            }
        }
    }
    ASSERT_GE(busiest, 0);
    int provenVisits = 0;
    for (size_t i : proven) {
        provenVisits += visitsBefore[i];
    }
    // On the uniform net the solver finishes fast (the first edge proven is
    // white's sit, which runs Bw out at once), so the stale edge holds a few
    // visits, not hundreds; what is checked is that they are all gone.
    ASSERT_GE(provenVisits, 2);
    EXPECT_GT(qBefore[busiest], 0.0f);
    const int rootVisits = tree->get_visits();
    Clocks later = clocks;
    later.clocks[BW] = 5000;
    Board walkBoard(root);
    const auto retimed = AgentContinuityPeer::retime(agent, tree, walkBoard,
                                                     live_clock(root, WHITE, later));
    EXPECT_GT(retimed.pruned, 0);
    const std::vector<int> visitsAfter = tree->get_child_visits();
    const std::vector<float> qAfter = tree->get_q_values();
    int removed = 0;
    for (size_t i : proven) {
        const std::shared_ptr<Node> child = tree->get_child(static_cast<int>(i));
        ASSERT_TRUE(child);
        EXPECT_EQ(child->get_node_type(), NodeType::UNSOLVED);
        EXPECT_FALSE(child->is_expanded());
        EXPECT_EQ(visitsAfter[i], 0);
        EXPECT_FLOAT_EQ(qAfter[i], SearchParams::Q_INIT);
        removed += visitsBefore[i];
    }
    EXPECT_EQ(tree->get_visits(), std::max(0, rootVisits - removed));
}

// ---- 1.7 ---------------------------------------------------------------------

TEST_F(TeamTreeReuse, BothSeatsOfTheTeam) {
    // Aw decides on A; Bw lands on B at 150, before Ab (1100): Bb's decision
    // on B adopts a node of the tree Aw's search built.
    Agent agent(1);
    Board root;
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    Clocks clocks;
    clocks.paces = {100, 1000, 150, 100};
    const TimeState t0 = live_clock(root, WHITE, clocks);
    search(agent, root, WHITE, options(t0, 800, true));
    const std::shared_ptr<Node> awTree = AgentContinuityPeer::root(agent);
    Line line{root, t0, awTree};
    ASSERT_TRUE(line.step({true, false}));   // Aw
    ASSERT_TRUE(line.step({false, true}));   // Bw
    ASSERT_EQ(line.node->get_team_to_play(), WHITE);
    ASSERT_TRUE(line.time.acting_boards(WHITE)[BOARD_B]);
    const std::shared_ptr<Node> expected = line.node;
    Board live = from_fens(line.board);
    ASSERT_EQ(live.side_to_move(BOARD_B), BLACK);   // Bb decides
    ASSERT_EQ(live.side_to_move(BOARD_A), BLACK);   // Ab still thinking
    search(agent, live, WHITE, options(live_at(live, WHITE, line.time, clocks, {1, 2, 3, 4}),
                                       64, true));
    EXPECT_TRUE(agent.team_tree_reuse().adopted) << agent.team_tree_reuse().reason;
    EXPECT_EQ(AgentContinuityPeer::root(agent), expected);
}

// ---- 1.9 ---------------------------------------------------------------------

TEST_F(TeamTreeReuse, MemoryBounded) {
    // The cap is TEAM_TREE_MAX_RETAINED (100k) in play; a 60-node cap on a
    // 2000-node tree checks the same walk without a 100k-node search.
    constexpr size_t cap = 60;
    const long baseline = Node::live_count();
    Agent agent(1);
    Board root;
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    Clocks clocks;
    clocks.paces = {100, 100, 5000, 100};
    const TimeState t0 = live_clock(root, WHITE, clocks);
    search(agent, root, WHITE, options(t0, 2000, true));
    std::shared_ptr<Node> tree = AgentContinuityPeer::root(agent);
    ASSERT_GT(reachable(tree.get()).size(), cap * 4);
    Board walkBoard(root);
    const auto retimed = AgentContinuityPeer::retime(agent, tree, walkBoard, t0, cap);
    // Every node still reachable was validated by the walk: at most the cap
    // of old nodes, plus the fresh (unexpanded) nodes that replaced the rest.
    std::unordered_set<const Node*> walked;
    size_t fresh = 0;
    for (const std::shared_ptr<Node>& node : retimed.walked) {
        walked.insert(node.get());
    }
    const std::vector<Node*> now = reachable(tree.get());
    for (Node* node : now) {
        EXPECT_TRUE(walked.contains(node));
        if (!node->is_expanded() && node->get_visits() == 0) {
            ++fresh;
        }
    }
    EXPECT_LE(now.size() - fresh, cap);
    EXPECT_EQ(static_cast<size_t>(retimed.pruned), fresh);
    // Once the agent lets go and the GC thread drains, only these are alive.
    const long alive = baseline + static_cast<long>(now.size());
    AgentContinuityPeer::release_tree(agent);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while ((AgentContinuityPeer::gc_pending(agent) > 0 || Node::live_count() > alive)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_LE(Node::live_count(), alive);
}

// ---- 1.11 --------------------------------------------------------------------

TEST_F(TeamTreeReuse, SelfPlayNoiseRefreshes) {
    // Self-play keeps the adopted root and gives it fresh root noise.
    Agent agent(1);
    Reached reached = white_then_black_reply(*this, agent, true);
    Line& line = *reached.line;
    ASSERT_TRUE(line.step({true, false}));
    ASSERT_TRUE(line.step({true, false}));
    const std::shared_ptr<Node> expected = line.node;
    Board live = from_fens(line.board);
    SearchOptions noisy = options(live_at(live, WHITE, line.time, reached.clocks), 64, true);
    noisy.search.rootDirichletAlpha = 0.3f;
    noisy.search.rootDirichletEpsilon = 0.25f;
    noisy.search.rootNoiseSeed = 11;
    noisy.reusedVisitsCount = true;
    search(agent, live, WHITE, noisy);
    EXPECT_TRUE(agent.team_tree_reuse().adopted) << agent.team_tree_reuse().reason;
    EXPECT_EQ(AgentContinuityPeer::root(agent), expected);
}

TEST_F(TeamTreeReuse, OldRootReadoptedWithoutASecondNoiseMix) {
    // Review fix: when no move landed since the last search (the partner
    // decides while our move is in flight), the old root itself is the
    // match. Its priors already hold that search's root noise: it is adopted
    // (oldRoot) but not re-noised, so the noise share stays what the
    // settings say instead of compounding with each re-adoption.
    Board root;
    root.set(std::string(kSmallWhite) + " | " + kSmallBlack);
    Clocks clocks;
    const TimeState t0 = live_clock(root, WHITE, clocks);
    SearchOptions noisy = options(t0, 200, true);
    noisy.search.rootDirichletAlpha = 0.3f;
    noisy.search.rootDirichletEpsilon = 0.25f;
    noisy.search.rootNoiseSeed = 11;
    Agent agent(1);
    search(agent, root, WHITE, noisy);
    const std::shared_ptr<Node> oldRoot = AgentContinuityPeer::root(agent);
    ASSERT_TRUE(oldRoot);
    const auto priorsA = oldRoot->board_priors(BOARD_A);
    const auto priorsB = oldRoot->board_priors(BOARD_B);
    // The partner's decision 1 ms later, same boards, a new noise seed.
    clocks.clocks[AW] -= 1;
    clocks.clocks[BB] -= 1;
    SearchOptions again = noisy;
    again.clock = live_clock(root, WHITE, clocks);
    again.search.rootNoiseSeed = 12;
    search(agent, root, WHITE, again);
    const TeamTreeReuseReport& reuse = agent.team_tree_reuse();
    ASSERT_TRUE(reuse.adopted) << reuse.reason;
    EXPECT_TRUE(reuse.oldRoot);
    EXPECT_EQ(AgentContinuityPeer::root(agent), oldRoot);
    for (const auto& [before, boardNumber] :
         {std::pair{priorsA, BOARD_A}, std::pair{priorsB, BOARD_B}}) {
        const auto after = oldRoot->board_priors(boardNumber);
        ASSERT_EQ(after.size(), before.size());
        for (size_t i = 0; i < after.size(); ++i) {
            EXPECT_EQ(after[i].first, before[i].first);
            EXPECT_FLOAT_EQ(after[i].second, before[i].second);
        }
    }
    // A node below an old root never had root noise: adopted after a
    // landing, it gets it (1.11, SelfPlayNoiseRefreshes), and is no oldRoot.
}

// ---- 1.12 --------------------------------------------------------------------

TEST_F(TeamTreeReuse, FourWorkersAfterAdoption) {
    int adopted = 0;
    for (int repeat = 0; repeat < 50; ++repeat) {
        Agent agent(4);
        Reached reached = white_then_black_reply(*this, agent, true, 400);
        Line& line = *reached.line;
        if (!line.step({true, false}) || !line.step({true, false})) {
            continue;
        }
        Board live = from_fens(line.board);
        const std::array<int32_t, 4> offsets = {repeat % 7, repeat % 5, repeat % 3, 1};
        search(agent, live, WHITE,
               options(live_at(live, WHITE, line.time, reached.clocks, offsets), 200, true));
        const TeamTreeReuseReport& reuse = agent.team_tree_reuse();
        if (!reuse.adopted) {
            continue;
        }
        ++adopted;
        const int visits = AgentContinuityPeer::root(agent)->get_visits();
        EXPECT_GE(visits, reuse.visits + 200);
        // Four workers may each have one batch in flight at the stop.
        EXPECT_LE(visits, reuse.visits + 200 + 8);
    }
    EXPECT_GT(adopted, 40);
}

// ---- 1.13 --------------------------------------------------------------------

/** Pieces in pockets and captures on both boards, so held-back drops and sits vary. */
constexpr const char* kBusyA = "r3k3/8/8/3p4/4P3/8/8/R3K3[Nn] w - - 0 1";
constexpr const char* kBusyB = "4k3/8/8/2p5/3P4/8/8/4K3[Pp] w - - 0 1";

TEST_F(TeamTreeReuse, RetainedExpansionsMatchFreshOnes) {
    std::mt19937 random(5);
    std::uniform_int_distribution<int32_t> offset(0, 200);
    int compared = 0;
    int pruned = 0;
    for (int adoption = 0; adoption < 20; ++adoption) {
        Agent agent(1);
        Board root;
        root.set(std::string(kBusyA) + " | " + kBusyB);
        Clocks clocks;
        // Paces close together, so offsets reorder landings and wake-ups.
        clocks.paces = {100, 120, 180, 100};
        clocks.elapsed = {0, 0, 60, 0};
        clocks.rootMoveMs = 120;
        search(agent, root, WHITE, options(live_clock(root, WHITE, clocks), 300, false));
        const std::shared_ptr<Node> tree = AgentContinuityPeer::root(agent);
        Clocks next = clocks;
        for (int seat = 0; seat < 4; ++seat) {
            next.clocks[seat] += offset(random);
        }
        next.elapsed[BW] = std::min(170, next.elapsed[BW] + offset(random) / 2);
        next.rootMoveMs += offset(random) / 4;
        const TimeState moved = live_clock(root, WHITE, next);
        Board walkBoard(root);
        pruned += AgentContinuityPeer::retime(agent, tree, walkBoard, moved).pruned;
        Board check(root);
        walk_once(tree.get(), check, moved, [&](Node* node, Board& at, const TimeState& time) {
            if (!node->is_expanded() || node->get_node_type() != NodeType::UNSOLVED) {
                return;
            }
            // The lists collect_batch would build for this node now.
            const Stockfish::Color team = node->get_team_to_play();
            const LeafTurn turn = leaf_turn(at, team, time);
            const JointActionRules rules = node->expanded_rules();
            EXPECT_EQ(rules.boardAOnTurn, turn.boardAOnTurn || turn.forcedWait);
            EXPECT_EQ(rules.boardBOnTurn, turn.boardBOnTurn);
            EXPECT_EQ(rules.teamHasTimeAdvantage, turn.forcedWait || time.can_sit(team));
            for (int boardNumber : {BOARD_A, BOARD_B}) {
                const bool onTurn = boardNumber == BOARD_A ? turn.boardAOnTurn : turn.boardBOnTurn;
                std::vector<Stockfish::Move> fresh =
                    leaf_board_actions(at, boardNumber, onTurn, time);
                std::vector<Stockfish::Move> kept = node->expanded_actions(boardNumber);
                std::sort(fresh.begin(), fresh.end());
                std::sort(kept.begin(), kept.end());
                EXPECT_EQ(kept, fresh) << "board " << boardNumber;
            }
            ++compared;
        });
    }
    EXPECT_GT(compared, 100);
    // The offsets did change some expansions, which the walk replaced.
    EXPECT_GT(pruned, 0);
}

// ---- 1.15 --------------------------------------------------------------------

TEST_F(TeamTreeReuse, TranspositionSplitWhenPathsDiverge) {
    // A node shared by two parents whose paths reach it in different states
    // gets a fresh node under the later parent. (Built here by sharing a
    // grandchild between two root children; the walk only sees the hashes.)
    Agent agent(1);
    Board root;
    root.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    Clocks clocks;
    clocks.paces = {100, 100, 5000, 100};
    const TimeState t0 = live_clock(root, WHITE, clocks);
    search(agent, root, WHITE, options(t0, 800, false));
    const std::shared_ptr<Node> tree = AgentContinuityPeer::root(agent);
    const std::vector<std::shared_ptr<Node>> children = tree->get_children();
    int first = -1;
    int second = -1;
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i] && children[i]->is_expanded() && children[i]->get_children().size() > 1) {
            (first < 0 ? first : second) = static_cast<int>(i);
            if (second >= 0) {
                break;
            }
        }
    }
    ASSERT_GE(second, 0);
    const std::shared_ptr<Node> shared = children[first]->get_child(0);
    const std::shared_ptr<Node> displaced = children[second]->get_child(0);
    children[second]->replace_child(0, shared);
    Board walkBoard(root);
    const auto retimed = AgentContinuityPeer::retime(agent, tree, walkBoard, t0);
    EXPECT_EQ(children[first]->get_child(0), shared);
    const std::shared_ptr<Node> split = children[second]->get_child(0);
    ASSERT_NE(split, shared);
    EXPECT_FALSE(split->is_expanded());
    // Each carries the hash of its own path, and the table holds both.
    const auto path_hash = [&](int rootChild) {
        Board board(root);
        const JointActionCandidate a = tree->get_joint_action(rootChild);
        TimeState time = after(t0, board, WHITE, a);
        board.make_moves(a.moveA, a.moveB);
        const JointActionCandidate b = children[rootChild]->get_joint_action(0);
        time = after(time, board, BLACK, b);
        board.make_moves(b.moveA, b.moveB);
        return board.search_hash_key(WHITE, time);
    };
    EXPECT_EQ(shared->get_hash(), path_hash(first));
    EXPECT_EQ(split->get_hash(), path_hash(second));
    TranspositionTable table;
    for (const std::shared_ptr<Node>& node : retimed.walked) {
        table.insertOrGet(node->get_hash(), node);
    }
    EXPECT_EQ(table.insertOrGet(path_hash(first), std::make_shared<Node>(WHITE)), shared);
    EXPECT_EQ(table.insertOrGet(path_hash(second), std::make_shared<Node>(WHITE)), split);
    (void)displaced;
}

// ---- 1.17 --------------------------------------------------------------------

TEST_F(TeamTreeReuse, NoOtherStateSurvives) {
    Agent agent(1);
    Reached reached = white_then_black_reply(*this, agent, true);
    EXPECT_EQ(AgentContinuityPeer::root_candidates(agent), 0u);
    EXPECT_TRUE(AgentContinuityPeer::team_tree_root(agent));
    Line& line = *reached.line;
    ASSERT_TRUE(line.step({true, false}));
    ASSERT_TRUE(line.step({true, false}));
    Board live = from_fens(line.board);
    // A single-board mate line kept from before is not re-timed: gone.
    AgentContinuityPeer::plant_mate_continuation(agent);
    const std::string out = search(agent, live, WHITE,
                                   options(live_at(live, WHITE, line.time, reached.clocks),
                                           64, true, true));
    EXPECT_EQ(AgentContinuityPeer::mate_continuations(agent), 0u);
    EXPECT_EQ(AgentContinuityPeer::root_candidates(agent), 0u);
    EXPECT_EQ(out.find("info string Tree reuse:"), std::string::npos) << out;
    EXPECT_NE(out.find("info string Team tree reuse:"), std::string::npos);
}

TEST_F(TeamTreeReuse, OptionOffAndNonDurativeDropTheTeamTree) {
    Agent agent(1);
    white_then_black_reply(*this, agent, true);
    ASSERT_TRUE(AgentContinuityPeer::team_tree_root(agent));
    // A durative search without the option keeps no team tree.
    Board board;
    board.set(std::string(kSmallWhite) + " | " + kSmallWhite);
    search(agent, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 32, false));
    EXPECT_FALSE(AgentContinuityPeer::team_tree_root(agent));
    // The option under a v3 clock does nothing.
    SearchOptions v3 = options(TimeState::clocked(WHITE, 500), 32, true);
    search(agent, board, WHITE, v3);
    EXPECT_FALSE(agent.team_tree_reuse().attempted);
    EXPECT_FALSE(AgentContinuityPeer::team_tree_root(agent));
}

// ---- 1.10, 1.11 (UCI) ----------------------------------------------------------

class TeamTreeReuseUci : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        const auto model = std::filesystem::path(__FILE__).parent_path()
            / "data" / "uniform.onnx";
        ASSERT_TRUE(uci.initializeEngines({0}, model.string(), 1));
        for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
                 {"TimeModel", "clock"}, {"DurativeTree", "true"}, {"SitGate", "false"},
                 {"FlagLoss", "true"}, {"InstantTransfer", "false"}, {"Team", "white"},
                 {"RootMoveTimeMs", "100"}, {"SeatPaceMs", "100 100 5000 100"},
                 {"CertifySelectedMove", "false"}, {"TeamTreeReuse", "true"}}) {
            send("setoption name " + name + " value " + value);
        }
    }

    void TearDown() override { uci.stop(); }

    /** One command, as the UCI loop would dispatch it; returns what it printed. */
    std::string send(const std::string& line) {
        std::ostringstream output;
        std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
        std::istringstream is(line);
        std::string command;
        is >> command;
        if (command == "setoption") uci.setoption(is);
        else if (command == "position") uci.position(is);
        else if (command == "go") {
            // A node-limited search ends by itself; then join its thread.
            uci.go(is);
            while (UCIContinuityPeer::searching(uci)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            uci.stop();
        }
        else if (command == "stop") uci.stop();
        else if (command == "ucinewgame") uci.new_game();
        std::cout.rdbuf(previous);
        return output.str();
    }

    UCI uci;
};

TEST_F(TeamTreeReuseUci, NoPermanentBrainOverTeamTree) {
    send("setoption name SeatClocksMs value 60000 60000 60000 60000");
    send(std::string("position fen ") + kSmallWhite + " | " + kSmallWhite);
    const std::string first = send("go nodes 400");
    ASSERT_NE(first.find("bestmove"), std::string::npos) << first;
    send("stop");
    // bestmove and stop: the tree is still there for the next search.
    EXPECT_TRUE(AgentContinuityPeer::team_tree_root(UCIContinuityPeer::agent(uci)));
}

TEST_F(TeamTreeReuseUci, OpeningNoiseDiscards) {
    send("setoption name OpeningNoise value true");
    send("setoption name SeatClocksMs value 60000 60000 60000 60000");
    send(std::string("position fen ") + kSmallWhite + " | " + kSmallWhite);
    send("go nodes 200");
    ASSERT_TRUE(AgentContinuityPeer::team_tree_root(UCIContinuityPeer::agent(uci)));
    send("setoption name SeatClocksMs value 59900 60000 59900 60000");
    const std::string second = send("go nodes 64");
    EXPECT_NE(second.find("info string Team tree reuse: none (no tree)"), std::string::npos)
        << second;
}

TEST_F(TeamTreeReuseUci, UcinewgameClearsTheTeamTree) {
    send("setoption name SeatClocksMs value 60000 60000 60000 60000");
    send(std::string("position fen ") + kSmallWhite + " | " + kSmallWhite);
    send("go nodes 200");
    ASSERT_TRUE(AgentContinuityPeer::team_tree_root(UCIContinuityPeer::agent(uci)));
    send("ucinewgame");
    EXPECT_FALSE(AgentContinuityPeer::team_tree_root(UCIContinuityPeer::agent(uci)));
}
#endif

}  // namespace
