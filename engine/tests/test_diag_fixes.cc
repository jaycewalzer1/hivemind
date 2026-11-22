#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/planes.h"
#include "environment/time_state.h"
#include "interface/uci.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"
#include "search/searchthread.h"
#include "tools/seat_clock_game.h"

// Fixes from the durative tree diagnosis at 261081b (docs/wave2_engine.md,
// "Diagnosis fixes"; spec v5, "Diagnosis fixes"). Each is an option that is
// off by default:
//  D1 OwnPaceBelowRoot: below the root the root team's seats act at their own
//     SeatPaceMs instead of RootMoveTimeMs; the root action keeps the budget.
//  D3 WaitNodeActingEval: a forced-wait node is evaluated from the team that
//     acts next, and the value negated.
//  D5 BlockableCheckNotMate: a check a drop could block is never mate, as in
//     the arena (arena/RULES.md, arena/state.py is_bughouse_mate).

class AgentDiagFixesPeer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
};

class UCIDiagFixesPeer {
public:
    static const DurativeConfig& durative(const UCI& uci) { return uci.durativeConfig; }
    static bool board_rule(const UCI& uci) { return uci.board.blockable_check_not_mate(); }
    static bool team_tree_reuse(const UCI& uci) { return uci.teamTreeReuse; }
    static bool plan_carry(const UCI& uci) { return uci.planCarry; }
    static std::optional<TimeState> clock(const UCI& uci) { return uci.current_clock(); }
};

namespace {

using Stockfish::BLACK;
using Stockfish::MOVE_NONE;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

// D5's position (diagnosis D5, confirmed against arena/state.py): after Aw's
// Rh1-h8 black's Ab is in check from a rook with six empty squares between,
// has no legal move and an empty pocket; its partner Bw (on turn on B) has
// no capture. The arena plays on (Ab waits); the engine at 261081b calls it
// mate in both trees ("score mate 1", it13).
constexpr const char* kRookCheckBefore = "k7/pp6/8/8/8/8/8/4K2R[] w - - 0 1";
constexpr const char* kRookCheckAfter = "k6R/pp6/8/8/8/8/8/4K3[] b - - 1 1";
constexpr const char* kQuietPartner = "4k3/8/8/8/8/8/8/4K3[] w - - 0 1";

Board boards(const std::string& a, const std::string& b) {
    Board board;
    board.set(a + " | " + b);
    return board;
}

struct Rules {
    std::array<int32_t, 4> clocks{60000, 60000, 60000, 60000};
    std::array<int32_t, 4> paces{300, 200, 1000, 400};
    std::array<int32_t, 4> elapsed{0, 0, 0, 0};
    int32_t rootMoveMs = 120;
    bool ownPace = false;
    bool jointPlans = false;
    bool waitEval = false;
};

/** A durative root of @p rootTeam with both boards white to move (Aw, Bw on turn). */
TimeState root_state(Stockfish::Color rootTeam, const Rules& r,
                     Stockfish::Color sideA = WHITE, Stockfish::Color sideB = WHITE) {
    DurativeConfig config;
    config.rootMoveTimeMs = r.rootMoveMs;
    config.seatClocksMs = r.clocks;
    config.seatPaceMs = r.paces;
    config.seatElapsedMs = r.elapsed;
    config.ownPaceBelowRoot = r.ownPace;
    config.jointPlans = r.jointPlans;
    config.waitNodeActingEval = r.waitEval;
    const TimeState base = TimeState::clocked(rootTeam, 0, 100, 100).with_sit_gate(false);
    return TimeState::durative(base, rootTeam, sideA, sideB, config);
}

/**
 * White decides on A at the root (budget 120); Ab answers after its 200 ms
 * pace; Aw then comes on turn below the root. Returns Aw's ready time there.
 */
int32_t aw_ready_after_ab(const Rules& r) {
    TimeState time = root_state(WHITE, r);
    EXPECT_EQ(time.seat_ready_ms(AW), r.rootMoveMs);
    time.advance_action(WHITE, true, false);   // root: Aw lands at the budget
    EXPECT_EQ(time.seat_ready_ms(AB), 200);
    time.advance_action(BLACK, true, false);   // Ab lands
    return time.seat_ready_ms(AW);
}

// ---- defaults and plumbing ----------------------------------------------------

TEST(DiagFixesDefaults, OptionsAdvertisedAndOff) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);
    for (const char* line : {"option name OwnPaceBelowRoot type check default false",
                             "option name WaitNodeActingEval type check default false",
                             "option name BlockableCheckNotMate type check default false"}) {
        EXPECT_NE(output.str().find(line), std::string::npos) << line;
    }
    const DurativeConfig config;
    EXPECT_FALSE(config.ownPaceBelowRoot);
    EXPECT_FALSE(config.waitNodeActingEval);
    EXPECT_FALSE(Board().blockable_check_not_mate());
    const Wave2GameOptions game;
    EXPECT_FALSE(game.ownPaceBelowRoot || game.waitNodeActingEval || game.blockableCheckNotMate);
    EXPECT_FALSE(UCIDiagFixesPeer::durative(uci).ownPaceBelowRoot);
    EXPECT_FALSE(UCIDiagFixesPeer::durative(uci).waitNodeActingEval);
    EXPECT_FALSE(UCIDiagFixesPeer::board_rule(uci));
}

TEST(DiagFixesDefaults, UciOptionsSetTheirSwitches) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    for (const char* text : {"name DurativeTree value true", "name TimeModel value clock",
                             "name SeatPaceMs value 300 200 1000 400"}) {
        std::istringstream option(text);
        uci.setoption(option);
    }
    const std::optional<TimeState> before = UCIDiagFixesPeer::clock(uci);
    for (const char* text : {"name OwnPaceBelowRoot value true",
                             "name WaitNodeActingEval value true",
                             "name BlockableCheckNotMate value true"}) {
        std::istringstream option(text);
        uci.setoption(option);
    }
    std::cout.rdbuf(previous);
    EXPECT_TRUE(UCIDiagFixesPeer::durative(uci).ownPaceBelowRoot);
    EXPECT_TRUE(UCIDiagFixesPeer::durative(uci).waitNodeActingEval);
    EXPECT_TRUE(UCIDiagFixesPeer::board_rule(uci));
    for (const char* reply : {"info string OwnPaceBelowRoot set to true",
                              "info string WaitNodeActingEval set to true",
                              "info string BlockableCheckNotMate set to true"}) {
        EXPECT_NE(output.str().find(reply), std::string::npos) << reply;
    }
    // A new position keeps the rule: it belongs to the board, not the FEN.
    std::istringstream position("fen " + std::string(kRookCheckAfter) + " | " + kQuietPartner);
    uci.position(position);
    EXPECT_TRUE(UCIDiagFixesPeer::board_rule(uci));
    // D1 enters the clock (our seats' paces are given), D3 does not key it.
    const std::optional<TimeState> after = UCIDiagFixesPeer::clock(uci);
    ASSERT_TRUE(before && after);
    EXPECT_NE(before->exact_key(), after->exact_key());
    EXPECT_TRUE(after->wait_node_acting_eval());
    EXPECT_FALSE(before->wait_node_acting_eval());
}

TEST(DiagFixesDefaults, ContinuityPresetCarriesD1AndD5NotD3) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    std::istringstream on("name Continuity value true");
    uci.setoption(on);
    std::cout.rdbuf(previous);
    EXPECT_TRUE(UCIDiagFixesPeer::durative(uci).ownPaceBelowRoot);
    EXPECT_TRUE(UCIDiagFixesPeer::board_rule(uci));
    EXPECT_FALSE(UCIDiagFixesPeer::durative(uci).waitNodeActingEval);
    EXPECT_TRUE(UCIDiagFixesPeer::team_tree_reuse(uci));
    EXPECT_NE(output.str().find("OwnPaceBelowRoot, BlockableCheckNotMate true"),
              std::string::npos) << output.str();
    std::istringstream off("name Continuity value false");
    previous = std::cout.rdbuf(output.rdbuf());
    uci.setoption(off);
    std::cout.rdbuf(previous);
    EXPECT_FALSE(UCIDiagFixesPeer::durative(uci).ownPaceBelowRoot);
    EXPECT_FALSE(UCIDiagFixesPeer::board_rule(uci));
    EXPECT_FALSE(UCIDiagFixesPeer::team_tree_reuse(uci));

    // Self-play and tournament: the same preset.
    Wave2GameOptions preset;
    EXPECT_TRUE(parse_wave2_option("--continuity", "true", preset));
    EXPECT_TRUE(preset.ownPaceBelowRoot && preset.blockableCheckNotMate);
    EXPECT_FALSE(preset.waitNodeActingEval);
    EXPECT_TRUE(parse_wave2_option("--continuity", "false", preset));
    EXPECT_FALSE(preset.ownPaceBelowRoot || preset.blockableCheckNotMate);
}

TEST(DiagFixesDefaults, SelfPlayFlagsParseAndValidate) {
    init_fairy_stockfish();  // Board below; ctest runs each test in its own process
    Wave2GameOptions options;
    EXPECT_TRUE(parse_wave2_option("--own-pace-below-root", "true", options));
    EXPECT_TRUE(options.ownPaceBelowRoot);
    EXPECT_TRUE(options.changes_search());
    // Durative only, like every tree option.
    EXPECT_THROW(options.validate(), std::invalid_argument);
    options.durativeTree = true;
    EXPECT_NO_THROW(options.validate());
    EXPECT_TRUE(options.durative_config().ownPaceBelowRoot);

    Wave2GameOptions wait;
    EXPECT_TRUE(parse_wave2_option("--wait-node-acting-eval", "true", wait));
    EXPECT_TRUE(wait.waitNodeActingEval);
    EXPECT_THROW(wait.validate(), std::invalid_argument);
    wait.durativeTree = true;
    EXPECT_NO_THROW(wait.validate());
    EXPECT_TRUE(wait.durative_config().waitNodeActingEval);

    // D5 is a rule of the board in both trees: no durative tree needed, and
    // alone it does not move a stock run onto the clock time model.
    Wave2GameOptions mate;
    EXPECT_TRUE(parse_wave2_option("--blockable-check-not-mate", "true", mate));
    EXPECT_TRUE(mate.blockableCheckNotMate);
    EXPECT_NO_THROW(mate.validate());
    EXPECT_FALSE(mate.changes_search());
    Board board;
    mate.apply_board_rules(board);
    EXPECT_TRUE(board.blockable_check_not_mate());
    Wave2GameOptions none;
    none.apply_board_rules(board);
    EXPECT_FALSE(board.blockable_check_not_mate());
}

// ---- D1: our own pace below the root ----------------------------------------------

TEST(OwnPaceBelowRoot, OffChargesTheRootBudgetBelowTheRoot) {
    Rules off;
    EXPECT_EQ(aw_ready_after_ab(off), 120);
}

TEST(OwnPaceBelowRoot, OnChargesTheSeatsOwnPace) {
    Rules on;
    on.ownPace = true;
    EXPECT_EQ(aw_ready_after_ab(on), 300);
    // The root action itself keeps the budget: Aw's clock ran 120 at the root.
    TimeState time = root_state(WHITE, on);
    time.advance_action(WHITE, true, false);
    EXPECT_EQ(time.seat_clock_ms(AW), 60000 - 120);
}

TEST(OwnPaceBelowRoot, FallsBackToTheBudgetWithoutAPace) {
    Rules on;
    on.ownPace = true;
    on.paces[AW] = 0;  // no SeatPaceMs for Aw: the team's v3 time is 100
    EXPECT_EQ(aw_ready_after_ab(on), 120);
    // Nothing given for our seats: the state is exactly the option off.
    Rules none;
    none.paces = {0, 0, 0, 0};
    Rules noneOn = none;
    noneOn.ownPace = true;
    EXPECT_EQ(root_state(WHITE, none), root_state(WHITE, noneOn));
    EXPECT_EQ(root_state(WHITE, none).exact_key(), root_state(WHITE, noneOn).exact_key());
    // With our paces given the key differs (nodes do not merge across it).
    Rules paced;
    Rules pacedOn = paced;
    pacedOn.ownPace = true;
    EXPECT_NE(root_state(WHITE, paced).exact_key(), root_state(WHITE, pacedOn).exact_key());
}

TEST(OwnPaceBelowRoot, PassCapAtTheRootKeepsTheBudgetBelowUsesThePace) {
    for (const bool ownPace : {false, true}) {
        Rules r;
        r.ownPace = ownPace;
        r.paces = {300, 200, 5000, 400};
        TimeState time = root_state(WHITE, r);
        time.advance_action(WHITE, false, false);   // root: Aw passes
        EXPECT_EQ(time.seat_ready_ms(AW), 8 * 120) << ownPace;
        time.advance_action(WHITE, false, false);   // Aw wakes and passes again
        EXPECT_EQ(time.seat_ready_ms(AW), ownPace ? 8 * 300 : 8 * 120) << ownPace;
    }
}

TEST(OwnPaceBelowRoot, PlannedHalfLandsAtTheSeatsOwnPace) {
    // Composition with fix 2. Bw lands during our root budget (ready 50);
    // Bb then comes on turn: at 50 (budget 100 less the 50 it is dated back)
    // with the option off, at 350 (its pace 400) with it on. On: Ab lands
    // first (200), Aw comes on turn at its pace (300), and white acts on B
    // at 350 while Aw is not ready: a planning node whose planned move lands
    // when Aw is ready, 150 later.
    Rules r;
    r.jointPlans = true;
    r.rootMoveMs = 100;
    r.elapsed[BW] = 950;
    for (const bool ownPace : {false, true}) {
        r.ownPace = ownPace;
        TimeState time = root_state(WHITE, r);
        time.advance_action(WHITE, true, false);   // root: Aw
        time.advance_action(BLACK, false, true);   // Bw's early landing
        EXPECT_EQ(time.seat_ready_ms(BB), ownPace ? 350 : 50) << ownPace;
        if (!ownPace) {
            continue;
        }
        time.advance_action(BLACK, true, false);   // Ab at 200
        EXPECT_EQ(time.seat_ready_ms(AW), 300);
        EXPECT_EQ(time.seat_ready_ms(BB), 150);
        const std::array<bool, 2> planned = time.planned_boards(WHITE);
        ASSERT_TRUE(planned[BOARD_A]);
        const Stockfish::Move plan = Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4);
        time.advance_action(WHITE, false, true, Stockfish::NO_PIECE_TYPE,
                            Stockfish::NO_PIECE_TYPE, plan, MOVE_NONE);
        EXPECT_EQ(time.committed_move(AW), plan);
        EXPECT_EQ(time.seat_ready_ms(AW), 150);
    }
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class DiagFixesSearch : public ::testing::Test {
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

    static SearchOptions options(const TimeState& clock, size_t nodes, bool teamTree = false,
                                 bool provers = false) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.clock = clock;
        opts.teamTreeReuse = teamTree;
        opts.search.enableRootMateSearch = provers;
        opts.search.certifySelectedMove = provers;
        opts.search.enableMateProbe = provers;
        return opts;
    }

    std::string search(Agent& agent, Board& board, Stockfish::Color team,
                       const SearchOptions& opts) {
        testing::internal::CaptureStdout();
        agent.run_search(board, {engine.get()}, team, false, opts);
        return testing::internal::GetCapturedStdout();
    }

    std::unique_ptr<Engine> engine;
};

/** Every node reachable from @p root, each once, with its board and path time. */
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
            const TimeState next = durative_time_after(at, board, node->get_team_to_play(), action);
            board.make_moves(action.moveA, action.moveB);
            visit(children[i].get(), next);
            board.unmake_moves(action.moveA, action.moveB);
        }
    };
    visit(root, time);
}

TEST_F(DiagFixesSearch, OwnPaceTreeReusedAndRetimedConsistently) {
    // Composition with fix 1: under D1 a team tree is adopted after a
    // landing, and every retained node carries the hash of its path time
    // charged at our own paces.
    constexpr const char* kSmall = "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1";
    Board root = boards(kSmall, kSmall);
    Rules r;
    r.ownPace = true;
    r.rootMoveMs = 100;
    r.paces = {180, 100, 5000, 140};
    Agent agent(1);
    const TimeState t0 = root_state(WHITE, r);
    search(agent, root, WHITE, options(t0, 800, true));
    std::shared_ptr<Node> node = AgentDiagFixesPeer::root(agent);
    Board line(root);
    TimeState time = t0;
    // White's most visited move on A, then black's most visited reply on A.
    for (int ply = 0; ply < 2; ++ply) {
        const std::vector<int> visits = node->get_child_visits();
        const std::vector<std::shared_ptr<Node>> children = node->get_children();
        int best = -1;
        for (size_t i = 0; i < children.size() && i < visits.size(); ++i) {
            const JointActionCandidate action = node->get_joint_action(static_cast<int>(i));
            if (children[i] && children[i]->is_expanded() && action.moveA != MOVE_NONE
                && action.moveB == MOVE_NONE && (best < 0 || visits[i] > visits[best])) {
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0) << "ply " << ply;
        const JointActionCandidate action = node->get_joint_action(best);
        time = durative_time_after(time, line, node->get_team_to_play(), action);
        line.make_moves(action.moveA, action.moveB);
        node = children[best];
    }
    ASSERT_EQ(node->get_team_to_play(), WHITE);
    // Aw came on turn at its own pace (180), not the budget (100).
    EXPECT_EQ(time.seat_ready_ms(AW), 180);
    // The live root a front end sends there: the path's clocks, Bw's elapsed
    // time matching its ready time, a new budget of 180.
    Rules live = r;
    live.rootMoveMs = 180;
    for (int seat = 0; seat < 4; ++seat) {
        live.clocks[seat] = time.seat_clock_ms(seat) + (seat == AB ? 4 : 0);
    }
    live.elapsed = {0, 0, live.paces[BW] - time.seat_ready_ms(BW), 0};
    Board liveBoard;
    liveBoard.set(line.fen(BOARD_A) + " | " + line.fen(BOARD_B));
    const TimeState liveTime = root_state(WHITE, live);
    search(agent, liveBoard, WHITE, options(liveTime, 64, true));
    const TeamTreeReuseReport& reuse = agent.team_tree_reuse();
    ASSERT_TRUE(reuse.adopted) << reuse.reason;
    EXPECT_EQ(AgentDiagFixesPeer::root(agent), node);
    int checked = 0;
    Board check(liveBoard);
    walk_once(node.get(), check, liveTime, [&](Node* n, Board& at, const TimeState& t) {
        if (n->get_hash() != 0) {
            EXPECT_EQ(n->get_hash(), at.search_hash_key(n->get_team_to_play(), t));
            ++checked;
        }
    });
    EXPECT_GT(checked, 1);
}

// ---- D3: wait nodes evaluated from the acting team ------------------------------

TEST(WaitNodeActingEval, PlanesTeamOnlySwapsAtAForcedWaitWithTheOption) {
    Rules off;
    Rules on;
    on.waitEval = true;
    const TimeState offTime = root_state(WHITE, off);
    const TimeState onTime = root_state(WHITE, on);
    EXPECT_FALSE(offTime.wait_node_acting_eval());
    EXPECT_TRUE(onTime.wait_node_acting_eval());
    // Not part of the clock's identity: an evaluation choice, not a time.
    EXPECT_EQ(offTime.exact_key(), onTime.exact_key());
    EXPECT_EQ(wait_node_planes_team(WHITE, true, offTime), WHITE);
    EXPECT_EQ(wait_node_planes_team(WHITE, false, onTime), WHITE);
    EXPECT_EQ(wait_node_planes_team(WHITE, true, onTime), BLACK);
    EXPECT_EQ(wait_node_planes_team(BLACK, true, onTime), WHITE);
    // Outside a durative clock there are no wait nodes.
    EXPECT_EQ(wait_node_planes_team(WHITE, true, TimeState::clocked(WHITE, 0)), WHITE);
}

class WaitNodeActingEvalIt13 : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        const auto model = std::filesystem::path(__FILE__).parent_path().parent_path()
            / "models" / "twin-s-clock-it13.onnx";
        if (!std::filesystem::exists(model)) {
            GTEST_SKIP() << "needs engine/models/twin-s-clock-it13.onnx";
        }
        engine = std::make_unique<Engine>(0, 1);
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    }

    /** The network's scalar value for @p team to play on @p board at @p time. */
    float value(Board& board, Stockfish::Color team, const TimeState& time) {
        std::vector<float> obs(NB_INPUT_VALUES());
        board_to_planes(board, obs.data(), team, time);
        std::vector<float> out(1 + 2 * NB_POLICY_VALUES() + 3 + 1);
        float* v = out.data();
        float* piA = v + 1;
        float* piB = piA + NB_POLICY_VALUES();
        float* wdl = piB + NB_POLICY_VALUES();
        EXPECT_TRUE(engine->runInference(obs.data(), v, piA, piB, wdl, wdl + 3, 0));
        return std::clamp(*v, -1.0f, 1.0f);
    }

    std::unique_ptr<Engine> engine;
};

TEST_F(WaitNodeActingEvalIt13, WaitLeafValueComesFromTheActingTeamNegated) {
    // White decides on A; Bw (black) lands at 50 during our budget and Ab at
    // 100: below the root black lands twice in a row, so white nodes there
    // are forced waits. At every wait leaf with one visit the node's value
    // is the network's: off, from the waiting team's planes; on, the
    // negation of the acting team's.
    Board root;
    root.set(std::string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1") + " | "
             + "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");
    int differing = 0;
    for (const bool on : {false, true}) {
        Rules r;
        r.waitEval = on;
        r.rootMoveMs = 100;
        r.paces = {100, 100, 100, 100};
        r.elapsed = {0, 0, 50, 0};
        const TimeState t0 = root_state(WHITE, r);
        Agent agent(1);
        SearchOptions opts;
        opts.targetNodes = 300;
        opts.enablePonder = false;
        opts.clock = t0;
        opts.search.enableRootMateSearch = false;
        opts.search.certifySelectedMove = false;
        opts.search.enableMateProbe = false;
        opts.search.enableWdlEval = false;
        opts.search.movesLeftDiscount = 0.0f;
        testing::internal::CaptureStdout();
        agent.run_search(root, {engine.get()}, WHITE, false, opts);
        testing::internal::GetCapturedStdout();
        int waits = 0;
        Board walk(root);
        walk_once(AgentDiagFixesPeer::root(agent).get(), walk, t0,
                  [&](Node* node, Board& at, const TimeState& t) {
            if (!node->is_expanded() || node->get_visits() != 1
                || node->get_node_type() != NodeType::UNSOLVED) {
                return;
            }
            const Stockfish::Color team = node->get_team_to_play();
            if (!leaf_turn(at, team, t).forcedWait) {
                return;
            }
            ++waits;
            const float waiting = value(at, team, t);
            const float acting = -value(at, ~team, t);
            differing += std::abs(waiting - acting) > 0.05f ? 1 : 0;
            EXPECT_NEAR(node->Q(), on ? acting : waiting, 0.02f)
                << (on ? "on" : "off") << " waiting " << waiting << " acting " << acting;
        });
        EXPECT_GT(waits, 0) << (on ? "on" : "off");
    }
    // The test separates the two: the network's tempo bias shows at some leaf.
    EXPECT_GT(differing, 0);
}
#endif

// ---- D5: a drop-blockable check is not mate --------------------------------------

TEST(BlockableCheckNotMate, OffTheEngineCallsTheRookCheckMate) {
    init_fairy_stockfish();
    Board board = boards(kRookCheckAfter, kQuietPartner);
    ASSERT_TRUE(board.is_in_check(BOARD_A));
    ASSERT_FALSE(board.has_any_legal_move(BOARD_A));
    // Bw (Ab's partner) is on turn but has nothing to capture.
    EXPECT_TRUE(board.is_checkmate(BLACK, true));
    EXPECT_TRUE(board.is_checkmate(BLACK, false));
}

TEST(BlockableCheckNotMate, OnMatchesTheArena) {
    init_fairy_stockfish();
    Board board = boards(kRookCheckAfter, kQuietPartner);
    board.set_blockable_check_not_mate(true);
    // Six empty squares between rook and king: a drop could block, so the
    // arena plays on and Ab waits.
    EXPECT_FALSE(board.is_checkmate(BLACK, true));
    EXPECT_TRUE(board.can_partner_provide_blocking_piece(BOARD_A, BLACK, false));
    // A copy keeps the rule (searches and provers work on copies).
    Board copy(board);
    EXPECT_FALSE(copy.is_checkmate(BLACK, true));
    // Unblockable checks stay mate (arena: contact, knight, pawn, double).
    for (const char* fen : {"kQ6/ppK5/8/8/8/8/8/8[] b - - 0 1",       // contact, defended
                            "kr6/ppN5/8/8/8/8/8/4K3[] b - - 0 1"}) {  // smothered knight
        Board unblockable = boards(fen, kQuietPartner);
        unblockable.set_blockable_check_not_mate(true);
        ASSERT_TRUE(unblockable.is_in_check(BOARD_A)) << fen;
        EXPECT_TRUE(unblockable.is_checkmate(BLACK, true)) << fen;
    }
}

TEST(BlockableCheckNotMate, SitGateStuckRuleStillApplies) {
    // The option changes the check verdict only. Under SitGate with no lead
    // (the team may not sit) a seat with no legal move whose partner cannot
    // free it still loses by the engine's stuck rule; the arena has no gate,
    // so this row stays the engine's own (docs/wave2_engine.md).
    init_fairy_stockfish();
    Board board = boards(kRookCheckAfter, kQuietPartner);
    board.set_blockable_check_not_mate(true);
    EXPECT_TRUE(board.is_checkmate(BLACK, false));
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
/** Whether a search from kRookCheckBefore proves Rh1-h8 (root WIN or a lost child). */
bool proves_rook_check(DiagFixesSearch& fixture, bool on, const TimeState& clock, bool provers,
                       const std::string& label) {
    Board board = boards(kRookCheckBefore, kQuietPartner);
    board.set_blockable_check_not_mate(on);
    Agent agent(1);
    fixture.search(agent, board, WHITE, DiagFixesSearch::options(clock, 200, false, provers));
    bool lostChild = false;
    for (const RootEdgeStats& edge : agent.root_edge_stats()) {
        if (edge.action.moveA != MOVE_NONE && board.uci_move(BOARD_A, edge.action.moveA) == "h1h8"
            && edge.childType == NodeType::LOSS) {
            lostChild = true;
        }
    }
    EXPECT_EQ(lostChild, agent.root_type() == NodeType::WIN) << label;
    return agent.root_type() == NodeType::WIN;
}

TEST_F(DiagFixesSearch, SearchNoLongerProvesTheRookCheck) {
    // The tree's own verdict in the arena's headline setting (durative tree,
    // SitGate off), root provers off.
    Rules level;
    level.paces = {100, 100, 100, 100};
    const TimeState durative = root_state(WHITE, level);
    EXPECT_TRUE(proves_rook_check(*this, false, durative, false, "durative off"));
    EXPECT_FALSE(proves_rook_check(*this, true, durative, false, "durative on"));
    // The v3 tree (DurativeTree off) with its provers, black ahead on the
    // lead so that black may sit there.
    const TimeState v3Ahead = TimeState::clocked(WHITE, -10000);
    EXPECT_TRUE(proves_rook_check(*this, false, v3Ahead, true, "v3 black ahead off"));
    EXPECT_FALSE(proves_rook_check(*this, true, v3Ahead, true, "v3 black ahead on"));
}

TEST_F(DiagFixesSearch, SitRightResidualsStayLosses) {
    // Pins what the option does not change (docs/wave2_engine.md, "Diagnosis
    // fixes", D5). The cause is not the check rule but the right to sit: a
    // team on turn on both boards that may not sit and has no move on one
    // of them has no legal action, which the engine scores as lost (the
    // stuck rule of Board::is_checkmate). The arena has no sit right: Ab
    // would wait. It holds
    //  - in the v3 tree at a lead of 0 (v3 always gates sitting on the lead);
    //  - in the durative tree's root provers whatever the lead: the
    //    single-board proofs model the defenders as never allowed to sit
    //    while the attacker's partner sits (search_single_board_forced_mate,
    //    and search_reduced_partner_mate on the v3 clock).
    const TimeState v3Level = TimeState::clocked(WHITE, 0).with_sit_gate(false);
    EXPECT_TRUE(proves_rook_check(*this, true, v3Level, false, "v3 level on"));
    Rules level;
    level.paces = {100, 100, 100, 100};
    Rules blackAhead = level;
    blackAhead.clocks = {50000, 60000, 60000, 50000};   // white's lead -10 s
    for (const Rules& r : {level, blackAhead}) {
        EXPECT_TRUE(proves_rook_check(*this, true, root_state(WHITE, r), true,
                                      "durative provers on"));
    }
}
#endif

}  // namespace
