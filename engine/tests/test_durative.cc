#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/planes.h"
#include "environment/time_state.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"

// Wave 2: the durative clock (UCI DurativeTree, SitGate, FlagLoss). Spec v4
// acceptance tests 1, 3, 4 and 7 (docs/wave2_engine.md); the stock
// bit-for-bit half of test 7 against the v3.0.3 binary is
// tests/stock_parity.sh.

class AgentDurativeTestPeer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
};

namespace {

using Stockfish::BLACK;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

/** A durative root for @p rootTeam; both boards at white to move, as at the start. */
TimeState durative_state(Stockfish::Color rootTeam,
                         std::array<int32_t, 4> clocks,
                         int32_t rootMoveMs,
                         std::array<int32_t, 4> paces = {100, 100, 100, 100},
                         bool flagLoss = true,
                         bool sitGate = false,
                         Stockfish::Color sideA = WHITE,
                         Stockfish::Color sideB = WHITE) {
    DurativeConfig config;
    config.flagLoss = flagLoss;
    config.rootMoveTimeMs = rootMoveMs;
    config.seatClocksMs = clocks;
    config.seatPaceMs = paces;
    const TimeState base = TimeState::clocked(rootTeam, 0, 100, 100)
        .with_sit_gate(sitGate);
    return TimeState::durative(base, rootTeam, sideA, sideB, config);
}

// ---- rules: who acts, how time moves --------------------------------------

TEST(DurativeClock, RootTeamDecidesAtTheRootThenEarliestReadySeatActs) {
    // White team on A (Aw), black team on B (Bw).
    TimeState time = durative_state(WHITE, {60000, 60000, 60000, 60000}, 1000,
                                     {100, 400, 300, 100});
    EXPECT_TRUE(time.at_root());
    EXPECT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    EXPECT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{false, false}));
    // Bw (black team) would land at 300, before our 1000 ms budget; at the
    // root we decide anyway, and our clock and theirs both run 1000 ms.
    time.advance_action(WHITE, true, false);
    EXPECT_FALSE(time.at_root());
    EXPECT_EQ(time.seat_clock_ms(AW), 59000);
    EXPECT_EQ(time.seat_clock_ms(BW), 59000);
    EXPECT_EQ(time.side_on(0), BLACK);
    // Now Ab (black team, pace 400) is on turn on A and Bw (ready 0) on B.
    EXPECT_EQ(time.seat_ready_ms(AB), 400);
    EXPECT_EQ(time.seat_ready_ms(BW), 0);
    EXPECT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{false, true}));
    EXPECT_FALSE(time.team_acts(WHITE));
    // Black's B seat moves; Bb (white team, our budget 1000) comes on turn.
    time.advance_action(BLACK, false, true);
    EXPECT_EQ(time.seat_ready_ms(BB), 1000);
    // A: Ab lands at 400, before Bb's 1000.
    EXPECT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
    time.advance_action(BLACK, true, false);
    // Ab and Bb only came on turn after the root move: 400 ms each.
    EXPECT_EQ(time.seat_clock_ms(AB), 60000 - 400);
    EXPECT_EQ(time.seat_clock_ms(BB), 60000 - 400);
    EXPECT_EQ(time.seat_ready_ms(BB), 600);
    // White team on both boards now: Aw at our budget 1000, Bb at 600.
    EXPECT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{false, true}));
}

TEST(DurativeClock, ExactTiesAlternateStartingWithTheRootTeamsOpponents) {
    // Every seat paced 500 and the root budget 500: after the root move the
    // two seats on turn (one per team) tie at every event.
    TimeState time = durative_state(WHITE, {60000, 60000, 60000, 60000}, 500,
                                     {500, 500, 500, 500});
    time.advance_action(WHITE, true, false);  // root: Aw moves; Ab ready 500, Bw ready 0
    time.advance_action(BLACK, false, true);  // Bw lands; Bb (ours) ready 500
    // Ab (black) and Bb (white) both land at 500: the first tie goes to the
    // root team's opponents.
    EXPECT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
    EXPECT_FALSE(time.team_acts(WHITE));
    time.advance_action(BLACK, true, false);
    // Bb now at 0, Aw at 500: no tie; Bb acts.
    EXPECT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{false, true}));
    time.advance_action(WHITE, false, true);
    // Aw (white) and Bw (black) both at 500: this tie goes to white.
    EXPECT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    EXPECT_FALSE(time.team_acts(BLACK));
}

TEST(DurativeClock, PassLastsUntilJustAfterTheOtherBoardsNextEventCapped) {
    TimeState time = durative_state(WHITE, {60000, 60000, 60000, 60000}, 200,
                                     {100, 100, 3000, 100});
    // Root passes on A; Bw lands at 3000 - 200 = 2800 from then, but the pass
    // is capped at PassCapK (8) x 200 = 1600.
    ASSERT_TRUE(time.can_sit(WHITE));
    time.advance_action(WHITE, false, false);
    EXPECT_TRUE(time.seat_sitting(AW));
    EXPECT_EQ(time.seat_ready_ms(AW), 1600);
    EXPECT_EQ(time.seat_ready_ms(BW), 2800);

    TimeState shortWait = durative_state(WHITE, {60000, 60000, 60000, 60000}, 200,
                                         {100, 100, 500, 100});
    shortWait.advance_action(WHITE, false, false);
    // Bw lands 300 ms later; the pass ends 1 ms after it.
    EXPECT_EQ(shortWait.seat_ready_ms(BW), 300);
    EXPECT_EQ(shortWait.seat_ready_ms(AW), 301);
    EXPECT_EQ(shortWait.acting_boards(BLACK), (std::array<bool, 2>{false, true}));

    // A pass never outlasts the passing seat's own flag.
    TimeState nearFlag = durative_state(WHITE, {500, 60000, 60000, 60000}, 200,
                                        {100, 100, 3000, 100});
    nearFlag.advance_action(WHITE, false, false);
    EXPECT_EQ(nearFlag.seat_ready_ms(AW), 300);
}

TEST(DurativeClock, SitGateOffMakesThePassLegalBehindOnTheClock) {
    // White team 5 s behind on the diagonal.
    const std::array<int32_t, 4> clocks{55000, 60000, 60000, 55000};
    const TimeState gated = durative_state(WHITE, clocks, 500, {100, 100, 100, 100},
                                           true, true);
    const TimeState open = durative_state(WHITE, clocks, 500);
    EXPECT_EQ(gated.lead_ms(WHITE), -5000);
    EXPECT_FALSE(gated.can_sit(WHITE));
    EXPECT_TRUE(open.can_sit(WHITE));
    // The network still sees upstream's sit right.
    EXPECT_FALSE(open.lead_sit_right(WHITE));
    EXPECT_TRUE(open.lead_sit_right(BLACK));
}

TEST(DurativeClock, SeatElapsedShortensTheOpponentsFirstLanding) {
    DurativeConfig config;
    config.rootMoveTimeMs = 700;
    config.seatPaceMs = {100, 100, 900, 100};
    config.seatElapsedMs = {0, 0, 600, 0};
    const TimeState time = TimeState::durative(
        TimeState::clocked(WHITE, 0), WHITE, WHITE, WHITE, config);
    EXPECT_EQ(time.seat_ready_ms(AW), 700);
    EXPECT_EQ(time.seat_ready_ms(BW), 300);
}

TEST(DurativeClock, LeadOnlyClocksDeriveFromTimeLeadMs) {
    DurativeConfig config;
    const TimeState time = TimeState::durative(
        TimeState::clocked(BLACK, 1501, 200, 300), BLACK, WHITE, WHITE, config);
    EXPECT_EQ(time.lead_ms(BLACK), 1501);
    EXPECT_EQ(time.lead_ms(WHITE), -1501);
    EXPECT_EQ(time.move_time_ms(BLACK), 200);
    EXPECT_EQ(time.move_time_ms(WHITE), 300);
    // Derived clocks sit far beyond any flag.
    EXPECT_GT(time.seat_clock_ms(AW), 1 << 28);
}

// ---- acceptance test 1: clocks 100 and 600 with a 1000 ms budget ----------

TEST(DurativeAcceptance, Test1_ClocksOf100And600WithA1000MsBudgetHaveOneLoser) {
    // Aw (white team) 100 ms, Bw (black team) 600 ms; both run while our
    // 1000 ms root move is in flight. 100 runs out first: white alone loses.
    TimeState low = durative_state(WHITE, {100, 60000, 600, 60000}, 1000);
    low.advance_action(WHITE, true, false);
    EXPECT_EQ(low.outcome(), TimeState::Outcome::BLACK_TEAM_WINS);
    EXPECT_EQ(low.outcome_for(WHITE), -1);
    EXPECT_EQ(low.outcome_for(BLACK), 1);

    TimeState high = durative_state(WHITE, {600, 60000, 100, 60000}, 1000);
    high.advance_action(WHITE, true, false);
    EXPECT_EQ(high.outcome(), TimeState::Outcome::WHITE_TEAM_WINS);

    // Same millisecond, different teams: a double flag, a draw (RULES.md).
    TimeState both = durative_state(WHITE, {100, 60000, 100, 60000}, 1000);
    both.advance_action(WHITE, true, false);
    EXPECT_EQ(both.outcome(), TimeState::Outcome::DRAW);

    // Without FlagLoss the clocks just run down, as in v3.
    TimeState noFlag = durative_state(WHITE, {100, 60000, 600, 60000}, 1000,
                                      {100, 100, 100, 100}, false);
    noFlag.advance_action(WHITE, true, false);
    EXPECT_EQ(noFlag.outcome(), TimeState::Outcome::NONE);
    EXPECT_EQ(noFlag.seat_clock_ms(AW), -900);
}

// ---- acceptance test 3: the symmetric sit-off -----------------------------

/**
 * White (root) passes on A; black's Bw then decides on B with @p bwDelta ms
 * more clock than Aw. Returns the state at Bw's decision.
 */
TimeState sit_off_at_black(int32_t bwDelta) {
    TimeState time = durative_state(WHITE, {30000, 60000, 30000 + bwDelta, 60000}, 500,
                                    {100, 100, 2000, 100});
    time.advance_action(WHITE, false, false);
    return time;
}

TEST(DurativeAcceptance, Test3_SymmetricSitOffIsADrawAndFlipsWithOneSecond) {
    // Equal clocks: black may sit too, and two seats sitting at equal clocks
    // across the teams is the arena's sit-off draw.
    TimeState equal = sit_off_at_black(0);
    ASSERT_EQ(equal.acting_boards(BLACK), (std::array<bool, 2>{false, true}));
    EXPECT_TRUE(equal.can_sit(BLACK));
    equal.advance_action(BLACK, false, false);
    EXPECT_EQ(equal.outcome(), TimeState::Outcome::DRAW);
    EXPECT_EQ(equal.outcome_for(WHITE), 0);

    // Black 1 s lower: it would flag first, so it must move.
    TimeState blackLower = sit_off_at_black(-1000);
    EXPECT_FALSE(blackLower.can_sit(BLACK));

    // Black 1 s higher: black may sit, and when white's pass ends white is
    // the lower clock and must move.
    TimeState whiteLower = sit_off_at_black(1000);
    EXPECT_TRUE(whiteLower.can_sit(BLACK));
    whiteLower.advance_action(BLACK, false, false);
    EXPECT_EQ(whiteLower.outcome(), TimeState::Outcome::NONE);
    ASSERT_EQ(whiteLower.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    EXPECT_FALSE(whiteLower.can_sit(WHITE));
    EXPECT_TRUE(whiteLower.can_sit(BLACK));
}

// ---- acceptance test 4: a seat with 3 s makes at most 3000/min moves ------

TEST(DurativeAcceptance, Test4_ASeatWith3sMakesAtMost30MovesInAnyRandomLine) {
    constexpr int32_t MIN_MOVE_MS = 100;
    std::mt19937 rng(20261119);
    int longest = 0;
    for (int line = 0; line < 2000; ++line) {
        // Root white; Aw has 3 s, everyone else plenty; all at the minimum.
        TimeState time = durative_state(
            WHITE, {3000, 600000, 600000, 600000}, MIN_MOVE_MS,
            {MIN_MOVE_MS, MIN_MOVE_MS, MIN_MOVE_MS, MIN_MOVE_MS});
        Stockfish::Color toPlay = WHITE;
        int awMoves = 0;
        for (int ply = 0; ply < 4000 && time.outcome() == TimeState::Outcome::NONE; ++ply) {
            const std::array<bool, 2> acts = time.acting_boards(toPlay);
            bool moved[2] = {false, false};
            if (acts[0] || acts[1]) {
                // Half the lines never pass, so the bound is reached.
                const bool mayPass = line % 2 == 1 && time.can_sit(toPlay)
                    && !(acts[0] && acts[1]);
                for (int board = 0; board < 2; ++board) {
                    moved[board] = acts[board] && (!mayPass || rng() % 4 != 0);
                }
                if (acts[0] && acts[1] && !moved[0] && !moved[1]) {
                    moved[rng() % 2] = true;
                }
            }
            // A move counts once it lands: one cut off by the flag does not.
            const bool awMoving = acts[0] && moved[0] && time.side_on(0) == WHITE;
            time.advance_action(toPlay, moved[0], moved[1]);
            if (awMoving && time.outcome() == TimeState::Outcome::NONE) {
                ++awMoves;
            }
            toPlay = ~toPlay;
        }
        EXPECT_NE(time.outcome(), TimeState::Outcome::NONE);
        longest = std::max(longest, awMoves);
    }
    // The 30th move would land with the clock at exactly 0: a flag.
    EXPECT_LE(longest, 3000 / MIN_MOVE_MS);
    EXPECT_EQ(longest, 3000 / MIN_MOVE_MS - 1);
}

// ---- acceptance test 7: defaults are stock v3.0.3 -------------------------

TEST(DurativeAcceptance, Test7_DefaultsLeaveTheV3ClockUnchanged) {
    const TimeState v3 = TimeState::clocked(WHITE, -1234, 250, 400, 700);
    const TimeState gated = v3.with_sit_gate(true);
    EXPECT_EQ(gated, v3);
    EXPECT_EQ(gated.exact_key(), v3.exact_key());
    EXPECT_FALSE(v3.is_durative());
    // Off, the gate changes who may sit and so must change the node key.
    EXPECT_NE(v3.with_sit_gate(false).exact_key(), v3.exact_key());
    // A legacy flag is always gated.
    const TimeState legacy = TimeState::legacy(BLACK, true);
    EXPECT_EQ(legacy.with_sit_gate(false), legacy);
    for (const TimeState& time : {v3, legacy, TimeState::clocked(BLACK, 900)}) {
        for (Stockfish::Color team : {WHITE, BLACK}) {
            EXPECT_EQ(time.lead_sit_right(team), time.can_sit(team));
            EXPECT_TRUE(time.team_acts(team));
            EXPECT_TRUE(time.instant_transfer());
            EXPECT_FALSE(time.mate_race_by_landing());
            EXPECT_FALSE(time.outcome_for(team).has_value());
        }
    }
}

class DurativePlanesTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

TEST_F(DurativePlanesTest, Test7_DefaultPlanesMatchV3AndDurativePlanesReadTheClocks) {
    Board board;
    std::vector<float> plain(NB_INPUT_VALUES()), gated(NB_INPUT_VALUES());
    for (int32_t lead : {-3000, 0, 1, 4500}) {
        const TimeState v3 = TimeState::clocked(WHITE, lead, 300, 200, 500);
        board_to_planes(board, plain.data(), WHITE, v3);
        board_to_planes(board, gated.data(), WHITE, v3.with_sit_gate(true));
        EXPECT_EQ(plain, gated);
    }
    // A durative state with the same lead and move times gives v3's planes:
    // the network's inputs do not change in wave 2.
    const TimeState v3 = TimeState::clocked(WHITE, 4000, 300, 200);
    DurativeConfig config;
    config.seatClocksMs = {62000, 60000, 60000, 62000};
    const TimeState durative = TimeState::durative(
        v3.with_sit_gate(false), WHITE, WHITE, WHITE, config);
    ASSERT_EQ(durative.lead_ms(WHITE), 2000);
    std::vector<float> fromClocks(NB_INPUT_VALUES());
    board_to_planes(board, fromClocks.data(), WHITE, durative);
    board_to_planes(board, plain.data(), WHITE, TimeState::clocked(WHITE, 2000, 300, 200));
    EXPECT_EQ(fromClocks, plain);
}

// ---- the same rules inside a real search ----------------------------------

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class DurativeSearchTest : public ::testing::Test {
protected:
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

    SearchOptions options(const TimeState& clock, size_t nodes) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.clock = clock;
        return opts;
    }

    std::unique_ptr<Engine> engine;
};

TEST_F(DurativeSearchTest, Test1_EveryRootActionOfTheFlaggingTeamLoses) {
    Board board;
    Agent agent(2);
    const TimeState clock = durative_state(WHITE, {100, 60000, 600, 60000}, 1000);
    agent.run_search(board, {engine.get()}, WHITE, false, options(clock, 200));
    const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
    ASSERT_FALSE(edges.empty());
    for (const RootEdgeStats& edge : edges) {
        // Child types are from the child's (black's) view: a black win.
        EXPECT_EQ(edge.childType, NodeType::WIN);
    }
    EXPECT_EQ(agent.root_type(), NodeType::LOSS);

    Agent mirrored(2);
    const TimeState other = durative_state(WHITE, {600, 60000, 100, 60000}, 1000);
    mirrored.run_search(board, {engine.get()}, WHITE, false, options(other, 200));
    EXPECT_EQ(mirrored.root_type(), NodeType::WIN);
}

/**
 * Longest count of Aw moves along any line of the searched tree, and how many
 * edges end the game on the clock. Wait nodes (pass, pass) are walked through.
 */
void walk(Node* node, Board& board, const TimeState& time, int awMoves,
          int& longest, int& clockEnds, int depth = 0) {
    longest = std::max(longest, awMoves);
    if (!node || depth > 200) {
        return;
    }
    const Stockfish::Color team = node->get_team_to_play();
    const std::vector<std::shared_ptr<Node>> children = node->get_children();
    for (size_t i = 0; i < children.size(); ++i) {
        if (!children[i]) {
            continue;
        }
        const JointActionCandidate action = node->get_joint_action(static_cast<int>(i));
        const bool awMoved = action.moveA != Stockfish::MOVE_NONE
            && board.side_to_move(BOARD_A) == WHITE;
        const TimeState next = time.after_action(
            team, action.moveA != Stockfish::MOVE_NONE, action.moveB != Stockfish::MOVE_NONE);
        if (next.outcome() != TimeState::Outcome::NONE) {
            ++clockEnds;
        }
        board.make_moves(action.moveA, action.moveB);
        walk(children[i].get(), board, next, awMoves + (awMoved ? 1 : 0),
             longest, clockEnds, depth + 1);
        board.unmake_moves(action.moveA, action.moveB);
    }
}

TEST_F(DurativeSearchTest, Test4_NoLineGivesASeatMoreMovesThanItsClockAllows) {
    // Aw has 300 ms at a 100 ms budget: two moves, and the third would land
    // at 0. Lines that try more end on the clock.
    Board board;
    const TimeState clock = durative_state(WHITE, {300, 600000, 600000, 600000}, 100);
    Agent agent(2);
    agent.run_search(board, {engine.get()}, WHITE, false, options(clock, 1500));
    int longest = 0;
    int clockEnds = 0;
    walk(AgentDurativeTestPeer::root(agent).get(), board, clock, 0, longest, clockEnds);
    EXPECT_LE(longest, 2);
}

TEST_F(DurativeSearchTest, Test4_ALowSeatsLinesEndOnTheClock) {
    // Bw (black team) has 150 ms: our 100 ms root move leaves it 50, and a
    // pass runs it out three plies later (pass, wait, landing). The uniform
    // test network spreads visits evenly, so it needs a few thousand nodes.
    Board board;
    const TimeState clock = durative_state(WHITE, {600000, 600000, 150, 600000}, 100);
    Agent agent(2);
    agent.run_search(board, {engine.get()}, WHITE, false, options(clock, 6000));
    int longest = 0;
    int clockEnds = 0;
    walk(AgentDurativeTestPeer::root(agent).get(), board, clock, 0, longest, clockEnds);
    EXPECT_GT(clockEnds, 0);
}

TEST_F(DurativeSearchTest, AWaitingTeamGetsTheSinglePassPassAction) {
    // Bw (black team) paced 50 ms, everyone else 1000: after our root move,
    // black lands on B first and white's nodes only wait.
    Board board;
    Agent agent(2);
    const TimeState clock = durative_state(WHITE, {60000, 60000, 60000, 60000}, 1000,
                                           {1000, 1000, 50, 1000});
    agent.run_search(board, {engine.get()}, WHITE, false, options(clock, 300));
    std::shared_ptr<Node> root = AgentDurativeTestPeer::root(agent);
    ASSERT_TRUE(root);
    int waits = 0;
    std::function<void(Node*, TimeState, int)> visit =
        [&](Node* node, TimeState time, int depth) {
            if (!node || depth > 6 || !node->is_expanded()) {
                return;
            }
            const Stockfish::Color team = node->get_team_to_play();
            const std::vector<std::shared_ptr<Node>> children = node->get_children();
            if (!time.team_acts(team)) {
                ++waits;
                ASSERT_EQ(children.size(), 1u);
                const JointActionCandidate action = node->get_joint_action(0);
                EXPECT_EQ(action.moveA, Stockfish::MOVE_NONE);
                EXPECT_EQ(action.moveB, Stockfish::MOVE_NONE);
            }
            for (size_t i = 0; i < children.size(); ++i) {
                const JointActionCandidate action =
                    node->get_joint_action(static_cast<int>(i));
                visit(children[i].get(),
                      time.after_action(team, action.moveA != Stockfish::MOVE_NONE,
                                        action.moveB != Stockfish::MOVE_NONE),
                      depth + 1);
            }
        };
    visit(root.get(), clock, 0);
    EXPECT_GT(waits, 0);
}
#endif

}  // namespace
