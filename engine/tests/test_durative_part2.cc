#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/time_state.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"
#include "search/searchthread.h"
#include "Fairy-Stockfish/src/uci.h"

// Wave 2, engine part 2: delayed transfer (InstantTransfer false), mate races
// by landing time (MateRaceByLanding), repetition under the durative clock,
// and spec v4 acceptance tests 2, 5 and 6 (docs/wave2_engine.md).

class AgentDurativePart2Peer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
};

namespace {

Stockfish::Move uci_move(Board& board, int boardNum, std::string text) {
    return Stockfish::UCI::to_move(*board.pos[boardNum], text);
}

using Stockfish::BLACK;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

struct Wave2Setup {
    std::array<int32_t, 4> clocks{60000, 60000, 60000, 60000};
    int32_t rootMoveMs = 100;
    std::array<int32_t, 4> paces{100, 100, 100, 100};
    bool flagLoss = true;
    bool sitGate = false;
    bool instantTransfer = false;
    bool mateRaceByLanding = true;
};

TimeState wave2_state(Stockfish::Color rootTeam, Stockfish::Color sideA,
                      Stockfish::Color sideB, const Wave2Setup& setup) {
    DurativeConfig config;
    config.flagLoss = setup.flagLoss;
    config.instantTransfer = setup.instantTransfer;
    config.mateRaceByLanding = setup.mateRaceByLanding;
    config.rootMoveTimeMs = setup.rootMoveMs;
    config.seatClocksMs = setup.clocks;
    config.seatPaceMs = setup.paces;
    const TimeState base = TimeState::clocked(rootTeam, 0, 100, 100)
        .with_sit_gate(setup.sitGate);
    return TimeState::durative(base, rootTeam, sideA, sideB, config);
}

// ---- delayed transfer, rules level ----------------------------------------

TEST(DurativeTransfer, ACaptureMidDecisionWaitsForThePartnersNextDecision) {
    // Root white team: Aw on A at 50 ms. Black team: Ab (pace 100) and Bw
    // (pace 160), who has been deciding since before the root.
    Wave2Setup setup;
    setup.rootMoveMs = 50;
    setup.paces = {100, 100, 160, 100};
    TimeState time = wave2_state(WHITE, WHITE, WHITE, setup);
    time.advance_action(WHITE, true, false);  // root: Aw lands at 50
    // Ab lands at 150, Bw at 160. Ab's capture sends a knight to Bw.
    ASSERT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
    time.advance_action(BLACK, true, false, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::KNIGHT), 1);
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::PAWN), 0);
    EXPECT_EQ(time.last_capture_team(), BLACK);
    // Bw lands 10 ms later, on the decision it started before the capture:
    // the knight is not its to drop.
    ASSERT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{false, true}));
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::KNIGHT), 1);
    time.advance_action(BLACK, false, true);
    // Aw, then Bb; when Bb hands B back, Bw's new decision has the knight.
    ASSERT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    time.advance_action(WHITE, true, false);
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::KNIGHT), 1);
    ASSERT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{false, true}));
    time.advance_action(WHITE, false, true);
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::KNIGHT), 0);
}

TEST(DurativeTransfer, ASeatBackFromAPassMayDropWhatLandedMeanwhile) {
    // White team on both boards at the root: Aw captures, Bb passes. The
    // knight reaches Bb during its pass; its next decision may drop it.
    Wave2Setup setup;
    TimeState time = wave2_state(WHITE, WHITE, BLACK, setup);
    time.advance_action(WHITE, true, false, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_TRUE(time.seat_sitting(BB));
    EXPECT_EQ(time.pending_in_hand(BB, Stockfish::KNIGHT), 0);
    // A seat in the middle of a decision does not: with Bb moving instead,
    // the knight lands during Bb's own landing and waits for its next turn.
    TimeState moved = wave2_state(WHITE, WHITE, BLACK, setup);
    moved.advance_action(WHITE, true, true, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_EQ(moved.pending_in_hand(BB, Stockfish::KNIGHT), 1);
    // Bw replies; B is Bb's again, a new decision: the knight is usable.
    ASSERT_EQ(moved.acting_boards(BLACK), (std::array<bool, 2>{true, true}));
    moved.advance_action(BLACK, true, true);
    EXPECT_EQ(moved.pending_in_hand(BB, Stockfish::KNIGHT), 0);
}

TEST(DurativeTransfer, InstantTransferAndV3StatesNeverHoldPieces) {
    Wave2Setup setup;
    setup.instantTransfer = true;
    setup.rootMoveMs = 50;
    setup.paces = {100, 100, 500, 100};
    TimeState time = wave2_state(WHITE, WHITE, WHITE, setup);
    time.advance_action(WHITE, true, false);
    time.advance_action(BLACK, true, false, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_EQ(time.pending_in_hand(BW, Stockfish::KNIGHT), 0);
    EXPECT_EQ(time.last_capture_team(), BLACK);
    const TimeState v3 = TimeState::clocked(WHITE, 0);
    EXPECT_EQ(v3.pending_in_hand(BW, Stockfish::KNIGHT), 0);
    EXPECT_FALSE(v3.last_capture_team().has_value());
}

TEST(DurativeTransfer, PendingPiecesSeparateNodeKeys) {
    Wave2Setup setup;
    setup.rootMoveMs = 50;
    setup.paces = {100, 100, 500, 100};
    TimeState time = wave2_state(WHITE, WHITE, WHITE, setup);
    time.advance_action(WHITE, true, false);
    const TimeState quiet = time.after_action(BLACK, true, false);
    const TimeState capture = time.after_action(
        BLACK, true, false, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_NE(quiet.exact_key(), capture.exact_key());
    EXPECT_FALSE(quiet == capture);
}

TEST(DurativeTransfer, HandPieceTypeIsWhatThePartnerReceives) {
    init_fairy_stockfish();
    Board board;
    // exd5 takes a knight; a promoted queen (Q~) goes over as a pawn.
    board.set_fen(BOARD_A, "4k3/8/8/3n4/4P3/8/8/4K3[] w - - 0 1");
    board.set_fen(BOARD_B, "3Q~k3/8/8/8/8/8/8/4K3[] b - - 0 1");
    const Stockfish::Move exd5 = uci_move(board, BOARD_A, "e4d5");
    const Stockfish::Move kxd8 = uci_move(board, BOARD_B, "e8d8");
    const Stockfish::Move quiet = uci_move(board, BOARD_A, "e1d1");
    EXPECT_EQ(board.hand_piece_type(BOARD_A, exd5), Stockfish::KNIGHT);
    EXPECT_EQ(board.hand_piece_type(BOARD_B, kxd8), Stockfish::PAWN);
    EXPECT_EQ(board.hand_piece_type(BOARD_A, quiet), Stockfish::NO_PIECE_TYPE);
    EXPECT_EQ(board.hand_piece_type(BOARD_A, Stockfish::MOVE_NONE), Stockfish::NO_PIECE_TYPE);
}

// ---- landing-time mate races, rules level ---------------------------------

TEST(DurativeMateRace, RootActionRecordsHowMuchEarlierTheOtherSeatLands) {
    Wave2Setup setup;
    setup.rootMoveMs = 500;
    setup.paces = {300, 100, 100, 100};
    // Black team root on B (Bw); Aw (white team) would land at 300.
    TimeState time = wave2_state(BLACK, WHITE, WHITE, setup);
    EXPECT_EQ(time.mate_race_early_ms(AW), -1);
    time.advance_action(BLACK, false, true);
    EXPECT_EQ(time.mate_race_early_ms(AW), 200);
    EXPECT_EQ(time.mate_race_early_ms(BB), -1);
    // Once Aw acts the race is over.
    time.advance_action(WHITE, true, false);
    EXPECT_EQ(time.mate_race_early_ms(AW), -1);
}

TEST(DurativeMateRace, OptionOffOrV3StatesHaveNoLandingRace) {
    Wave2Setup setup;
    setup.mateRaceByLanding = false;
    EXPECT_FALSE(wave2_state(WHITE, WHITE, WHITE, setup).mate_race_by_landing());
    EXPECT_FALSE(TimeState::clocked(WHITE, 0).mate_race_by_landing());
    EXPECT_EQ(TimeState::clocked(WHITE, 0).mate_race_early_ms(AW), -1);
}

// ---- repetition under the durative clock (acceptance test 6, rules) ----

TEST(DurativeRepetition, AThreefoldIsADrawInV3ButNotUnderTreeClocks) {
    init_fairy_stockfish();
    Board board;
    board.set_fen(BOARD_A, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");
    board.set_fen(BOARD_B, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");
    for (int cycle = 0; cycle < 2; ++cycle) {
        for (const char* uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
            board.push_move(BOARD_A, uci_move(board, BOARD_A, uci));
        }
    }
    ASSERT_TRUE(board.is_draw());
    const std::array<int, 2> plies{1, 0};
    const TimeState v3 = TimeState::clocked(WHITE, 0).with_sit_gate(false);
    EXPECT_EQ(classify_terminal_position(board, WHITE, WHITE, v3, plies),
              TerminalOutcome::DRAW);
    Wave2Setup setup;
    const TimeState durative = wave2_state(WHITE, WHITE, WHITE, setup);
    EXPECT_EQ(classify_terminal_position(board, WHITE, WHITE, durative, plies),
              TerminalOutcome::NONE);
}

// ---- inside a real search ---------------------------------------------------

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class DurativePart2SearchTest : public ::testing::Test {
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

    static SearchOptions options(const TimeState& clock, size_t nodes) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.clock = clock;
        return opts;
    }

    std::unique_ptr<Engine> engine;
};

/** Walks the searched tree, charging each edge as the search did. */
void walk_tree(Node* node, Board& board, const TimeState& time, int depth,
               const std::function<void(Node*, Board&, const TimeState&,
                                        const JointActionCandidate&)>& onEdge) {
    if (!node || depth > 40 || !node->is_expanded()) {
        return;
    }
    const Stockfish::Color team = node->get_team_to_play();
    const std::vector<std::shared_ptr<Node>> children = node->get_children();
    for (size_t i = 0; i < children.size(); ++i) {
        const JointActionCandidate action = node->get_joint_action(static_cast<int>(i));
        onEdge(node, board, time, action);
        if (!children[i]) {
            continue;
        }
        const TimeState next = time.is_durative()
            ? time.after_action(team, action.moveA != Stockfish::MOVE_NONE,
                                action.moveB != Stockfish::MOVE_NONE,
                                board.hand_piece_type(BOARD_A, action.moveA),
                                board.hand_piece_type(BOARD_B, action.moveB))
            : time.after(team, board.on_turn_on_both_boards(team));
        board.make_moves(action.moveA, action.moveB);
        walk_tree(children[i].get(), board, next, depth + 1, onEdge);
        board.unmake_moves(action.moveA, action.moveB);
    }
}

constexpr const char* kKnightHangsA = "4k3/8/4p3/3N4/8/8/8/4K3[] w - - 0 1";
constexpr const char* kBareB = "4k3/8/8/8/8/8/8/4K3[] w - - 0 1";

/**
 * Bw's knight drops in the tree at nodes where the knight in its pocket
 * landed mid-decision (@p blocked), and anywhere (@p dropped).
 */
void count_knight_drops(Agent& agent, Board& board, const TimeState& clock,
                        int& blockedNodes, int& blockedDrops, int& dropped) {
    walk_tree(AgentDurativePart2Peer::root(agent).get(), board, clock, 0,
              [&](Node*, Board& at, const TimeState& time, const JointActionCandidate& action) {
                  const bool bwOnTurn = at.side_to_move(BOARD_B) == WHITE;
                  const bool isKnightDrop = action.moveB != Stockfish::MOVE_NONE
                      && type_of(action.moveB) == Stockfish::DROP
                      && Stockfish::in_hand_piece_type(action.moveB) == Stockfish::KNIGHT;
                  dropped += isKnightDrop ? 1 : 0;
                  if (!bwOnTurn) {
                      return;
                  }
                  const int pending = time.pending_in_hand(BW, Stockfish::KNIGHT);
                  if (pending > 0
                      && at.count_in_hand(BOARD_B, WHITE, Stockfish::KNIGHT) <= pending) {
                      ++blockedNodes;
                      blockedDrops += isKnightDrop ? 1 : 0;
                  }
              });
}

TEST_F(DurativePart2SearchTest, DelayedTransferKeepsALandedKnightOutOfTheCurrentDecision) {
    // Root white team: Aw's knight on d5 hangs to ...exd5, which sends it to
    // Bw, who has been deciding since before the root (pace 500).
    Board board;
    board.set_fen(BOARD_A, kKnightHangsA);
    board.set_fen(BOARD_B, kBareB);
    Wave2Setup setup;
    setup.rootMoveMs = 50;
    setup.paces = {100, 100, 500, 100};
    const TimeState delayed = wave2_state(WHITE, WHITE, WHITE, setup);
    Agent agent(2);
    agent.run_search(board, {engine.get()}, WHITE, false, options(delayed, 3000));
    int blockedNodes = 0;
    int blockedDrops = 0;
    int dropped = 0;
    count_knight_drops(agent, board, delayed, blockedNodes, blockedDrops, dropped);
    EXPECT_GT(blockedNodes, 0);
    EXPECT_EQ(blockedDrops, 0);
    (void)dropped;

    setup.instantTransfer = true;
    const TimeState instant = wave2_state(WHITE, WHITE, WHITE, setup);
    Agent instantAgent(2);
    instantAgent.run_search(board, {engine.get()}, WHITE, false, options(instant, 3000));
    // The same nodes, with the knight usable at once: some edges drop it there.
    int instantDropsWhileDeciding = 0;
    walk_tree(AgentDurativePart2Peer::root(instantAgent).get(), board, instant, 0,
              [&](Node*, Board& at, const TimeState& time, const JointActionCandidate& action) {
                  if (at.side_to_move(BOARD_B) != WHITE
                      || action.moveB == Stockfish::MOVE_NONE
                      || type_of(action.moveB) != Stockfish::DROP
                      || time.seat_sitting(BW)) {
                      return;
                  }
                  // Bw still on its pre-root decision: B has not changed hands.
                  if (time.side_on(BOARD_B) == WHITE && at.count_in_hand(BOARD_B, WHITE, Stockfish::KNIGHT) > 0
                      && time.seat_clock_ms(BW) > 60000 - 500) {
                      ++instantDropsWhileDeciding;
                  }
              });
    EXPECT_GT(instantDropsWhileDeciding, 0);
}

// Spec v4 acceptance test 5: both teams have a mate in one, each on its own
// board (smothered Nf7#). Aw lands at 300 ms, Bw at 500 ms, from either root.
constexpr const char* kSmotherA = "6rk/6pp/8/6N1/8/8/8/6K1[] w - - 0 1";
constexpr const char* kSmotherB = "6rk/6pp/8/6N1/8/8/8/6K1[] w - - 0 1";
// The same with the mate a capture (Nxf7#), for the exact-tie rule.
constexpr const char* kSmotherTakesA = "6rk/5ppp/8/6N1/8/8/8/6K1[] w - - 0 1";
constexpr const char* kSmotherTakesB = "6rk/5ppp/8/6N1/8/8/8/6K1[] w - - 0 1";

NodeType mirror_root(Engine* engine, Stockfish::Color rootTeam, const char* fenA,
                     const char* fenB, int32_t awMs, int32_t bwMs, bool byLanding) {
    Board board;
    board.set_fen(BOARD_A, fenA);
    board.set_fen(BOARD_B, fenB);
    Wave2Setup setup;
    setup.mateRaceByLanding = byLanding;
    setup.paces = {awMs, 100, bwMs, 100};
    setup.rootMoveMs = rootTeam == WHITE ? awMs : bwMs;
    const TimeState clock = wave2_state(rootTeam, WHITE, WHITE, setup);
    Agent agent(2);
    SearchOptions opts;
    opts.targetNodes = 800;
    opts.enablePonder = false;
    opts.clock = clock;
    agent.run_search(board, {engine}, rootTeam, false, opts);
    return agent.root_type();
}

TEST_F(DurativePart2SearchTest, Test5_MirrorMateInOneEachSideIsWonByTheEarlierLanding) {
    // White team lands first: it wins from its own root and loses from black's.
    EXPECT_EQ(mirror_root(engine.get(), WHITE, kSmotherA, kSmotherB, 300, 500, true),
              NodeType::WIN);
    EXPECT_EQ(mirror_root(engine.get(), BLACK, kSmotherA, kSmotherB, 300, 500, true),
              NodeType::LOSS);
    // And mirrored: black team lands first.
    EXPECT_EQ(mirror_root(engine.get(), WHITE, kSmotherA, kSmotherB, 500, 300, true),
              NodeType::LOSS);
    EXPECT_EQ(mirror_root(engine.get(), BLACK, kSmotherA, kSmotherB, 500, 300, true),
              NodeType::WIN);
}

TEST_F(DurativePart2SearchTest, Test5_WithoutLandingRacesBothRootsClaimTheWin) {
    // The ablation: each root mates first in its own tree, V(A) = V(B) = +1.
    EXPECT_EQ(mirror_root(engine.get(), WHITE, kSmotherA, kSmotherB, 300, 500, false),
              NodeType::WIN);
    EXPECT_EQ(mirror_root(engine.get(), BLACK, kSmotherA, kSmotherB, 300, 500, false),
              NodeType::WIN);
}

TEST_F(DurativePart2SearchTest, ExactTieGoesToTheOpponentOfTheLastCapturer) {
    // Both mates are captures landing at the same millisecond: the root
    // team's capture is the last one, so the other team wins the tie, for
    // either team at the root.
    EXPECT_EQ(mirror_root(engine.get(), WHITE, kSmotherTakesA, kSmotherTakesB, 300, 300, true),
              NodeType::LOSS);
    EXPECT_EQ(mirror_root(engine.get(), BLACK, kSmotherTakesA, kSmotherTakesB, 300, 300, true),
              NodeType::LOSS);
    // A tie with no capture known keeps upstream's rule: the root's mate stands.
    EXPECT_EQ(mirror_root(engine.get(), WHITE, kSmotherA, kSmotherB, 300, 300, true),
              NodeType::WIN);
    EXPECT_EQ(mirror_root(engine.get(), BLACK, kSmotherA, kSmotherB, 300, 300, true),
              NodeType::WIN);
}

// Spec v4 acceptance test 2, "sit for the knight", as wave 2 rules make it.
// A: a knight drop on f7 mates (Aw needs a knight). B: our Bb wins Bw's knight
// in two moves (a queen check, Qa4+ or Qd4+, then Qxh4). Passing on A works only if Bw has to answer
// the check: with the sit gate off it may sit in check, and the sit-off rule
// then makes the lower clock move. Our lead makes that Bw; our deficit makes
// it Aw, which must move before the knight comes.
constexpr const char* kSitForKnightA = "6rk/6pp/8/8/8/8/8/6K1[] w - - 0 1";
constexpr const char* kSitForKnightB = "4k3/3q4/8/8/7N/8/8/K7[] b - - 0 1";

std::pair<NodeType, JointActionCandidate> sit_for_knight(
    Engine* engine, int32_t ourClock, int32_t theirClock, bool sitGate, size_t nodes) {
    Board board;
    board.set_fen(BOARD_A, kSitForKnightA);
    board.set_fen(BOARD_B, kSitForKnightB);
    Wave2Setup setup;
    setup.clocks = {ourClock, theirClock, theirClock, ourClock};
    setup.sitGate = sitGate;
    setup.rootMoveMs = 200;
    setup.paces = {200, 200, 200, 200};
    const TimeState clock = wave2_state(WHITE, WHITE, BLACK, setup);
    Agent agent(2);
    SearchOptions opts;
    opts.targetNodes = nodes;
    opts.enablePonder = false;
    opts.clock = clock;
    const JointActionCandidate choice = agent.run_search(board, {engine}, WHITE, false, opts);
    return {agent.root_type(), choice};
}

TEST_F(DurativePart2SearchTest, Test2_SitForTheKnightWithALeadNotWithADeficit) {
    const auto model = std::filesystem::path(__FILE__).parent_path().parent_path()
        / "models" / "twin-s-clock-it13.onnx";
    if (!std::filesystem::exists(model)) {
        GTEST_SKIP() << "needs engine/models/twin-s-clock-it13.onnx";
    }
    engine = std::make_unique<Engine>(0, 16);
    ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    // 10 s diagonal lead: the root passes on A and checks on B, a proven win.
    // Needs the real network: on the uniform test net the proof is too deep.
    const auto [leadType, leadChoice] = sit_for_knight(engine.get(), 65000, 55000, false, 20000);
    EXPECT_EQ(leadType, NodeType::WIN);
    EXPECT_EQ(leadChoice.moveA, Stockfish::MOVE_NONE);
    EXPECT_NE(leadChoice.moveB, Stockfish::MOVE_NONE);
    // 10 s deficit: Bw sits in check and Aw has to move first; no win.
    const auto [deficitType, deficitChoice] =
        sit_for_knight(engine.get(), 55000, 65000, false, 20000);
    EXPECT_NE(deficitType, NodeType::WIN);
    // With upstream's gate and a deficit, the pass beside a quiet move is
    // not even legal.
    const auto [gatedType, gatedChoice] = sit_for_knight(engine.get(), 55000, 65000, true, 2000);
    EXPECT_FALSE(gatedChoice.moveA == Stockfish::MOVE_NONE
                 && gatedChoice.moveB != Stockfish::MOVE_NONE);
    (void)gatedType;
    (void)deficitChoice;
}

TEST_F(DurativePart2SearchTest, Test6_RepetitionIsNeverASolverDrawUnderTreeClocks) {
    // Board A is one knight move (...Ng8, the black team's root action) from
    // a threefold; v3's tree calls lines that complete it draws, the
    // durative tree does not (clocks differ, so no sit-off draw can occur
    // either).
    Board board;
    board.set_fen(BOARD_A, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");
    board.set_fen(BOARD_B, kBareB);
    for (const char* uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"}) {
        board.push_move(BOARD_A, uci_move(board, BOARD_A, uci));
    }
    auto draws = [&](const TimeState& clock) {
        Agent agent(2);
        agent.run_search(board, {engine.get()}, BLACK, false, options(clock, 4000));
        int count = 0;
        std::function<void(Node*, int)> visit = [&](Node* node, int depth) {
            if (!node || depth > 40) {
                return;
            }
            count += node->get_node_type() == NodeType::DRAW ? 1 : 0;
            if (!node->is_expanded()) {
                return;
            }
            for (const std::shared_ptr<Node>& child : node->get_children()) {
                visit(child.get(), depth + 1);
            }
        };
        visit(AgentDurativePart2Peer::root(agent).get(), 0);
        return count;
    };
    Wave2Setup setup;
    setup.clocks = {61000, 60000, 60000, 61000};
    setup.flagLoss = false;
    EXPECT_EQ(draws(wave2_state(BLACK, BLACK, WHITE, setup)), 0);
    EXPECT_GT(draws(TimeState::clocked(BLACK, 1000).with_sit_gate(false)), 0);
}
#endif

}  // namespace
