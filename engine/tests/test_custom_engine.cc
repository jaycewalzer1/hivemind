// Custom-engine additions (docs/custom_engine_spec.md section 1): FreePass,
// clock planes, 'position ... clocks', TreeMoveMs tree clocks and flags, the
// 'go ... sample' serializer. Every feature is off by default; the first test
// of each group checks the stock behaviour is untouched.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/joint_action.h"
#include "environment/planes.h"
#include "interface/uci.h"
#include "search/searchthread.h"
#include "tools/training_sample.h"

class UCIClockTestPeer {
public:
    static Board& board(UCI& uci) { return uci.board; }
    static int tree_move_ms(UCI& uci) { return uci.treeMoveMs; }
    static SearchParams::RuntimeConfig config(UCI& uci) {
        return uci.current_search_config();
    }
};

namespace {

void initialize_engine_state() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        init_fairy_stockfish();
        init_policy_index();
    });
}

void uci_command(UCI& uci, const std::string& command) {
    std::istringstream input(command);
    std::string token;
    input >> token;
    if (token == "position") {
        uci.position(input);
    } else if (token == "setoption") {
        uci.setoption(input);
    }
}

class CustomEngineTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { initialize_engine_state(); }
    void TearDown() override {
        g_freePass = false;
        g_requiredMoveBoard = REQUIRE_MOVE_NONE;
    }
};

float plane_at(const std::vector<float>& planes, int channel, int square = 0) {
    return planes[static_cast<size_t>(channel) * 64 + static_cast<size_t>(square)];
}

// ---------------------------------------------------------------- FreePass

TEST_F(CustomEngineTest, FreePassOffKeepsStockPassRules) {
    g_freePass = false;
    // Single pass with both boards on turn needs time advantage or a capture.
    EXPECT_FALSE(is_single_pass_legal(false, true, true, false));
    EXPECT_TRUE(is_single_pass_legal(false, true, true, true));
    EXPECT_TRUE(is_single_pass_legal(true, true, true, false));
    // Double sit needs time advantage and exactly one board on turn.
    EXPECT_FALSE(is_double_sit_legal(false, true, false));
    EXPECT_FALSE(is_double_sit_legal(true, true, true));
    EXPECT_TRUE(is_double_sit_legal(true, true, false));

    const JointActionRules bothOnTurn{true, true, false, true, true};
    const Stockfish::Move someMove = Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4);
    EXPECT_FALSE(is_joint_action_legal(
        bothOnTurn, Stockfish::MOVE_NONE, someMove, false, false));
    EXPECT_FALSE(is_joint_action_legal(
        bothOnTurn, Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, false, false));
}

TEST_F(CustomEngineTest, FreePassOnAllowsAnyPassWithoutTimeAdvantage) {
    g_freePass = true;
    EXPECT_TRUE(is_single_pass_legal(false, true, true, false));
    EXPECT_TRUE(is_double_sit_legal(false, true, true));
    EXPECT_TRUE(is_double_sit_legal(false, true, false));
    EXPECT_TRUE(is_double_sit_legal(false, false, true));
    // Not on turn anywhere: the pass is the only action and stays legal.
    EXPECT_TRUE(is_double_sit_legal(false, false, false));

    const Stockfish::Move someMove = Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4);
    const JointActionRules bothOnTurn{true, true, false, true, true};
    EXPECT_TRUE(is_joint_action_legal(
        bothOnTurn, Stockfish::MOVE_NONE, someMove, false, false));
    EXPECT_TRUE(is_joint_action_legal(
        bothOnTurn, someMove, Stockfish::MOVE_NONE, false, false));
    EXPECT_TRUE(is_joint_action_legal(
        bothOnTurn, Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, false, false));
    EXPECT_TRUE(is_joint_action_legal(bothOnTurn, someMove, someMove, false, false));
}

TEST_F(CustomEngineTest, FreePassStillHonoursRequireMoveOn) {
    g_freePass = true;
    g_requiredMoveBoard = REQUIRE_MOVE_BOARD_A;
    const Stockfish::Move someMove = Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4);
    const JointActionRules bothOnTurn{true, true, false, true, true};
    EXPECT_FALSE(is_joint_action_legal(
        bothOnTurn, Stockfish::MOVE_NONE, someMove, false, false));
}

TEST_F(CustomEngineTest, FreePassJointGeneratorOffersEveryPassCombination) {
    // Start position, white team: on turn on A only (B: white to move = the
    // other team). With FreePass a pass on A is offered even without time
    // advantage; without it the team must move on A.
    Board board;
    const std::vector<Stockfish::Move> movesA = board.legal_moves(BOARD_A);
    std::vector<Stockfish::Move> actionsA = movesA;
    actionsA.push_back(Stockfish::MOVE_NONE);
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};
    std::vector<float> priorsA(actionsA.size(), 1.0f / actionsA.size());
    std::vector<float> priorsB = {1.0f};

    for (bool freePass : {false, true}) {
        g_freePass = freePass;
        const JointActionRules rules{true, false, false, true, false};
        size_t legal = 0;
        bool passLegal = false;
        for (Stockfish::Move moveA : actionsA) {
            if (is_joint_action_legal(rules, moveA, Stockfish::MOVE_NONE, false, false)) {
                ++legal;
                passLegal |= moveA == Stockfish::MOVE_NONE;
            }
        }
        EXPECT_EQ(passLegal, freePass);
        EXPECT_EQ(legal, movesA.size() + (freePass ? 1 : 0));
    }
}

TEST_F(CustomEngineTest, FreePassOptionDefaultsOffAndToggles) {
    UCI uci;
    EXPECT_FALSE(g_freePass);
    uci_command(uci, "setoption name FreePass value true");
    EXPECT_TRUE(g_freePass);
    uci_command(uci, "setoption name FreePass value false");
    EXPECT_FALSE(g_freePass);
}

TEST_F(CustomEngineTest, MateProbeAnyTimeAndTreeMoveMsOptions) {
    UCI uci;
    EXPECT_FALSE(UCIClockTestPeer::config(uci).mateProbeAnyTime);
    EXPECT_EQ(UCIClockTestPeer::tree_move_ms(uci), 0);
    uci_command(uci, "setoption name MateProbeAnyTime value true");
    uci_command(uci, "setoption name TreeMoveMs value 750");
    EXPECT_TRUE(UCIClockTestPeer::config(uci).mateProbeAnyTime);
    EXPECT_EQ(UCIClockTestPeer::tree_move_ms(uci), 750);
}

// ---------------------------------------------------- position ... clocks

TEST_F(CustomEngineTest, PositionWithoutClocksHasNone) {
    UCI uci;
    uci_command(uci, "position startpos moves 1e2e4 2d2d4");
    Board& board = UCIClockTestPeer::board(uci);
    EXPECT_FALSE(board.has_clocks());
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::BLACK);
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::BLACK);
}

TEST_F(CustomEngineTest, PositionParsesClocksAfterFenAndAfterMoves) {
    UCI uci;
    const std::string fen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | "
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    uci_command(uci, "position fen " + fen + " clocks 120000 119500 100000 90000.4");
    Board& board = UCIClockTestPeer::board(uci);
    ASSERT_TRUE(board.has_clocks());
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::WHITE), 120000);
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::BLACK), 119500);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::WHITE), 100000);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::BLACK), 90000);
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::WHITE);

    uci_command(uci, "position fen " + fen + " moves 1e2e4 clocks 1 2 3 4");
    ASSERT_TRUE(board.has_clocks());
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::BLACK);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::BLACK), 4);

    uci_command(uci, "position startpos clocks 5 6 7 8 ");
    ASSERT_TRUE(board.has_clocks());
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::BLACK), 6);

    // A later position without the suffix drops the clocks again.
    uci_command(uci, "position startpos moves 1e2e4");
    EXPECT_FALSE(board.has_clocks());
    // A malformed suffix is ignored rather than half-applied.
    uci_command(uci, "position startpos clocks 1 2 x 4");
    EXPECT_FALSE(board.has_clocks());
}

// --------------------------------------------------------------- planes

TEST_F(CustomEngineTest, ClocksNeverChangeStockChannels) {
    Board withClocks;
    withClocks.push_move(BOARD_A, Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4));
    Board noClocks(withClocks);
    withClocks.set_clocks(1000, 250000, 30000, 7000);
    withClocks.treeMoveMs = 500;

    for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
        for (bool timeAdvantage : {false, true}) {
            std::vector<float> a(NB_INPUT_VALUES()), b(NB_INPUT_VALUES());
            board_to_planes(withClocks, a.data(), team, timeAdvantage);
            board_to_planes(noClocks, b.data(), team, timeAdvantage);
            for (int boardBlock = 0; boardBlock < 2; ++boardBlock) {
                const int base = boardBlock * NB_INPUT_CHANNELS_PER_BOARD;
                for (int channel = 0; channel < NB_LEGACY_INPUT_CHANNELS_PER_BOARD; ++channel) {
                    for (int square = 0; square < 64; ++square) {
                        ASSERT_EQ(plane_at(a, base + channel, square),
                                  plane_at(b, base + channel, square))
                            << "board " << boardBlock << " channel " << channel;
                    }
                }
            }
            if constexpr (!HAS_CLOCK_PLANES) {
                EXPECT_EQ(a, b);
            }
        }
    }
}

TEST_F(CustomEngineTest, ClockPlanesAreTeamRelativePerBoard) {
    if constexpr (!HAS_CLOCK_PLANES) {
        EXPECT_EQ(NB_INPUT_CHANNELS_PER_BOARD, 37);
        GTEST_SKIP() << "clock planes exist only in the hivemind-clock build";
    }
    Board board;
    // A: white 150 s, black 600 s; B: white 5 s, black 0.
    board.set_clocks(150000, 600000, 5000, 0);
    const auto expect_block = [](const std::vector<float>& planes, int boardBlock,
                                 double ourSeconds, double theirSeconds) {
        const int base = boardBlock * NB_INPUT_CHANNELS_PER_BOARD + 37;
        const float expected[4] = {
            static_cast<float>(std::min(ourSeconds / 300.0, 1.0)),
            static_cast<float>(std::exp(-ourSeconds / 10.0)),
            static_cast<float>(std::min(theirSeconds / 300.0, 1.0)),
            static_cast<float>(std::exp(-theirSeconds / 10.0)),
        };
        for (int channel = 0; channel < 4; ++channel) {
            for (int square = 0; square < 64; ++square) {
                ASSERT_FLOAT_EQ(plane_at(planes, base + channel, square), expected[channel])
                    << "board " << boardBlock << " clock channel " << (37 + channel);
            }
        }
    };

    std::vector<float> planes(NB_INPUT_VALUES());
    // White team: White on A, Black on B.
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_block(planes, 0, 150.0, 600.0);
    expect_block(planes, 1, 0.0, 5.0);
    // Black team: Black on A, White on B.
    board_to_planes(board, planes.data(), Stockfish::BLACK, false);
    expect_block(planes, 0, 600.0, 150.0);
    expect_block(planes, 1, 5.0, 0.0);

    // A negative clock is clamped to 0 seconds.
    board.set_clocks(-2000, 1000, 1000, 1000);
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_block(planes, 0, 0.0, 1.0);

    // No clocks: all four planes are 0 on both boards.
    board.clear_clocks();
    board_to_planes(board, planes.data(), Stockfish::WHITE, true);
    for (int boardBlock = 0; boardBlock < 2; ++boardBlock) {
        for (int channel = 37; channel <= 40; ++channel) {
            EXPECT_FLOAT_EQ(
                plane_at(planes, boardBlock * NB_INPUT_CHANNELS_PER_BOARD + channel, 5), 0.0f);
        }
    }
}

// ------------------------------------------------------------ tree clocks

TEST_F(CustomEngineTest, TreeClocksOffLeaveBoardAndHashUnchanged) {
    Board plain;
    Board clocked(plain);
    clocked.set_clocks(1000, 1000, 1000, 1000);  // TreeMoveMs stays 0
    const Stockfish::Move e4 = Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4);
    clocked.make_moves(e4, Stockfish::MOVE_NONE);
    plain.make_moves(e4, Stockfish::MOVE_NONE);
    EXPECT_EQ(clocked.clock_ms(BOARD_A, Stockfish::WHITE), 1000);
    EXPECT_EQ(clocked.flagged_teams(), 0);
    if constexpr (!HAS_CLOCK_PLANES) {
        // Frozen clocks do not touch the node key in the stock-width build.
        EXPECT_EQ(clocked.hash_key(false), plain.hash_key(false));
        EXPECT_EQ(clocked.search_hash_key(Stockfish::BLACK, true),
                  plain.search_hash_key(Stockfish::BLACK, true));
    }
}

TEST_F(CustomEngineTest, TreeMoveMsChargesSeatOnTurnOnEachBoardAndRefunds) {
    Board board;
    // Board B: give black the move so the two boards have different sides.
    board.push_move(BOARD_B, Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4));
    board.set_clocks(10000, 10000, 10000, 10000);
    board.treeMoveMs = 300;
    const uint64_t rootHash = board.hash_key();

    const Stockfish::Move d4 = Stockfish::make_move(Stockfish::SQ_D2, Stockfish::SQ_D4);
    board.make_moves(d4, Stockfish::MOVE_NONE);
    // A: white was on turn and moved; B: black was on turn and passed.
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::WHITE), 9700);
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::BLACK), 10000);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::WHITE), 10000);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::BLACK), 9700);
    EXPECT_NE(board.hash_key(), rootHash);

    // A double pass is a macro-ply too.
    board.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.clock_ms(BOARD_A, Stockfish::BLACK), 9700);
    EXPECT_EQ(board.clock_ms(BOARD_B, Stockfish::BLACK), 9400);

    // Copies carry the clocks.
    Board copy(board);
    EXPECT_EQ(copy.clock_ms(BOARD_B, Stockfish::BLACK), 9400);
    EXPECT_EQ(copy.treeMoveMs, 300);

    board.unmake_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE);
    board.unmake_moves(d4, Stockfish::MOVE_NONE);
    for (int b = 0; b < 2; ++b) {
        for (Stockfish::Color c : {Stockfish::WHITE, Stockfish::BLACK}) {
            EXPECT_EQ(board.clock_ms(b, c), 10000);
        }
    }
    EXPECT_EQ(board.treeClockPlies, 0);
    EXPECT_EQ(board.hash_key(), rootHash);
}

TEST_F(CustomEngineTest, TreeFlagIsTerminalLikeMate) {
    Board board;
    // White team's seat on B is Black; give it 250 ms.
    board.push_move(BOARD_B, Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4));
    board.set_clocks(60000, 60000, 60000, 250);
    board.treeMoveMs = 300;

    // At the root nothing is terminal yet.
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::WHITE, Stockfish::WHITE, false,
                  std::array<int, 2>{0, 0}),
              TerminalOutcome::NONE);

    // One macro-ply (white team passes on B): B black drops to -50 ms.
    const Stockfish::Move d4 = Stockfish::make_move(Stockfish::SQ_D2, Stockfish::SQ_D4);
    board.make_moves(d4, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.flagged_teams(), 1 << Stockfish::WHITE);
    int endInPly = -1;
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::WHITE, false,
                  std::array<int, 2>{1, 0}, &endInPly),
              TerminalOutcome::WIN);
    EXPECT_EQ(endInPly, 1);
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::WHITE, Stockfish::WHITE, false,
                  std::array<int, 2>{1, 0}),
              TerminalOutcome::LOSS);

    // Without TreeMoveMs the clocks are frozen and nothing flags.
    board.unmake_moves(d4, Stockfish::MOVE_NONE);
    board.treeMoveMs = 0;
    board.make_moves(d4, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.flagged_teams(), 0);
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::WHITE, false,
                  std::array<int, 2>{1, 0}),
              TerminalOutcome::NONE);
}

TEST_F(CustomEngineTest, BothTeamsFlaggingOnOnePlyIsADraw) {
    // Start position on both boards: A white is the white team's seat, B
    // white the black team's. Both are on turn and both run out together.
    Board fresh;
    fresh.set_clocks(100, 60000, 100, 60000);
    fresh.treeMoveMs = 100;
    fresh.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE);
    EXPECT_EQ(fresh.flagged_teams(), (1 << Stockfish::WHITE) | (1 << Stockfish::BLACK));
    EXPECT_EQ(classify_terminal_position(
                  fresh, Stockfish::BLACK, Stockfish::WHITE, false,
                  std::array<int, 2>{0, 0}),
              TerminalOutcome::DRAW);
}

// ------------------------------------------------- FreePass mate rule

// Board A positions with the arena's verdict, arena.state.is_bughouse_mate
// (arena/tests/test_custom_engine.py pins the same table on the Python side).
struct ArenaMateCase {
    const char* name;
    const char* fen;
    bool arenaMate;
};

const ArenaMateCase kArenaMateCases[] = {
    {"stalemate", "7k/8/8/8/8/8/5q2/7K[] w - - 0 1", false},
    {"blockable_no_moves", "k7/8/8/8/5n2/5n2/8/r6K[] w - - 0 1", false},
    {"back_rank_blockable", "R5k1/5ppp/8/8/8/8/8/6K1[] b - - 0 1", false},
    {"smothered", "6rk/5Npp/8/8/8/8/8/K7[] b - - 0 1", true},
    {"contact_queen", "k7/1Q6/1K6/8/8/8/8/8[] b - - 0 1", true},
    {"double_check", "4R2k/5Npp/8/8/8/8/8/K7[] b - - 0 1", true},
    {"quiet", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1", false},
};

// The other board: the team is on turn there with quiet moves only and no
// capture available, so stock rules would call a stuck board a loss.
const char* kAfterE4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1";
const char* kStart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1";

TEST_F(CustomEngineTest, FreePassMateFollowsArenaRule) {
    g_freePass = true;
    for (const ArenaMateCase& c : kArenaMateCases) {
        for (int placed : {BOARD_A, BOARD_B}) {
            Board board;
            const int other = 1 - placed;
            board.set_fen(placed, c.fen);
            const Stockfish::Color stm = board.side_to_move(placed);
            board.set_fen(other, stm == Stockfish::WHITE ? kAfterE4 : kStart);
            // is_checkmate names a team by its colour on board A.
            const Stockfish::Color team = placed == BOARD_A ? stm : ~stm;
            for (bool timeAdvantage : {false, true}) {
                for (bool assume : {false, true}) {
                    EXPECT_EQ(board.is_checkmate(team, timeAdvantage, nullptr, assume), c.arenaMate)
                        << c.name << " board " << placed << " ta " << timeAdvantage
                        << " assume " << assume;
                }
                // Not mate: the team keeps its moves on the other board.
                EXPECT_EQ(board.legal_moves(team, timeAdvantage).empty(), c.arenaMate)
                    << c.name << " board " << placed << " ta " << timeAdvantage;
            }
        }
    }
}

TEST_F(CustomEngineTest, FreePassOffKeepsStockMateRule) {
    g_freePass = false;
    Board board;
    // Stuck board A without TimeAdvantage: a loss under stock rules.
    board.set_fen(BOARD_A, "7k/8/8/8/8/8/5q2/7K[] w - - 0 1");
    board.set_fen(BOARD_B, kAfterE4);
    EXPECT_TRUE(board.is_checkmate(Stockfish::WHITE, false));
    EXPECT_FALSE(board.is_checkmate(Stockfish::WHITE, true));
    // Blockable check with no move and no partner capture: mate under stock rules.
    board.set_fen(BOARD_A, "k7/8/8/8/5n2/5n2/8/r6K[] w - - 0 1");
    EXPECT_TRUE(board.is_checkmate(Stockfish::WHITE, false));
    EXPECT_TRUE(board.is_checkmate(Stockfish::WHITE, true));
    EXPECT_TRUE(board.legal_moves(Stockfish::WHITE, true).empty());
}

// ---------------------------------------------------------------- sample

training::TrainingSample make_sample() {
    Board board;
    board.set_clocks(90000, 80000, 70000, 60000);
    training::TrainingSample sample;
    sample.gameId = 0x0102030405060708ULL;
    sample.nodes = 801;
    sample.macroPly = 17;
    sample.movesLeft = 0;
    sample.team = 1;
    sample.hasTimeAdvantage = 1;
    sample.outcome = -1;
    sample.wdl = 0;
    sample.rootQ = -0.375f;
    sample.planes = training::encode_planes(board, Stockfish::BLACK, true);
    sample.policyA = {{12, 0.25f}, {300, 0.75f}};
    sample.policyB = {{0, 1.0f}};
    sample.jointPolicy = {{12, training::PASS_POLICY_INDEX, 0.25f}, {300, 7, 0.75f}};
    return sample;
}

template <typename T>
T read_le(const std::string& bytes, size_t& offset) {
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

TEST_F(CustomEngineTest, Base64MatchesRfc4648Vectors) {
    const std::pair<std::string, std::string> vectors[] = {
        {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"},
        {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="}, {"foobar", "Zm9vYmFy"},
        {std::string("\x00\xff\x10", 3), "AP8Q"},
    };
    for (const auto& [plain, encoded] : vectors) {
        EXPECT_EQ(training::base64_encode(plain), encoded);
        EXPECT_EQ(training::base64_decode(encoded), plain);
    }
    EXPECT_THROW(training::base64_decode("abc"), std::invalid_argument);
    EXPECT_THROW(training::base64_decode("ab!d"), std::invalid_argument);
}

TEST_F(CustomEngineTest, SampleRoundTripsThroughBase64InShardLayout) {
    const training::TrainingSample sample = make_sample();
    const std::string bytes = training::base64_decode(
        training::base64_encode(training::serialize_sample(sample)));

    const size_t expectedSize = 8 + 4 + 2 + 2 + 1 + 1 + 1 + 1 + 4
        + NB_INPUT_VALUES() + (2 + 2 * 6) + (2 + 1 * 6) + (2 + 2 * 8);
    ASSERT_EQ(bytes.size(), expectedSize);

    size_t at = 0;
    EXPECT_EQ(read_le<uint64_t>(bytes, at), sample.gameId);
    EXPECT_EQ(read_le<uint32_t>(bytes, at), sample.nodes);
    EXPECT_EQ(read_le<uint16_t>(bytes, at), sample.macroPly);
    EXPECT_EQ(read_le<uint16_t>(bytes, at), sample.movesLeft);
    EXPECT_EQ(read_le<uint8_t>(bytes, at), sample.team);
    EXPECT_EQ(read_le<uint8_t>(bytes, at), sample.hasTimeAdvantage);
    EXPECT_EQ(read_le<int8_t>(bytes, at), sample.outcome);
    EXPECT_EQ(read_le<uint8_t>(bytes, at), sample.wdl);
    EXPECT_FLOAT_EQ(read_le<float>(bytes, at), sample.rootQ);
    EXPECT_EQ(0, std::memcmp(bytes.data() + at, sample.planes.data(), sample.planes.size()));
    at += sample.planes.size();
    for (const auto* policy : {&sample.policyA, &sample.policyB}) {
        ASSERT_EQ(read_le<uint16_t>(bytes, at), policy->size());
        for (const auto& entry : *policy) {
            EXPECT_EQ(read_le<uint16_t>(bytes, at), entry.index);
            EXPECT_FLOAT_EQ(read_le<float>(bytes, at), entry.probability);
        }
    }
    ASSERT_EQ(read_le<uint16_t>(bytes, at), sample.jointPolicy.size());
    for (const auto& entry : sample.jointPolicy) {
        EXPECT_EQ(read_le<uint16_t>(bytes, at), entry.indexA);
        EXPECT_EQ(read_le<uint16_t>(bytes, at), entry.indexB);
        EXPECT_FLOAT_EQ(read_le<float>(bytes, at), entry.probability);
    }
    EXPECT_EQ(at, bytes.size());

    // A shard is the header followed by these exact records.
    std::ostringstream shard(std::ios::binary);
    training::write_chunk_header(shard, 2);
    training::write_sample(shard, sample);
    training::write_sample(shard, sample);
    const std::string shardBytes = shard.str();
    const size_t headerSize = 4 + 4 + 2 + 2 + 8;
    ASSERT_EQ(shardBytes.size(), headerSize + 2 * bytes.size());
    EXPECT_EQ(shardBytes.substr(0, 4), "HVM5");
    size_t headerAt = 8;
    EXPECT_EQ(read_le<uint16_t>(shardBytes, headerAt), NB_INPUT_CHANNELS);
    EXPECT_EQ(shardBytes.substr(headerSize, bytes.size()), bytes);
    EXPECT_EQ(shardBytes.substr(headerSize + bytes.size()), bytes);
}

TEST_F(CustomEngineTest, SamplePoliciesFollowRootVisits) {
    Board board;
    const std::vector<Stockfish::Move> moves = board.legal_moves(BOARD_A);
    ASSERT_GE(moves.size(), 2U);
    const JointActionRules rules{true, false, false, true, false};
    std::vector<RootEdgeStats> edges = {
        {JointActionCandidate(moves[0], 0.5f, 0, Stockfish::MOVE_NONE, 1.0f, 0,
                              rules, false, false), 3},
        {JointActionCandidate(moves[1], 0.5f, 1, Stockfish::MOVE_NONE, 1.0f, 0,
                              rules, false, false), 1},
        {JointActionCandidate(moves[1], 0.5f, 1, Stockfish::MOVE_NONE, 1.0f, 0,
                              rules, false, false), 0},
    };
    const auto policyA = training::marginal_policy(board, BOARD_A, edges);
    ASSERT_EQ(policyA.size(), 2U);
    float total = 0.0f;
    for (const auto& entry : policyA) {
        total += entry.probability;
        EXPECT_TRUE(entry.probability == 0.75f || entry.probability == 0.25f);
    }
    EXPECT_FLOAT_EQ(total, 1.0f);
    const auto policyB = training::marginal_policy(board, BOARD_B, edges);
    ASSERT_EQ(policyB.size(), 1U);
    EXPECT_FLOAT_EQ(policyB[0].probability, 1.0f);
    const auto joint = training::joint_policy(board, edges);
    ASSERT_EQ(joint.size(), 2U);
    for (const auto& entry : joint) {
        EXPECT_EQ(entry.indexB, training::PASS_POLICY_INDEX);
    }
}

}  // namespace
