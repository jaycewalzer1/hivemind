#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/planes.h"
#include "environment/time_state.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "tools/seat_clock_game.h"
#include "tools/selfplay.h"

// Wave 2b: the wide planes (seat clocks, thinking time, pocket presence,
// recent landings) and the arrivals head. Plane values on hand-built
// positions, the landing history in the durative tree, the engine choosing
// narrow or wide by the network's input width, and self-play's sample
// fields. docs/wave2_engine.md, "Wave 2b".

namespace {

using Stockfish::BLACK;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;
constexpr int PER = NB_WIDE_EXTRA_CHANNELS_PER_BOARD;

SeatLanding landing(int seat, Stockfish::PieceType type, int32_t ago) {
    return SeatLanding{static_cast<int8_t>(seat), static_cast<int8_t>(type), ago};
}

// Aw holds N and Q, Ab a pawn; Bw a rook, Bb two bishops.
void set_pockets(Board& board) {
    board.set_fen(BOARD_A, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[NQp] w KQkq - 0 1");
    board.set_fen(BOARD_B, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[Rbb] w KQkq - 0 1");
}

DurativeConfig wide_config() {
    DurativeConfig config;
    config.wideInputs = true;
    config.rootMoveTimeMs = 1000;
    config.seatClocksMs = {60000, 30000, 400000, 5000};
    config.seatPaceMs = {300, 300, 300, 300};
    config.boardElapsedMs = {2000, 20000};
    config.recentLandings = {landing(AW, Stockfish::KNIGHT, 1000),
                             landing(AW, Stockfish::KNIGHT, 2000),
                             landing(AW, Stockfish::KNIGHT, 3000),
                             landing(BB, Stockfish::PAWN, 4999),
                             landing(AB, Stockfish::QUEEN, 6000),  // too old
                             landing(BW, Stockfish::ROOK, 0)};
    return config;
}

TimeState wide_state(Stockfish::Color rootTeam, const DurativeConfig& config = wide_config()) {
    const TimeState base = TimeState::clocked(rootTeam, 0, 300, 300).with_sit_gate(false);
    return TimeState::durative(base, rootTeam, WHITE, WHITE, config);
}

std::array<float, NB_WIDE_EXTRA_VALUES> extras_of(Board& board, Stockfish::Color team,
                                                  const TimeState& time) {
    std::array<float, NB_WIDE_EXTRA_VALUES> extras{};
    wide_extra_planes(board, team, time, extras.data());
    return extras;
}

class Wave2bTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

// ---- plane values ----------------------------------------------------------

TEST_F(Wave2bTest, WidePlanesOnAHandBuiltPositionFromTheWhiteTeam) {
    Board board;
    set_pockets(board);
    const TimeState time = wide_state(WHITE);
    ASSERT_TRUE(time.wide_timing());
    const auto x = extras_of(board, WHITE, time);
    const float* a = x.data();
    const float* b = x.data() + PER;
    // Board A: ours Aw (60 s), theirs Ab (30 s).
    EXPECT_FLOAT_EQ(a[0], 0.2f);
    EXPECT_FLOAT_EQ(a[1], static_cast<float>(std::exp(-6.0)));
    EXPECT_FLOAT_EQ(a[2], 0.1f);
    EXPECT_FLOAT_EQ(a[3], static_cast<float>(std::exp(-3.0)));
    // Board A thinking time 2 s.
    EXPECT_FLOAT_EQ(a[4], 0.2f);
    EXPECT_FLOAT_EQ(a[5], static_cast<float>(1.0 - std::exp(-1.0)));
    // Presence: ours (Aw) N and Q, theirs (Ab) P.
    EXPECT_EQ(std::vector<float>(a + 6, a + 11), (std::vector<float>{0, 1, 0, 0, 1}));
    EXPECT_EQ(std::vector<float>(a + 11, a + 16), (std::vector<float>{1, 0, 0, 0, 0}));
    // Three knights landed in Aw's pocket in the last 5 s: min(3/2, 1).
    EXPECT_EQ(std::vector<float>(a + 16, a + 21), (std::vector<float>{0, 1, 0, 0, 0}));
    // Ab's queen landed 6 s ago: out of the window.
    EXPECT_EQ(std::vector<float>(a + 21, a + 26), (std::vector<float>{0, 0, 0, 0, 0}));
    // Board B: ours Bb (5 s), theirs Bw (400 s, saturated).
    EXPECT_FLOAT_EQ(b[0], 5.0f / 300.0f);
    EXPECT_FLOAT_EQ(b[1], static_cast<float>(std::exp(-0.5)));
    EXPECT_FLOAT_EQ(b[2], 1.0f);
    EXPECT_FLOAT_EQ(b[3], static_cast<float>(std::exp(-40.0)));
    EXPECT_FLOAT_EQ(b[4], 1.0f);
    EXPECT_FLOAT_EQ(b[5], static_cast<float>(1.0 - std::exp(-10.0)));
    EXPECT_EQ(std::vector<float>(b + 6, b + 11), (std::vector<float>{0, 0, 1, 0, 0}));
    EXPECT_EQ(std::vector<float>(b + 11, b + 16), (std::vector<float>{0, 0, 0, 1, 0}));
    EXPECT_EQ(std::vector<float>(b + 16, b + 21), (std::vector<float>{0.5f, 0, 0, 0, 0}));
    EXPECT_EQ(std::vector<float>(b + 21, b + 26), (std::vector<float>{0, 0, 0, 0.5f, 0}));
}

TEST_F(Wave2bTest, TheBlackTeamSeesTheSameSeatsTheOtherWayRound) {
    Board board;
    set_pockets(board);
    const TimeState time = wide_state(BLACK);
    const auto x = extras_of(board, BLACK, time);
    const float* a = x.data();
    const float* b = x.data() + PER;
    // Board A: ours Ab, theirs Aw. Board B: ours Bw, theirs Bb.
    EXPECT_FLOAT_EQ(a[0], 0.1f);
    EXPECT_FLOAT_EQ(a[2], 0.2f);
    EXPECT_EQ(std::vector<float>(a + 6, a + 11), (std::vector<float>{1, 0, 0, 0, 0}));
    EXPECT_EQ(std::vector<float>(a + 11, a + 16), (std::vector<float>{0, 1, 0, 0, 1}));
    EXPECT_EQ(std::vector<float>(a + 21, a + 26), (std::vector<float>{0, 1, 0, 0, 0}));
    EXPECT_FLOAT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[2], 5.0f / 300.0f);
    // The thinking time belongs to the board, not to a team.
    EXPECT_FLOAT_EQ(b[4], 1.0f);
    EXPECT_EQ(std::vector<float>(b + 16, b + 21), (std::vector<float>{0, 0, 0, 0.5f, 0}));
    EXPECT_EQ(std::vector<float>(b + 21, b + 26), (std::vector<float>{0.5f, 0, 0, 0, 0}));
}

TEST_F(Wave2bTest, WithoutTimingOnlyThePresencePlanesAreSet) {
    Board board;
    set_pockets(board);
    // v3's clocked state, a legacy flag, a narrow durative state and a wide
    // one whose clocks are derived from a lead: no timing given.
    DurativeConfig narrow = wide_config();
    narrow.wideInputs = false;
    DurativeConfig derived = wide_config();
    derived.seatClocksMs = {0, 0, 0, 0};
    for (const TimeState& time : {TimeState::clocked(WHITE, 5000, 300, 300),
                                  TimeState::legacy(WHITE, true), wide_state(WHITE, narrow),
                                  wide_state(WHITE, derived)}) {
        EXPECT_FALSE(time.wide_timing());
        const auto x = extras_of(board, WHITE, time);
        for (int board_ = 0; board_ < 2; ++board_) {
            const float* p = x.data() + board_ * PER;
            for (int channel : {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}) {
                EXPECT_EQ(p[channel], 0.0f) << "board " << board_ << " +" << channel;
            }
        }
        EXPECT_EQ(x[7], 1.0f);        // our knight on A
        EXPECT_EQ(x[PER + 8], 1.0f);  // our bishop on B
    }
}

TEST_F(Wave2bTest, TheWideLayoutAppendsEachBoardsPlanesAfterItsBlock) {
    Board board;
    set_pockets(board);
    const TimeState time = wide_state(WHITE);
    std::vector<float> narrow(NB_INPUT_VALUES());
    std::vector<float> wide(NB_WIDE_INPUT_CHANNELS * 64);
    board_to_planes(board, narrow.data(), WHITE, time);
    board_to_wide_planes(board, wide.data(), WHITE, time);
    const auto x = extras_of(board, WHITE, time);
    constexpr size_t block = NB_INPUT_CHANNELS_PER_BOARD * 64;
    constexpr size_t wideBlock = NB_WIDE_INPUT_CHANNELS_PER_BOARD * 64;
    for (size_t board_ = 0; board_ < 2; ++board_) {
        for (size_t i = 0; i < block; ++i) {
            ASSERT_EQ(wide[board_ * wideBlock + i], narrow[board_ * block + i]);
        }
        for (size_t channel = 0; channel < PER; ++channel) {
            for (size_t square = 0; square < 64; ++square) {
                ASSERT_EQ(wide[board_ * wideBlock + block + channel * 64 + square],
                          x[board_ * PER + channel]);
            }
        }
    }
    EXPECT_EQ(NB_WIDE_INPUT_CHANNELS, 130);
}

// ---- the landing history in the tree ---------------------------------------

TEST_F(Wave2bTest, TheTreeAgesLandingsAndAddsTheCapturesItPlays) {
    TimeState time = wide_state(WHITE);
    EXPECT_EQ(time.recent_landings(AW, Stockfish::KNIGHT), 3);
    EXPECT_EQ(time.recent_landings(BB, Stockfish::PAWN), 1);
    EXPECT_EQ(time.recent_landings(AB, Stockfish::QUEEN), 0);
    // Our root move on A (1000 ms) captures a knight: it lands in Bb's pocket.
    time.advance_action(WHITE, true, false, Stockfish::KNIGHT, Stockfish::NO_PIECE_TYPE);
    EXPECT_EQ(time.recent_landings(BB, Stockfish::KNIGHT), 1);
    EXPECT_EQ(time.recent_landings(BB, Stockfish::PAWN), 0);  // 5999 ms old now
    EXPECT_EQ(time.recent_landings(AW, Stockfish::KNIGHT), 3);
    EXPECT_EQ(time.board_elapsed_ms(0), 0);      // a move landed on A
    EXPECT_EQ(time.board_elapsed_ms(1), 21000);  // B kept thinking
    // Bw (ready at 0 after the root) passes: no landing, A's thinking runs on.
    TimeState passed = time.after_action(BLACK, false, false);
    EXPECT_EQ(passed.board_elapsed_ms(1), 21000);
    // The knights age out 2 s later than the one landing at 3 s ago.
    TimeState aged = time;
    for (int step = 0; step < 40 && aged.recent_landings(AW, Stockfish::KNIGHT) == 3; ++step) {
        const auto boardsW = aged.acting_boards(WHITE);
        const Stockfish::Color team = boardsW[0] || boardsW[1] ? WHITE : BLACK;
        const auto boards = aged.acting_boards(team);
        aged.advance_action(team, boards[0], boards[1]);
    }
    EXPECT_LT(aged.recent_landings(AW, Stockfish::KNIGHT), 3);
}

TEST_F(Wave2bTest, LandingsSeparateWideNodeKeysButNeverNarrowOnes) {
    DurativeConfig wideA = wide_config();
    DurativeConfig wideB = wide_config();
    wideB.recentLandings[0].agoMs = 1500;
    EXPECT_NE(wide_state(WHITE, wideA).exact_key(), wide_state(WHITE, wideB).exact_key());

    // Narrow: whatever the landings and thinking times say, the state and
    // its key are exactly the wave 2 durative state.
    DurativeConfig plain;
    plain.rootMoveTimeMs = 1000;
    plain.seatClocksMs = wideA.seatClocksMs;
    plain.seatPaceMs = wideA.seatPaceMs;
    DurativeConfig narrow = wideA;
    narrow.wideInputs = false;
    TimeState p = wide_state(WHITE, plain);
    TimeState n = wide_state(WHITE, narrow);
    EXPECT_EQ(p, n);
    EXPECT_EQ(p.exact_key(), n.exact_key());
    p.advance_action(WHITE, true, false, Stockfish::QUEEN);
    n.advance_action(WHITE, true, false, Stockfish::QUEEN);
    EXPECT_EQ(p, n);
    EXPECT_EQ(p.exact_key(), n.exact_key());
    EXPECT_EQ(n.recent_landings(BB, Stockfish::QUEEN), 0);
}

// ---- the game's landing log and labels -------------------------------------

TEST_F(Wave2bTest, PocketLandingCountsUseTheHeadsOrderAndAHalfOpenWindow) {
    const std::vector<SeatClockGame::Landing> log = {
        {1000, AW, Stockfish::KNIGHT}, {2000, BB, Stockfish::PAWN},
        {3000, AB, Stockfish::QUEEN}, {6000, BW, Stockfish::ROOK}};
    // White: ours A = Aw, theirs A = Ab, ours B = Bb, theirs B = Bw.
    auto counts = pocket_landing_counts(log, WHITE, 1000, 6000);
    EXPECT_EQ(counts[0 * 5 + 1], 0);  // the knight at 1000 is not after 1000
    EXPECT_EQ(counts[1 * 5 + 4], 1);  // Ab's queen
    EXPECT_EQ(counts[2 * 5 + 0], 1);  // Bb's pawn
    EXPECT_EQ(counts[3 * 5 + 3], 1);  // Bw's rook at exactly 6000
    counts = pocket_landing_counts(log, BLACK, 0, 1000);
    EXPECT_EQ(counts[1 * 5 + 1], 1);  // Aw is theirs on A for black
}

TEST_F(Wave2bTest, TheSeatClockGameLogsCapturesIntoThePartnersPocket) {
    Wave2GameOptions options;
    options.durativeTree = true;
    options.baseMix = {{60000, 1.0}};
    options.minMoveMs = 1;
    options.lagMaxMs = 0;
    options.paceJitter = 0.0;
    SeatClockGame::Start start;
    start.baseMs = 60000;
    start.clocksMs = {60000, 60000, 60000, 60000};
    start.moveTimesMs = {300, 300, 300, 300};
    SeatClockGame game(options, start);
    game.set_wide_inputs(true);
    std::array<Stockfish::Color, 2> sides{WHITE, WHITE};
    SeatClockGame::Event event = game.next(sides);
    ASSERT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    ASSERT_TRUE(event.seats[AW]);
    std::mt19937_64 random(1);
    game.plan_cost(AW, random);
    game.commit_move(AW);
    event = game.next(sides);
    ASSERT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    ASSERT_TRUE(event.seats[BW]);
    game.plan_cost(BW, random);
    game.commit_pass(BW, sides);
    event = game.next(sides);
    ASSERT_EQ(event.type, SeatClockGame::EventType::LAND);
    ASSERT_TRUE(event.seats[AW]);
    // Aw captured a bishop: it goes to Bb, Aw's partner.
    game.landed(AW, true, Stockfish::BISHOP);
    ASSERT_EQ(game.landings().size(), 1u);
    EXPECT_EQ(game.landings()[0].seat, BB);
    EXPECT_EQ(game.landings()[0].timeMs, 300);
    EXPECT_EQ(game.board_elapsed_ms(0), 0);
    EXPECT_EQ(game.board_elapsed_ms(1), 300);
    sides = {BLACK, WHITE};
    event = game.next(sides);
    ASSERT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    // The black team's search sees the landing in its wide state.
    game.plan_cost(AB, random);
    const TimeState clock = game.search_clock(BLACK, sides, event.seats);
    EXPECT_TRUE(clock.wide_timing());
    EXPECT_EQ(clock.recent_landings(BB, Stockfish::BISHOP), 1);
    EXPECT_EQ(clock.board_elapsed_ms(0), static_cast<int32_t>(game.now_ms() - 300));
    game.set_wide_inputs(false);
    EXPECT_FALSE(game.search_clock(BLACK, sides, event.seats).wide_timing());
}

// ---- the engine: narrow or wide by input width ------------------------------

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)

std::filesystem::path data_file(const char* name) {
    return std::filesystem::path(__FILE__).parent_path() / "data" / name;
}

TEST_F(Wave2bTest, TheEngineHandsAWideNetworkTheExtraPlanes) {
    // wide_probe.onnx echoes 20 of the wide planes as its arrivals output.
    Engine engine(0, 2);
    ASSERT_TRUE(engine.loadNetwork(data_file("wide_probe.onnx").string(), ""));
    EXPECT_EQ(engine.inputChannels(), NB_WIDE_INPUT_CHANNELS);
    EXPECT_TRUE(engine.wideInputs());
    EXPECT_TRUE(engine.hasArrivals());

    Board board;
    set_pockets(board);
    std::vector<float> obs(2 * NB_INPUT_VALUES());
    std::vector<float> extras(2 * NB_WIDE_EXTRA_VALUES);
    const TimeState white = wide_state(WHITE);
    const TimeState black = wide_state(BLACK);
    board_to_planes(board, obs.data(), WHITE, white);
    board_to_planes(board, obs.data() + NB_INPUT_VALUES(), BLACK, black);
    wide_extra_planes(board, WHITE, white, extras.data());
    wide_extra_planes(board, BLACK, black, extras.data() + NB_WIDE_EXTRA_VALUES);
    std::vector<float> value(2), piA(2 * NB_POLICY_VALUES()), piB(2 * NB_POLICY_VALUES()),
        wdl(6), movesLeft(2), arrivals(2 * NB_ARRIVALS_VALUES, -1.0f);
    ASSERT_TRUE(engine.runInference(obs.data(), value.data(), piA.data(), piB.data(), wdl.data(),
                                    movesLeft.data(), 0, extras.data(), arrivals.data()));
    const int offsets[10] = {0, 1, 2, 3, 4, 5, 6, 11, 16, 21};
    for (int row = 0; row < 2; ++row) {
        for (int index = 0; index < NB_ARRIVALS_VALUES; ++index) {
            const int board_ = index / 10;
            const float expected = extras[row * NB_WIDE_EXTRA_VALUES + board_ * PER
                                          + offsets[index % 10]];
            // FP16 on the way in.
            EXPECT_NEAR(arrivals[row * NB_ARRIVALS_VALUES + index], expected, 1e-3)
                << "row " << row << " index " << index;
        }
    }
    // No extras given: the extra planes are 0.
    ASSERT_TRUE(engine.runInference(obs.data(), value.data(), piA.data(), piB.data(), wdl.data(),
                                    movesLeft.data(), 0, nullptr, arrivals.data()));
    for (float a : arrivals) {
        EXPECT_EQ(a, 0.0f);
    }
}

TEST_F(Wave2bTest, ANarrowNetworkIgnoresTheExtrasBitForBit) {
    const std::filesystem::path it13 = std::filesystem::path(__FILE__).parent_path().parent_path()
        / "models" / "twin-s-clock-it13.onnx";
    Engine engine(0, 1);
    const bool real = std::filesystem::exists(it13);
    ASSERT_TRUE(engine.loadNetwork(real ? it13.string() : data_file("uniform.onnx").string(), ""));
    EXPECT_EQ(engine.inputChannels(), NB_INPUT_CHANNELS);
    EXPECT_FALSE(engine.wideInputs());
    EXPECT_FALSE(engine.hasArrivals());
    Board board;
    set_pockets(board);
    // The narrow planes of a wide durative state equal those of the same
    // state built narrow: the wide bookkeeping never reaches them.
    DurativeConfig narrowConfig = wide_config();
    narrowConfig.wideInputs = false;
    std::vector<float> obs(NB_INPUT_VALUES()), narrowObs(NB_INPUT_VALUES());
    board_to_planes(board, obs.data(), WHITE, wide_state(WHITE));
    board_to_planes(board, narrowObs.data(), WHITE, wide_state(WHITE, narrowConfig));
    ASSERT_EQ(obs, narrowObs);
    std::vector<float> garbage(NB_WIDE_EXTRA_VALUES, 0.75f);
    auto run = [&](const float* extras) {
        std::vector<float> out(1 + 2 * NB_POLICY_VALUES() + 3 + 1);
        float* value = out.data();
        float* piA = value + 1;
        float* piB = piA + NB_POLICY_VALUES();
        float* wdl = piB + NB_POLICY_VALUES();
        float* movesLeft = wdl + 3;
        std::vector<float> arrivals(NB_ARRIVALS_VALUES, -1.0f);
        EXPECT_TRUE(engine.runInference(obs.data(), value, piA, piB, wdl, movesLeft, 0, extras,
                                        arrivals.data()));
        // No arrivals head: the buffer is left alone.
        EXPECT_EQ(arrivals[0], -1.0f);
        return out;
    };
    const std::vector<float> plain = run(nullptr);
    EXPECT_EQ(plain, run(garbage.data()));
    if (!real) {
        GTEST_LOG_(INFO) << "it13 not found; checked on uniform.onnx only";
    }
}

TEST_F(Wave2bTest, ADurativeSearchRunsOnAWideNetwork) {
    Engine engine(0, 4);
    ASSERT_TRUE(engine.loadNetwork(data_file("wide_probe.onnx").string(), ""));
    Board board;
    Agent agent(2);
    SearchOptions options;
    options.targetNodes = 300;
    options.enablePonder = false;
    options.clock = wide_state(WHITE);
    agent.run_search(board, {&engine}, WHITE, false, options);
    EXPECT_FALSE(agent.root_edge_stats().empty());
}

// ---- self-play sample fields -----------------------------------------------

TEST_F(Wave2bTest, SelfPlayWritesThinkingTimesRecentLandingsAndArrivalLabels) {
    Engine engine(0, 1);
    ASSERT_TRUE(engine.loadNetwork(data_file("wide_probe.onnx").string(), ""));
    const std::filesystem::path output = std::filesystem::temp_directory_path()
        / ("hivemind_wave2b_selfplay_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    SelfPlayConfig config;
    config.games = 3;
    config.nodes = 16;
    config.maxMacroPlies = 400;
    config.batchSize = 1;
    config.seed = 11;
    config.outputDirectory = output;
    config.rawPolicyMeanMacroPlies = 0.0;
    config.resignThreshold = 1.0f;
    config.fairyStockfishMateNodes = 0;
    config.clockModel = true;
    config.clock.moveTimeMinMs = 100;
    config.clock.moveTimeMaxMs = 400;
    Wave2GameOptions& w = config.wave2;
    w.durativeTree = true;
    w.sitGate = false;
    w.flagLoss = true;
    w.instantTransfer = false;
    w.baseMix = {{8000, 1.0}};
    w.seatMoveTimes = true;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({&engine}, config), 0);
    testing::internal::GetCapturedStdout();

    std::ifstream stream(output / "seat_clocks.tsv");
    std::string header;
    std::getline(stream, header);
    std::vector<std::string> names;
    {
        std::istringstream fields(header);
        for (std::string name; std::getline(fields, name, '\t');) {
            names.push_back(name);
        }
    }
    ASSERT_EQ(names.size(), 17u + 2u + 40u);
    EXPECT_EQ(names[17], "board_elapsed_a");
    EXPECT_EQ(names[19], "recent_ours_a_p");
    EXPECT_EQ(names[39], "arrivals_ours_a_p");
    EXPECT_EQ(names.back(), "arrivals_theirs_b_q");
    // Every row has every column; thinking times lie within the game so
    // far; counts are never negative. (The windows themselves are checked
    // by PocketLandingCountsUseTheHeadsOrderAndAHalfOpenWindow.)
    int rows = 0;
    long arrivals = 0;
    long recent = 0;
    for (std::string line; std::getline(stream, line); ++rows) {
        std::istringstream fields(line);
        std::vector<long> values;
        for (std::string field; std::getline(fields, field, '\t');) {
            values.push_back(std::stol(field));
        }
        ASSERT_EQ(values.size(), names.size());
        const long timeMs = values[3];
        EXPECT_GE(values[17], 0);
        EXPECT_LE(values[17], timeMs);
        EXPECT_LE(values[18], timeMs);
        for (size_t i = 19; i < names.size(); ++i) {
            EXPECT_GE(values[i], 0);
            (i < 39 ? recent : arrivals) += values[i];
        }
    }
    EXPECT_GT(rows, 0);
    GTEST_LOG_(INFO) << rows << " rows, " << recent << " recent and " << arrivals
                     << " arriving pieces counted";
    std::error_code ignored;
    std::filesystem::remove_all(output, ignored);
}

#endif

}  // namespace
