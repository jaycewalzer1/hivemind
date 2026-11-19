#include <gtest/gtest.h>

#include <map>
#include <random>

#include "tools/seat_clock_game.h"

// Wave 2 self-play plumbing: the selfplay/tournament options and the
// seat-clock game they play (tools/seat_clock_game.h).

namespace {

using Stockfish::BLACK;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;
using Sides = std::array<Stockfish::Color, 2>;

Wave2GameOptions wave2_options() {
    Wave2GameOptions options;
    const std::vector<std::pair<std::string, std::string>> args = {
        {"--durative-tree", "true"}, {"--sit-gate", "false"}, {"--flag-loss", "true"},
        {"--instant-transfer", "false"}, {"--mate-race-by-landing", "true"},
        {"--pass-cap-k", "8"}, {"--base-mix", "60000:0.25,120000:0.25,180000:0.35,300000:0.15"},
        {"--seat-move-times", "true"}};
    for (const auto& [option, value] : args) {
        EXPECT_TRUE(parse_wave2_option(option, value, options)) << option;
    }
    options.validate();
    return options;
}

/** A fixed start: every seat at @p clock with move time @p move. */
SeatClockGame::Start fixed_start(int32_t clock, std::array<int32_t, 4> moves) {
    SeatClockGame::Start start;
    start.baseMs = clock;
    start.clocksMs = {clock, clock, clock, clock};
    start.moveTimesMs = moves;
    return start;
}

/** No jitter, floor or lag: costs are the move times exactly. */
Wave2GameOptions exact(Wave2GameOptions options) {
    options.minMoveMs = 1;
    options.lagMaxMs = 0;
    options.paceJitter = 0.0;
    return options;
}

TEST(Wave2Options, TheWave2PresetParsesAndStockIsTheDefault) {
    const Wave2GameOptions options = wave2_options();
    EXPECT_TRUE(options.durativeTree);
    EXPECT_FALSE(options.sitGate);
    EXPECT_TRUE(options.flagLoss);
    EXPECT_FALSE(options.instantTransfer);
    EXPECT_TRUE(options.mateRaceByLanding);
    EXPECT_TRUE(options.seatMoveTimes);
    ASSERT_EQ(options.baseMix.size(), 4u);
    EXPECT_EQ(options.baseMix[2].first, 180000);
    EXPECT_DOUBLE_EQ(options.baseMix[2].second, 0.35);
    EXPECT_TRUE(options.seat_clocks());
    EXPECT_TRUE(options.changes_search());

    const Wave2GameOptions stock;
    EXPECT_FALSE(stock.seat_clocks());
    EXPECT_FALSE(stock.changes_search());
    const DurativeConfig config = options.durative_config();
    EXPECT_TRUE(config.flagLoss);
    EXPECT_FALSE(config.instantTransfer);
    EXPECT_TRUE(config.mateRaceByLanding);

    Wave2GameOptions other;
    EXPECT_FALSE(parse_wave2_option("--nodes", "800", other));
    EXPECT_THROW(parse_wave2_option("--flag-loss", "maybe", other), std::invalid_argument);
    EXPECT_THROW(parse_base_mix("60000"), std::invalid_argument);
    EXPECT_TRUE(parse_base_mix("off").empty());
    other.passCapK = 0;
    EXPECT_THROW(other.validate(), std::invalid_argument);
}

TEST(Wave2Options, StartsFollowTheBaseMixAndTheLeadOdds) {
    const Wave2GameOptions options = wave2_options();
    ClockSampling clock;
    clock.moveTimeMinMs = 100;
    clock.moveTimeMaxMs = 3000;
    std::mt19937_64 random(7);
    std::map<int32_t, int> bases;
    int odds = 0;
    int ownMoveTimes = 0;
    constexpr int GAMES = 4000;
    for (int game = 0; game < GAMES; ++game) {
        const SeatClockGame::Start start = SeatClockGame::sample_start(options, clock, random);
        ++bases[start.baseMs];
        // White seats +lead/2, black seats -lead/2: the drawn lead is v3's.
        SeatClockGame state(options, start);
        EXPECT_EQ(state.white_lead_ms(), start.whiteLeadMs);
        EXPECT_EQ(start.clocksMs[AW], start.clocksMs[BB]);
        EXPECT_EQ(start.clocksMs[AB], start.clocksMs[BW]);
        odds += start.whiteLeadMs != 0 ? 1 : 0;
        ownMoveTimes += start.moveTimesMs[AW] != start.moveTimesMs[BB] ? 1 : 0;
        for (int seat = 0; seat < 4; ++seat) {
            EXPECT_GE(start.moveTimesMs[seat], 100);
            EXPECT_LE(start.moveTimesMs[seat], 3000);
            EXPECT_GE(start.clocksMs[seat], 1000);
        }
    }
    EXPECT_NEAR(bases[60000] / double(GAMES), 0.25, 0.03);
    EXPECT_NEAR(bases[180000] / double(GAMES), 0.35, 0.03);
    EXPECT_NEAR(bases[300000] / double(GAMES), 0.15, 0.03);
    EXPECT_NEAR(odds / double(GAMES), 0.5, 0.05);  // levelStartFraction 0.5
    EXPECT_GT(ownMoveTimes, GAMES / 2);
}

TEST(SeatClockGame, CostsHaveAFloorLagAndJitter) {
    Wave2GameOptions options = wave2_options();
    SeatClockGame game(options, fixed_start(60000, {50, 1000, 1000, 1000}));
    std::mt19937_64 random(3);
    int32_t low = 1 << 30;
    int32_t high = 0;
    for (int i = 0; i < 2000; ++i) {
        const int32_t fast = game.plan_cost(AW, random);
        EXPECT_GE(fast, 100);                  // the 100 ms floor
        EXPECT_LE(fast, 100 + 150);            // plus at most 150 ms lag
        const int32_t slow = game.plan_cost(BW, random);
        low = std::min(low, slow);
        high = std::max(high, slow);
        EXPECT_GE(slow, 800 - 1);              // 1000 / 1.25
        EXPECT_LE(slow, 1250 + 150);
    }
    EXPECT_LT(low, 850);
    EXPECT_GT(high, 1300);
}

TEST(SeatClockGame, DecisionsLandLaterAndClocksOnTurnRun) {
    const Wave2GameOptions options = exact(wave2_options());
    SeatClockGame game(options, fixed_start(60000, {300, 200, 500, 400}));
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    // t = 0: Aw and Bw both decide; ties go to white first, then black.
    SeatClockGame::Event event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    EXPECT_EQ(event.team, WHITE);
    EXPECT_TRUE(event.seats[AW]);
    EXPECT_FALSE(event.seats[BW]);
    game.plan_cost(AW, random);
    game.commit_move(AW);
    event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    EXPECT_EQ(event.team, BLACK);
    EXPECT_TRUE(event.seats[BW]);
    game.plan_cost(BW, random);
    game.commit_move(BW);
    // Aw lands at 300.
    event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::LAND);
    EXPECT_EQ(event.timeMs, 300);
    EXPECT_TRUE(event.seats[AW]);
    EXPECT_EQ(game.clock_ms(AW), 59700);
    EXPECT_EQ(game.clock_ms(BW), 59700);
    EXPECT_EQ(game.clock_ms(AB), 60000);
    game.landed(AW, false);
    sides[0] = BLACK;
    // Ab comes on turn and decides at once; Bw is 300 ms into its thinking.
    event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::DECIDE);
    EXPECT_TRUE(event.seats[AB]);
    EXPECT_EQ(game.elapsed_ms(BW), 300);
    const TimeState clock = game.search_clock(BLACK, sides, event.seats);
    ASSERT_TRUE(clock.is_durative());
    EXPECT_EQ(clock.seat_clock_ms(AW), 59700);
    EXPECT_EQ(clock.seat_clock_ms(BW), 59700);
    EXPECT_FALSE(clock.sit_gate());
    EXPECT_TRUE(clock.mate_race_by_landing());
    EXPECT_FALSE(clock.instant_transfer());
    // Black is the root team on both boards: Ab at its planned cost, Bw too.
    EXPECT_EQ(clock.acting_boards(BLACK), (std::array<bool, 2>{true, true}));
    // Aw's observed pace is its landing: 300.
    EXPECT_EQ(game.pace_ms(AW), 300);
}

TEST(SeatClockGame, TheOtherTeamsElapsedThinkingReachesTheSearch) {
    const Wave2GameOptions options = exact(wave2_options());
    SeatClockGame game(options, fixed_start(60000, {300, 200, 500, 400}));
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    game.next(sides);
    game.plan_cost(AW, random);
    game.commit_move(AW);
    game.next(sides);
    game.plan_cost(BW, random);
    game.commit_move(BW);
    game.next(sides);  // Aw lands at 300
    game.landed(AW, false);
    sides[0] = BLACK;
    game.next(sides);  // Ab decides
    game.plan_cost(AB, random);
    game.commit_move(AB);
    // Ab lands at 500 together with Bw: both black, one event.
    SeatClockGame::Event event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::LAND);
    EXPECT_EQ(event.timeMs, 500);
    EXPECT_TRUE(event.seats[AB]);
    EXPECT_TRUE(event.seats[BW]);
    game.landed(AB, true);
    game.landed(BW, false);
    sides = {WHITE, BLACK};
    event = game.next(sides);
    // White on both boards decides at 500: one search for the team.
    EXPECT_TRUE(event.seats[AW]);
    EXPECT_TRUE(event.seats[BB]);
    game.plan_cost(AW, random);
    game.plan_cost(BB, random);
    const TimeState clock = game.search_clock(WHITE, sides, event.seats);
    EXPECT_EQ(clock.lead_ms(WHITE), game.white_lead_ms());
}

TEST(SeatClockGame, APassLastsUntilJustAfterTheOtherBoardsNextEvent) {
    const Wave2GameOptions options = exact(wave2_options());
    SeatClockGame game(options, fixed_start(60000, {300, 200, 500, 400}));
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    game.next(sides);
    game.plan_cost(AW, random);
    // Bw has not decided yet: no landing planned there, so up to the cap.
    game.commit_pass(AW, sides);
    EXPECT_EQ(game.event_at_ms(AW), 8 * 300);
    EXPECT_TRUE(game.sitting(AW));
    game.next(sides);  // Bw decides at 0
    game.plan_cost(BW, random);
    game.commit_move(BW);  // lands at 500
    SeatClockGame::Event event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::LAND);
    EXPECT_EQ(event.timeMs, 500);
    game.landed(BW, false);
    // Bw's landing wakes Aw 1 ms later.
    EXPECT_EQ(game.event_at_ms(AW), 501);
    sides[1] = BLACK;
    event = game.next(sides);  // Bb comes on turn at 500 and decides first
    EXPECT_TRUE(event.seats[BB]);
    game.plan_cost(BB, random);
    game.commit_move(BB);  // lands at 900
    event = game.next(sides);
    EXPECT_EQ(event.timeMs, 501);
    EXPECT_TRUE(event.seats[AW]);
    game.commit_pass(AW, sides);
    EXPECT_EQ(game.event_at_ms(AW), 901);
    // A pass is capped at PassCapK x the seat's move time.
    Wave2GameOptions capped = options;
    capped.passCapK = 1;
    SeatClockGame short_(capped, fixed_start(60000, {300, 200, 5000, 400}));
    short_.next(sides);
    short_.plan_cost(AW, random);
    short_.commit_move(AW);
    short_.next(sides);
    short_.plan_cost(BW, random);
    short_.commit_move(BW);  // lands at 5000
    short_.next(sides);      // Aw lands at 300
    short_.landed(AW, false);
    sides[0] = BLACK;
    short_.next(sides);      // Ab decides at 300
    short_.commit_pass(AB, sides);
    EXPECT_EQ(short_.event_at_ms(AB), 300 + 200);
}

TEST(SeatClockGame, AFlagEndsTheGameAndADoubleFlagIsADraw) {
    const Wave2GameOptions options = exact(wave2_options());
    SeatClockGame::Start start = fixed_start(60000, {500, 200, 500, 400});
    start.clocksMs[AW] = 300;
    SeatClockGame game(options, start);
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    game.next(sides);
    game.plan_cost(AW, random);
    game.commit_move(AW);  // would land at 500, after its flag at 300
    game.next(sides);
    game.plan_cost(BW, random);
    game.commit_move(BW);
    const SeatClockGame::Event flag = game.next(sides);
    EXPECT_EQ(flag.type, SeatClockGame::EventType::FLAG);
    EXPECT_EQ(flag.timeMs, 300);
    EXPECT_TRUE(flag.seats[AW]);
    EXPECT_EQ(flag.loser, static_cast<int>(WHITE));
    EXPECT_EQ(game.clock_ms(AW), 0);
    EXPECT_EQ(game.clock_ms(BW), 59700);

    SeatClockGame::Start both = start;
    both.clocksMs[BW] = 300;
    SeatClockGame tie(options, both);
    tie.next(sides);
    tie.plan_cost(AW, random);
    tie.commit_move(AW);
    tie.next(sides);
    tie.plan_cost(BW, random);
    tie.commit_move(BW);
    EXPECT_EQ(tie.next(sides).type, SeatClockGame::EventType::DOUBLE_FLAG);

    // Without FlagLoss the clock stops at 0 and the move lands.
    Wave2GameOptions noFlag = options;
    noFlag.flagLoss = false;
    SeatClockGame runs(noFlag, start);
    runs.next(sides);
    runs.plan_cost(AW, random);
    runs.commit_move(AW);
    runs.next(sides);
    runs.plan_cost(BW, random);
    runs.commit_move(BW);
    const SeatClockGame::Event landing = runs.next(sides);
    EXPECT_EQ(landing.type, SeatClockGame::EventType::LAND);
    EXPECT_EQ(runs.clock_ms(AW), 0);
}

TEST(SeatClockGame, BothSeatsSittingIsAdjudicatedOnTheClocks) {
    Wave2GameOptions options = exact(wave2_options());
    options.sitoffMs = 1000;
    SeatClockGame::Start start = fixed_start(60000, {300, 200, 500, 400});
    start.clocksMs[BW] = 50000;
    SeatClockGame game(options, start);
    Sides sides{WHITE, WHITE};
    SeatClockGame::Event event;
    for (int step = 0; step < 100; ++step) {
        event = game.next(sides);
        if (event.type != SeatClockGame::EventType::DECIDE) {
            break;
        }
        for (int seat = 0; seat < 4; ++seat) {
            if (event.seats[seat]) {
                game.commit_pass(seat, sides);
            }
        }
    }
    ASSERT_EQ(event.type, SeatClockGame::EventType::SITOFF);
    EXPECT_EQ(event.timeMs, 1000);
    EXPECT_EQ(event.loser, static_cast<int>(BLACK));  // Bw has the lower clock

    start.clocksMs[BW] = 60000;
    SeatClockGame level(options, start);
    for (int step = 0; step < 100; ++step) {
        event = level.next(sides);
        if (event.type != SeatClockGame::EventType::DECIDE) {
            break;
        }
        for (int seat = 0; seat < 4; ++seat) {
            if (event.seats[seat]) {
                level.commit_pass(seat, sides);
            }
        }
    }
    ASSERT_EQ(event.type, SeatClockGame::EventType::SITOFF);
    EXPECT_EQ(event.loser, -1);
}

TEST(SeatClockGame, LandingTiesGoToTheOpponentOfTheLastCapturer) {
    const Wave2GameOptions options = exact(wave2_options());
    // Every seat 300 ms: after the opening decisions, Aw and Bw land together.
    SeatClockGame game(options, fixed_start(60000, {300, 300, 300, 300}));
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    game.next(sides);
    game.plan_cost(AW, random);
    game.commit_move(AW);
    game.next(sides);
    game.plan_cost(BW, random);
    game.commit_move(BW);
    // No capture yet: alternation (white went first at 0, so black now).
    SeatClockGame::Event event = game.next(sides);
    EXPECT_EQ(event.team, BLACK);
    game.landed(BW, true);  // black captures
    sides[1] = BLACK;
    // Aw still lands at 300; Bb decides at 300. Landings come first.
    event = game.next(sides);
    EXPECT_EQ(event.type, SeatClockGame::EventType::LAND);
    EXPECT_TRUE(event.seats[AW]);
}

TEST(SeatClockGame, StockSearchClockIsV3sClockAtTheClocksLead) {
    Wave2GameOptions options = exact(wave2_options());
    options.durativeTree = false;
    options.sitGate = true;
    SeatClockGame::Start start = fixed_start(60000, {300, 200, 500, 400});
    start.clocksMs[AW] = 62000;
    start.clocksMs[BB] = 62000;
    SeatClockGame game(options, start);
    std::mt19937_64 random(1);
    Sides sides{WHITE, WHITE};
    const SeatClockGame::Event event = game.next(sides);
    game.plan_cost(AW, random);
    const TimeState clock = game.search_clock(WHITE, sides, event.seats);
    EXPECT_FALSE(clock.is_durative());
    EXPECT_EQ(clock, TimeState::clocked(WHITE, 2000, (300 + 400) / 2, (200 + 500) / 2, 300));
}

}  // namespace
