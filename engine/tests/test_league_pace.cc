#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "nn/engine.h"
#include "tools/distill_data.h"
#include "tools/seat_clock_game.h"
#include "tools/selfplay.h"

// Spec v7 section 5 (WP3): pace mode (each team's k sets every decision's
// cost and budget), league games (the opponent team searches with its own
// network, only the learner's records are written) and the think policy
// hook, all off by default.

namespace {

Wave2GameOptions plain_options() {
    Wave2GameOptions options;
    options.baseMix = {{60000, 1.0}};
    options.paceJitter = 0.0;
    options.lagMaxMs = 0;
    options.minMoveMs = 100;
    options.passCapK = 8;
    return options;
}

SeatClockGame::Start level_start(int32_t clockMs) {
    SeatClockGame::Start start;
    start.baseMs = 60000;
    start.clocksMs = {clockMs, clockMs, clockMs, clockMs};
    start.moveTimesMs = {500, 500, 500, 500};
    return start;
}

}  // namespace

TEST(PaceCost, FollowsTheClockOverKAtFullAndHalfClock) {
    std::mt19937_64 random(1);
    SeatClockGame full(plain_options(), level_start(60000));
    full.apply_pace({110.0, 220.0});
    // Apply_pace: the move times (and first observed paces) are clock / k.
    EXPECT_EQ(full.move_time_ms(TimeState::SEAT_AW), 545);
    EXPECT_EQ(full.move_time_ms(TimeState::SEAT_AB), 273);
    EXPECT_EQ(full.pace_ms(TimeState::SEAT_AB), 273);
    EXPECT_EQ(full.start().moveTimesMs[TimeState::SEAT_AW], 545);
    EXPECT_EQ(full.plan_pace_cost(TimeState::SEAT_AW, 110.0, 1.0, random), 545);
    EXPECT_EQ(full.plan_pace_cost(TimeState::SEAT_AW, 110.0, 2.0, random), 1091);
    EXPECT_EQ(full.move_time_ms(TimeState::SEAT_AW), 1091);

    SeatClockGame half(plain_options(), level_start(30000));
    EXPECT_EQ(half.plan_pace_cost(TimeState::SEAT_AW, 110.0, 1.0, random), 273);
    EXPECT_EQ(half.plan_pace_cost(TimeState::SEAT_AW, 110.0, 0.5, random), 136);
    // Floored at the minimum move time.
    SeatClockGame low(plain_options(), level_start(5000));
    EXPECT_EQ(low.plan_pace_cost(TimeState::SEAT_AW, 400.0, 1.0, random), 100);
    EXPECT_EQ(low.move_time_ms(TimeState::SEAT_AW), 13);
}

TEST(PaceCost, DrawsWhatPlanCostDrawsAndRescalesWithTheSameDraws) {
    Wave2GameOptions options = plain_options();
    options.paceJitter = 0.25;
    options.lagMaxMs = 150;
    std::mt19937_64 a(7);
    std::mt19937_64 b(7);
    SeatClockGame stock(options, level_start(60000));
    SeatClockGame paced(options, level_start(60000));
    stock.plan_cost(TimeState::SEAT_AW, a);
    const int32_t first = paced.plan_pace_cost(TimeState::SEAT_AW, 110.0, 1.0, b);
    // The game's stream is where it would be without pace mode.
    EXPECT_EQ(a(), b());
    // Rescaling reuses the jitter and lag: scale 2 roughly doubles the cost.
    const int32_t doubled = paced.rescale_pace_cost(TimeState::SEAT_AW, 110.0, 2.0);
    EXPECT_GE(first, 545 / 1.25 - 1);
    EXPECT_LE(first, 545 * 1.25 + 151);
    EXPECT_GT(doubled, first);
    EXPECT_EQ(paced.rescale_pace_cost(TimeState::SEAT_AW, 110.0, 1.0), first);
}

TEST(PaceCost, PassCapFollowsThePace) {
    std::mt19937_64 random(3);
    SeatClockGame game(plain_options(), level_start(60000));
    const std::array<Stockfish::Color, 2> sides{Stockfish::WHITE, Stockfish::WHITE};
    // Board B's white seat decides at the same moment and has no landing
    // planned, so A's pass lasts PassCapK x its (paced) move time.
    game.plan_pace_cost(TimeState::SEAT_AW, 110.0, 1.0, random);
    game.commit_pass(TimeState::SEAT_AW, sides);
    EXPECT_EQ(game.event_at_ms(TimeState::SEAT_AW) - game.now_ms(), 8 * 545);
    game.plan_pace_cost(TimeState::SEAT_BW, 400.0, 1.0, random);
    game.commit_pass(TimeState::SEAT_BW, sides);
    EXPECT_EQ(game.event_at_ms(TimeState::SEAT_BW) - game.now_ms(), 8 * 150);
}

TEST(PaceBudget, FormulaAndClamps) {
    PaceSampling pace;
    pace.kMin = 80;
    pace.kMax = 400;
    EXPECT_EQ(pace_node_budget(pace, 800, 110.0, 1.0, 60000, 60000), 800u);
    EXPECT_EQ(pace_node_budget(pace, 800, 110.0, 1.0, 30000, 60000), 400u);
    EXPECT_EQ(pace_node_budget(pace, 800, 400.0, 1.0, 60000, 60000), 220u);
    EXPECT_EQ(pace_node_budget(pace, 800, 80.0, 2.0, 60000, 60000), 2200u);
    EXPECT_EQ(pace_node_budget(pace, 800, 400.0, 1.0, 6000, 60000), 200u);   // min
    EXPECT_EQ(pace_node_budget(pace, 2000, 80.0, 2.0, 60000, 60000), 3200u); // max
    // A lead puts a clock above the base: more than N at k = ref.
    EXPECT_EQ(pace_node_budget(pace, 800, 110.0, 1.0, 66000, 60000), 880u);
}

TEST(PaceBudget, KIsLogUniformInRange) {
    std::mt19937_64 random(11);
    std::vector<int32_t> draws;
    for (int i = 0; i < 4000; ++i) {
        draws.push_back(sample_log_uniform(80, 400, random));
    }
    EXPECT_EQ(*std::min_element(draws.begin(), draws.end()) >= 80, true);
    EXPECT_EQ(*std::max_element(draws.begin(), draws.end()) <= 400, true);
    std::nth_element(draws.begin(), draws.begin() + 2000, draws.end());
    // Median of log-uniform [80, 400]: sqrt(80 x 400) = 179.
    EXPECT_NEAR(draws[2000], 179, 10);
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::stringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::vector<std::string> game_lines(const std::string& out) {
    std::vector<std::string> lines;
    std::istringstream stream(out);
    for (std::string line; std::getline(stream, line);) {
        if (line.rfind("selfplay game ", 0) == 0) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::string field(const std::string& line, const std::string& key) {
    const size_t at = line.find(" " + key + " ");
    if (at == std::string::npos) {
        return "";
    }
    const size_t start = at + key.size() + 2;
    return line.substr(start, line.find(' ', start) - start);
}

size_t count_records(const std::filesystem::path& directory) {
    size_t total = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".dst") {
            total += distill::read_chunk(entry.path()).size();
        }
    }
    return total;
}

class LeaguePaceSelfPlay : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        const auto model = std::filesystem::path(__FILE__).parent_path() / "data" / "uniform.onnx";
        engine = std::make_unique<Engine>(0, 1);
        opponent = std::make_unique<Engine>(0, 1);
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
        ASSERT_TRUE(opponent->loadNetwork(model.string(), ""));
        root = std::filesystem::temp_directory_path()
            / ("hivemind_league_pace_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    // Small seat-clock games as rl-loop plays them (hybrid gate).
    SelfPlayConfig config(const std::string& name, size_t games) const {
        SelfPlayConfig c;
        c.games = games;
        c.nodes = 16;
        c.maxMacroPlies = 60;
        c.batchSize = 1;
        c.seed = 77;
        c.outputDirectory = root / name / "games";
        c.distillOutputDirectory = root / name;
        c.writeTrainingChunks = false;
        c.resignThreshold = 0.0f;
        c.fairyStockfishMateNodes = 0;
        c.rootScans = false;
        c.rawPolicyMeanMacroPlies = 0.0;
        c.clockModel = true;
        Wave2GameOptions& w = c.wave2;
        w.flagLoss = true;
        w.instantTransfer = false;
        w.baseMix = {{20000, 1.0}};
        w.seatMoveTimes = true;
        w.hybrid.mode = HybridMode::GATE;
        return c;
    }

    SelfPlayConfig paced(const std::string& name, size_t games) const {
        SelfPlayConfig c = config(name, games);
        c.pace.kMin = 80;
        c.pace.kMax = 400;
        c.pace.minNodes = 4;
        c.pace.maxNodes = 64;
        c.wave2.paceJitter = 0.0;
        c.wave2.lagMaxMs = 0;
        return c;
    }

    std::string run(const SelfPlayConfig& c, bool league = false) {
        testing::internal::CaptureStdout();
        const int status = league ? run_selfplay({engine.get()}, c, {opponent.get()})
                                  : run_selfplay({engine.get()}, c);
        const std::string out = testing::internal::GetCapturedStdout();
        EXPECT_EQ(status, 0);
        return out;
    }

    std::unique_ptr<Engine> engine;
    std::unique_ptr<Engine> opponent;
    std::filesystem::path root;
};

TEST_F(LeaguePaceSelfPlay, OffModeKeepsBudgetsOutputsAndColumns) {
    SelfPlayConfig c = config("off", 2);
    std::vector<SearchObservation> seen;
    c.searchObserver = [&seen](const SearchObservation& s) { seen.push_back(s); };
    const std::string out = run(c);
    ASSERT_FALSE(seen.empty());
    for (const SearchObservation& s : seen) {
        EXPECT_FALSE(s.opponent);
        EXPECT_EQ(s.k, 0.0);
        EXPECT_EQ(s.targetNodes, s.jitteredNodes);
        EXPECT_GE(s.targetNodes, 15u);
        EXPECT_LE(s.targetNodes, 17u);
    }
    for (const std::string& line : game_lines(out)) {
        EXPECT_EQ(line.find("k_white_team"), std::string::npos) << line;
        EXPECT_EQ(line.find("learner_team"), std::string::npos) << line;
    }
    const std::string rows = read_text(root / "off" / "games" / "seat_clocks.tsv");
    EXPECT_EQ(rows.find("k_team"), std::string::npos);
    EXPECT_EQ(read_text(root / "off" / "games" / "games.pgn").find("KWhiteTeam"), std::string::npos);
    // The observer is only a probe: the same run without it writes the same records.
    SelfPlayConfig again = config("off2", 2);
    run(again);
    EXPECT_EQ(count_records(root / "off"), count_records(root / "off2"));
}

TEST_F(LeaguePaceSelfPlay, PaceSetsCostAndNodesPerDecision) {
    SelfPlayConfig c = paced("pace", 3);
    c.recordStateOutput = root / "pace" / "games" / "states.jsonl";
    std::vector<SearchObservation> seen;
    c.searchObserver = [&seen](const SearchObservation& s) { seen.push_back(s); };
    const std::string out = run(c);
    ASSERT_FALSE(seen.empty());
    for (const SearchObservation& s : seen) {
        EXPECT_GE(s.k, 80.0);
        EXPECT_LE(s.k, 400.0);
        int32_t clockNow = 0;
        for (int seat = 0; seat < 4; ++seat) {
            if (!s.deciding[seat]) {
                continue;
            }
            clockNow = std::max(clockNow, s.clocksMs[seat]);
            const int32_t paced = static_cast<int32_t>(std::llround(s.clocksMs[seat] / s.k));
            EXPECT_EQ(s.plannedMs[seat], std::max<int32_t>(100, paced));
            EXPECT_EQ(s.moveTimesMs[seat], std::max<int32_t>(1, paced));
        }
        EXPECT_EQ(s.targetNodes, pace_node_budget(c.pace, s.jitteredNodes, s.k, 1.0, clockNow, 20000));
    }
    // Game lines and PGN tags carry the paces; rows carry the record team's k.
    const std::vector<std::string> lines = game_lines(out);
    ASSERT_EQ(lines.size(), 3u);
    for (const std::string& line : lines) {
        const int kWhite = std::stoi(field(line, "k_white_team"));
        EXPECT_GE(kWhite, 80);
        EXPECT_LE(kWhite, 400);
        EXPECT_EQ(line.find("learner_team"), std::string::npos);
    }
    EXPECT_NE(read_text(root / "pace" / "games" / "games.pgn").find("[KWhiteTeam \""), std::string::npos);
    const std::string rows = read_text(root / "pace" / "games" / "seat_clocks.tsv");
    EXPECT_NE(rows.find("\tk_team\tthink_scale\n"), std::string::npos);
    // The state dump records the paced budgets reanalyse rebuilds.
    const std::string states = read_text(c.recordStateOutput);
    size_t records = 0;
    for (size_t at = states.find("\"target_nodes\": "); at != std::string::npos;
         at = states.find("\"target_nodes\": ", at + 1)) {
        const size_t target = std::stoull(states.substr(at + 16));
        EXPECT_GE(target, 4u);
        EXPECT_LE(target, 64u);
        ++records;
    }
    EXPECT_EQ(records, seen.size());
    EXPECT_EQ(records, count_records(root / "pace"));
}

TEST_F(LeaguePaceSelfPlay, LeagueWritesOnlyTheLearnersSearches) {
    SelfPlayConfig c = paced("league", 4);
    c.recordStateOutput = root / "league" / "games" / "states.jsonl";
    size_t learner = 0;
    size_t opponentSearches = 0;
    c.searchObserver = [&](const SearchObservation& s) {
        (s.opponent ? opponentSearches : learner) += 1;
    };
    const std::string out = run(c, true);
    EXPECT_GT(opponentSearches, 0u);
    EXPECT_EQ(count_records(root / "league"), learner);
    const std::vector<std::string> lines = game_lines(out);
    ASSERT_EQ(lines.size(), 4u);
    std::vector<int> learnerTeams;
    size_t announced = 0;
    for (const std::string& line : lines) {
        const std::string team = field(line, "learner_team");
        ASSERT_TRUE(team == "white" || team == "black") << line;
        learnerTeams.push_back(team == "white" ? 0 : 1);
        const std::string result = field(line, "learner_result");
        EXPECT_TRUE(result == "win" || result == "draw" || result == "loss") << line;
        announced += std::stoull(field(line, "samples"));
    }
    EXPECT_EQ(announced, learner);
    // Every seat row (one per record) is the learner team's.
    std::istringstream rows(read_text(root / "league" / "games" / "seat_clocks.tsv"));
    std::string header;
    std::getline(rows, header);
    size_t rowCount = 0;
    for (std::string row; std::getline(rows, row); ++rowCount) {
        std::istringstream fields(row);
        size_t game = 0, record = 0;
        int team = -1;
        fields >> game >> record >> team;
        ASSERT_GE(game, 1u);
        EXPECT_EQ(team, learnerTeams[game - 1]) << row;
    }
    EXPECT_EQ(rowCount, learner);
    const std::string states = read_text(c.recordStateOutput);
    EXPECT_EQ(static_cast<size_t>(std::count(states.begin(), states.end(), '\n')),
              1 + lines.size() + learner);
    EXPECT_NE(out.find("selfplay league: learner records only, " + std::to_string(opponentSearches)
                       + " opponent searches"), std::string::npos) << out;
    EXPECT_NE(out.find("selfplay complete 4 games"), std::string::npos);
}

TEST_F(LeaguePaceSelfPlay, LearnerTeamSplitsAndOpponentKIsFixed) {
    SelfPlayConfig c = paced("split", 40);
    c.maxMacroPlies = 2;
    c.opponentK = 250;
    const std::string out = run(c, true);
    int white = 0;
    for (const std::string& line : game_lines(out)) {
        const bool learnerWhite = field(line, "learner_team") == "white";
        white += learnerWhite ? 1 : 0;
        EXPECT_EQ(field(line, learnerWhite ? "k_black_team" : "k_white_team"), "250") << line;
    }
    // Binomial(40, 0.5): 10..30 holds with probability > 0.998.
    EXPECT_GE(white, 10);
    EXPECT_LE(white, 30);
}

TEST_F(LeaguePaceSelfPlay, FixedLearnerTeamAndOpponentEngine) {
    SelfPlayConfig c = paced("black", 2);
    c.learnerTeam = 1;
    std::vector<SearchObservation> seen;
    c.searchObserver = [&seen](const SearchObservation& s) { seen.push_back(s); };
    const std::string out = run(c, true);
    for (const SearchObservation& s : seen) {
        // White, the opponent, always searches on the opponent's engines.
        EXPECT_EQ(s.opponent, s.team == 0);
    }
    for (const std::string& line : game_lines(out)) {
        EXPECT_EQ(field(line, "learner_team"), "black");
    }
}

TEST_F(LeaguePaceSelfPlay, ThinkPolicyWithoutAHeadSearchesAtScaleOne) {
#if __has_include("tools/think_policy.h")
    SelfPlayConfig c = paced("think", 2);
    c.thinkPolicy = true;
    std::vector<SearchObservation> seen;
    c.searchObserver = [&seen](const SearchObservation& s) { seen.push_back(s); };
    const std::string out = run(c);
    ASSERT_FALSE(seen.empty());
    for (const SearchObservation& s : seen) {
        EXPECT_EQ(s.thinkScale, 1.0);
    }
    for (const std::string& line : game_lines(out)) {
        EXPECT_EQ(field(line, "mean_s"), "1.000");
        EXPECT_GT(std::stoull(field(line, "think_missing")), 0u) << line;
    }
    EXPECT_NE(out.find("selfplay think policy: mean scale 1.000"), std::string::npos) << out;
#else
    GTEST_SKIP() << "no tools/think_policy.h";
#endif
}

TEST_F(LeaguePaceSelfPlay, Validation) {
    auto refused = [this](const SelfPlayConfig& c, bool league) {
        try {
            if (league) {
                run_selfplay({engine.get()}, c, {opponent.get()});
            } else {
                run_selfplay({engine.get()}, c);
            }
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    SelfPlayConfig noClocks = paced("v1", 1);
    noClocks.wave2.baseMix.clear();
    noClocks.wave2.hybrid.mode = HybridMode::OFF;
    EXPECT_TRUE(refused(noClocks, false));
    SelfPlayConfig badRange = paced("v2", 1);
    badRange.pace.kMax = 50;
    EXPECT_TRUE(refused(badRange, false));
    SelfPlayConfig onlyMax = config("v3", 1);
    onlyMax.pace.kMax = 400;
    EXPECT_TRUE(refused(onlyMax, false));
    SelfPlayConfig noLeague = paced("v4", 1);
    noLeague.opponentK = 110;
    EXPECT_TRUE(refused(noLeague, false));
    SelfPlayConfig learner = config("v5", 1);
    learner.learnerTeam = 0;
    EXPECT_TRUE(refused(learner, false));
    SelfPlayConfig opponentKNoPace = config("v6", 1);
    opponentKNoPace.opponentK = 110;
    EXPECT_TRUE(refused(opponentKNoPace, true));
    // Same engine for both sides is refused.
    SelfPlayConfig same = paced("v7", 1);
    bool sameRefused = false;
    try {
        run_selfplay({engine.get()}, same, {engine.get()});
    } catch (const std::invalid_argument&) {
        sameRefused = true;
    }
    EXPECT_TRUE(sameRefused);
}

}  // namespace
#endif
