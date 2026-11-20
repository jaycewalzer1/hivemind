#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "nn/engine.h"
#include "tools/selfplay.h"
#include "tools/tournament.h"

// Wave 2 self-play plumbing end to end: the options reach a real run_selfplay
// on the uniform test network, games are played on four seat clocks and the
// outputs (HVM6 records, seat_clocks.tsv, PGN tags) agree with each other.

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::stringstream text;
    text << stream.rdbuf();
    return text.str();
}

uint64_t hvm_records(const std::filesystem::path& directory) {
    uint64_t total = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::ifstream stream(entry.path(), std::ios::binary);
        char header[12];
        stream.read(header, sizeof(header));
        uint64_t count = 0;
        stream.read(reinterpret_cast<char*>(&count), sizeof(count));
        total += count;
    }
    return total;
}

std::vector<std::string> tag_values(const std::string& pgn, const std::string& tag) {
    std::vector<std::string> values;
    const std::string key = "[" + tag + " \"";
    for (size_t at = pgn.find(key); at != std::string::npos; at = pgn.find(key, at + 1)) {
        const size_t start = at + key.size();
        values.push_back(pgn.substr(start, pgn.find('"', start) - start));
    }
    return values;
}

class SelfPlayWave2Test : public ::testing::Test {
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
        output = std::filesystem::temp_directory_path()
            / ("hivemind_wave2_selfplay_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::remove_all(output, ignored);
    }

    SelfPlayConfig wave2_config(size_t games) const {
        SelfPlayConfig config;
        config.games = games;
        config.nodes = 16;
        config.maxMacroPlies = 400;
        config.batchSize = 1;
        config.seed = 5;
        config.outputDirectory = output;
        config.rawPolicyMeanMacroPlies = 0.0;
        config.resignThreshold = 1.0f;  // never resign
        config.fairyStockfishMateNodes = 0;
        config.clockModel = true;
        config.clock.moveTimeMinMs = 200;
        config.clock.moveTimeMaxMs = 2000;
        Wave2GameOptions& w = config.wave2;
        w.durativeTree = true;
        w.sitGate = false;
        w.flagLoss = true;
        w.instantTransfer = false;
        w.mateRaceByLanding = true;
        w.baseMix = {{4000, 1.0}};
        w.seatMoveTimes = true;
        return config;
    }

    std::unique_ptr<Engine> engine;
    std::filesystem::path output;
};

TEST_F(SelfPlayWave2Test, SeatClockGamesWriteMatchingRecordsClocksAndTags) {
    const SelfPlayConfig config = wave2_config(3);
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, config), 0);
    const std::string stdoutText = testing::internal::GetCapturedStdout();

    const std::string pgn = read_file(output / "games.pgn");
    EXPECT_EQ(tag_values(pgn, "BaseMs"), (std::vector<std::string>{"4000", "4000", "4000"}));
    EXPECT_EQ(tag_values(pgn, "SeatMoveTimesMs").size(), 3u);
    EXPECT_EQ(tag_values(pgn, "FinalClocksMs").size(), 3u);
    const std::set<std::string> endings{"flag", "double_flag", "sitoff_adjudicated",
                                        "checkmate", "draw", "no legal action", "macro-ply limit"};
    for (const std::string& termination : tag_values(pgn, "Termination")) {
        EXPECT_TRUE(endings.contains(termination)) << termination;
    }
    // A 4 s base at 200-2000 ms a move: some seat runs out.
    EXPECT_NE(stdoutText.find("termination flag"), std::string::npos) << stdoutText;
    EXPECT_NE(stdoutText.find(" flag-seat "), std::string::npos);
    EXPECT_NE(stdoutText.find(" clocks "), std::string::npos);

    // One seat_clocks.tsv row per HVM6 record, and the game lines' sample
    // counts add up to them.
    const std::string rows = read_file(output / "seat_clocks.tsv");
    const size_t lines = static_cast<size_t>(std::count(rows.begin(), rows.end(), '\n'));
    const uint64_t records = hvm_records(output / "training_data");
    EXPECT_GT(records, 0u);
    EXPECT_EQ(lines - 1, records);
    uint64_t announced = 0;
    std::istringstream gameLines(stdoutText);
    for (std::string line; std::getline(gameLines, line);) {
        const size_t at = line.find(" samples ");
        if (line.rfind("selfplay game ", 0) == 0 && at != std::string::npos) {
            announced += std::stoull(line.substr(at + 9));
        }
    }
    EXPECT_EQ(announced, records);
    // Every row's clocks are at most the base plus half the largest lead
    // odds (2 x (base - 1 s)), and never below 0.
    std::istringstream rowStream(rows);
    std::string header;
    std::getline(rowStream, header);
    EXPECT_EQ(header.rfind("game\trecord\tteam\ttime_ms\tnodes\tclock_aw", 0), 0u);
    for (std::string row; std::getline(rowStream, row);) {
        std::istringstream fields(row);
        int64_t game, record, team, time, nodes;
        fields >> game >> record >> team >> time >> nodes;
        for (int seat = 0; seat < 4; ++seat) {
            int64_t clock = -1;
            fields >> clock;
            EXPECT_GE(clock, 0);
            EXPECT_LE(clock, 4000 + 3000);
        }
    }
}

TEST_F(SelfPlayWave2Test, TeamTreeReuseInSeatClockGames) {
    // Wave 2c, fix 1 (spec v5 test 1.8): each team's agent keeps its tree
    // between decisions. The tolerance is the widest allowed: this checks
    // the plumbing; how often the default tolerance hits is the probe's
    // question. Hits are rare anyway: a tree only holds the landings in the
    // order its paces predicted (docs/wave2_engine.md, Wave 2c), so the
    // games are 20 s and 256 nodes to give each team a few.
    SelfPlayConfig config = wave2_config(4);
    config.nodes = 256;
    config.wave2.baseMix = {{20000, 1.0}};
    config.wave2.teamTreeReuse = true;
    config.wave2.teamTreeReuseToleranceMs = 5000;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, config), 0);
    const std::string stdoutText = testing::internal::GetCapturedStdout();
    const size_t at = stdoutText.find("selfplay team tree reuse: ");
    ASSERT_NE(at, std::string::npos) << stdoutText;
    const std::string line = stdoutText.substr(at, stdoutText.find('\n', at) - at);
    std::istringstream summary(line);
    std::string word;
    for (int skip = 0; skip < 4; ++skip) {
        summary >> word;   // "selfplay team tree reuse:"
    }
    double percent = 0.0;
    summary >> percent;
    EXPECT_GT(percent, 0.0) << line;
    std::array<int, 4> hits{};
    const char* const seats[4] = {"Aw", "Ab", "Bw", "Bb"};
    for (int seat = 0; seat < 4; ++seat) {
        const size_t named = line.find(std::string(" ") + seats[seat] + " ");
        ASSERT_NE(named, std::string::npos) << line;
        hits[seat] = std::stoi(line.substr(named + 4));
    }
    // Every adopting search counted its deciding seat(s). Which seats adopt
    // varies run to run (four search workers); that one seat adopts the
    // tree its partner's search built is TeamTreeReuse.BothSeatsOfTheTeam.
    const size_t open = line.find('(');
    const int adopted = std::stoi(line.substr(open + 1));
    EXPECT_GT(adopted, 0) << line;
    EXPECT_GE(hits[0] + hits[1] + hits[2] + hits[3], adopted) << line;
    EXPECT_NE(line.find(", refused"), std::string::npos);
    // One seat_clocks.tsv row per record, as without the option.
    const std::string rows = read_file(output / "seat_clocks.tsv");
    const size_t lines = static_cast<size_t>(std::count(rows.begin(), rows.end(), '\n'));
    EXPECT_EQ(lines - 1, hvm_records(output / "training_data"));
    EXPECT_NE(stdoutText.find("selfplay complete 4 games"), std::string::npos);
}

TEST_F(SelfPlayWave2Test, TeamTreeReuseNeedsDurativeSeatClocks) {
    Wave2GameOptions options;
    options.teamTreeReuse = true;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    options.durativeTree = true;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    options.baseMix = {{4000, 1.0}};
    EXPECT_NO_THROW(options.validate());
    options.teamTreeReuseToleranceMs = 5001;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    Wave2GameOptions parsed;
    EXPECT_TRUE(parse_wave2_option("--team-tree-reuse", "true", parsed));
    EXPECT_TRUE(parse_wave2_option("--team-tree-reuse-tolerance-ms", "1000", parsed));
    EXPECT_TRUE(parsed.teamTreeReuse);
    EXPECT_EQ(parsed.teamTreeReuseToleranceMs, 1000);
    EXPECT_TRUE(parsed.changes_search());
}

TEST_F(SelfPlayWave2Test, ContinuityKnobsNeedTheirSwitches) {
    // A knob alone does nothing, so it must neither count as a search change
    // (main.cc would move a stock run onto the clock time model) nor pass.
    Wave2GameOptions tolerance;
    tolerance.teamTreeReuseToleranceMs = 400;
    Wave2GameOptions free;
    free.jointPlanFreePermille = 400;
    Wave2GameOptions carry;
    carry.planCarryPermille = 200;
    for (const Wave2GameOptions* options : {&tolerance, &free, &carry}) {
        EXPECT_FALSE(options->changes_search());
        EXPECT_THROW(options->validate(), std::invalid_argument);
    }
    Wave2GameOptions parsed;
    EXPECT_TRUE(parse_wave2_option("--plan-carry-permille", "200", parsed));
    EXPECT_THROW(parsed.validate(), std::invalid_argument);
    // With the switches (here the preset) the knobs are accepted.
    EXPECT_TRUE(parse_wave2_option("--durative-tree", "true", parsed));
    EXPECT_TRUE(parse_wave2_option("--base-mix", "4000:1", parsed));
    EXPECT_TRUE(parse_wave2_option("--continuity", "true", parsed));
    EXPECT_TRUE(parse_wave2_option("--joint-plan-free-permille", "400", parsed));
    EXPECT_TRUE(parse_wave2_option("--team-tree-reuse-tolerance-ms", "400", parsed));
    EXPECT_NO_THROW(parsed.validate());
    EXPECT_TRUE(parsed.changes_search());
}

TEST_F(SelfPlayWave2Test, TreeOptionsAloneKeepTheLeadOnlyGame) {
    SelfPlayConfig config = wave2_config(1);
    config.wave2.baseMix.clear();
    config.wave2.flagLoss = false;
    config.maxMacroPlies = 12;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, config), 0);
    const std::string stdoutText = testing::internal::GetCapturedStdout();
    EXPECT_NE(stdoutText.find("selfplay complete 1 games"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(output / "seat_clocks.tsv"));
    EXPECT_TRUE(tag_values(read_file(output / "games.pgn"), "BaseMs").empty());
}

TEST_F(SelfPlayWave2Test, TournamentPlaysSeatClockPairsWithTheSameOptions) {
    TournamentConfig config;
    config.games = 2;
    config.nodes = 16;
    config.maxMacroPlies = 400;
    config.contenderBatchSize = 1;
    config.baselineBatchSize = 1;
    config.seed = 9;
    config.outputDirectory = output;
    config.clockModel = true;
    config.clock.moveTimeMinMs = 200;
    config.clock.moveTimeMaxMs = 2000;
    config.wave2 = wave2_config(1).wave2;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_tournament(engine.get(), engine.get(), "uniform-a", "uniform-b", config), 0);
    const std::string stdoutText = testing::internal::GetCapturedStdout();
    EXPECT_NE(stdoutText.find("tournament game 2/2"), std::string::npos) << stdoutText;
    const std::string pgn = read_file(output / "games.pgn");
    const std::set<std::string> endings{"flag", "double_flag", "sitoff_adjudicated",
                                        "checkmate", "draw", "no legal action", "macro-ply limit"};
    const std::vector<std::string> terminations = tag_values(pgn, "Termination");
    ASSERT_EQ(terminations.size(), 2u);
    for (const std::string& termination : terminations) {
        EXPECT_TRUE(endings.contains(termination)) << termination;
    }
}

TEST_F(SelfPlayWave2Test, PlanCarryInSeatClockGames) {
    // Wave 2c, fix 3 (spec v5 test 3.6): the half of a team's chosen action
    // for its on-turn seat that is not deciding (in these games a sitting
    // seat: a deciding seat's move is played, one in flight lands) is
    // remembered, and that seat's next search applies it. Two games on the
    // same agents also run the game-start clear (3.8).
    // Passes are what leave a seat undecided, and the uniform net passes
    // rarely: 6 games at 20 s give a few.
    SelfPlayConfig config = wave2_config(6);
    config.nodes = 64;
    config.wave2.baseMix = {{20000, 1.0}};
    config.wave2.planCarry = true;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, config), 0);
    const std::string stdoutText = testing::internal::GetCapturedStdout();
    const size_t at = stdoutText.find("selfplay plan carry: ");
    ASSERT_NE(at, std::string::npos) << stdoutText;
    const std::string line = stdoutText.substr(at, stdoutText.find('\n', at) - at);
    std::istringstream summary(line);
    std::string word;
    for (int skip = 0; skip < 3; ++skip) {
        summary >> word;   // "selfplay plan carry:"
    }
    uint64_t recorded = 0, applied = 0, searches = 0;
    summary >> recorded >> word >> applied >> word >> word >> searches;
    EXPECT_GT(recorded, 0u) << line;
    EXPECT_GT(applied, 0u) << line;
    EXPECT_GE(searches, applied) << line;
    // One agent per team that never changes team: no plan of the other team.
    EXPECT_EQ(line.find("other team"), std::string::npos) << line;
    EXPECT_NE(stdoutText.find("selfplay complete 6 games"), std::string::npos);
}

TEST_F(SelfPlayWave2Test, PlanCarryNeedsDurativeSeatClocks) {
    Wave2GameOptions options;
    options.planCarry = true;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    options.durativeTree = true;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    options.baseMix = {{4000, 1.0}};
    EXPECT_NO_THROW(options.validate());
    options.planCarryPermille = 1001;
    EXPECT_THROW(options.validate(), std::invalid_argument);
    Wave2GameOptions parsed;
    EXPECT_TRUE(parse_wave2_option("--plan-carry", "true", parsed));
    EXPECT_TRUE(parse_wave2_option("--plan-carry-permille", "450", parsed));
    EXPECT_TRUE(parsed.planCarry);
    EXPECT_EQ(parsed.planCarryPermille, 450);
    EXPECT_TRUE(parsed.changes_search());
    // The preset sets the three switches and leaves the knobs.
    Wave2GameOptions preset;
    EXPECT_TRUE(parse_wave2_option("--continuity", "true", preset));
    EXPECT_TRUE(preset.teamTreeReuse && preset.jointPlans && preset.planCarry);
    EXPECT_EQ(preset.planCarryPermille, 300);
    EXPECT_TRUE(parse_wave2_option("--continuity", "false", preset));
    EXPECT_FALSE(preset.teamTreeReuse || preset.jointPlans || preset.planCarry);
}

/** The numbers after each label in a tournament's per-game continuity lines, summed. */
std::map<std::string, uint64_t> tournament_continuity(const std::string& stdoutText, size_t games,
                                                      const std::vector<std::string>& labels) {
    std::map<std::string, uint64_t> sums;
    for (size_t game = 1; game <= games; ++game) {
        const std::string key = "tournament continuity game " + std::to_string(game) + "/"
            + std::to_string(games) + ": ";
        const size_t at = stdoutText.find(key);
        EXPECT_NE(at, std::string::npos) << stdoutText;
        if (at == std::string::npos) {
            continue;
        }
        const std::string line = stdoutText.substr(at, stdoutText.find('\n', at) - at);
        for (const std::string& label : labels) {
            const size_t named = line.find(label);
            EXPECT_NE(named, std::string::npos) << label << " in " << line;
            if (named != std::string::npos) {
                sums[label] += std::stoull(line.substr(named + label.size()));
            }
        }
        // Both agents start every game with no tree and no plan.
        EXPECT_NE(line.find(", left at start 0"), std::string::npos) << line;
    }
    return sums;
}

TEST_F(SelfPlayWave2Test, TournamentWithContinuity) {
    // Spec v5 3.8: contender and baseline swap teams between the pair's two
    // games; each game starts with both agents' trees and plans cleared.
    // The per-game continuity line says what was left after the clear and
    // whether fix 1 and fix 3 fired. The games start from a position with
    // few moves on each board, so seats sit (fix 3 remembers a sitting
    // seat's half) and trees hold the lines played.
    const std::filesystem::path positions = output.string() + "_positions.tsv";
    {
        std::ofstream file(positions);
        file << "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1|4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1\twhite\n";
    }
    const auto run = [&](bool teamTree, bool plans, const std::filesystem::path& directory) {
        TournamentConfig config;
        config.games = 2;
        config.nodes = 64;
        config.maxMacroPlies = 400;
        config.contenderBatchSize = 1;
        config.baselineBatchSize = 1;
        config.seed = 9;
        config.outputDirectory = directory;
        config.positionsFile = positions;
        config.clockModel = true;
        config.clock.moveTimeMinMs = 200;
        config.clock.moveTimeMaxMs = 2000;
        config.wave2 = wave2_config(1).wave2;
        config.wave2.baseMix = {{20000, 1.0}};
        config.wave2.teamTreeReuse = teamTree;
        config.wave2.teamTreeReuseToleranceMs = teamTree ? 5000 : 250;
        config.wave2.jointPlans = teamTree && plans;
        config.wave2.planCarry = plans;
        config.wave2.validate();
        testing::internal::CaptureStdout();
        EXPECT_EQ(run_tournament(engine.get(), engine.get(), "uniform-a", "uniform-b", config), 0);
        const std::string stdoutText = testing::internal::GetCapturedStdout();
        EXPECT_NE(stdoutText.find("tournament game 2/2"), std::string::npos) << stdoutText;
        return stdoutText;
    };
    // All three fixes: trees are adopted (the old root among them, when the
    // partner decides before anything lands) and halves are remembered.
    // With fix 1 on, the next search of the team mostly re-adopts the old
    // root, whose own visits carry the half, so it is not mixed in again.
    const std::string all = run(true, true, output / "continuity");
    const auto sums = tournament_continuity(all, 2, {"team tree adopted ", "plans recorded ",
                                                     " other team "});
    EXPECT_GT(sums.at("team tree adopted "), 0u) << all;
    EXPECT_GT(sums.at("plans recorded "), 0u) << all;
    // A plan never reaches the other team (the sides swap teams).
    EXPECT_EQ(sums.at(" other team "), 0u) << all;
    // Fix 3 alone: the remembered halves are mixed into the next search.
    const std::string carry = run(false, true, output / "carry");
    const auto carried = tournament_continuity(carry, 2, {"plans recorded ", " applied "});
    EXPECT_GT(carried.at("plans recorded "), 0u) << carry;
    EXPECT_GT(carried.at(" applied "), 0u) << carry;
    std::error_code ignored;
    std::filesystem::remove(positions, ignored);
}

}  // namespace
#endif
