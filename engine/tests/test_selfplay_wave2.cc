#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "nn/engine.h"
#include "tools/selfplay.h"

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

}  // namespace
#endif
