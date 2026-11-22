#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "nn/engine.h"
#include "tools/distill_data.h"
#include "tools/reanalyse.h"
#include "tools/selfplay.h"

// docs/search_balance.md section 6: the playout cap (deep / fast searches in
// self-play), the state dump and `reanalyse`.

namespace {

TEST(PlayoutCap, OffLeavesTheBudgetAndNoiseAlone) {
    SelfPlayConfig config;
    EXPECT_FALSE(config.cap_active());
    for (size_t decision = 0; decision < 1000; ++decision) {
        const CapDecision cap = cap_decision(config, 1234, 7, decision, 790 + decision % 20);
        EXPECT_EQ(cap.targetNodes, 790 + decision % 20);
        EXPECT_EQ(cap.dirichletEpsilon, config.dirichletEpsilon);
        EXPECT_FALSE(cap.deep);
        EXPECT_FALSE(cap.masked);
        EXPECT_TRUE(cap.feedsResign);
    }
    // Setting the defaults explicitly keeps it off.
    config.fastNodes = config.nodes;
    config.deepMultiplier = 4;
    EXPECT_FALSE(config.cap_active());
}

TEST(PlayoutCap, DeepShareIsBinomialAndDeterministic) {
    SelfPlayConfig config;
    config.deepFraction = 0.125;
    config.deepMultiplier = 2;
    ASSERT_TRUE(config.cap_active());
    EXPECT_FALSE(config.fast_policy_masked());
    size_t deep = 0;
    const size_t games = 200;
    const size_t decisions = 150;
    for (size_t game = 0; game < games; ++game) {
        for (size_t decision = 0; decision < decisions; ++decision) {
            const CapDecision cap = cap_decision(config, 1000, game, decision, 800);
            const CapDecision again = cap_decision(config, 1000, game, decision, 800);
            EXPECT_EQ(cap.deep, again.deep);
            EXPECT_EQ(cap.targetNodes, cap.deep ? 1600u : 800u);
            // b = nodes: every search keeps its noise, policy and resign vote.
            EXPECT_EQ(cap.dirichletEpsilon, config.dirichletEpsilon);
            EXPECT_FALSE(cap.masked);
            EXPECT_TRUE(cap.feedsResign);
            deep += cap.deep ? 1 : 0;
        }
    }
    const double n = static_cast<double>(games * decisions);
    const double sigma = std::sqrt(n * 0.125 * 0.875);
    EXPECT_NEAR(static_cast<double>(deep), n * 0.125, 4.0 * sigma);
    // Another salt or seed draws other coins.
    SelfPlayConfig salted = config;
    salted.capSeedSalt = 99;
    size_t differ = 0;
    for (size_t decision = 0; decision < 2000; ++decision) {
        differ += cap_decision(config, 1000, 0, decision, 800).deep
            != cap_decision(salted, 1000, 0, decision, 800).deep ? 1 : 0;
    }
    EXPECT_GT(differ, 100u);
}

TEST(PlayoutCap, FastSearchesAreMaskedNoiselessAndDoNotResign) {
    SelfPlayConfig config;
    config.deepFraction = 0.25;
    config.deepMultiplier = 4;
    config.fastNodes = 200;
    ASSERT_TRUE(config.fast_policy_masked());
    bool sawDeep = false;
    bool sawFast = false;
    for (size_t decision = 0; decision < 200; ++decision) {
        const CapDecision cap = cap_decision(config, 5, 3, decision, 820);
        if (cap.deep) {
            sawDeep = true;
            EXPECT_EQ(cap.targetNodes, 3280u);
            EXPECT_EQ(cap.dirichletEpsilon, config.dirichletEpsilon);
            EXPECT_FALSE(cap.masked);
            EXPECT_TRUE(cap.feedsResign);
        } else {
            sawFast = true;
            EXPECT_EQ(cap.targetNodes, 205u);  // the base's jitter carries over
            EXPECT_EQ(cap.dirichletEpsilon, 0.0f);
            EXPECT_TRUE(cap.masked);
            EXPECT_FALSE(cap.feedsResign);
        }
    }
    EXPECT_TRUE(sawDeep && sawFast);
    // Probe mode: full-budget fast searches, policy masked by request, no
    // noise on deep searches.
    SelfPlayConfig probe;
    probe.deepFraction = 0.125;
    probe.deepMultiplier = 16;
    probe.fastPolicyMask = 1;
    probe.deepDirichlet = false;
    ASSERT_TRUE(probe.cap_active());
    for (size_t decision = 0; decision < 200; ++decision) {
        const CapDecision cap = cap_decision(probe, 5, 3, decision, 800);
        EXPECT_EQ(cap.masked, !cap.deep);
        EXPECT_EQ(cap.dirichletEpsilon, cap.deep ? 0.0f : probe.dirichletEpsilon);
        EXPECT_TRUE(cap.feedsResign);
    }
}

TEST(PlayoutCap, RecordsWithoutPolicyRoundTrip) {
    const auto path = std::filesystem::temp_directory_path()
        / ("hivemind_cap_chunk_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".dst");
    std::vector<distill::Record> records(3);
    records[0].policy[0] = {{3, 0x3C00}};
    records[0].value = 0.25f;
    records[1].value = -0.5f;  // masked: no policy on either board
    records[1].wdl = {1.0f, 0.0f, 0.0f};
    records[2].policy[1] = {{7, 0x3800}, {9, 0x3800}};
    distill::write_chunk(path, records);
    const std::vector<distill::Record> read = distill::read_chunk(path);
    std::filesystem::remove(path);
    ASSERT_EQ(read.size(), 3u);
    EXPECT_EQ(read[0].policy[0].size(), 1u);
    EXPECT_TRUE(read[1].policy[0].empty());
    EXPECT_TRUE(read[1].policy[1].empty());
    EXPECT_EQ(read[1].value, -0.5f);
    EXPECT_EQ(read[1].wdl[0], 1.0f);
    EXPECT_EQ(read[2].policy[1].size(), 2u);
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::stringstream text;
    text << stream.rdbuf();
    return text.str();
}

uint64_t json_count(const std::string& text, const std::string& key) {
    const size_t at = text.find("\"" + key + "\": ");
    return at == std::string::npos ? UINT64_MAX : std::stoull(text.substr(at + key.size() + 4));
}

std::vector<distill::Record> read_dir(const std::filesystem::path& directory, const std::string& prefix) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with(prefix) && name.ends_with(".dst")) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    std::vector<distill::Record> records;
    for (const auto& file : files) {
        for (distill::Record& record : distill::read_chunk(file)) {
            records.push_back(std::move(record));
        }
    }
    return records;
}

class SearchBalanceSelfPlay : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        engine = std::make_unique<Engine>(0, 1);
        const auto model = std::filesystem::path(__FILE__).parent_path() / "data" / "uniform.onnx";
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
        root = std::filesystem::temp_directory_path()
            / ("hivemind_search_balance_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    // Seat-clock games as the RL loop plays them (wave 2, hybrid gate), small.
    SelfPlayConfig config(const std::string& name, size_t games) const {
        SelfPlayConfig c;
        c.games = games;
        c.nodes = 32;
        c.maxMacroPlies = 120;
        c.batchSize = 1;
        c.seed = 1000;
        c.outputDirectory = root / name / "games";
        c.distillOutputDirectory = root / name;
        c.writeTrainingChunks = false;
        c.resignThreshold = 0.0f;
        c.fairyStockfishMateNodes = 0;
        c.rootScans = false;  // the time-budgeted root scans are not reproducible
        c.clockModel = true;
        Wave2GameOptions& w = c.wave2;
        w.flagLoss = true;
        w.instantTransfer = false;
        w.baseMix = {{20000, 1.0}};
        w.seatMoveTimes = true;
        w.hybrid.mode = HybridMode::GATE;
        return c;
    }

    std::unique_ptr<Engine> engine;
    std::filesystem::path root;
};

TEST_F(SearchBalanceSelfPlay, CappedRunCountsMatchItsRecords) {
    SelfPlayConfig c = config("capped", 3);
    c.deepFraction = 0.25;
    c.deepMultiplier = 2;
    c.fastNodes = 16;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, c), 0);
    const std::string out = testing::internal::GetCapturedStdout();
    const std::vector<distill::Record> records = read_dir(root / "capped", "search_");
    ASSERT_FALSE(records.empty());
    uint64_t masked = 0;
    for (const distill::Record& record : records) {
        masked += record.policy[0].empty() && record.policy[1].empty() ? 1 : 0;
    }
    const std::string counts = read_file(root / "capped" / "cap_counts.json");
    EXPECT_EQ(json_count(counts, "records"), records.size());
    EXPECT_EQ(json_count(counts, "masked"), masked);
    EXPECT_EQ(json_count(counts, "deep") + json_count(counts, "masked"), records.size());
    EXPECT_GT(json_count(counts, "deep"), 0u);
    EXPECT_GT(masked, 0u);
    // One seat row per record, game lines intact for the label parsers.
    const std::string rows = read_file(root / "capped" / "games" / "seat_clocks.tsv");
    EXPECT_EQ(static_cast<size_t>(std::count(rows.begin(), rows.end(), '\n')) - 1, records.size());
    EXPECT_NE(out.find("selfplay cap game 1 records "), std::string::npos) << out;
    EXPECT_NE(out.find("selfplay cap: "), std::string::npos);
    EXPECT_NE(out.find("selfplay complete 3 games"), std::string::npos);
}

TEST_F(SearchBalanceSelfPlay, StateDumpDoesNotChangeTheRunAndReanalyseReproducesIt) {
    SelfPlayConfig plain = config("plain", 2);
    SelfPlayConfig dumped = config("dumped", 2);
    dumped.recordStateOutput = root / "dumped" / "games" / "states.jsonl";
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, plain), 0);
    ASSERT_EQ(run_selfplay({engine.get()}, dumped), 0);
    testing::internal::GetCapturedStdout();
    const std::vector<distill::Record> a = read_dir(root / "plain", "search_");
    const std::vector<distill::Record> b = read_dir(root / "dumped", "search_");
    ASSERT_EQ(a.size(), b.size());
    // The dump is a side output only: same records, PGN and seat clocks.
    for (size_t index = 0; index < a.size(); ++index) {
        EXPECT_EQ(a[index].planeBits, b[index].planeBits);
        EXPECT_EQ(a[index].value, b[index].value);
    }
    EXPECT_EQ(read_file(root / "plain" / "games" / "games.pgn"),
              read_file(root / "dumped" / "games" / "games.pgn"));
    EXPECT_FALSE(std::filesystem::exists(root / "plain" / "cap_counts.json"));
    // The dump itself is a function of the seed (TimeState padding zeroed).
    SelfPlayConfig again = config("again", 2);
    again.recordStateOutput = root / "again.jsonl";
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, again), 0);
    testing::internal::GetCapturedStdout();
    EXPECT_EQ(read_file(dumped.recordStateOutput), read_file(again.recordStateOutput));

    ReanalyseConfig re;
    re.statesFile = dumped.recordStateOutput;
    re.outputDirectory = root / "reanalysed";
    re.replayNoise = true;
    re.nodes = 0;  // the recorded budgets
    re.wave2 = dumped.wave2;
    re.rootScans = false;
    for (const auto& entry : std::filesystem::directory_iterator(root / "dumped")) {
        if (entry.path().extension() == ".dst") {
            re.sourceFiles.push_back(entry.path());
        }
    }
    std::sort(re.sourceFiles.begin(), re.sourceFiles.end());
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_reanalyse({engine.get()}, re), 0);  // throws on any plane mismatch
    testing::internal::GetCapturedStdout();
    const std::string summary = read_file(root / "reanalysed" / "reanalyse.json");
    EXPECT_EQ(json_count(summary, "reanalysed"), b.size());
    EXPECT_EQ(json_count(summary, "planes_equal"), b.size());
    // Fidelity gate: >= 99% of records reproduce policy and value exactly.
    EXPECT_GE(json_count(summary, "policy_and_value_equal") * 100, b.size() * 99) << summary;
    const std::vector<distill::Record> r = read_dir(root / "reanalysed", "reanalyse_");
    ASSERT_EQ(r.size(), b.size());
    for (size_t index = 0; index < r.size(); ++index) {
        EXPECT_EQ(r[index].outcome, b[index].outcome);
        EXPECT_EQ(r[index].wdl, b[index].wdl);
        EXPECT_EQ(r[index].movesLeft, b[index].movesLeft);
        EXPECT_EQ(r[index].ply, b[index].ply);
    }
}

TEST_F(SearchBalanceSelfPlay, ReanalyseAtDeeperBudgetSelectsAShare) {
    SelfPlayConfig c = config("source", 2);
    c.recordStateOutput = root / "source.jsonl";
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_selfplay({engine.get()}, c), 0);
    ReanalyseConfig re;
    re.statesFile = c.recordStateOutput;
    re.outputDirectory = root / "deep";
    re.nodes = 128;
    re.select = 0.5;
    re.wave2 = c.wave2;
    re.rootScans = false;
    ASSERT_EQ(run_reanalyse({engine.get()}, re), 0);
    testing::internal::GetCapturedStdout();
    const std::string summary = read_file(root / "deep" / "reanalyse.json");
    const uint64_t all = json_count(summary, "records");
    const uint64_t chosen = json_count(summary, "reanalysed");
    EXPECT_GT(chosen, 0u);
    EXPECT_LT(chosen, all);
    // The deeper budget: about 4x the 32-node searches' visits.
    EXPECT_GT(json_count(summary, "visits"), chosen * 64);
    const std::string index = read_file(root / "deep" / "reanalysed.tsv");
    EXPECT_EQ(static_cast<uint64_t>(std::count(index.begin(), index.end(), '\n')) - 1, chosen);
    // Every reanalysed record carries a policy for the boards on turn.
    for (const distill::Record& record : read_dir(root / "deep", "reanalyse_")) {
        EXPECT_FALSE(record.policy[0].empty() && record.policy[1].empty());
    }
}

#endif

}  // namespace
