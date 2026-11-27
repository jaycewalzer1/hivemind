#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/time_state.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "tools/think_policy.h"

// Spec v7 6.3: the think_gain head's engine side (think_policy.h, the
// think_gain output binding, the UCI info line).

TEST(ThinkPolicy, ScaleIsClampedAndOneAtAFreshHead) {
    const ThinkPolicy policy;
    EXPECT_DOUBLE_EQ(think_scale(policy, 0.0f), 0.5);
    EXPECT_DOUBLE_EQ(think_scale(policy, 1.0f), 2.0);
    EXPECT_DOUBLE_EQ(think_scale(policy, -1.0f), 0.5);
    EXPECT_DOUBLE_EQ(think_scale(policy, 4.0f), 2.0);
    EXPECT_NEAR(think_scale(policy, 1.0f / 3.0f), 1.0, 1e-6);
    EXPECT_NEAR(think_scale(policy, 0.5f), 1.25, 1e-9);
    const ThinkPolicy narrow{0.8, 1.2};
    EXPECT_DOUBLE_EQ(think_scale(narrow, 0.0f), 0.8);
    EXPECT_DOUBLE_EQ(think_scale(narrow, 1.0f), 1.2);
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
namespace {

std::filesystem::path data_file(const char* name) {
    return std::filesystem::path(__FILE__).parent_path() / "data" / name;
}

// think_probe.onnx: uniform.onnx plus constant think_gain logits
// [-ln 2, ln 3], probabilities 1/3 and 3/4 (make_think_probe.py).
constexpr float PROBE_A = 1.0f / 3.0f;
constexpr float PROBE_B = 0.75f;

std::vector<std::string> think_gain_lines(const std::string& output) {
    std::vector<std::string> lines;
    std::istringstream in(output);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("info string think_gain", 0) == 0) {
            lines.push_back(line);
        }
    }
    return lines;
}

class ThinkGainEngine : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

TEST_F(ThinkGainEngine, TheOrtEngineReadsAHeadedNetworksThinkGain) {
    Engine engine(0, 1);
    ASSERT_TRUE(engine.loadNetwork(data_file("think_probe.onnx").string(), ""));
    ASSERT_TRUE(engine.hasThinkGain());
    EXPECT_FALSE(engine.hasArrivals());

    Board board;
    std::array<float, 2> gain{-1.0f, -1.0f};
    ASSERT_TRUE(evaluate_think_gain(engine, board, Stockfish::WHITE,
                                    TimeState::legacy(Stockfish::WHITE, false), gain));
    // FP16 on the way out.
    EXPECT_NEAR(gain[0], PROBE_A, 1e-3);
    EXPECT_NEAR(gain[1], PROBE_B, 1e-3);

    // runInference hands out the logits when asked, and is unchanged when not.
    std::vector<float> obs(NB_INPUT_VALUES()), value(1), piA(NB_POLICY_VALUES()),
        piB(NB_POLICY_VALUES()), wdl(3), movesLeft(1), logits(2, 9.0f);
    ASSERT_TRUE(engine.runInference(obs.data(), value.data(), piA.data(), piB.data(), wdl.data(),
                                    movesLeft.data(), 0, nullptr, nullptr, logits.data()));
    EXPECT_NEAR(logits[0], -std::log(2.0f), 1e-3);
    EXPECT_NEAR(logits[1], std::log(3.0f), 1e-3);
    ASSERT_TRUE(engine.runInference(obs.data(), value.data(), piA.data(), piB.data(), wdl.data(),
                                    movesLeft.data(), 0));
}

TEST_F(ThinkGainEngine, ANetworkWithoutTheHeadReportsNone) {
    const std::filesystem::path it13 = std::filesystem::path(__FILE__).parent_path().parent_path()
        / "models" / "twin-s-clock-it13.onnx";
    const bool real = std::filesystem::exists(it13);
    Engine engine(0, 1);
    ASSERT_TRUE(engine.loadNetwork(real ? it13.string() : data_file("uniform.onnx").string(), ""));
    EXPECT_FALSE(engine.hasThinkGain());
    Board board;
    std::array<float, 2> gain{-1.0f, -2.0f};
    EXPECT_FALSE(evaluate_think_gain(engine, board, Stockfish::WHITE,
                                     TimeState::legacy(Stockfish::WHITE, false), gain));
    EXPECT_EQ(gain[0], -1.0f);
    EXPECT_EQ(gain[1], -2.0f);
    // A logits buffer is left alone.
    std::vector<float> obs(NB_INPUT_VALUES()), value(1), piA(NB_POLICY_VALUES()),
        piB(NB_POLICY_VALUES()), wdl(3), movesLeft(1), logits(2, 9.0f);
    ASSERT_TRUE(engine.runInference(obs.data(), value.data(), piA.data(), piB.data(), wdl.data(),
                                    movesLeft.data(), 0, nullptr, nullptr, logits.data()));
    EXPECT_EQ(logits[0], 9.0f);
    if (!real) {
        GTEST_LOG_(INFO) << "it13 not found; checked on uniform.onnx only";
    }
}

std::string verbose_search(const char* model) {
    Engine engine(0, 1);
    EXPECT_TRUE(engine.loadNetwork(data_file(model).string(), ""));
    SearchOptions options = SearchOptions::uci(64, 1);
    options.enablePonder = false;
    Board board;
    Agent agent(1);
    testing::internal::CaptureStdout();
    agent.run_search(board, {&engine}, Stockfish::WHITE, false, options);
    return testing::internal::GetCapturedStdout();
}

TEST_F(ThinkGainEngine, AVerboseSearchPrintsThinkGainBeforeBestmove) {
    const std::string out = verbose_search("think_probe.onnx");
    const auto lines = think_gain_lines(out);
    ASSERT_EQ(lines.size(), 1u) << out;
    EXPECT_EQ(lines[0], "info string think_gain 0.3333 0.7500");
    const auto at = out.find("info string think_gain");
    const auto best = out.find("bestmove ");
    ASSERT_NE(best, std::string::npos);
    EXPECT_LT(at, best);
}

TEST_F(ThinkGainEngine, AVerboseSearchWithoutTheHeadPrintsNothing) {
    const std::string out = verbose_search("uniform.onnx");
    EXPECT_TRUE(think_gain_lines(out).empty()) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos);
}

}  // namespace
#endif
