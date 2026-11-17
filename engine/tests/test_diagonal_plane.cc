// Spec v3 (docs/custom_engine_spec_v3.md, audit WP-F inputs 1): the diagonal
// clock plane, channel 43 of each board block in the 44-channel clock build,
// tanh((t_ourA - t_oppB) / s), identical in both blocks. The stock build keeps
// 37 channels and never writes it.

#include <gtest/gtest.h>

#include <cmath>
#include <mutex>
#include <vector>

#include "common/utils.h"
#include "environment/planes.h"

namespace {

void initialize_once() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        init_fairy_stockfish();
        init_policy_index();
    });
}

class DiagonalPlaneTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { initialize_once(); }
};

constexpr int DIAGONAL_CHANNEL = 43;

float at(const std::vector<float>& planes, int channel, int square = 0) {
    return planes[static_cast<size_t>(channel) * 64 + static_cast<size_t>(square)];
}

// Expected value from the spec, in seconds, negative clocks counted as 0.
float expected(double oursMs, double diagonalMs) {
    const double ours = std::max(0.0, oursMs) / 1000.0;
    const double diagonal = std::max(0.0, diagonalMs) / 1000.0;
    return static_cast<float>(std::tanh((ours - diagonal) / DIAGONAL_PLANE_SCALE_S));
}

void expect_diagonal(const std::vector<float>& planes, float value) {
    for (int block = 0; block < 2; ++block) {
        const int channel = block * NB_INPUT_CHANNELS_PER_BOARD + DIAGONAL_CHANNEL;
        for (int square = 0; square < 64; ++square) {
            ASSERT_FLOAT_EQ(at(planes, channel, square), value)
                << "block " << block << " square " << square;
        }
    }
}

}  // namespace

TEST_F(DiagonalPlaneTest, ChannelCountsPerBuild) {
    if constexpr (HAS_CLOCK_PLANES) {
        EXPECT_EQ(NB_INPUT_CHANNELS_PER_BOARD, 44);
        EXPECT_EQ(NB_INPUT_CHANNELS, 88);
        EXPECT_TRUE(HAS_ELAPSED_PLANES);
        EXPECT_TRUE(HAS_DIAGONAL_PLANE);
    } else {
        EXPECT_EQ(NB_INPUT_CHANNELS_PER_BOARD, 37);
        EXPECT_FALSE(HAS_DIAGONAL_PLANE);
    }
}

TEST_F(DiagonalPlaneTest, OurBoardAClockAgainstPartnersOpponent) {
    if constexpr (!HAS_DIAGONAL_PLANE) {
        GTEST_SKIP() << "the diagonal plane exists only in the hivemind-clock build";
    }
    Board board;
    // A: white 40 s, black 100 s; B: white 25 s, black 70 s.
    board.set_clocks(40000, 100000, 25000, 70000);
    std::vector<float> planes(NB_INPUT_VALUES());

    // White team: we are white on A, our diagonal is white on B (partner's opponent).
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_diagonal(planes, expected(40000, 25000));
    EXPECT_GT(at(planes, DIAGONAL_CHANNEL), 0.0f);

    // Black team: black on A against black on B.
    board_to_planes(board, planes.data(), Stockfish::BLACK, true);
    expect_diagonal(planes, expected(100000, 70000));

    // The sign flips with the lead.
    board.set_clocks(25000, 1000, 40000, 1000);
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_diagonal(planes, expected(25000, 40000));
    EXPECT_LT(at(planes, DIAGONAL_CHANNEL), 0.0f);

    // A 15 s lead at the default scale is tanh(1).
    if (DIAGONAL_PLANE_SCALE_S == 15.0) {
        board.set_clocks(30000, 1000, 15000, 1000);
        board_to_planes(board, planes.data(), Stockfish::WHITE, false);
        EXPECT_NEAR(at(planes, DIAGONAL_CHANNEL), std::tanh(1.0), 1e-6);
    }
}

TEST_F(DiagonalPlaneTest, NegativeClocksClampAndNoClocksGiveZero) {
    if constexpr (!HAS_DIAGONAL_PLANE) {
        GTEST_SKIP() << "the diagonal plane exists only in the hivemind-clock build";
    }
    Board board;
    board.set_clocks(-3000, 5000, 2000, 5000);
    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_diagonal(planes, expected(0, 2000));

    board.set_clocks(5000, 5000, 5000, 5000);
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_diagonal(planes, 0.0f);

    board.clear_clocks();
    board_to_planes(board, planes.data(), Stockfish::WHITE, true);
    expect_diagonal(planes, 0.0f);
}

TEST_F(DiagonalPlaneTest, OtherChannelsUnchangedByTheDiagonal) {
    if constexpr (!HAS_DIAGONAL_PLANE) {
        GTEST_SKIP() << "the diagonal plane exists only in the hivemind-clock build";
    }
    // Two positions whose clocks differ only in the diagonal seat: every
    // channel but 37..40 (that seat's own clock planes) and 43 agrees.
    Board a;
    Board b;
    a.set_clocks(60000, 60000, 60000, 60000);
    b.set_clocks(60000, 60000, 45000, 60000);
    std::vector<float> pa(NB_INPUT_VALUES()), pb(NB_INPUT_VALUES());
    board_to_planes(a, pa.data(), Stockfish::WHITE, false);
    board_to_planes(b, pb.data(), Stockfish::WHITE, false);
    for (int block = 0; block < 2; ++block) {
        for (int channel = 0; channel < NB_INPUT_CHANNELS_PER_BOARD; ++channel) {
            const int c = block * NB_INPUT_CHANNELS_PER_BOARD + channel;
            const bool clockOfDiagonalSeat = block == 1 && channel >= 39 && channel <= 40;
            if (channel == DIAGONAL_CHANNEL || clockOfDiagonalSeat) {
                continue;
            }
            EXPECT_FLOAT_EQ(at(pa, c), at(pb, c)) << "block " << block << " channel " << channel;
        }
    }
    expect_diagonal(pa, 0.0f);
    expect_diagonal(pb, expected(60000, 45000));
}

TEST_F(DiagonalPlaneTest, FollowsTreeClocks) {
    if constexpr (!HAS_DIAGONAL_PLANE) {
        GTEST_SKIP() << "the diagonal plane exists only in the hivemind-clock build";
    }
    Board board;
    board.set_clocks(20000, 20000, 20000, 20000);
    board.treeMoveMs = 1000;
    // White on A moves (charged 1 s); B: white is on turn there and is charged too.
    board.make_moves(Stockfish::make_move(Stockfish::SQ_E2, Stockfish::SQ_E4), Stockfish::MOVE_NONE);
    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    expect_diagonal(planes, expected(board.clock_ms(BOARD_A, Stockfish::WHITE),
                                     board.clock_ms(BOARD_B, Stockfish::WHITE)));
}
