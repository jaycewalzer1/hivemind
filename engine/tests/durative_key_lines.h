#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "environment/time_state.h"

// Spec v5 test 0.3: random durative lines, folded to one key per line. It
// reads only the TimeState API that 261081b already had, so the same file
// built against that commit's headers records
// tests/data/durative_keys_261081b.txt (see the comment there), and the
// test compares the current build's keys, with JointPlans off, against it.

/**
 * @brief The exact_key() of every node along @p lines random durative lines,
 * folded into one 64-bit value per line. Each line draws its rules (FlagLoss,
 * InstantTransfer, MateRaceByLanding, PassCapK, SitGate, wide inputs), clocks,
 * paces, elapsed times, sitting seats and sides, then plays up to 60 actions:
 * each acting board moves or passes at random, a move sends a random piece
 * type (or none) to the partner.
 */
inline std::vector<uint64_t> durative_key_lines(int lines, uint32_t seed = 20261120) {
    std::mt19937 rng(seed);
    const auto draw = [&rng](int32_t low, int32_t high) {
        return static_cast<int32_t>(low + static_cast<int64_t>(rng() % static_cast<uint32_t>(high - low + 1)));
    };
    std::vector<uint64_t> folded;
    folded.reserve(static_cast<size_t>(lines));
    for (int line = 0; line < lines; ++line) {
        const Stockfish::Color rootTeam = rng() % 2 ? Stockfish::WHITE : Stockfish::BLACK;
        DurativeConfig config;
        config.flagLoss = rng() % 2 == 0;
        config.instantTransfer = rng() % 2 == 0;
        config.mateRaceByLanding = rng() % 3 == 0;
        config.passCapK = draw(1, 12);
        config.rootMoveTimeMs = draw(1, 3000);
        for (int seat = 0; seat < 4; ++seat) {
            config.seatClocksMs[seat] = draw(200, 120000);
            config.seatPaceMs[seat] = draw(1, 4000);
            config.seatElapsedMs[seat] = draw(0, 3000);
            config.seatSitting[seat] = rng() % 4 == 0;
        }
        config.wideInputs = rng() % 3 == 0;
        if (config.wideInputs) {
            config.boardElapsedMs = {draw(-1, 4000), draw(-1, 4000)};
            for (int slot = 0; slot < 3; ++slot) {
                config.recentLandings[slot] = SeatLanding{
                    static_cast<int8_t>(draw(0, 3)), static_cast<int8_t>(draw(1, 5)),
                    draw(0, 6000)};
            }
        }
        const Stockfish::Color sideA = rng() % 2 ? Stockfish::WHITE : Stockfish::BLACK;
        const Stockfish::Color sideB = rng() % 2 ? Stockfish::WHITE : Stockfish::BLACK;
        const TimeState base = TimeState::clocked(rootTeam, draw(-5000, 5000), draw(1, 2000),
                                                  draw(1, 2000))
                                   .with_sit_gate(rng() % 2 == 0);
        TimeState time = TimeState::durative(base, rootTeam, sideA, sideB, config);
        uint64_t hash = 0x6b657973ULL ^ static_cast<uint64_t>(line);
        const auto fold = [&hash](uint64_t key) {
            hash = (hash ^ key) * 0x100000001b3ULL;
            hash ^= hash >> 29;
        };
        fold(time.exact_key());
        Stockfish::Color toPlay = rootTeam;
        for (int ply = 0; ply < 60 && time.outcome() == TimeState::Outcome::NONE; ++ply) {
            const std::array<bool, 2> acts = time.acting_boards(toPlay);
            bool moved[2] = {false, false};
            Stockfish::PieceType hand[2] = {Stockfish::NO_PIECE_TYPE, Stockfish::NO_PIECE_TYPE};
            for (int board = 0; board < 2; ++board) {
                moved[board] = acts[board] && rng() % 4 != 0;
                if (moved[board] && rng() % 3 == 0) {
                    hand[board] = static_cast<Stockfish::PieceType>(draw(1, 5));
                }
            }
            time.advance_action(toPlay, moved[0], moved[1], hand[0], hand[1]);
            fold(time.exact_key());
            fold(static_cast<uint64_t>(time.can_sit(Stockfish::WHITE))
                 | (static_cast<uint64_t>(time.can_sit(Stockfish::BLACK)) << 1));
            toPlay = ~toPlay;
        }
        folded.push_back(hash);
    }
    return folded;
}
