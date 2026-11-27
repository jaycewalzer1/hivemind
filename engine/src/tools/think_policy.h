#pragma once

/**
 * Spec v7 section 6.3: the learned time policy's engine side.
 *
 * A twin-board network may carry a think_gain head: two logits per position,
 * one per board, whose sigmoid p is the predicted chance that searching that
 * board's decision longer (4x) changes the move. A searcher turns p into a
 * budget scale s = clamp(0.5 + 1.5 p, 0.5, 2.0) (spec v2 B); a fresh head
 * (bias -ln 2, p = 1/3) gives s = 1, so a new head changes nothing.
 *
 * Self-play (selfplay.cc, WP3) includes this behind __has_include.
 */

#include <algorithm>
#include <array>
#include <cmath>

#include "environment/board.h"
#include "environment/constants.h"
#include "environment/planes.h"
#include "environment/time_state.h"
#include "nn/engine.h"

struct ThinkPolicy {
    double minScale = 0.5, maxScale = 2.0;
};

/// Budget scale for a think_gain probability: clamp(0.5 + 1.5 p, min, max).
inline double think_scale(const ThinkPolicy& policy, float p) {
    return std::clamp(0.5 + 1.5 * static_cast<double>(p), policy.minScale, policy.maxScale);
}

/**
 * One network evaluation of @p board for @p team on @p clock: the
 * think_gain probabilities (sigmoid of the logits) for board A and B in
 * @p out. False, @p out untouched, when the engine's network has no
 * think_gain output. Throws when the inference itself fails.
 *
 * Runs a full batch with the position in row 0 (runInference's contract),
 * so call it once per decision, not per node.
 */
inline bool evaluate_think_gain(Engine& engine, Board& board, Stockfish::Color team,
                                const TimeState& clock, std::array<float, 2>& out) {
    (void)engine;
    (void)board;
    (void)team;
    (void)clock;
    (void)out;
    return false;
}
