#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "tools/seat_clock_game.h"

class Engine;

/**
 * @brief `hivemind reanalyse` (docs/search_balance.md section 6): re-searches
 * the roots a self-play run dumped with `selfplay --record-state-output` and
 * writes them as HDST records: the new root visits as the policy (the same
 * edge filters as self-play), the new root Q as the value, and the outcome,
 * WDL and moves-left the original record got at game end.
 *
 * Each root is rebuilt exactly: the board by replaying the game's moves from
 * the start under the run's wave 2 board rules, the clock and the hybrid's
 * durative clock from their dumped bytes (same build only: the header's
 * time_state_bytes is checked). Every search starts from an empty tree, so a
 * run made with --team-tree-reuse or --plan-carry cannot be reproduced bit
 * for bit (the fidelity test needs them off).
 *
 * With `sourceFiles` (the run's HDST chunks, in order) the recomputed input
 * planes must equal the source record's on every reanalysed record, and the
 * run reports how many reanalysed records reproduce the source policy and
 * value exactly (the fidelity gate: --nodes equal to the run's,
 * --replay-noise true).
 *
 * Outputs in `outputDirectory`: reanalyse_<seed>_NNNNN.dst chunks,
 * reanalysed.tsv (game, record, source index of every record written, in
 * chunk order) for the trainer's dedupe, and reanalyse.json (counts).
 */
struct ReanalyseConfig {
    std::filesystem::path statesFile;
    std::vector<std::filesystem::path> sourceFiles;
    std::filesystem::path outputDirectory;
    // 0: each record's recorded target. Otherwise the record's jittered base
    // budget scaled by nodes / the run's --nodes.
    size_t nodes = 0;
    // Replay each record's recorded root noise (seed and epsilon). Off, the
    // root has no noise unless `dirichlet` is on (then the configured
    // epsilon with the recorded seed).
    bool replayNoise = false;
    bool dirichlet = false;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
    // Share of records reanalysed, chosen by a hash of (seed, game, record).
    double select = 1.0;
    uint64_t selectSalt = 0x5E1EC7ULL;
    // Leave out the records that already had a deep search.
    bool skipDeep = false;
    uint64_t fairyStockfishMateNodes = 0;
    bool rootScans = true;
    Wave2GameOptions wave2;
    size_t chunkPositions = 250'000;
    size_t gamesPerBatch = 64;
};

int run_reanalyse(const std::vector<Engine*>& engines, const ReanalyseConfig& config);

/// Whether record (game, record) of a run with seed `seed` is in the
/// `select` share (exposed for tests and the Python side's bookkeeping).
bool reanalyse_selected(uint64_t seed, uint64_t salt, size_t game, size_t record, double select);
