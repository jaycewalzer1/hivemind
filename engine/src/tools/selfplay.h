#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "search/search_params.h"
#include "tools/clock_sampling.h"
#include "tools/seat_clock_game.h"

class Engine;

struct SelfPlayConfig {
    size_t games = 1;
    size_t nodes = 800;
    size_t maxMacroPlies = 400;
    size_t chunkSamples = 16384;
    double rawPolicyMeanMacroPlies = 8.0;
    size_t rawPolicyMaxMacroPlies = 30;
    double rawPolicyHighTemperatureProbability = 0.05;
    double mctsTemperature = 1.0;
    double mctsTemperatureDecay = 0.93;
    size_t mctsTemperaturePlies = 20;
    float resignThreshold = -0.90f;
    size_t resignConsecutivePlies = 3;
    double resignDisableFraction = 0.10;
    double qValueRatio = 0.15;
    double nodeRandomFactor = 0.05;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
    uint64_t fairyStockfishMateNodes = SearchParams::MATE_PROBE_ROOT_NODE_BUDGET;
    float initialClockSeconds = 180.0f;
    int batchSize = SearchParams::BATCH_SIZE;
    uint64_t seed = 0;
    std::filesystem::path outputDirectory = "selfplay_games";
    // When set, every searched position is also written as an HDST
    // distillation record (tools/distill_data.h): the visit distribution over
    // all legal moves of each board on turn, root Q as the value, and for
    // finished games the result as WDL plus remaining team decisions.
    // Games played at once, each on its own engine (see main: one engine per
    // game on each device). 1 keeps a single game spread over all engines.
    size_t parallelGames = 1;
    // Write the HVM6 training chunks (off when only distillation data is wanted).
    bool writeTrainingChunks = true;
    // Root mate scan and selected-move certification before each search.
    bool rootScans = true;
    // Keep the subtree below each played move for the next search, which tops
    // the reused root up to the node budget, adding at least this fraction of
    // it as new nodes. Off, every search starts from an empty tree.
    bool treeReuse = false;
    double treeReuseMinNewFraction = 0.25;
    // Every this many games, save the finished games so an interrupted run can
    // pick up from there with `resume` (0: no checkpoints). Needs
    // writeTrainingChunks off: only the distillation output and the PGN are
    // checkpointed.
    size_t checkpointGames = 0;
    bool resume = false;
    // Clock model: each game draws a starting lead and per-team move times
    // from `clock`, and every search and adjudication follows that lead.
    // Off, one team may sit for the whole game (the TimeAdvantage flag).
    bool clockModel = false;
    ClockSampling clock;
    // Wave 2 (docs/wave2_training.md): the tree options of every search and,
    // with a base mix, games on four real seat clocks. Defaults: stock v3.
    Wave2GameOptions wave2;
    std::filesystem::path distillOutputDirectory;
    size_t distillChunkPositions = 250'000;

    // Playout cap randomisation (docs/search_balance.md section 6). All off by
    // default, and off a run is byte-identical to a build without them: the
    // coin never draws from a game's random stream.
    // Share of searched decisions that are deep (budget deepMultiplier x the
    // jittered base). The deep search also picks the move played.
    double deepFraction = 0.0;
    size_t deepMultiplier = 1;
    // Budget of the other decisions; 0 is the same as `nodes`.
    size_t fastNodes = 0;
    // Write non-deep records with empty policy on both boards (value, WDL
    // and moves-left only). -1: on exactly when fastNodes < nodes.
    int fastPolicyMask = -1;
    // Root Dirichlet noise on non-deep searches whose budget is below
    // `nodes` (full-budget non-deep searches keep it), and on deep searches.
    bool fastDirichlet = false;
    bool deepDirichlet = true;
    uint64_t capSeedSalt = 0x5EEDCA9ULL;
    // One JSONL line per game and per record (see write_state_lines in
    // selfplay.cc) from which `hivemind reanalyse` rebuilds each search's
    // root exactly. Empty: off.
    std::filesystem::path recordStateOutput;

    size_t fast_nodes() const { return fastNodes == 0 ? nodes : fastNodes; }
    bool cap_active() const {
        return deepFraction > 0.0 || fast_nodes() != nodes || fastPolicyMask == 1;
    }
    bool fast_policy_masked() const {
        return fastPolicyMask < 0 ? fast_nodes() < nodes : fastPolicyMask == 1;
    }
};

/// What the playout cap decided for one searched decision.
struct CapDecision {
    bool deep = false;
    bool masked = false;      // record written without a policy target
    bool feedsResign = true;  // may update the resign counter
    size_t targetNodes = 0;
    float dirichletEpsilon = 0.0f;
};

/**
 * @brief The playout cap's choice for decision `decision` of game
 * `gameIndex`, given the jittered base budget the decision drew. With the cap
 * off it returns the base and the configured noise unchanged. The coin is a
 * pure function of (seed, salt, game, decision).
 */
CapDecision cap_decision(const SelfPlayConfig& config, uint64_t runId, size_t gameIndex,
                         size_t decision, size_t baseNodes);

// Game-indexed streams stay independent of worker assignment.
inline uint64_t selfplay_seed(uint64_t seed, uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (value ^ (value >> 31));
}

int run_selfplay(const std::vector<Engine*>& engines, const SelfPlayConfig& config);
class Board;
struct RootEdgeStats;
namespace distill { struct PolicyEntry; }
/// Self-play's HDST policy target for one board (every legal move and the
/// pass, zero for unvisited ones; empty when the board is not on turn), for
/// `hivemind reanalyse`.
std::vector<distill::PolicyEntry> selfplay_distill_policy(
    Board& board, int boardNumber, bool onTurn, const std::vector<RootEdgeStats>& edges);
