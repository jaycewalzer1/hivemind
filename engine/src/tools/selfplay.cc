#include "tools/selfplay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <atomic>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "search/agent.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "nn/engine.h"
#include "common/globals.h"
#include "environment/planes.h"
#include "common/utils.h"
#include "tools/distill_data.h"
#include "tools/selfplay_policy.h"
#include "tools/seat_clock_game.h"

namespace {

// HVM6: the clock planes close each board's block, stored as (v + 1) / 2
// since a chunk keeps planes as unsigned bytes.
constexpr std::array<char, 4> CHUNK_MAGIC = {'H', 'V', 'M', '6'};
constexpr uint32_t CHUNK_VERSION = 6;
constexpr uint16_t PASS_POLICY_INDEX = static_cast<uint16_t>(NB_POLICY_VALUES());

struct SparsePolicyEntry {
    uint16_t index;
    float probability;
};

struct SparseJointPolicyEntry {
    uint16_t indexA;
    uint16_t indexB;
    float probability;
};

struct TrainingSample {
    uint64_t gameId = 0;
    uint32_t nodes = 0;
    uint16_t macroPly = 0;
    uint16_t movesLeft = 0;
    uint8_t team = 0;
    uint8_t hasTimeAdvantage = 0;
    int8_t outcome = 0;
    uint8_t wdl = 1;
    float rootQ = 0.0f;
    std::array<uint8_t, NB_INPUT_VALUES()> planes{};
    std::vector<SparsePolicyEntry> policyA;
    std::vector<SparsePolicyEntry> policyB;
    std::vector<SparseJointPolicyEntry> jointPolicy;
};

struct PgnMove {
    std::string token;
    std::string san;
    float remainingSeconds = 0.0f;
};

template <typename T>
void write_scalar(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to write self-play chunk");
    }
}

class ChunkWriter {
public:
    ChunkWriter(std::filesystem::path directory, size_t samplesPerChunk, uint64_t runId)
        : directory_(std::move(directory)),
          samplesPerChunk_(std::max<size_t>(1, samplesPerChunk)),
          runId_(runId) {
        std::filesystem::create_directories(directory_);
    }

    void append(std::vector<TrainingSample> samples) {
        for (TrainingSample& sample : samples) {
            pending_.push_back(std::move(sample));
            if (pending_.size() >= samplesPerChunk_) {
                flush(samplesPerChunk_);
            }
        }
    }

    void finish() {
        if (!pending_.empty()) {
            flush(pending_.size());
        }
    }

private:
    void write_policy(std::ostream& stream, const std::vector<SparsePolicyEntry>& policy) {
        if (policy.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Sparse policy is too large for HVM6");
        }
        write_scalar(stream, static_cast<uint16_t>(policy.size()));
        for (const SparsePolicyEntry& entry : policy) {
            write_scalar(stream, entry.index);
            write_scalar(stream, entry.probability);
        }
    }

    void write_joint_policy(
        std::ostream& stream,
        const std::vector<SparseJointPolicyEntry>& policy) {
        if (policy.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Sparse joint policy is too large for HVM6");
        }
        write_scalar(stream, static_cast<uint16_t>(policy.size()));
        for (const SparseJointPolicyEntry& entry : policy) {
            write_scalar(stream, entry.indexA);
            write_scalar(stream, entry.indexB);
            write_scalar(stream, entry.probability);
        }
    }

    void flush(size_t count) {
        std::ostringstream filename;
        filename << "chunk_" << runId_ << '_' << std::setw(6) << std::setfill('0')
                 << chunkIndex_++ << ".hvm";
        const std::filesystem::path finalPath = directory_ / filename.str();
        const std::filesystem::path temporaryPath = finalPath.string() + ".tmp";

        std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Unable to create " + temporaryPath.string());
        }
        stream.write(CHUNK_MAGIC.data(), CHUNK_MAGIC.size());
        write_scalar(stream, CHUNK_VERSION);
        write_scalar(stream, static_cast<uint16_t>(NB_INPUT_CHANNELS));
        write_scalar(stream, static_cast<uint16_t>(NB_POLICY_VALUES()));
        write_scalar(stream, static_cast<uint64_t>(count));

        for (size_t index = 0; index < count; ++index) {
            const TrainingSample& sample = pending_[index];
            write_scalar(stream, sample.gameId);
            write_scalar(stream, sample.nodes);
            write_scalar(stream, sample.macroPly);
            write_scalar(stream, sample.movesLeft);
            write_scalar(stream, sample.team);
            write_scalar(stream, sample.hasTimeAdvantage);
            write_scalar(stream, sample.outcome);
            write_scalar(stream, sample.wdl);
            write_scalar(stream, sample.rootQ);
            stream.write(
                reinterpret_cast<const char*>(sample.planes.data()),
                static_cast<std::streamsize>(sample.planes.size()));
            write_policy(stream, sample.policyA);
            write_policy(stream, sample.policyB);
            write_joint_policy(stream, sample.jointPolicy);
        }
        stream.close();
        if (!stream) {
            throw std::runtime_error("Failed to finalize " + temporaryPath.string());
        }

        std::error_code error;
        std::filesystem::rename(temporaryPath, finalPath, error);
        if (error) {
            std::filesystem::remove(temporaryPath);
            throw std::runtime_error("Unable to publish " + finalPath.string() + ": " + error.message());
        }
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(count));
    }

    std::filesystem::path directory_;
    size_t samplesPerChunk_;
    uint64_t runId_;
    size_t chunkIndex_ = 0;
    std::vector<TrainingSample> pending_;
};


void apply_temperature(std::vector<float>& probabilities, double temperature) {
    if (probabilities.empty() || temperature <= 0.0) {
        throw std::invalid_argument("Policy temperature must be positive");
    }
    const double exponent = 1.0 / temperature;
    double total = 0.0;
    for (float& probability : probabilities) {
        probability = static_cast<float>(std::pow(std::max(0.0f, probability), exponent));
        total += probability;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        const float uniform = 1.0f / static_cast<float>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
        return;
    }
    for (float& probability : probabilities) {
        probability = static_cast<float>(probability / total);
    }
}

size_t sample_initialization_length(const SelfPlayConfig& config,
                                    std::mt19937_64& randomEngine) {
    if (config.rawPolicyMeanMacroPlies <= 0.0 || config.rawPolicyMaxMacroPlies == 0) {
        return 0;
    }
    std::exponential_distribution<double> distribution(
        1.0 / config.rawPolicyMeanMacroPlies);
    size_t length = static_cast<size_t>(std::llround(distribution(randomEngine)));
    if (length > config.rawPolicyMaxMacroPlies) {
        std::uniform_int_distribution<size_t> clipped(0, config.rawPolicyMaxMacroPlies);
        length = clipped(randomEngine);
    }
    return length;
}

double sample_raw_policy_temperature(const SelfPlayConfig& config,
                                     std::mt19937_64& randomEngine) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(randomEngine) >= config.rawPolicyHighTemperatureProbability) {
        return 1.0;
    }
    const double choice = unit(randomEngine);
    if (choice < 0.75) {
        return 2.0;
    }
    if (choice < 0.95) {
        return 5.0;
    }
    return 10.0;
}

size_t randomized_node_budget(const SelfPlayConfig& config,
                              std::mt19937_64& randomEngine) {
    std::uniform_real_distribution<double> jitter(
        -config.nodeRandomFactor, config.nodeRandomFactor);
    return std::max<size_t>(1, static_cast<size_t>(std::llround(
        static_cast<double>(config.nodes) * (1.0 + jitter(randomEngine)))));
}

double mcts_temperature(const SelfPlayConfig& config, size_t macroPly) {
    if (config.mctsTemperaturePlies > 0 && macroPly >= config.mctsTemperaturePlies) {
        return 0.0;
    }
    return config.mctsTemperature
        * std::pow(config.mctsTemperatureDecay, static_cast<double>(macroPly / 2));
}

class RawPolicyEvaluator {
public:
    explicit RawPolicyEvaluator(size_t batchSize)
        : batchSize_(batchSize),
          observations_(batchSize * NB_INPUT_VALUES()),
          wideExtras_(batchSize * NB_WIDE_EXTRA_VALUES),
          values_(batchSize),
          policyA_(batchSize * NB_POLICY_VALUES()),
          policyB_(batchSize * NB_POLICY_VALUES()),
          wdl_(batchSize * 3),
          movesLeft_(batchSize) {}

    void evaluate(Engine& engine,
                  Board& board,
                  Stockfish::Color team,
                  const TimeState& time) {
        std::array<float, NB_INPUT_VALUES()> planes{};
        board_to_planes(board, planes.data(), team, time);
        for (size_t batch = 0; batch < batchSize_; ++batch) {
            std::copy(planes.begin(), planes.end(),
                      observations_.begin() + static_cast<std::ptrdiff_t>(batch * planes.size()));
        }
        if (engine.wideInputs()) {
            std::array<float, NB_WIDE_EXTRA_VALUES> extras{};
            wide_extra_planes(board, team, time, extras.data());
            for (size_t batch = 0; batch < batchSize_; ++batch) {
                std::copy(extras.begin(), extras.end(),
                          wideExtras_.begin() + static_cast<std::ptrdiff_t>(batch * extras.size()));
            }
        }
        if (!engine.runInference(
                observations_.data(), values_.data(), policyA_.data(), policyB_.data(),
                wdl_.data(), movesLeft_.data(), 0, wideExtras_.data())) {
            throw std::runtime_error("Raw-policy inference failed");
        }
    }

    float* policy(int boardNumber) {
        return boardNumber == BOARD_A ? policyA_.data() : policyB_.data();
    }

private:
    size_t batchSize_;
    std::vector<float> observations_;
    std::vector<float> wideExtras_;
    std::vector<float> values_;
    std::vector<float> policyA_;
    std::vector<float> policyB_;
    std::vector<float> wdl_;
    std::vector<float> movesLeft_;
};

void prepare_raw_policy(
    Board& board,
    int boardNumber,
    bool boardOnTurn,
    float* policyOutput,
    double temperature,
    std::vector<Stockfish::Move>& actions,
    std::vector<float>& probabilities) {
    if (boardOnTurn) {
        actions = board.legal_moves(boardNumber);
        std::erase_if(actions, [&board, boardNumber](Stockfish::Move move) {
            return !is_policy_move_representable(board, boardNumber, move);
        });
    }
    if (actions.empty()) {
        actions.push_back(Stockfish::MOVE_NONE);
        probabilities.push_back(1.0f);
        return;
    }
    actions.push_back(Stockfish::MOVE_NONE);
    probabilities = get_normalized_probability(
        policyOutput, actions, boardNumber, board);
    apply_temperature(probabilities, temperature);
}

JointActionCandidate sample_raw_policy_action(
    Engine& engine,
    RawPolicyEvaluator& evaluator,
    Board& board,
    Stockfish::Color team,
    const TimeState& time,
    double temperature,
    std::mt19937_64& randomEngine) {
    evaluator.evaluate(engine, board, team, time);
    const bool hasTimeAdvantage = time.can_sit(team);
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == team;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~team;
    std::vector<Stockfish::Move> actionsA;
    std::vector<Stockfish::Move> actionsB;
    std::vector<float> probabilitiesA;
    std::vector<float> probabilitiesB;
    prepare_raw_policy(
        board, BOARD_A, boardAOnTurn, evaluator.policy(BOARD_A), temperature,
        actionsA, probabilitiesA);
    prepare_raw_policy(
        board, BOARD_B, boardBOnTurn, evaluator.policy(BOARD_B), temperature,
        actionsB, probabilitiesB);

    auto capture_flags = [&board](const std::vector<Stockfish::Move>& actions, int boardNumber) {
        std::vector<uint8_t> captures;
        captures.reserve(actions.size());
        for (Stockfish::Move move : actions) {
            captures.push_back(move != Stockfish::MOVE_NONE
                && board.is_capture(boardNumber, move) ? 1 : 0);
        }
        return captures;
    };
    const std::vector<uint8_t> capturesA = capture_flags(actionsA, BOARD_A);
    const std::vector<uint8_t> capturesB = capture_flags(actionsB, BOARD_B);

    JointActionRules rules;
    rules.boardAOnTurn = boardAOnTurn;
    rules.boardBOnTurn = boardBOnTurn;
    rules.teamHasTimeAdvantage = hasTimeAdvantage;
    rules.boardACanMove = boardAOnTurn && actionsA.size() > 1;
    rules.boardBCanMove = boardBOnTurn && actionsB.size() > 1;

    auto make_candidate = [&](size_t iA, size_t iB) {
        return JointActionCandidate(
            actionsA[iA], probabilitiesA[iA], iA,
            actionsB[iB], probabilitiesB[iB], iB,
            rules, capturesA[iA] != 0, capturesB[iB] != 0);
    };

    std::discrete_distribution<size_t> sampleA(probabilitiesA.begin(), probabilitiesA.end());
    std::discrete_distribution<size_t> sampleB(probabilitiesB.begin(), probabilitiesB.end());
    JointActionCandidate candidate = make_candidate(sampleA(randomEngine), sampleB(randomEngine));
    if (candidate.jointPrior >= 0.0f) {
        return candidate;
    }

    // The independent sample produced an illegal joint action, so resample from
    // the legal pairs only.
    std::vector<std::pair<size_t, size_t>> legalPairs;
    std::vector<double> weights;
    for (size_t iA = 0; iA < actionsA.size(); ++iA) {
        for (size_t iB = 0; iB < actionsB.size(); ++iB) {
            if (make_candidate(iA, iB).jointPrior < 0.0f) {
                continue;
            }
            legalPairs.emplace_back(iA, iB);
            weights.push_back(
                static_cast<double>(probabilitiesA[iA]) * static_cast<double>(probabilitiesB[iB]));
        }
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        throw std::runtime_error("Raw policy produced no legal joint action");
    }
    std::discrete_distribution<size_t> legalSample(weights.begin(), weights.end());
    const auto [indexA, indexB] = legalPairs[legalSample(randomEngine)];
    return make_candidate(indexA, indexB);
}

bool action_leads_to_terminal(
    Board& board,
    const JointActionCandidate& action,
    Stockfish::Color team,
    const TimeState& time) {
    TimeState next = time;
    next.advance(team, board.on_turn_on_both_boards(team));
    Board future(board);
    future.make_moves(action.moveA, action.moveB);
    const Stockfish::Color nextTeam = ~team;
    return future.is_checkmate(nextTeam, next.can_sit(nextTeam))
        || future.is_checkmate(team, next.can_sit(team))
        || future.is_draw();
}

int policy_index(Board& board, int boardNumber, Stockfish::Move move) {
    const int idx = get_fast_policy_index(move, board.side_to_move(boardNumber));
    if (idx < 0) {
        throw std::runtime_error("Move is absent from policy map: " + board.uci_move(boardNumber, move));
    }
    return idx;
}

std::vector<SparsePolicyEntry> marginal_policy(
    Board& board,
    int boardNumber,
    const std::vector<RootEdgeStats>& edges) {
    std::map<uint16_t, uint64_t> visitsByMove;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const Stockfish::Move move = boardNumber == BOARD_A
            ? edge.action.moveA
            : edge.action.moveB;
        const int index = policy_index(board, boardNumber, move);
        visitsByMove[static_cast<uint16_t>(index)] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }

    std::vector<SparsePolicyEntry> policy;
    policy.reserve(visitsByMove.size());
    for (const auto& [index, visits] : visitsByMove) {
        policy.push_back({index, static_cast<float>(visits) / static_cast<float>(totalVisits)});
    }
    return policy;
}

/// A board's search policy over all of its legal moves and the pass (zero for
/// the ones the search did not visit), for distillation. Empty when the board
/// is not on turn.
std::vector<distill::PolicyEntry> distill_search_policy(
    Board& board,
    int boardNumber,
    bool onTurn,
    const std::vector<RootEdgeStats>& edges) {
    std::vector<distill::PolicyEntry> entries;
    if (!onTurn) {
        return entries;
    }
    std::map<Stockfish::Move, uint64_t> visitsByMove;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const Stockfish::Move move = boardNumber == BOARD_A ? edge.action.moveA : edge.action.moveB;
        visitsByMove[move] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }
    const std::vector<Stockfish::Move> actions = distill::policy_actions(board, boardNumber);
    entries.reserve(actions.size());
    uint64_t covered = 0;
    for (const Stockfish::Move move : actions) {
        const auto found = visitsByMove.find(move);
        const uint64_t visits = found == visitsByMove.end() ? 0 : found->second;
        covered += visits;
        entries.push_back({static_cast<uint16_t>(policy_index(board, boardNumber, move)),
                           distill::half_bits(__float2half_rn(
                               static_cast<float>(visits) / static_cast<float>(totalVisits)))});
    }
    if (covered != totalVisits) {
        throw std::runtime_error("Search visited a move outside the legal policy moves");
    }
    return entries;
}

uint16_t joint_policy_index(Board& board, int boardNumber,
                            Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return PASS_POLICY_INDEX;
    }
    return static_cast<uint16_t>(policy_index(board, boardNumber, move));
}

std::vector<SparseJointPolicyEntry> joint_policy(
    Board& board, const std::vector<RootEdgeStats>& edges) {
    std::map<std::pair<uint16_t, uint16_t>, uint64_t> visitsByAction;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const auto key = std::make_pair(
            joint_policy_index(board, BOARD_A, edge.action.moveA),
            joint_policy_index(board, BOARD_B, edge.action.moveB));
        visitsByAction[key] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }

    std::vector<SparseJointPolicyEntry> policy;
    policy.reserve(visitsByAction.size());
    for (const auto& [indices, visits] : visitsByAction) {
        policy.push_back({
            indices.first,
            indices.second,
            static_cast<float>(visits) / static_cast<float>(totalVisits),
        });
    }
    return policy;
}

JointActionCandidate select_action(
    const std::vector<RootEdgeStats>& edges,
    double temperature,
    std::mt19937_64& randomEngine) {
    if (edges.empty()) {
        throw std::runtime_error("Cannot select from an empty root");
    }
    if (temperature <= 1e-6) {
        return std::max_element(
            edges.begin(), edges.end(),
            [](const RootEdgeStats& left, const RootEdgeStats& right) {
                return left.visits < right.visits;
            })->action;
    }

    std::vector<double> weights;
    weights.reserve(edges.size());
    const int maxVisits = std::max_element(
        edges.begin(), edges.end(),
        [](const RootEdgeStats& left, const RootEdgeStats& right) {
            return left.visits < right.visits;
        })->visits;
    for (const RootEdgeStats& edge : edges) {
        weights.push_back(edge.visits > 0 && maxVisits > 0
            ? std::exp((std::log(static_cast<double>(edge.visits))
                        - std::log(static_cast<double>(maxVisits))) / temperature)
            : 0.0);
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        return edges.front().action;
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return edges[distribution(randomEngine)].action;
}

std::array<uint8_t, NB_INPUT_VALUES()> encode_planes(
    Board& board,
    Stockfish::Color team,
    const TimeState& time) {
    std::array<float, NB_INPUT_VALUES()> raw{};
    std::array<uint8_t, NB_INPUT_VALUES()> encoded{};
    board_to_planes(board, raw.data(), team, time);
    constexpr size_t square = 64;
    constexpr size_t block = NB_INPUT_CHANNELS_PER_BOARD * square;
    constexpr size_t clockStart = NB_LEGACY_INPUT_CHANNELS_PER_BOARD * square;
    for (size_t index = 0; index < raw.size(); ++index) {
        // The clock planes are signed; the chunk keeps them as (v + 1) / 2.
        const bool clockPlane = index % block >= clockStart;
        const float value = clockPlane ? (raw[index] + 1.0f) / 2.0f : raw[index];
        encoded[index] = static_cast<uint8_t>(std::lround(
            std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }
    return encoded;
}

std::string current_date() {
    const std::time_t now = std::time(nullptr);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif
    std::ostringstream date;
    date << std::put_time(&localTime, "%Y.%m.%d");
    return date.str();
}

// Wave 2 seat-clock games (--base-mix): the clocks a game started and ended
// on, and per searched record the clocks the search saw.
struct SeatClockRow {
    size_t sample = 0;
    int team = 0;
    int64_t timeMs = 0;
    uint32_t nodes = 0;
    std::array<int32_t, 4> clocks{};
    std::array<int32_t, 4> elapsed{};
    std::array<int32_t, 4> paces{};
    // Wave 2b: ms since the last landing on A and B; pieces landed in the
    // last 5 s and (written at game end) in the next 5 s, both in the
    // arrivals head's order from the record team's view (pocket_landing_counts).
    std::array<int32_t, 2> boardElapsed{};
    std::array<int, 20> recent{};
    std::array<int, 20> arrivals{};
};

struct SeatClockSummary {
    SeatClockGame::Start start;
    std::array<int32_t, 4> finalClocks{};
    int32_t finalWhiteLeadMs = 0;
    int64_t gameMs = 0;
    int flagSeat = -1;
    std::vector<SeatClockRow> rows;
};

const char* seat_name(int seat) {
    static const char* const names[4] = {"Aw", "Ab", "Bw", "Bb"};
    return names[seat];
}

template <typename Values>
std::string four(const Values& values) {
    std::ostringstream text;
    for (size_t seat = 0; seat < 4; ++seat) {
        text << (seat ? " " : "") << values[seat];
    }
    return text.str();
}

void write_seat_tags(std::ostream& stream, const SeatClockSummary& seats) {
    stream << "[BaseMs \"" << seats.start.baseMs << "\"]\n"
           << "[StartClocksMs \"" << four(seats.start.clocksMs) << "\"]\n"
           << "[SeatMoveTimesMs \"" << four(seats.start.moveTimesMs) << "\"]\n"
           << "[FinalClocksMs \"" << four(seats.finalClocks) << "\"]\n"
           << "[GameMs \"" << seats.gameMs << "\"]\n";
}

/**
 * One line per searched record of a seat-clock game, in record order (HVM6
 * and HDST have no room for them): game, record index in the game, team,
 * game time, nodes, then the four seat clocks, the elapsed thinking of the
 * seats on turn (0 for the others) and the observed paces, Aw Ab Bw Bb;
 * then (wave 2b) ms since the last landing on A and B, and 20 recent and 20
 * arrivals counts, pockets ours A, theirs A, ours B, theirs B from the
 * record team's view, each P N B R Q (SeatClockRow).
 */
void write_seat_rows(const std::filesystem::path& path, size_t gameNumber,
                     const SeatClockSummary& seats) {
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("Unable to append " + path.string());
    }
    if (fresh) {
        stream << "game\trecord\tteam\ttime_ms\tnodes"
                  "\tclock_aw\tclock_ab\tclock_bw\tclock_bb"
                  "\telapsed_aw\telapsed_ab\telapsed_bw\telapsed_bb"
                  "\tpace_aw\tpace_ab\tpace_bw\tpace_bb"
                  "\tboard_elapsed_a\tboard_elapsed_b";
        for (const char* kind : {"recent", "arrivals"}) {
            for (const char* pocket : {"ours_a", "theirs_a", "ours_b", "theirs_b"}) {
                for (const char* piece : {"p", "n", "b", "r", "q"}) {
                    stream << '\t' << kind << '_' << pocket << '_' << piece;
                }
            }
        }
        stream << '\n';
    }
    for (const SeatClockRow& row : seats.rows) {
        stream << gameNumber + 1 << '\t' << row.sample << '\t' << row.team << '\t'
               << row.timeMs << '\t' << row.nodes;
        for (const auto* values : {&row.clocks, &row.elapsed, &row.paces}) {
            for (int32_t value : *values) {
                stream << '\t' << value;
            }
        }
        stream << '\t' << row.boardElapsed[0] << '\t' << row.boardElapsed[1];
        for (const auto* counts : {&row.recent, &row.arrivals}) {
            for (int count : *counts) {
                stream << '\t' << count;
            }
        }
        stream << '\n';
    }
}

void append_pgn(
    const std::filesystem::path& path,
    size_t round,
    const std::vector<PgnMove>& moves,
    int winner,
    int startingTeam,
    size_t rawPolicyMacroPlies,
    size_t rawPolicyEvents,
    float initialClockSeconds,
    const std::string& termination,
    const TimeState& initialTime,
    const TimeState& finalTime,
    const SeatClockSummary* seats = nullptr) {
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("Unable to append " + path.string());
    }
    const std::string result = winner == 0 ? "1-0" : winner == 1 ? "0-1" : "1/2-1/2";
    const std::string winnerName = winner == 0 ? "Hivemind-A" : winner == 1 ? "Hivemind-B" : "Draw";
    stream << "[Event \"Hivemind Self-Play\"]\n"
           << "[Site \"Hivemind Engine\"]\n"
           << "[Date \"" << current_date() << "\"]\n"
           << "[Round \"" << round << "\"]\n"
           << "[Variant \"bughouse\"]\n"
           << "[TimeControl \"" << static_cast<int>(initialClockSeconds) << "\"]\n"
           << "[WhiteTeam \"Hivemind-A\"]\n"
           << "[BlackTeam \"Hivemind-B\"]\n"
           << "[WhiteA \"Hivemind-A1\"]\n"
           << "[BlackA \"Hivemind-B2\"]\n"
           << "[WhiteB \"Hivemind-B1\"]\n"
           << "[BlackB \"Hivemind-A2\"]\n";
    if (initialTime.is_clocked()) {
        // Leads are Hivemind-A's (the white team's) over its diagonal opponents.
        stream << "[TimeModel \"clock\"]\n"
               << "[TimeLeadMs \"" << initialTime.lead_ms(Stockfish::WHITE) << "\"]\n"
               << "[FinalTimeLeadMs \"" << finalTime.lead_ms(Stockfish::WHITE) << "\"]\n"
               << "[WhiteTeamMoveTimeMs \"" << initialTime.move_time_ms(Stockfish::WHITE) << "\"]\n"
               << "[BlackTeamMoveTimeMs \"" << initialTime.move_time_ms(Stockfish::BLACK) << "\"]\n";
    } else {
        stream << "[TimeAdvantage \"" << (startingTeam == 0 ? "Hivemind-B" : "Hivemind-A") << "\"]\n";
    }
    if (seats) {
        write_seat_tags(stream, *seats);
    }
    stream << "[RawPolicyMacroPlies \"" << rawPolicyMacroPlies << "\"]\n"
           << "[RawPolicyEvents \"" << rawPolicyEvents << "\"]\n"
           << "[PlyCount \"" << moves.size() << "\"]\n"
           << "[Result \"" << result << "\"]\n"
           << "[Termination \"" << termination << "\"]\n\n";

    stream << std::fixed << std::setprecision(1);
    for (const PgnMove& move : moves) {
        stream << move.token << ". " << move.san << " {" << move.remainingSeconds << "} ";
    }
    stream << "{C:" << termination << ' ' << result << "}\n"
           << '{' << winnerName << (winner < 0 ? " game" : " won") << " by " << termination << "} *\n\n";
}

// How far a checkpointed run got; see run_selfplay.
struct SelfPlayProgress {
    uint64_t seed = 0;
    size_t games = 0;         // the run's game count
    size_t done = 0;          // games saved, in game order
    size_t flushedGames = 0;  // of those, the games in finished chunks
    size_t chunks = 0;        // distillation chunks written
    uintmax_t pgnBytes = 0;   // PGN size after the saved games
};

void write_progress(const std::filesystem::path& path, const SelfPlayProgress& progress) {
    const std::filesystem::path temporaryPath = path.string() + ".tmp";
    {
        std::ofstream stream(temporaryPath, std::ios::trunc);
        stream << "seed " << progress.seed << '\n'
               << "games " << progress.games << '\n'
               << "done " << progress.done << '\n'
               << "flushed_games " << progress.flushedGames << '\n'
               << "chunks " << progress.chunks << '\n'
               << "pgn_bytes " << progress.pgnBytes << '\n';
        if (!stream) {
            throw std::runtime_error("Failed to write " + temporaryPath.string());
        }
    }
    std::filesystem::rename(temporaryPath, path);
}

SelfPlayProgress read_progress(const std::filesystem::path& path) {
    std::ifstream stream(path);
    SelfPlayProgress progress;
    std::string key;
    std::map<std::string, uintmax_t> values;
    uintmax_t value = 0;
    while (stream >> key >> value) {
        values[key] = value;
    }
    for (const char* required : {"seed", "games", "done", "flushed_games", "chunks", "pgn_bytes"}) {
        if (!values.contains(required)) {
            throw std::runtime_error("Self-play checkpoint " + path.string() + " lacks " + required);
        }
    }
    progress.seed = values["seed"];
    progress.games = values["games"];
    progress.done = values["done"];
    progress.flushedGames = values["flushed_games"];
    progress.chunks = values["chunks"];
    progress.pgnBytes = values["pgn_bytes"];
    return progress;
}

// distill::ChunkWriter's name for chunk `index`.
std::string distill_chunk_name(const std::string& prefix, size_t index) {
    std::ostringstream name;
    name << prefix << '_' << std::setw(5) << std::setfill('0') << index << ".dst";
    return name.str();
}

std::string checkpoint_file_name(size_t firstGame) {
    std::ostringstream name;
    name << "games_" << std::setw(8) << std::setfill('0') << firstGame << ".dst";
    return name.str();
}

// Checkpoint files by the first game each holds, in game order.
std::map<size_t, std::filesystem::path> checkpoint_files(const std::filesystem::path& directory) {
    std::map<size_t, std::filesystem::path> files;
    if (!std::filesystem::is_directory(directory)) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name.size() == 18 && name.starts_with("games_") && name.ends_with(".dst")) {
            files.emplace(std::stoull(name.substr(6, 8)), entry.path());
        }
    }
    return files;
}

void append_pgn_move(
    Board& board,
    int boardNumber,
    Stockfish::Move move,
    std::array<int, 2>& moveNumbers,
    size_t eventIndex,
    float initialClockSeconds,
    std::vector<PgnMove>& pgnMoves,
    float remainingSeconds = -1.0f) {
    if (move == Stockfish::MOVE_NONE) {
        return;
    }
    const bool whiteToMove = board.side_to_move(boardNumber) == Stockfish::WHITE;
    const char boardLetter = boardNumber == BOARD_A
        ? (whiteToMove ? 'A' : 'a')
        : (whiteToMove ? 'B' : 'b');
    PgnMove pgnMove;
    pgnMove.token = std::to_string(moveNumbers[boardNumber]) + boardLetter;
    pgnMove.san = board.san_move(boardNumber, move);
    // Seat-clock games give the mover's real clock; otherwise a placeholder.
    pgnMove.remainingSeconds = remainingSeconds >= 0.0f
        ? remainingSeconds
        : std::max(0.0f, initialClockSeconds - 0.1f * static_cast<float>(eventIndex + 1));
    pgnMoves.push_back(std::move(pgnMove));
    if (!whiteToMove) {
        moveNumbers[boardNumber]++;
    }
}

/// What one seat-clock game produced (see play_seat_clock_game).
struct SeatClockGameResult {
    std::vector<TrainingSample> samples;
    std::vector<distill::Record> distillRecords;
    std::vector<PgnMove> pgnMoves;
    int winner = -1;
    std::string termination = "macro-ply limit";
    size_t rawPolicyMacroPlies = 0;
    size_t rawPolicyEvents = 0;
    int wouldResignTeam = -1;
    SeatClockSummary seats;
};

/**
 * @brief A wave 2 self-play game on four real seat clocks (--base-mix).
 *
 * SeatClockGame drives it: each decision is one search by the deciding
 * seat's team (one record), rooted at that moment with the seat clocks,
 * observed paces, the other team's elapsed thinking and the wave 2 tree
 * options; only the deciding seats' boards of the chosen action are played,
 * when they land. A flag (FlagLoss), a double flag, a sit-off, a mate or a
 * draw on landing ends the game; config.maxMacroPlies caps the decisions.
 * Tree reuse is not used: durative trees do not transpose across searches.
 */
SeatClockGameResult play_seat_clock_game(
    const SelfPlayConfig& config,
    const std::vector<Engine*>& threadEngines,
    Agent (&agents)[2],
    RawPolicyEvaluator& rawPolicyEvaluator,
    std::vector<__half>& distillPlanes,
    bool writeDistill,
    uint64_t runId,
    size_t gameIndex,
    std::mt19937_64& randomEngine) {
    SeatClockGameResult result;
    Engine& rawPolicyEngine = *threadEngines.front();
    Board board;
    SeatClockGame game(config.wave2,
                       SeatClockGame::sample_start(config.wave2, config.clock, randomEngine));
    result.seats.start = game.start();
    game.set_wide_inputs(rawPolicyEngine.wideInputs());
    std::array<Stockfish::Move, 4> committed{};
    std::array<int, 2> moveNumbers = {1, 1};
    const size_t initializationLength = sample_initialization_length(config, randomEngine);
    bool rawInitializationActive = initializationLength > 0;
    const bool canResign = config.resignThreshold < 0.0f
        && (config.resignDisableFraction <= 0.0
            || std::uniform_real_distribution<double>(0.0, 1.0)(randomEngine) >= config.resignDisableFraction);
    const bool checksResign = !canResign && config.resignThreshold < 0.0f;
    std::array<size_t, 2> consecutiveResignPlies = {0, 0};
    size_t decisions = 0;
    for (Agent& teamAgent : agents) {
        teamAgent.reset_search_state();
    }

    auto sides_now = [&board]() {
        return std::array<Stockfish::Color, 2>{
            board.side_to_move(BOARD_A), board.side_to_move(BOARD_B)};
    };
    for (;;) {
        const std::array<Stockfish::Color, 2> sides = sides_now();
        const SeatClockGame::Event event = game.next(sides);
        if (event.type == SeatClockGame::EventType::FLAG) {
            result.winner = event.loser == 0 ? 1 : 0;
            result.termination = "flag";
            for (int seat = 0; seat < 4; ++seat) {
                result.seats.flagSeat = event.seats[seat] ? seat : result.seats.flagSeat;
            }
            break;
        }
        if (event.type == SeatClockGame::EventType::DOUBLE_FLAG) {
            result.winner = -1;
            result.termination = "double_flag";
            break;
        }
        if (event.type == SeatClockGame::EventType::SITOFF) {
            result.winner = event.loser < 0 ? -1 : 1 - event.loser;
            result.termination = "sitoff_adjudicated";
            break;
        }
        const Stockfish::Color team = event.team;
        if (event.type == SeatClockGame::EventType::LAND) {
            for (int seat = 0; seat < 4; ++seat) {
                if (!event.seats[seat]) {
                    continue;
                }
                const int boardNumber = seat / 2;
                const Stockfish::Move move = committed[seat];
                if (!board.is_legal_move(boardNumber, move)) {
                    throw std::runtime_error("A committed seat-clock move is no longer legal");
                }
                append_pgn_move(board, boardNumber, move, moveNumbers, result.pgnMoves.size(),
                                config.initialClockSeconds, result.pgnMoves,
                                static_cast<float>(game.clock_ms(seat)) / 1000.0f);
                const bool capture = board.is_capture(boardNumber, move);
                const Stockfish::PieceType hand = board.hand_piece_type(boardNumber, move);
                board.push_move(boardNumber, move);
                game.landed(seat, capture, hand);
            }
            if (rawInitializationActive) {
                result.rawPolicyEvents = result.pgnMoves.size();
            }
            // A mate ends the game when it lands, whichever board.
            if (board.is_checkmate(~team, true)) {
                result.winner = static_cast<int>(team);
                result.termination = "checkmate";
                break;
            }
            if (board.is_checkmate(team, true)) {
                result.winner = static_cast<int>(~team);
                result.termination = "checkmate";
                break;
            }
            if (board.is_draw()) {
                result.termination = "draw";
                break;
            }
            continue;
        }

        // A decision: the seats in event.seats start thinking now.
        if (decisions >= config.maxMacroPlies) {
            break;
        }
        for (int seat = 0; seat < 4; ++seat) {
            if (event.seats[seat]) {
                game.plan_cost(seat, randomEngine);
            }
        }
        const TimeState clock = game.search_clock(team, sides, event.seats);
        const bool hasTimeAdvantage = clock.can_sit(team);
        JointActionCandidate action;
        bool searched = false;
        if (rawInitializationActive && decisions < initializationLength) {
            const JointActionCandidate rawAction = sample_raw_policy_action(
                rawPolicyEngine, rawPolicyEvaluator, board, team, clock,
                sample_raw_policy_temperature(config, randomEngine), randomEngine);
            if (!action_leads_to_terminal(board, rawAction, team, clock)) {
                action = rawAction;
                ++result.rawPolicyMacroPlies;
            } else {
                rawInitializationActive = false;
            }
        } else {
            rawInitializationActive = false;
        }
        if (!rawInitializationActive) {
            const GameStatus status = adjudicate_game(board, team, clock);
            if (status != GameStatus::ONGOING) {
                result.winner = adjudicated_winner(status, team);
                result.termination = adjudicated_termination(status);
                break;
            }
            TrainingSample sample;
            sample.gameId = gameIndex;
            sample.macroPly = static_cast<uint16_t>(std::min<size_t>(
                decisions, std::numeric_limits<uint16_t>::max()));
            sample.team = team == Stockfish::WHITE ? 0 : 1;
            sample.hasTimeAdvantage = hasTimeAdvantage ? 1 : 0;
            if (config.writeTrainingChunks) {
                sample.planes = encode_planes(board, team, clock);
            }
            const size_t teamIndex = team == Stockfish::WHITE ? 0 : 1;
            Agent& agent = agents[teamIndex];
            agent.reset_search_state();
            SearchOptions searchOptions;
            searchOptions.targetNodes = randomized_node_budget(config, randomEngine);
            searchOptions.retainForReuse = false;
            searchOptions.search.enableMateProbe = config.fairyStockfishMateNodes > 0;
            searchOptions.mateProbeNodes = config.fairyStockfishMateNodes;
            searchOptions.completeMateProbe = config.fairyStockfishMateNodes > 0;
            searchOptions.search.enableGumbelRootSearch = false;
            if (!config.rootScans) {
                searchOptions.search.enableRootMateSearch = false;
                searchOptions.search.certifySelectedMove = false;
            }
            searchOptions.search.rootDirichletAlpha = config.dirichletAlpha;
            searchOptions.search.rootDirichletEpsilon = config.dirichletEpsilon;
            searchOptions.search.rootNoiseSeed = selfplay_seed(
                runId, gameIndex * config.maxMacroPlies + decisions);
            searchOptions.clock = clock;
            const JointActionCandidate choice = agent.run_search(
                board, threadEngines, team, hasTimeAdvantage, searchOptions);
            if (agent.search_status() != GameStatus::ONGOING) {
                result.winner = adjudicated_winner(agent.search_status(), team);
                result.termination = adjudicated_termination(agent.search_status());
                break;
            }
            std::vector<RootEdgeStats> edges = agent.root_edge_stats();
            if (edges.empty()) {
                throw std::runtime_error("Ongoing search returned no root edges");
            }
            const uint64_t actualVisits = std::accumulate(
                edges.begin(), edges.end(), uint64_t{0},
                [](uint64_t total, const RootEdgeStats& edge) {
                    return total + static_cast<uint64_t>(std::max(0, edge.visits));
                });
            sample.nodes = static_cast<uint32_t>(std::min<uint64_t>(
                actualVisits, std::numeric_limits<uint32_t>::max()));
            edges = selfplay_policy_edges(edges, agent.root_type(), choice, agent.certified_choice());
            const float rootQ = agent.root_q();
            sample.rootQ = rootQ;
            if (config.writeTrainingChunks) {
                sample.policyA = marginal_policy(board, BOARD_A, edges);
                sample.policyB = marginal_policy(board, BOARD_B, edges);
                sample.jointPolicy = joint_policy(board, edges);
            }
            if (writeDistill) {
                distill::Record record;
                board_to_planes(board, distillPlanes.data(), team, clock);
                distill::pack_planes(distillPlanes.data(), record);
                record.value = rootQ;
                record.ply = sample.macroPly;
                record.policy[0] = distill_search_policy(
                    board, BOARD_A, board.side_to_move(BOARD_A) == team, edges);
                record.policy[1] = distill_search_policy(
                    board, BOARD_B, board.side_to_move(BOARD_B) == ~team, edges);
                result.distillRecords.push_back(std::move(record));
            }
            SeatClockRow row;
            row.sample = result.samples.size();
            row.team = sample.team;
            row.timeMs = game.now_ms();
            row.nodes = sample.nodes;
            row.clocks = game.clocks_ms();
            row.paces = game.paces_ms();
            for (int board_ = 0; board_ < 2; ++board_) {
                const int seat = TimeState::seat_of(board_, sides[board_]);
                row.elapsed[seat] = game.elapsed_ms(seat);
                row.boardElapsed[board_] = game.board_elapsed_ms(board_);
            }
            row.recent = pocket_landing_counts(game.landings(), team,
                                               game.now_ms() - RECENT_LANDING_WINDOW_MS,
                                               game.now_ms());
            result.seats.rows.push_back(row);
            result.samples.push_back(std::move(sample));

            if (checksResign && result.wouldResignTeam < 0) {
                consecutiveResignPlies[teamIndex] = rootQ <= config.resignThreshold
                    ? consecutiveResignPlies[teamIndex] + 1 : 0;
                if (consecutiveResignPlies[teamIndex] >= config.resignConsecutivePlies) {
                    result.wouldResignTeam = static_cast<int>(teamIndex);
                }
            }
            if (canResign) {
                if (rootQ <= config.resignThreshold) {
                    if (++consecutiveResignPlies[teamIndex] >= config.resignConsecutivePlies) {
                        result.winner = 1 - static_cast<int>(teamIndex);
                        result.termination = "resignation";
                        break;
                    }
                } else {
                    consecutiveResignPlies[teamIndex] = 0;
                }
            }
            action = select_action(edges, mcts_temperature(config, decisions), randomEngine);
            searched = true;
        }
        (void)searched;
        ++decisions;
        // Only the seats deciding now act; the team's other seat (if on
        // turn) is still on a decision of its own.
        for (int seat = 0; seat < 4; ++seat) {
            if (!event.seats[seat]) {
                continue;
            }
            const Stockfish::Move move = seat / 2 == BOARD_A ? action.moveA : action.moveB;
            if (move == Stockfish::MOVE_NONE) {
                game.commit_pass(seat, sides);
            } else {
                committed[seat] = move;
                game.commit_move(seat);
            }
        }
    }
    // Wave 2b arrivals labels: what landed in each pocket in the 5 s after
    // each record's search started (the game's end cuts the window short).
    for (SeatClockRow& row : result.seats.rows) {
        row.arrivals = pocket_landing_counts(
            game.landings(), row.team == 0 ? Stockfish::WHITE : Stockfish::BLACK,
            row.timeMs, row.timeMs + RECENT_LANDING_WINDOW_MS);
    }
    result.seats.finalClocks = game.clocks_ms();
    result.seats.finalWhiteLeadMs = game.white_lead_ms();
    result.seats.gameMs = game.now_ms();
    return result;
}

} // namespace

int run_selfplay(const std::vector<Engine*>& engines, const SelfPlayConfig& config) {
    if (config.games == 0 || config.nodes == 0 || config.maxMacroPlies == 0) {
        throw std::invalid_argument("games, nodes, and max-macro-plies must be positive");
    }
    if (engines.empty()) {
        throw std::invalid_argument("Self-play requires at least one inference engine");
    }
    if (config.rawPolicyMeanMacroPlies < 0.0
        || config.rawPolicyHighTemperatureProbability < 0.0
        || config.rawPolicyHighTemperatureProbability > 1.0
        || config.mctsTemperature <= 0.0
        || config.mctsTemperatureDecay <= 0.0
        || config.mctsTemperatureDecay > 1.0
        || config.nodeRandomFactor < 0.0
        || config.nodeRandomFactor >= 1.0
        || !(config.treeReuseMinNewFraction >= 0.0 && config.treeReuseMinNewFraction <= 1.0)) {
        throw std::invalid_argument("Invalid self-play exploration configuration");
    }
    const bool checkpointing = config.checkpointGames > 0;
    if ((checkpointing || config.resume) && config.writeTrainingChunks) {
        throw std::invalid_argument("Self-play checkpoints need --training-chunks false");
    }
    if (config.resume && (config.seed == 0 || !checkpointing)) {
        throw std::invalid_argument("Resuming self-play needs a fixed --seed and --checkpoint-games");
    }
    const size_t gameThreads = std::max<size_t>(1, config.parallelGames);
    std::set<Engine*> uniqueEngines(engines.begin(), engines.end());
    if (uniqueEngines.size() != engines.size() || uniqueEngines.count(nullptr)
        || engines.size() < gameThreads) {
        throw std::invalid_argument("Self-play requires a distinct engine per game thread");
    }
    const uint64_t runId = config.seed != 0
        ? config.seed
        : static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    // Checkpoints: every checkpointGames finished games, in game order, the
    // games not yet in a chunk go to checkpoint/games_<first game>.dst, and
    // checkpoint/progress records how far the chunks, those files and the PGN
    // reach. A resumed run cuts the PGN back to there, reloads the games not
    // yet in a chunk and plays on from the first game missing, each game on
    // the seed it always had.
    const std::filesystem::path checkpointDirectory = config.outputDirectory / "checkpoint";
    const std::filesystem::path progressPath = checkpointDirectory / "progress";
    const std::filesystem::path pgnPath = config.outputDirectory / "games.pgn";
    // Seat-clock games: the clocks behind every record (see write_seat_rows).
    const std::filesystem::path seatRowsPath = config.outputDirectory / "seat_clocks.tsv";
    const std::string distillPrefix = "search_" + std::to_string(runId);
    std::optional<SelfPlayProgress> resumed;
    if (config.resume && std::filesystem::exists(progressPath)) {
        resumed = read_progress(progressPath);
        if (resumed->seed != runId || resumed->games != config.games) {
            throw std::invalid_argument(
                "Self-play checkpoint is for another seed or game count");
        }
    }
    if (!resumed) {
        if (std::filesystem::exists(config.outputDirectory)
            && !std::filesystem::is_empty(config.outputDirectory)) {
            throw std::invalid_argument("Self-play output directory must be fresh");
        }
        if (!config.distillOutputDirectory.empty()
            && std::filesystem::exists(config.distillOutputDirectory)
            && !std::filesystem::is_empty(config.distillOutputDirectory)) {
            throw std::invalid_argument("Self-play distillation directory must be fresh");
        }
    }
    const std::filesystem::path trainingDirectory = config.outputDirectory / "training_data";
    std::filesystem::create_directories(config.outputDirectory);
    ChunkWriter chunkWriter(trainingDirectory, config.chunkSamples, runId);
    std::unique_ptr<distill::ChunkWriter> distillWriter;
    if (resumed) {
        std::filesystem::resize_file(pgnPath, resumed->pgnBytes);
        if (!config.distillOutputDirectory.empty()) {
            // A chunk written after the last progress record holds games the
            // checkpoint files hold too.
            for (size_t index = resumed->chunks;
                 std::filesystem::remove(config.distillOutputDirectory
                     / distill_chunk_name(distillPrefix, index));
                 ++index) {}
        }
    }
    if (!config.distillOutputDirectory.empty()) {
        distillWriter = std::make_unique<distill::ChunkWriter>(
            config.distillOutputDirectory, distillPrefix,
            config.distillChunkPositions, resumed.has_value());
    }
    std::vector<distill::Record> unchunked;  // since the last chunk or checkpoint file
    size_t unchunkedFirst = 0;
    size_t flushedGames = 0;
    if (resumed) {
        flushedGames = resumed->flushedGames;
        unchunkedFirst = resumed->done;
        for (const auto& [first, path] : checkpoint_files(checkpointDirectory)) {
            if (first < resumed->flushedGames || first >= resumed->done) {
                std::filesystem::remove(path);
                continue;
            }
            for (distill::Record& record : distill::read_chunk(path)) {
                if (distillWriter) {
                    distillWriter->append_unflushed(std::move(record));
                }
            }
        }
        std::cout << "selfplay resumed after " << resumed->done << '/' << config.games
                  << " games" << std::endl;
    }
    if (checkpointing) {
        std::filesystem::create_directories(checkpointDirectory);
    }
    const auto save_progress = [&](size_t done) {
        SelfPlayProgress progress;
        progress.seed = runId;
        progress.games = config.games;
        progress.done = done;
        progress.flushedGames = flushedGames;
        progress.chunks = distillWriter ? distillWriter->chunks_written() : 0;
        progress.pgnBytes = std::filesystem::exists(pgnPath)
            ? std::filesystem::file_size(pgnPath) : 0;
        write_progress(progressPath, progress);
    };
    // Games run in `parallelGames` threads, each with its own engine, so the
    // GPU overlaps many small searches; outputs are serialised.
    const size_t firstGame = resumed ? resumed->done : 0;
    std::atomic<size_t> nextGame{firstGame};
    std::mutex outputMutex;
    std::exception_ptr failure;
    // Finished games waiting for every earlier game to be written (guarded by
    // outputMutex), so the corpus order depends only on the seed.
    struct FinishedGame {
        std::vector<TrainingSample> samples;
        std::vector<distill::Record> distillRecords;
        std::vector<PgnMove> pgnMoves;
        int winner;
        int startingTeam;
        size_t rawPolicyMacroPlies;
        size_t rawPolicyEvents;
        std::string termination;
        TimeState initialTime;
        TimeState finalTime;
        // In a game exempt from resigning, the team the rule would have
        // resigned for (-1: none): what the resignation threshold is checked by.
        int wouldResignTeam;
        // Seat-clock games (--base-mix) only.
        std::optional<SeatClockSummary> seats;
    };
    std::map<size_t, FinishedGame> finishedGames;
    size_t nextOutput = firstGame;
    // Guarded by outputMutex.
    size_t resignChecks = 0;
    size_t wrongResignations = 0;
    std::atomic<uint64_t> searches{0};
    std::atomic<uint64_t> retainedVisits{0};
    std::atomic<uint64_t> searchedVisits{0};
    const auto writeGame = [&](size_t gameNumber, FinishedGame& game) {
        if (distillWriter) {
            // Finished games give WDL (loss, draw, win) and moves-left targets
            // (remaining team decisions, capped at 100, over 100, the scale
            // the search assumes). A game cut off at the macro-ply limit has
            // no real result: its WDL stays zero, which the trainer masks.
            // A resignation ends the game early, so the decisions left in it
            // say nothing about how long the position had to go: its records
            // keep the result and drop the moves-left target.
            const bool finished = game.termination != "macro-ply limit";
            const bool resigned = game.termination == "resignation";
            for (size_t index = 0; index < game.distillRecords.size(); ++index) {
                distill::Record& record = game.distillRecords[index];
                record.outcome = game.samples[index].outcome;
                if (finished) {
                    record.wdl = {0.0f, 0.0f, 0.0f};
                    record.wdl[static_cast<size_t>(game.samples[index].outcome + 1)] = 1.0f;
                    const size_t remaining = std::min<size_t>(game.samples.size() - index, 100);
                    record.movesLeft = resigned
                        ? distill::MOVES_LEFT_UNKNOWN
                        : distill::half_bits(
                            __float2half_rn(static_cast<float>(remaining) / 100.0f));
                }
                if (checkpointing) {
                    // A game stays in one chunk, so a checkpoint never splits it.
                    unchunked.push_back(record);
                    distillWriter->append_unflushed(std::move(record));
                } else {
                    distillWriter->append(std::move(record));
                }
            }
        }
        const size_t sampleCount = game.samples.size();
        if (config.writeTrainingChunks) {
            chunkWriter.append(std::move(game.samples));
        }
        append_pgn(
            pgnPath, gameNumber + 1, game.pgnMoves, game.winner, game.startingTeam,
            game.rawPolicyMacroPlies, game.rawPolicyEvents,
            config.initialClockSeconds, game.termination,
            game.initialTime, game.finalTime, game.seats ? &*game.seats : nullptr);
        if (game.seats) {
            write_seat_rows(seatRowsPath, gameNumber, *game.seats);
        }
        if (checkpointing) {
            const size_t done = gameNumber + 1;
            if (distillWriter && distillWriter->full()) {
                distillWriter->flush();
                flushedGames = done;
                unchunked.clear();
                unchunkedFirst = done;
                save_progress(done);
                for (const auto& [first, path] : checkpoint_files(checkpointDirectory)) {
                    std::filesystem::remove(path);
                }
            } else if (done % config.checkpointGames == 0) {
                if (!unchunked.empty()) {
                    distill::write_chunk(
                        checkpointDirectory / checkpoint_file_name(unchunkedFirst), unchunked);
                }
                unchunked.clear();
                unchunkedFirst = done;
                save_progress(done);
            }
        }
        std::cout << "selfplay game " << (gameNumber + 1) << '/' << config.games
                  << " raw " << game.rawPolicyMacroPlies
                  << " samples " << sampleCount
                  << " events " << game.pgnMoves.size()
                  << " termination " << game.termination;
        if (game.seats) {
            const SeatClockSummary& seats = *game.seats;
            std::cout << " lead " << seats.start.whiteLeadMs << "->"
                      << seats.finalWhiteLeadMs << " ms";
            if (seats.flagSeat >= 0) {
                std::cout << " flag-seat " << seat_name(seats.flagSeat);
            }
            std::cout << " clocks " << four(seats.finalClocks);
        } else if (game.initialTime.is_clocked()) {
            std::cout << " lead " << game.initialTime.lead_ms(Stockfish::WHITE)
                      << "->" << game.finalTime.lead_ms(Stockfish::WHITE) << " ms";
        }
        if (game.wouldResignTeam >= 0) {
            // Right when the team the rule picked went on to lose.
            const bool right = game.winner >= 0 && game.winner != game.wouldResignTeam;
            ++resignChecks;
            wrongResignations += right ? 0 : 1;
            std::cout << " resign-check " << (right ? "right" : "wrong");
        }
        // Piped to `hivemind contribute`, which counts these lines.
        std::cout << '\n' << std::flush;
    };
    auto playGames = [&](const std::vector<Engine*>& threadEngines) {
        std::vector<__half> distillPlanes(NB_INPUT_VALUES());
        Engine& rawPolicyEngine = *threadEngines.front();
        RawPolicyEvaluator rawPolicyEvaluator(
            static_cast<size_t>(rawPolicyEngine.getBatchSize()));
        // One agent per team in each game thread: their search workers and GC
        // threads outlive the game, and each game starts them from empty
        // trees. With tree reuse a team's next search continues its own tree,
        // through its move and the reply: the tree's terminal verdicts (mate
        // races, waiting-board mates) are judged for its root team, so a tree
        // is never handed to the other team.
        Agent agents[2];

        for (;;) {
            const size_t gameIndex = nextGame.fetch_add(1);
            if (gameIndex >= config.games) {
                break;
            }
            std::mt19937_64 randomEngine(selfplay_seed(runId, gameIndex));
            Board board;
            // Per team: the position its last search was rooted at and the
            // moves played since (its own, then the reply).
            std::array<std::optional<Board>, 2> lineRoots;
            std::array<std::vector<JointActionCandidate>, 2> lines;
            for (Agent& teamAgent : agents) {
                teamAgent.reset_search_state();
            }
            const int startingTeam = static_cast<int>(randomEngine() & 1ULL);
            Stockfish::Color team = startingTeam == 0 ? Stockfish::WHITE : Stockfish::BLACK;
            // The flag model lets the team that does not start sit all game.
            // Only the clock model draws from the random stream here, so flag
            // games replay exactly as before.
            TimeState time = config.clockModel
                ? sample_clock(config.clock, randomEngine)
                : TimeState::legacy(team, false);
            if (config.clockModel && !config.wave2.sitGate) {
                // SitGate off: a pass is always legal, in the game and the tree.
                time = time.with_sit_gate(false);
            }
            const TimeState initialTime = time;
            int winner = -1;
            std::string termination = "macro-ply limit";
            std::vector<TrainingSample> samples;
            std::vector<distill::Record> distillRecords;
            std::vector<PgnMove> pgnMoves;
            std::array<int, 2> moveNumbers = {1, 1};
            const size_t initializationLength = sample_initialization_length(config, randomEngine);
            bool rawInitializationActive = initializationLength > 0;
            size_t rawPolicyMacroPlies = 0;
            size_t rawPolicyEvents = 0;
            const bool canResign = config.resignThreshold < 0.0f
                && (config.resignDisableFraction <= 0.0
                    || std::uniform_real_distribution<double>(0.0, 1.0)(randomEngine) >= config.resignDisableFraction);
            std::array<size_t, 2> consecutiveResignPlies = {0, 0};
            // A game exempt from resigning still runs the rule, to check it.
            const bool checksResign = !canResign && config.resignThreshold < 0.0f;
            int wouldResignTeam = -1;

            std::optional<SeatClockSummary> seats;
            if (config.wave2.seat_clocks()) {
                SeatClockGameResult played = play_seat_clock_game(
                    config, threadEngines, agents, rawPolicyEvaluator, distillPlanes,
                    distillWriter != nullptr, runId, gameIndex, randomEngine);
                samples = std::move(played.samples);
                distillRecords = std::move(played.distillRecords);
                pgnMoves = std::move(played.pgnMoves);
                winner = played.winner;
                termination = std::move(played.termination);
                rawPolicyMacroPlies = played.rawPolicyMacroPlies;
                rawPolicyEvents = played.rawPolicyEvents;
                wouldResignTeam = played.wouldResignTeam;
                searches.fetch_add(samples.size(), std::memory_order_relaxed);
                seats = std::move(played.seats);
            }

            for (size_t macroPly = 0; !seats && macroPly < config.maxMacroPlies; ++macroPly) {
                const bool hasTimeAdvantage = time.can_sit(team);
                const GameStatus status = adjudicate_game(board, team, time);
                if (status != GameStatus::ONGOING) {
                    winner = adjudicated_winner(status, team);
                    termination = adjudicated_termination(status);
                    break;
                }

                if (rawInitializationActive && macroPly < initializationLength) {
                    const JointActionCandidate rawAction = sample_raw_policy_action(
                        rawPolicyEngine, rawPolicyEvaluator, board, team, time,
                        sample_raw_policy_temperature(config, randomEngine), randomEngine);
                    if (!action_leads_to_terminal(board, rawAction, team, time)) {
                        time.advance(team, board.on_turn_on_both_boards(team));
                        if (rawAction.moveA != Stockfish::MOVE_NONE) {
                            append_pgn_move(
                                board, BOARD_A, rawAction.moveA, moveNumbers, pgnMoves.size(),
                                config.initialClockSeconds, pgnMoves);
                            board.push_move(BOARD_A, rawAction.moveA);
                        }
                        if (rawAction.moveB != Stockfish::MOVE_NONE) {
                            append_pgn_move(
                                board, BOARD_B, rawAction.moveB, moveNumbers, pgnMoves.size(),
                                config.initialClockSeconds, pgnMoves);
                            board.push_move(BOARD_B, rawAction.moveB);
                        }
                        rawPolicyMacroPlies++;
                        rawPolicyEvents = pgnMoves.size();
                        team = ~team;
                        continue;
                    }
                    rawInitializationActive = false;
                }

                TrainingSample sample;
                sample.gameId = gameIndex;
                sample.macroPly = static_cast<uint16_t>(std::min<size_t>(
                    macroPly, std::numeric_limits<uint16_t>::max()));
                sample.team = team == Stockfish::WHITE ? 0 : 1;
                sample.hasTimeAdvantage = hasTimeAdvantage ? 1 : 0;
                if (config.writeTrainingChunks) {
                    sample.planes = encode_planes(board, team, time);
                }

                const size_t teamIndex = team == Stockfish::WHITE ? 0 : 1;
                Agent& agent = agents[teamIndex];
                if (!config.treeReuse) {
                    agent.reset_search_state();
                } else if (lineRoots[teamIndex]) {
                    retainedVisits.fetch_add(
                        static_cast<uint64_t>(std::max(
                            0, agent.retain_played_line(*lineRoots[teamIndex], lines[teamIndex]))),
                        std::memory_order_relaxed);
                }
                SearchOptions searchOptions;
                searchOptions.targetNodes = randomized_node_budget(config, randomEngine);
                // The moves played are sampled, so the subtree worth keeping is
                // the one they lead to, retained before this team's next search.
                searchOptions.retainForReuse = false;
                if (config.treeReuse) {
                    searchOptions.reusedVisitsCount = true;
                    searchOptions.minNewNodes = std::max<size_t>(1, static_cast<size_t>(std::ceil(
                        config.treeReuseMinNewFraction
                        * static_cast<double>(searchOptions.targetNodes))));
                }
                searchOptions.search.enableMateProbe = config.fairyStockfishMateNodes > 0;
                searchOptions.mateProbeNodes = config.fairyStockfishMateNodes;
                searchOptions.completeMateProbe = config.fairyStockfishMateNodes > 0;
                // Self-play policy targets and temperature sampling are visit-based.
                // Keep that data-generation contract until it has a dedicated
                // Gumbel-improved policy target rather than biased halving visits.
                searchOptions.search.enableGumbelRootSearch = false;
                if (!config.rootScans) {
                    searchOptions.search.enableRootMateSearch = false;
                    searchOptions.search.certifySelectedMove = false;
                }
                searchOptions.search.rootDirichletAlpha = config.dirichletAlpha;
                searchOptions.search.rootDirichletEpsilon = config.dirichletEpsilon;
                searchOptions.search.rootNoiseSeed = selfplay_seed(runId, gameIndex * config.maxMacroPlies + macroPly);
                if (config.clockModel) {
                    searchOptions.clock = time;
                    if (config.wave2.durativeTree) {
                        // The durative tree on the lead-only clock (no
                        // --base-mix): seat clocks derived from the lead.
                        DurativeConfig durative = config.wave2.durative_config();
                        durative.rootMoveTimeMs = time.next_move_time_ms(team);
                        searchOptions.clock = TimeState::durative(
                            time, team, board.side_to_move(BOARD_A),
                            board.side_to_move(BOARD_B), durative);
                    }
                }
                const JointActionCandidate choice = agent.run_search(
                    board, threadEngines, team, hasTimeAdvantage, searchOptions);
                std::vector<RootEdgeStats> edges = agent.root_edge_stats();
                if (agent.search_status() != GameStatus::ONGOING) {
                    winner = adjudicated_winner(agent.search_status(), team);
                    termination = adjudicated_termination(agent.search_status());
                    break;
                }
                if (edges.empty()) {
                    throw std::runtime_error("Ongoing search returned no root edges");
                }
                const uint64_t actualVisits = std::accumulate(
                    edges.begin(), edges.end(), uint64_t{0},
                    [](uint64_t total, const RootEdgeStats& edge) {
                        return total + static_cast<uint64_t>(std::max(0, edge.visits));
                    });
                sample.nodes = static_cast<uint32_t>(std::min<uint64_t>(
                    actualVisits, std::numeric_limits<uint32_t>::max()));
                searches.fetch_add(1, std::memory_order_relaxed);
                searchedVisits.fetch_add(actualVisits, std::memory_order_relaxed);

                edges = selfplay_policy_edges(edges, agent.root_type(), choice, agent.certified_choice());
                const float rootQ = agent.root_q();
                sample.rootQ = rootQ;
                if (config.writeTrainingChunks) {
                    sample.policyA = marginal_policy(board, BOARD_A, edges);
                    sample.policyB = marginal_policy(board, BOARD_B, edges);
                    sample.jointPolicy = joint_policy(board, edges);
                }
                if (distillWriter) {
                    // WDL stays zero: search has a value but no WDL distribution,
                    // so these records carry no WDL or moves-left target.
                    distill::Record record;
                    board_to_planes(board, distillPlanes.data(), team, time);
                    distill::pack_planes(distillPlanes.data(), record);
                    record.value = rootQ;
                    record.ply = sample.macroPly;
                    record.policy[0] = distill_search_policy(
                        board, BOARD_A, board.side_to_move(BOARD_A) == team, edges);
                    record.policy[1] = distill_search_policy(
                        board, BOARD_B, board.side_to_move(BOARD_B) == ~team, edges);
                    distillRecords.push_back(std::move(record));
                }
                samples.push_back(std::move(sample));

                if (checksResign && wouldResignTeam < 0) {
                    consecutiveResignPlies[teamIndex] = rootQ <= config.resignThreshold
                        ? consecutiveResignPlies[teamIndex] + 1 : 0;
                    if (consecutiveResignPlies[teamIndex] >= config.resignConsecutivePlies) {
                        wouldResignTeam = team == Stockfish::WHITE ? 0 : 1;
                    }
                }
                if (canResign) {
                    if (rootQ <= config.resignThreshold) {
                        consecutiveResignPlies[teamIndex]++;
                        if (consecutiveResignPlies[teamIndex] >= config.resignConsecutivePlies) {
                            winner = team == Stockfish::WHITE ? 1 : 0;
                            termination = "resignation";
                            break;
                        }
                    } else {
                        consecutiveResignPlies[teamIndex] = 0;
                    }
                }

                const JointActionCandidate action = select_action(
                    edges, mcts_temperature(config, macroPly), randomEngine);
                if (config.treeReuse) {
                    lineRoots[teamIndex].emplace(board);
                    lines[teamIndex] = {action};
                    if (lineRoots[1 - teamIndex]) {
                        lines[1 - teamIndex].push_back(action);
                    }
                }
                time.advance(team, board.on_turn_on_both_boards(team));
                if (action.moveA == Stockfish::MOVE_NONE && action.moveB == Stockfish::MOVE_NONE) {
                    team = ~team;
                    continue;
                }

                if (action.moveA != Stockfish::MOVE_NONE) {
                    append_pgn_move(
                        board, BOARD_A, action.moveA, moveNumbers, pgnMoves.size(),
                        config.initialClockSeconds, pgnMoves);
                    board.push_move(BOARD_A, action.moveA);
                }
                if (action.moveB != Stockfish::MOVE_NONE) {
                    append_pgn_move(
                        board, BOARD_B, action.moveB, moveNumbers, pgnMoves.size(),
                        config.initialClockSeconds, pgnMoves);
                    board.push_move(BOARD_B, action.moveB);
                }

                team = ~team;
            }

            const GameStatus finalStatus = seats
                ? GameStatus::ONGOING : adjudicate_game(board, team, time);
            if (winner < 0 && finalStatus != GameStatus::ONGOING) {
                winner = adjudicated_winner(finalStatus, team);
                termination = adjudicated_termination(finalStatus);
            }

            for (size_t index = 0; index < samples.size(); ++index) {
                TrainingSample& sample = samples[index];
                sample.outcome = winner < 0 ? 0 : (sample.team == winner ? 1 : -1);
                sample.wdl = static_cast<uint8_t>(sample.outcome + 1);
                sample.movesLeft = static_cast<uint16_t>(std::min<size_t>(
                    samples.size() - index, std::numeric_limits<uint16_t>::max()));
            }
            // Hand the game to the writer; games are written in index order, but a
            // finished thread starts its next game instead of waiting for earlier
            // (possibly much longer) games to finish.
            const std::lock_guard<std::mutex> lock(outputMutex);
            if (failure) {
                return;
            }
            finishedGames.emplace(gameIndex, FinishedGame{
                std::move(samples), std::move(distillRecords), std::move(pgnMoves), winner,
                startingTeam, rawPolicyMacroPlies, rawPolicyEvents, std::move(termination),
                initialTime, time, wouldResignTeam, std::move(seats)});
            for (auto next = finishedGames.find(nextOutput); next != finishedGames.end();
                 next = finishedGames.find(nextOutput)) {
                writeGame(next->first, next->second);
                finishedGames.erase(next);
                ++nextOutput;
            }
        }
    };

    if (gameThreads == 1) {
        playGames(engines);
    } else {
        std::vector<std::thread> threads;
        for (size_t threadIndex = 0; threadIndex < gameThreads; ++threadIndex) {
            threads.emplace_back([&, threadIndex] {
                try {
                    playGames({engines[threadIndex]});
                } catch (...) {
                    const std::lock_guard<std::mutex> lock(outputMutex);
                    if (!failure) {
                        failure = std::current_exception();
                    }
                    nextGame = config.games;  // stop the other threads
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

    chunkWriter.finish();
    if (distillWriter) {
        distillWriter->flush();
    }
    if (checkpointing) {
        std::filesystem::remove_all(checkpointDirectory);
    }
    if (config.treeReuse && searchedVisits > 0) {
        std::cout << "selfplay tree reuse: " << std::fixed << std::setprecision(1)
                  << 100.0 * static_cast<double>(retainedVisits)
                         / static_cast<double>(searchedVisits)
                  << "% of root visits carried over, " << searches << " searches" << '\n';
    }
    if (resignChecks > 0) {
        std::cout << "selfplay resign check: " << resignChecks
                  << " games exempt from resigning would have resigned, "
                  << wrongResignations << " of them wrongly" << '\n';
    }
    // `hivemind rl-loop` checks this line rather than counting game lines,
    // which a resumed run prints only for the games it played.
    std::cout << "selfplay complete " << config.games << " games" << std::endl;
    return 0;
}
