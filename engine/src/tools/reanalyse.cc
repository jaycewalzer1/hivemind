#include "tools/reanalyse.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "environment/board.h"
#include "environment/constants.h"
#include "environment/planes.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "tools/distill_data.h"
#include "tools/selfplay.h"
#include "tools/selfplay_policy.h"

namespace {

// ---- the state file's lines (written by selfplay's write_state_lines) ----

size_t field_start(const std::string& line, const char* key) {
    const std::string needle = std::string("\"") + key + "\": ";
    const size_t at = line.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error(std::string("State line lacks \"") + key + "\": "
                                 + line.substr(0, 120));
    }
    return at + needle.size();
}

uint64_t field_u64(const std::string& line, const char* key) {
    return std::stoull(line.substr(field_start(line, key), 24));
}

int64_t field_i64(const std::string& line, const char* key) {
    return std::stoll(line.substr(field_start(line, key), 24));
}

std::string field_string(const std::string& line, const char* key) {
    const size_t start = field_start(line, key);
    if (line.compare(start, 4, "null") == 0) {
        return {};
    }
    if (line[start] != '"') {
        throw std::runtime_error(std::string("State field ") + key + " is not a string");
    }
    return line.substr(start + 1, line.find('"', start + 1) - start - 1);
}

TimeState time_state_from_hex(const std::string& hex) {
    if (hex.size() != 2 * sizeof(TimeState)) {
        throw std::runtime_error("State clock has another size than this build's TimeState");
    }
    TimeState state;
    auto* bytes = reinterpret_cast<unsigned char*>(&state);
    for (size_t index = 0; index < sizeof(TimeState); ++index) {
        bytes[index] = static_cast<unsigned char>(std::stoi(hex.substr(2 * index, 2), nullptr, 16));
    }
    return state;
}

struct StateRecord {
    size_t record = 0;
    size_t sourceIndex = 0;  // position among all records of the run
    size_t prefix = 0;
    uint16_t ply = 0;
    int team = 0;
    bool timeAdvantage = false;
    bool clockSet = false;
    TimeState clock;
    std::optional<TimeState> hybridClock;
    uint64_t noiseSeed = 0;
    size_t baseNodes = 0;
    size_t targetNodes = 0;
    float epsilon = 0.0f;
    bool deep = false;
    int outcome = 0;
    std::array<float, 3> wdl{};
    uint16_t movesLeft = 0;
};

struct StateGame {
    size_t game = 0;
    std::vector<std::pair<int, int>> moves;
    std::vector<StateRecord> records;  // the selected ones
    size_t recordCount = 0;            // all of the game's records
};

std::vector<std::pair<int, int>> parse_moves(const std::string& line) {
    std::vector<std::pair<int, int>> moves;
    size_t at = field_start(line, "moves");
    if (line[at] != '[') {
        throw std::runtime_error("State game line has no moves list");
    }
    ++at;
    while (true) {
        at = line.find_first_of("[]", at);
        if (at == std::string::npos || line[at] == ']') {
            break;
        }
        size_t used = 0;
        const int boardNumber = std::stoi(line.substr(at + 1), &used);
        const size_t comma = line.find(',', at);
        const int move = std::stoi(line.substr(comma + 1), &used);
        moves.emplace_back(boardNumber, move);
        at = line.find(']', comma) + 1;
    }
    return moves;
}

StateRecord parse_record(const std::string& line) {
    StateRecord record;
    record.record = field_u64(line, "record");
    record.prefix = field_u64(line, "prefix");
    record.ply = static_cast<uint16_t>(field_u64(line, "ply"));
    record.team = static_cast<int>(field_i64(line, "team"));
    record.timeAdvantage = field_i64(line, "time_advantage") != 0;
    record.clockSet = field_i64(line, "clock_set") != 0;
    record.clock = time_state_from_hex(field_string(line, "clock"));
    const std::string hybrid = field_string(line, "hybrid_clock");
    if (!hybrid.empty()) {
        record.hybridClock = time_state_from_hex(hybrid);
    }
    record.noiseSeed = field_u64(line, "noise_seed");
    record.baseNodes = field_u64(line, "base_nodes");
    record.targetNodes = field_u64(line, "target_nodes");
    const uint32_t epsilonBits = static_cast<uint32_t>(field_u64(line, "epsilon_bits"));
    std::memcpy(&record.epsilon, &epsilonBits, sizeof(epsilonBits));
    record.deep = field_i64(line, "deep") != 0;
    record.outcome = static_cast<int>(field_i64(line, "outcome"));
    size_t at = field_start(line, "wdl") + 1;
    for (float& value : record.wdl) {
        size_t used = 0;
        value = static_cast<float>(std::stoi(line.substr(at), &used));
        at = line.find_first_of(",]", at) + 1;
    }
    record.movesLeft = static_cast<uint16_t>(field_u64(line, "moves_left"));
    return record;
}

// ---- the source run's HDST records, read in order ----

class SourceCursor {
public:
    explicit SourceCursor(std::vector<std::filesystem::path> files) : files_(std::move(files)) {}

    bool enabled() const { return !files_.empty(); }

    /// The record at run position `index`; positions must not go backwards.
    const distill::Record& at(size_t index) {
        while (index >= loadedFirst_ + loaded_.size()) {
            if (next_ >= files_.size()) {
                throw std::runtime_error("The source chunks hold fewer records than the state file");
            }
            loadedFirst_ += loaded_.size();
            loaded_ = distill::read_chunk(files_[next_++]);
        }
        if (index < loadedFirst_) {
            throw std::runtime_error("Source records read out of order");
        }
        return loaded_[index - loadedFirst_];
    }

private:
    std::vector<std::filesystem::path> files_;
    size_t next_ = 0;
    size_t loadedFirst_ = 0;
    std::vector<distill::Record> loaded_;
};

bool same_policy(const std::vector<distill::PolicyEntry>& left,
                 const std::vector<distill::PolicyEntry>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 0; index < left.size(); ++index) {
        if (left[index].index != right[index].index
            || left[index].probability != right[index].probability) {
            return false;
        }
    }
    return true;
}

struct Counts {
    uint64_t records = 0;    // in the state file
    uint64_t selected = 0;   // reanalysed and written
    uint64_t compared = 0;   // checked against a source record
    uint64_t planesEqual = 0;
    uint64_t policyEqual = 0;
    uint64_t valueEqual = 0;
    uint64_t bothEqual = 0;
    uint64_t visits = 0;
};

struct GameOutput {
    std::vector<distill::Record> records;
    std::vector<size_t> searchVisits;
};

GameOutput reanalyse_game(const ReanalyseConfig& config, size_t runNodes, const StateGame& game,
                          Engine& engine, Agent& agent, std::vector<__half>& planes) {
    GameOutput output;
    Board board;
    config.wave2.apply_board_rules(board);
    size_t played = 0;
    const std::vector<Engine*> engines = {&engine};
    for (const StateRecord& state : game.records) {
        if (state.prefix > game.moves.size() || state.prefix < played) {
            throw std::runtime_error("State record prefix out of range");
        }
        for (; played < state.prefix; ++played) {
            const auto [boardNumber, move] = game.moves[played];
            board.push_move(boardNumber, static_cast<Stockfish::Move>(move));
        }
        Board root(board);
        const Stockfish::Color team = state.team == 0 ? Stockfish::WHITE : Stockfish::BLACK;
        agent.reset_search_state();
        agent.clear_plans();
        SearchOptions options;
        options.targetNodes = config.nodes == 0
            ? state.targetNodes
            : std::max<size_t>(1, static_cast<size_t>(std::llround(
                  static_cast<double>(state.baseNodes) * static_cast<double>(config.nodes)
                  / static_cast<double>(runNodes))));
        options.retainForReuse = false;
        options.search.enableMateProbe = config.fairyStockfishMateNodes > 0;
        options.mateProbeNodes = config.fairyStockfishMateNodes;
        options.completeMateProbe = config.fairyStockfishMateNodes > 0;
        options.search.enableGumbelRootSearch = false;
        if (!config.rootScans) {
            options.search.enableRootMateSearch = false;
            options.search.certifySelectedMove = false;
        }
        options.search.rootDirichletAlpha = config.dirichletAlpha;
        options.search.rootDirichletEpsilon = config.replayNoise
            ? state.epsilon : (config.dirichlet ? config.dirichletEpsilon : 0.0f);
        options.search.rootNoiseSeed = state.noiseSeed;
        if (state.clockSet) {
            options.clock = state.clock;
        }
        if (config.wave2.hybrid.mode != HybridMode::OFF && state.hybridClock) {
            options.hybrid.knobs = config.wave2.hybrid;
            options.hybrid.clock = state.hybridClock;
        }
        const JointActionCandidate choice = agent.run_search(
            root, engines, team, state.timeAdvantage, options);
        std::vector<RootEdgeStats> edges = agent.root_edge_stats();
        if (agent.search_status() != GameStatus::ONGOING || edges.empty()) {
            throw std::runtime_error("A reanalysed root is no longer an ongoing search (game "
                                     + std::to_string(game.game) + " record "
                                     + std::to_string(state.record) + ")");
        }
        size_t visits = 0;
        for (const RootEdgeStats& edge : edges) {
            visits += static_cast<size_t>(std::max(0, edge.visits));
        }
        // Self-play's target edges: the hybrid's sit never is one.
        if (agent.hybrid_report().trigger.mask != 0) {
            edges = hybrid_policy_edges(edges, agent.root_type(), choice, agent.certified_choice());
        } else {
            edges = selfplay_policy_edges(edges, agent.root_type(), choice, agent.certified_choice());
        }
        distill::Record record;
        board_to_planes(root, planes.data(), team, state.clock);
        distill::pack_planes(planes.data(), record);
        record.value = agent.root_q();
        record.ply = state.ply;
        record.outcome = static_cast<int8_t>(state.outcome);
        record.wdl = state.wdl;
        record.movesLeft = state.movesLeft;
        record.policy[0] = selfplay_distill_policy(
            root, BOARD_A, root.side_to_move(BOARD_A) == team, edges);
        record.policy[1] = selfplay_distill_policy(
            root, BOARD_B, root.side_to_move(BOARD_B) == ~team, edges);
        output.records.push_back(std::move(record));
        output.searchVisits.push_back(visits);
    }
    return output;
}

}  // namespace

bool reanalyse_selected(uint64_t seed, uint64_t salt, size_t game, size_t record, double select) {
    if (select >= 1.0) {
        return true;
    }
    const uint64_t bits = selfplay_seed(0, selfplay_seed(seed ^ salt, (static_cast<uint64_t>(game) << 20) ^ record));
    return static_cast<double>(bits >> 11) * 0x1.0p-53 < select;
}

int run_reanalyse(const std::vector<Engine*>& engines, const ReanalyseConfig& config) {
    if (engines.empty()) {
        throw std::invalid_argument("reanalyse needs at least one engine");
    }
    if (!(config.select > 0.0 && config.select <= 1.0)) {
        throw std::invalid_argument("--select must be in (0, 1]");
    }
    std::ifstream states(config.statesFile);
    if (!states) {
        throw std::runtime_error("Cannot open " + config.statesFile.string());
    }
    std::string line;
    if (!std::getline(states, line) || line.find("\"type\": \"header\"") == std::string::npos) {
        throw std::runtime_error(config.statesFile.string() + " has no state header");
    }
    if (field_u64(line, "time_state_bytes") != sizeof(TimeState)
        || field_u64(line, "input_channels") != static_cast<uint64_t>(NB_INPUT_CHANNELS)) {
        throw std::runtime_error("The state file was written by a build with another TimeState or plane layout");
    }
    const uint64_t seed = field_u64(line, "seed");
    const size_t runNodes = field_u64(line, "nodes");
    if (std::filesystem::exists(config.outputDirectory)
        && !std::filesystem::is_empty(config.outputDirectory)) {
        throw std::invalid_argument("reanalyse output directory must be fresh");
    }
    std::filesystem::create_directories(config.outputDirectory);
    distill::ChunkWriter writer(config.outputDirectory, "reanalyse_" + std::to_string(seed),
                                config.chunkPositions);
    std::ofstream index(config.outputDirectory / "reanalysed.tsv");
    index << "game\trecord\tsource_index\n";
    SourceCursor source(config.sourceFiles);
    Counts counts;
    size_t sourceIndex = 0;

    std::vector<StateGame> batch;
    std::optional<StateGame> current;
    auto flush_batch = [&]() {
        std::vector<GameOutput> outputs(batch.size());
        std::atomic<size_t> next{0};
        std::mutex failureMutex;
        std::exception_ptr failure;
        const size_t threads = std::min(engines.size(), std::max<size_t>(1, batch.size()));
        auto work = [&](size_t thread) {
            Agent agent;
            std::vector<__half> planes(NB_INPUT_VALUES());
            try {
                for (size_t at = next.fetch_add(1); at < batch.size(); at = next.fetch_add(1)) {
                    outputs[at] = reanalyse_game(config, runNodes, batch[at], *engines[thread],
                                                 agent, planes);
                }
            } catch (...) {
                std::lock_guard lock(failureMutex);
                failure = failure ? failure : std::current_exception();
                next = batch.size();
            }
        };
        if (threads == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            for (size_t thread = 0; thread < threads; ++thread) {
                pool.emplace_back(work, thread);
            }
            for (std::thread& thread : pool) {
                thread.join();
            }
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        for (size_t at = 0; at < batch.size(); ++at) {
            const StateGame& game = batch[at];
            for (size_t r = 0; r < game.records.size(); ++r) {
                distill::Record& record = outputs[at].records[r];
                const StateRecord& state = game.records[r];
                counts.visits += outputs[at].searchVisits[r];
                if (source.enabled()) {
                    const distill::Record& original = source.at(state.sourceIndex);
                    ++counts.compared;
                    const bool planes = original.planeBits == record.planeBits
                        && original.planeValues == record.planeValues;
                    if (!planes) {
                        throw std::runtime_error(
                            "Recomputed planes differ from the source record (game "
                            + std::to_string(game.game) + " record " + std::to_string(state.record)
                            + "): the state file and source chunks do not belong together");
                    }
                    ++counts.planesEqual;
                    const bool policy = same_policy(original.policy[0], record.policy[0])
                        && same_policy(original.policy[1], record.policy[1]);
                    const bool value = original.value == record.value;
                    counts.policyEqual += policy ? 1 : 0;
                    counts.valueEqual += value ? 1 : 0;
                    counts.bothEqual += policy && value ? 1 : 0;
                }
                index << game.game << '\t' << state.record << '\t' << state.sourceIndex << '\n';
                writer.append(std::move(record));
                ++counts.selected;
            }
        }
        std::cout << "reanalyse: " << counts.selected << " records from "
                  << (batch.empty() ? 0 : batch.back().game + 1) << " games" << std::endl;
        batch.clear();
    };

    while (std::getline(states, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.find("\"type\": \"game\"") != std::string::npos) {
            if (current) {
                batch.push_back(std::move(*current));
                if (batch.size() >= config.gamesPerBatch * engines.size()) {
                    flush_batch();
                }
            }
            current.emplace();
            current->game = field_u64(line, "game");
            current->moves = parse_moves(line);
            continue;
        }
        if (!current) {
            throw std::runtime_error("State record before its game line");
        }
        if (field_u64(line, "game") != current->game) {
            throw std::runtime_error("State record of another game than the last game line");
        }
        ++counts.records;
        ++current->recordCount;
        const size_t thisIndex = sourceIndex++;
        const size_t recordNumber = field_u64(line, "record");
        if (!reanalyse_selected(seed, config.selectSalt, current->game, recordNumber, config.select)) {
            continue;
        }
        StateRecord record = parse_record(line);
        if (config.skipDeep && record.deep) {
            continue;
        }
        record.sourceIndex = thisIndex;
        current->records.push_back(std::move(record));
    }
    if (current) {
        batch.push_back(std::move(*current));
    }
    flush_batch();
    writer.flush();
    index.close();

    std::ofstream summary(config.outputDirectory / "reanalyse.json");
    summary << "{\"seed\": " << seed << ", \"run_nodes\": " << runNodes
            << ", \"nodes\": " << config.nodes << ", \"replay_noise\": "
            << (config.replayNoise ? "true" : "false") << ", \"select\": " << config.select
            << ", \"records\": " << counts.records << ", \"reanalysed\": " << counts.selected
            << ", \"visits\": " << counts.visits << ", \"compared\": " << counts.compared
            << ", \"planes_equal\": " << counts.planesEqual
            << ", \"policy_equal\": " << counts.policyEqual
            << ", \"value_equal\": " << counts.valueEqual
            << ", \"policy_and_value_equal\": " << counts.bothEqual << "}\n";
    std::cout << "reanalyse complete " << counts.selected << " of " << counts.records << " records";
    if (counts.compared > 0) {
        std::cout << ", fidelity: planes " << counts.planesEqual << '/' << counts.compared
                  << ", policy " << counts.policyEqual << ", value " << counts.valueEqual
                  << ", both " << counts.bothEqual;
    }
    std::cout << std::endl;
    return 0;
}
