#include "tools/training_sample.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "common/utils.h"
#include "environment/planes.h"

namespace training {

namespace {

void write_policy(std::ostream& stream, const std::vector<SparsePolicyEntry>& policy) {
    if (policy.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("Sparse policy is too large for HVM5");
    }
    write_scalar(stream, static_cast<uint16_t>(policy.size()));
    for (const SparsePolicyEntry& entry : policy) {
        write_scalar(stream, entry.index);
        write_scalar(stream, entry.probability);
    }
}

void write_joint_policy(std::ostream& stream,
                        const std::vector<SparseJointPolicyEntry>& policy) {
    if (policy.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("Sparse joint policy is too large for HVM5");
    }
    write_scalar(stream, static_cast<uint16_t>(policy.size()));
    for (const SparseJointPolicyEntry& entry : policy) {
        write_scalar(stream, entry.indexA);
        write_scalar(stream, entry.indexB);
        write_scalar(stream, entry.probability);
    }
}

int policy_index(Board& board, int boardNumber, Stockfish::Move move) {
    const int idx = get_fast_policy_index(move, board.side_to_move(boardNumber));
    if (idx < 0) {
        throw std::runtime_error("Move is absent from policy map: " + board.uci_move(boardNumber, move));
    }
    return idx;
}

uint16_t joint_policy_index(Board& board, int boardNumber, Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return PASS_POLICY_INDEX;
    }
    return static_cast<uint16_t>(policy_index(board, boardNumber, move));
}

constexpr char BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

}  // namespace

void write_chunk_header(std::ostream& stream, uint64_t sampleCount) {
    stream.write(CHUNK_MAGIC.data(), CHUNK_MAGIC.size());
    write_scalar(stream, CHUNK_VERSION);
    write_scalar(stream, static_cast<uint16_t>(NB_INPUT_CHANNELS));
    write_scalar(stream, static_cast<uint16_t>(NB_POLICY_VALUES()));
    write_scalar(stream, sampleCount);
}

void write_sample(std::ostream& stream, const TrainingSample& sample) {
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

std::string serialize_sample(const TrainingSample& sample) {
    std::ostringstream stream(std::ios::binary);
    write_sample(stream, sample);
    return stream.str();
}

std::array<uint8_t, NB_INPUT_VALUES()> encode_planes(
    Board& board, Stockfish::Color team, bool hasTimeAdvantage) {
    std::array<float, NB_INPUT_VALUES()> raw{};
    std::array<uint8_t, NB_INPUT_VALUES()> encoded{};
    board_to_planes(board, raw.data(), team, hasTimeAdvantage);
    for (size_t index = 0; index < raw.size(); ++index) {
        encoded[index] = static_cast<uint8_t>(std::lround(
            std::clamp(raw[index], 0.0f, 1.0f) * 255.0f));
    }
    return encoded;
}

std::vector<SparsePolicyEntry> marginal_policy(
    Board& board, int boardNumber, const std::vector<RootEdgeStats>& edges) {
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

std::string base64_encode(const std::string& bytes) {
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t index = 0;
    for (; index + 2 < bytes.size(); index += 3) {
        const uint32_t chunk = (static_cast<uint8_t>(bytes[index]) << 16)
            | (static_cast<uint8_t>(bytes[index + 1]) << 8)
            | static_cast<uint8_t>(bytes[index + 2]);
        out.push_back(BASE64_ALPHABET[(chunk >> 18) & 63]);
        out.push_back(BASE64_ALPHABET[(chunk >> 12) & 63]);
        out.push_back(BASE64_ALPHABET[(chunk >> 6) & 63]);
        out.push_back(BASE64_ALPHABET[chunk & 63]);
    }
    const size_t rest = bytes.size() - index;
    if (rest > 0) {
        uint32_t chunk = static_cast<uint8_t>(bytes[index]) << 16;
        if (rest == 2) {
            chunk |= static_cast<uint8_t>(bytes[index + 1]) << 8;
        }
        out.push_back(BASE64_ALPHABET[(chunk >> 18) & 63]);
        out.push_back(BASE64_ALPHABET[(chunk >> 12) & 63]);
        out.push_back(rest == 2 ? BASE64_ALPHABET[(chunk >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

std::string base64_decode(const std::string& text) {
    if (text.size() % 4 != 0) {
        throw std::invalid_argument("base64 length is not a multiple of 4");
    }
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    out.reserve(text.size() / 4 * 3);
    for (size_t index = 0; index < text.size(); index += 4) {
        const bool last = index + 4 == text.size();
        const int padding = last
            ? (text[index + 3] == '=' ? (text[index + 2] == '=' ? 2 : 1) : 0)
            : 0;
        uint32_t chunk = 0;
        for (int offset = 0; offset < 4; ++offset) {
            const char c = text[index + offset];
            int value = 0;
            if (offset >= 4 - padding) {
                if (c != '=') {
                    throw std::invalid_argument("bad base64 padding");
                }
            } else {
                value = value_of(c);
                if (value < 0) {
                    throw std::invalid_argument("bad base64 character");
                }
            }
            chunk = (chunk << 6) | static_cast<uint32_t>(value);
        }
        out.push_back(static_cast<char>((chunk >> 16) & 0xFF));
        if (padding < 2) {
            out.push_back(static_cast<char>((chunk >> 8) & 0xFF));
        }
        if (padding < 1) {
            out.push_back(static_cast<char>(chunk & 0xFF));
        }
    }
    return out;
}

}  // namespace training
