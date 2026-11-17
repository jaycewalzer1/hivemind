#pragma once

// One self-play training sample and its HVM5 serialization. Shared by the
// self-play shard writer (selfplay.cc) and 'go ... sample' in the UCI loop, so
// both emit byte-identical sample records.

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "environment/board.h"
#include "environment/constants.h"
#include "search/agent.h"

namespace training {

constexpr std::array<char, 4> CHUNK_MAGIC = {'H', 'V', 'M', '5'};
constexpr uint32_t CHUNK_VERSION = 5;
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

template <typename T>
void write_scalar(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to write self-play chunk");
    }
}

/// Chunk header: magic, version, input channels, policy values, sample count.
void write_chunk_header(std::ostream& stream, uint64_t sampleCount);

/// One sample record exactly as it follows the header in an HVM5 chunk.
void write_sample(std::ostream& stream, const TrainingSample& sample);

/// write_sample into a byte string.
std::string serialize_sample(const TrainingSample& sample);

/// Network planes quantized to bytes: round(clamp(x, 0, 1) * 255).
std::array<uint8_t, NB_INPUT_VALUES()> encode_planes(
    Board& board, Stockfish::Color team, bool hasTimeAdvantage);

/// Root visit share per move on one board (pass included at its policy index).
std::vector<SparsePolicyEntry> marginal_policy(
    Board& board, int boardNumber, const std::vector<RootEdgeStats>& edges);

/// Root visit share per joint action; a pass is PASS_POLICY_INDEX.
std::vector<SparseJointPolicyEntry> joint_policy(
    Board& board, const std::vector<RootEdgeStats>& edges);

/// Standard base64 (RFC 4648 alphabet, '=' padding).
std::string base64_encode(const std::string& bytes);
/// Inverse of base64_encode; throws std::invalid_argument on malformed input.
std::string base64_decode(const std::string& text);

}  // namespace training
