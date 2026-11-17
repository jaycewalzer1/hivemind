// ONNX Runtime inference backend.
//
// A drop-in replacement for the TensorRT backend in engine.cc that runs the
// same exported graph on any ONNX Runtime execution provider. It exists so the
// engine can be built and shipped without CUDA: TensorRT plus its per-SM
// builder resources is ~2 GB and NVIDIA-only, while ORT's CPU build is ~28 MB
// and runs everywhere.
//
// `enqueue`/`synchronize` overlap is preserved with a worker thread per
//     search thread, which is what keeps batch N's inference running while the
//     search walks the tree for batch N+1.

#include "nn/engine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <future>
#include <stdexcept>

#include "environment/constants.h"
#include "search/search_params.h"

namespace {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hivemind");
    return env;
}

std::string lowered(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name;
}

// The exported graph names its heads differently across training runs
// (`pi_a` / `policy_a`, `wdl_out` / `wdl`, ...), so match on a normalized
// substring rather than an exact string.
bool matches(const std::string& normalized,
             std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (normalized.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

struct Engine::OrtState {
    // One in-flight request per search thread.
    struct Worker {
        std::vector<__half> inputHalf;
        std::vector<float> inputFloat;
        std::vector<Ort::Value> outputs;
        std::vector<std::vector<__half>> convertedOutputs;
        std::future<bool> pending;
        bool hasPending = false;
    };

    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Worker> workers;

    // Names owned here; the C API wants stable `const char*`.
    std::vector<std::string> outputNameStorage;
    std::vector<const char*> outputNames;
    std::array<const char*, 1> inputNames{nullptr};
    bool usesFp16 = true;
    std::vector<ONNXTensorElementDataType> outputTypes;

    // Index into `outputs` for each head, or -1 when the graph omits it.
    int valueIdx = -1;
    int policyAIdx = -1;
    int policyBIdx = -1;
    int wdlIdx = -1;
    int movesLeftIdx = -1;
    int jointFactorsAIdx = -1;
    int jointFactorsBIdx = -1;
    int thinkGainIdx = -1;

    // Channels per board the network reads. Equal to the build's
    // NB_INPUT_CHANNELS_PER_BOARD, except a v1 clock network (41) in the
    // 43-channel clock build, whose input drops each block's last 2 planes.
    int netChannelsPerBoard = NB_INPUT_CHANNELS_PER_BOARD;
    bool dynamicBatch = false;

    size_t net_values_per_row() const {
        return static_cast<size_t>(2 * netChannelsPerBoard) * BOARD_HEIGHT * BOARD_WIDTH;
    }

    // Copies `rows` rows of build-layout planes into network layout.
    template <typename Src, typename Dst, typename Convert>
    void pack_rows(const Src* src, Dst* dst, size_t rows, Convert convert) const {
        constexpr size_t planeSize = BOARD_HEIGHT * BOARD_WIDTH;
        const size_t srcBlock = NB_INPUT_CHANNELS_PER_BOARD * planeSize;
        const size_t dstBlock = static_cast<size_t>(netChannelsPerBoard) * planeSize;
        for (size_t block = 0; block < rows * 2; ++block) {
            std::transform(src + block * srcBlock, src + block * srcBlock + dstBlock,
                           dst + block * dstBlock, convert);
        }
    }
};

const char* Engine::backendName() { return "ONNX Runtime (CPU)"; }

Engine::Engine(int deviceId, int batchSize)
    : m_deviceId(deviceId),
      m_batchSize(batchSize > 0 ? batchSize : SearchParams::BATCH_SIZE),
      m_ort(std::make_unique<OrtState>()) {}

Engine::~Engine() {
    if (!m_ort) return;
    // Drain anything still running so worker threads never outlive the session.
    for (auto& worker : m_ort->workers) {
        if (worker.hasPending && worker.pending.valid()) {
            try {
                worker.pending.wait();
            } catch (...) {
            }
            worker.hasPending = false;
        }
    }
}

bool Engine::loadNetwork(const std::string& onnxFile,
                         const std::string& /*engineFile*/) {
    try {
        auto& state = *m_ort;

        state.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Concurrent Run calls share this session's intra-op pool. Let ORT
        // choose its topology-aware default instead of dividing the one shared
        // pool by the number of search workers and leaving cores idle.
        const int intraOp = 0;
        state.options.SetIntraOpNumThreads(intraOp);
        state.options.SetInterOpNumThreads(1);
        state.options.SetExecutionMode(ORT_SEQUENTIAL);

        const std::filesystem::path modelPath(onnxFile);
        state.session = std::make_unique<Ort::Session>(
            ort_env(), modelPath.c_str(), state.options);

        Ort::AllocatorWithDefaultOptions allocator;

        if (state.session->GetInputCount() != 1) {
            std::cerr << "Expected exactly one network input, found "
                      << state.session->GetInputCount() << std::endl;
            return false;
        }

        auto inputName = state.session->GetInputNameAllocated(0, allocator);
        m_inputName = inputName.get();

        const auto inputType = state.session->GetInputTypeInfo(0)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetElementType();
        if (inputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            inputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "Unsupported network input type: " << inputType << std::endl;
            return false;
        }
#if !defined(HIVEMIND_ORT_FP16)
        if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            std::cerr << "This is an FP16 network. Rebuild with "
                         "-DHIVEMIND_ORT_FP16=ON to use it directly."
                      << std::endl;
            return false;
        }
#endif
        state.usesFp16 = inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        if (state.usesFp16) {
            std::cout
                << "info string warning FP16 inference is usually slow on CPU; "
                   "convert the model with engine/scripts/convert_onnx_fp32.py"
                << std::endl;
        }

        // The network's input channels: the build's layout, or a v1 clock
        // network (41 per board) in the 43-channel clock build.
        {
            const auto inputShape = state.session->GetInputTypeInfo(0)
                                        .GetTensorTypeAndShapeInfo()
                                        .GetShape();
            const int64_t channels = inputShape.size() == 4 ? inputShape[1] : -1;
            state.dynamicBatch = inputShape.size() == 4 && inputShape[0] <= 0;
            if (channels == NB_INPUT_CHANNELS || channels <= 0) {
                state.netChannelsPerBoard = NB_INPUT_CHANNELS_PER_BOARD;
            } else if (HAS_ELAPSED_PLANES
                       && channels == 2 * NB_V1_CLOCK_INPUT_CHANNELS_PER_BOARD) {
                state.netChannelsPerBoard = NB_V1_CLOCK_INPUT_CHANNELS_PER_BOARD;
                std::cout << "info string v1 clock network (" << channels
                          << " input channels): elapsed planes are not given to it"
                          << std::endl;
            } else {
                std::cerr << "Network expects " << channels
                          << " input channels; this build makes "
                          << NB_INPUT_CHANNELS << std::endl;
                return false;
            }
        }

        // Classify every output, then run only the ones the engine reads:
        // extra heads such as 'arrivals' (spec v2 C) are training targets.
        const size_t graphOutputCount = state.session->GetOutputCount();
        std::vector<std::string> graphNames;
        std::vector<ONNXTensorElementDataType> graphTypes;
        for (size_t i = 0; i < graphOutputCount; ++i) {
            auto name = state.session->GetOutputNameAllocated(i, allocator);
            graphNames.emplace_back(name.get());
            graphTypes.push_back(state.session->GetOutputTypeInfo(i)
                                     .GetTensorTypeAndShapeInfo()
                                     .GetElementType());
        }
        auto use_output = [&](size_t graphIndex, int& slot, std::string& name) -> bool {
            const auto type = graphTypes[graphIndex];
            if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                std::cerr << "Unsupported network output type for "
                          << graphNames[graphIndex] << ": " << type << std::endl;
                return false;
            }
            slot = static_cast<int>(state.outputNameStorage.size());
            state.outputNameStorage.push_back(graphNames[graphIndex]);
            state.outputTypes.push_back(type);
            name = graphNames[graphIndex];
            return true;
        };
        for (size_t i = 0; i < graphOutputCount; ++i) {
            const std::string norm = lowered(graphNames[i]);
            bool ok = true;
            // Order matters: "jointfactors_a" also contains "_a".
            if (matches(norm, {"think_gain", "thinkgain"})) {
                const auto shape = state.session->GetOutputTypeInfo(i)
                                       .GetTensorTypeAndShapeInfo()
                                       .GetShape();
                if (shape.size() == 2 && shape[1] == 2) {
                    ok = use_output(i, state.thinkGainIdx, m_thinkGainName);
                } else {
                    std::cout << "info string think_gain output is not [batch, 2];"
                              << " ignored" << std::endl;
                }
            } else if (matches(norm, {"arrival"})) {
                // Training target only; the engine never reads it.
            } else if (matches(norm, {"jointfactor", "joint_factor"})) {
                if (matches(norm, {"_a", "a_"}) || norm.back() == 'a') {
                    ok = use_output(i, state.jointFactorsAIdx, m_jointFactorsAName);
                } else {
                    ok = use_output(i, state.jointFactorsBIdx, m_jointFactorsBName);
                }
            } else if (matches(norm, {"wdl"})) {
                ok = use_output(i, state.wdlIdx, m_wdlName);
            } else if (matches(norm, {"moves_left", "movesleft"})) {
                ok = use_output(i, state.movesLeftIdx, m_movesLeftName);
            } else if (matches(norm, {"pi_a", "policy_a", "policya"})) {
                ok = use_output(i, state.policyAIdx, m_policyAName);
            } else if (matches(norm, {"pi_b", "policy_b", "policyb"})) {
                ok = use_output(i, state.policyBIdx, m_policyBName);
            } else if (matches(norm, {"value"})) {
                ok = use_output(i, state.valueIdx, m_valueName);
            } else {
                // Unknown heads used to be fetched and never read; now
                // they are simply not computed.
            }
            if (!ok) {
                return false;
            }
        }
        for (const auto& name : state.outputNameStorage) {
            state.outputNames.push_back(name.c_str());
        }
        state.inputNames[0] = m_inputName.c_str();
        const size_t outputCount = state.outputNameStorage.size();

        if (state.valueIdx < 0 || state.policyAIdx < 0 || state.policyBIdx < 0) {
            std::cerr << "Network is missing a required head (value/pi_a/pi_b)."
                      << std::endl;
            return false;
        }

        // Joint-factor rank comes from the head's own shape when present.
        m_jointFactorRank = 0;
        if (state.jointFactorsAIdx >= 0) {
            const auto shape =
                state.session->GetOutputTypeInfo(
                        static_cast<size_t>(std::find(graphNames.begin(), graphNames.end(),
                                                      m_jointFactorsAName)
                                            - graphNames.begin()))
                    .GetTensorTypeAndShapeInfo()
                    .GetShape();
            // [batch, rank, vocabulary]
            if (shape.size() == 3 && shape[1] > 0) {
                m_jointFactorRank = static_cast<size_t>(shape[1]);
            }
        }

        const int workerCount = std::max(1, SearchParams::NUM_SEARCH_THREADS);
        state.workers.resize(workerCount);
        for (auto& worker : state.workers) {
            const size_t inputElements =
                static_cast<size_t>(m_batchSize) * state.net_values_per_row();
            if (state.usesFp16) {
                worker.inputHalf.resize(inputElements);
            } else {
                worker.inputFloat.resize(inputElements);
            }
            worker.convertedOutputs.resize(outputCount);
        }

        std::cout << "info string backend " << backendName() << " model "
                  << onnxFile << " batch " << m_batchSize << " workers "
                  << workerCount << " precision "
                  << (state.usesFp16 ? "fp16" : "fp32")
                  << " intra-op threads "
                  << (intraOp == 0 ? "auto" : std::to_string(intraOp))
                  << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime failed to load the network: " << e.what()
                  << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load the network: " << e.what() << std::endl;
        return false;
    }
}

bool Engine::enqueueInferenceHalf(const __half* obs, size_t workerIndex) {
    if (!obs || !m_ort || !m_ort->session ||
        workerIndex >= m_ort->workers.size()) {
        return false;
    }
    auto& worker = m_ort->workers[workerIndex];
    if (worker.hasPending) {
        // One request in flight per worker, per the header contract.
        return false;
    }

    // Copy rather than alias the caller's buffer: the search reuses its
    // double-buffered observation slabs as soon as it has enqueued.
    const bool sameLayout = m_ort->netChannelsPerBoard == NB_INPUT_CHANNELS_PER_BOARD;
    if (m_ort->usesFp16) {
        if (sameLayout) {
            std::memcpy(worker.inputHalf.data(), obs,
                        worker.inputHalf.size() * sizeof(__half));
        } else {
            m_ort->pack_rows(obs, worker.inputHalf.data(),
                             static_cast<size_t>(m_batchSize),
                             [](__half v) { return v; });
        }
    } else if (sameLayout) {
        std::transform(obs, obs + worker.inputFloat.size(),
                       worker.inputFloat.begin(), __half2float);
    } else {
        m_ort->pack_rows(obs, worker.inputFloat.data(),
                         static_cast<size_t>(m_batchSize), __half2float);
    }

    auto& state = *m_ort;
    const int64_t batch = m_batchSize;
    worker.pending = std::async(std::launch::async, [&state, &worker, batch]() {
        try {
            const std::array<int64_t, 4> shape{
                batch, 2 * state.netChannelsPerBoard, BOARD_HEIGHT, BOARD_WIDTH};
            Ort::Value input{nullptr};
            if (state.usesFp16) {
                input = Ort::Value::CreateTensor(
                    state.memoryInfo, worker.inputHalf.data(),
                    worker.inputHalf.size() * sizeof(__half),
                    shape.data(), shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            } else {
                input = Ort::Value::CreateTensor<float>(
                    state.memoryInfo, worker.inputFloat.data(),
                    worker.inputFloat.size(), shape.data(), shape.size());
            }

            worker.outputs = state.session->Run(
                Ort::RunOptions{nullptr}, state.inputNames.data(), &input, 1,
                state.outputNames.data(), state.outputNames.size());
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime inference failed: " << e.what()
                      << std::endl;
            return false;
        }
    });
    worker.hasPending = true;
    return true;
}

bool Engine::synchronizeInferenceHalf(HalfInferenceOutputs& outputs,
                                      size_t workerIndex) {
    if (!m_ort || workerIndex >= m_ort->workers.size()) return false;
    auto& worker = m_ort->workers[workerIndex];
    if (!worker.hasPending) return false;

    worker.hasPending = false;
    bool ok = false;
    try {
        ok = worker.pending.get();
    } catch (const std::exception& e) {
        std::cerr << "Inference worker threw: " << e.what() << std::endl;
        return false;
    }
    if (!ok) return false;

    auto& state = *m_ort;
    auto at = [&](int idx) -> const __half* {
        if (idx < 0 || static_cast<size_t>(idx) >= worker.outputs.size()) {
            return nullptr;
        }
        const size_t outputIndex = static_cast<size_t>(idx);
        if (state.outputTypes[outputIndex] ==
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return static_cast<const __half*>(
                worker.outputs[outputIndex].GetTensorRawData());
        }
        const auto* values = worker.outputs[outputIndex].GetTensorData<float>();
        auto& converted = worker.convertedOutputs[outputIndex];
        const size_t count = worker.outputs[outputIndex]
                                 .GetTensorTypeAndShapeInfo()
                                 .GetElementCount();
        converted.resize(count);
        std::transform(values, values + count, converted.begin(), __float2half_rn);
        return converted.data();
    };

    outputs.value = at(state.valueIdx);
    outputs.policyA = at(state.policyAIdx);
    outputs.policyB = at(state.policyBIdx);
    outputs.wdl = at(state.wdlIdx);
    outputs.movesLeft = at(state.movesLeftIdx);
    outputs.jointFactorsA = at(state.jointFactorsAIdx);
    outputs.jointFactorsB = at(state.jointFactorsBIdx);
    outputs.jointFactorRank = outputs.jointFactorsA ? m_jointFactorRank : 0;
    outputs.thinkGain = at(state.thinkGainIdx);
    return outputs.value && outputs.policyA && outputs.policyB;
}

bool Engine::evaluateThinkGain(const float* planes, float logits[2]) {
    if (!planes || !m_ort || !m_ort->session || m_ort->thinkGainIdx < 0) {
        return false;
    }
    auto& state = *m_ort;
    try {
        // One row when the graph's batch is dynamic, else a full batch of
        // copies. Only the think_gain output is requested, so ORT skips the
        // policy and value heads.
        const int64_t rows = state.dynamicBatch ? 1 : m_batchSize;
        const size_t rowValues = state.net_values_per_row();
        const std::array<int64_t, 4> shape{
            rows, 2 * state.netChannelsPerBoard, BOARD_HEIGHT, BOARD_WIDTH};
        std::vector<float> inputFloat(static_cast<size_t>(rows) * rowValues);
        state.pack_rows(planes, inputFloat.data(), 1, [](float v) { return v; });
        for (int64_t row = 1; row < rows; ++row) {
            std::copy_n(inputFloat.data(), rowValues, inputFloat.data() + row * rowValues);
        }
        std::vector<__half> inputHalf;
        Ort::Value input{nullptr};
        if (state.usesFp16) {
            inputHalf.resize(inputFloat.size());
            std::transform(inputFloat.begin(), inputFloat.end(), inputHalf.begin(),
                           __float2half_rn);
            input = Ort::Value::CreateTensor(
                state.memoryInfo, inputHalf.data(), inputHalf.size() * sizeof(__half),
                shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        } else {
            input = Ort::Value::CreateTensor<float>(
                state.memoryInfo, inputFloat.data(), inputFloat.size(),
                shape.data(), shape.size());
        }
        const char* outputName = state.outputNames[static_cast<size_t>(state.thinkGainIdx)];
        auto outputs = state.session->Run(Ort::RunOptions{nullptr},
                                          state.inputNames.data(), &input, 1,
                                          &outputName, 1);
        if (outputs.empty() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() < 2) {
            return false;
        }
        if (state.outputTypes[static_cast<size_t>(state.thinkGainIdx)]
            == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* values = static_cast<const __half*>(outputs[0].GetTensorRawData());
            logits[0] = __half2float(values[0]);
            logits[1] = __half2float(values[1]);
        } else {
            const auto* values = outputs[0].GetTensorData<float>();
            logits[0] = values[0];
            logits[1] = values[1];
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "think_gain inference failed: " << e.what() << std::endl;
        return false;
    }
}

bool Engine::runInferenceHalf(const __half* obs, HalfInferenceOutputs& outputs,
                              size_t workerIndex) {
    if (!enqueueInferenceHalf(obs, workerIndex)) return false;
    return synchronizeInferenceHalf(outputs, workerIndex);
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB,
                          float* wdl, float* movesLeft, size_t workerIndex) {
    const size_t inputElements =
        static_cast<size_t>(m_batchSize) * NB_INPUT_VALUES();
    std::vector<__half> halfInput(inputElements);
    std::transform(obs, obs + inputElements, halfInput.begin(), __float2half_rn);

    HalfInferenceOutputs outputs;
    if (!runInferenceHalf(halfInput.data(), outputs, workerIndex)) return false;

    const size_t batch = static_cast<size_t>(m_batchSize);
    auto copy = [](const __half* src, float* dst, size_t count) {
        if (src && dst) {
            std::transform(src, src + count, dst, __half2float);
        }
    };
    copy(outputs.value, value, batch);
    copy(outputs.policyA, piA, batch * NB_POLICY_VALUES());
    copy(outputs.policyB, piB, batch * NB_POLICY_VALUES());
    copy(outputs.wdl, wdl, batch * 3);
    copy(outputs.movesLeft, movesLeft, batch);
    return true;
}
