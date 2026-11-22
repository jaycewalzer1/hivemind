#include "interface/uci.h"
#include "nn/embedded_files.h"

#include <algorithm>
#include <cctype>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <random>

#include "nn/onnx_utils.h"
#include "environment/planes.h"
#include "common/utils.h"
#include "common/globals.h"
#include "search/mate_probe.h"
#include "search/single_board.h"

using namespace std;

UCI::UCI() : mainSearchThread(nullptr) {
    std::random_device entropy;
    const uint64_t clockSeed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::seed_seq seed{
        entropy(), entropy(), entropy(), entropy(),
        static_cast<unsigned int>(clockSeed),
        static_cast<unsigned int>(clockSeed >> 32)};
    openingNoiseGenerator.seed(seed);
}

UCI::~UCI() {
    stop();
}

bool UCI::initializeEngines(
    const std::vector<int>& deviceIdsToUse,
    const std::string& networkPathToUse,
    int batchSizeToUse) {
    deviceIds = deviceIdsToUse;
    networkPath = networkPathToUse;
    batchSize = batchSizeToUse > 0 ? batchSizeToUse : SearchParams::BATCH_SIZE;
    return reload_engines();
}

bool UCI::reload_engines() {
    stop();

    // Clear any existing engines.
    engines.clear();

    const std::string onnxFile = resolveModelPath(networkPath);
    if (onnxFile.empty()) {
        std::cerr << "Error: No ONNX model found; pass --model <onnx> or --network <onnx>." << std::endl;
        return false;
    }
    if (!embedded::is_model(onnxFile) && !std::filesystem::is_regular_file(onnxFile)) {
        std::cerr << "Error: ONNX model not found: " << onnxFile << std::endl;
        return false;
    }
    // For each device ID, create a new Engine, load the network, and store it.
    for (int deviceId : deviceIds) {
        const std::string engineFile = getEnginePath(onnxFile, "fp16", batchSize, deviceId, "v3");

        // Create a new engine instance on the given GPU.
        auto enginePtr = std::make_unique<Engine>(deviceId, batchSize);

        // Attempt to load the network (build or deserialize).
        if (!enginePtr->loadNetwork(onnxFile, engineFile)) {
            std::cerr << "Error: Failed to load engine on device " << deviceId << std::endl;
        } else {
            engines.push_back(std::move(enginePtr));
        }
    }

    networkReadsClock = !engines.empty()
        && (engines.front()->inputChannels() == NB_INPUT_CHANNELS
            || engines.front()->wideInputs());
    networkWide = !engines.empty() && engines.front()->wideInputs();

    // Create the single-threaded Agent
    agent = std::make_unique<Agent>();
    return !engines.empty();
}


void UCI::stop() {
    singleStop.store(true);
    singlePonder.store(false);
    if (agent) {
        agent->set_is_running(false);
    }
    if (mainSearchThread) {
        if (mainSearchThread->joinable()) {
            mainSearchThread->join();
        }
        delete mainSearchThread;
        mainSearchThread = nullptr;
    }
    ongoingSearch.store(false, std::memory_order_release);
}

void UCI::new_game() {
    stop();
    if (agent) {
        agent->reset_search_state();
        // Fix 3's plans live outside the tree; a new game starts without.
        agent->clear_plans();
    }
}

void UCI::position(istringstream& is) {
    stop();

    std::string token;
    is >> token;
    
    // Set the board position
    if (token == "startpos") {
        // Use a predefined starting FEN for the initial position.
        board.set(board.is_single_board() ? board.initial_fen()
            : board.startingFen + "|" + board.startingFen);
        is >> token;
    }
    else if (token == "fen") {
        // Build the FEN string until we hit "moves" or end of stream
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token + " ";
        }
        board.set(fen);
        
    }
    else {
        return;
    }
    
    if (token == "moves") {
        // Parse move list (if any)
        int moveCount = 0;
        while (is >> token) {
            if (!board.is_single_board() && (token.empty() || token[0] < '1' || token[0] > '2')) {
                std::cerr << "Error: Invalid board indicator in move '" << token 
                          << "' at move " << (moveCount + 1) << std::endl;
                break;
            }
            int boardNum = board.is_single_board() ? BOARD_A : token[0] - '1';
            std::string moveStr = board.is_single_board() ? token : token.substr(1);
            Stockfish::Move m = Stockfish::UCI::to_move(*board.pos[boardNum], moveStr);
            if (m == Stockfish::MOVE_NONE) {
                std::cerr << "Error: Invalid move '" << moveStr << "' on board " 
                          << (boardNum + 1) << " at move " << (moveCount + 1) << std::endl;
                std::cerr << "       Current FEN: " << board.fen(boardNum) << std::endl;
                std::cerr << "       Legal moves: ";
                auto legalMoves = board.legal_moves(boardNum);
                for (const auto& lm : legalMoves) {
                    std::cerr << board.uci_move(boardNum, lm) << " ";
                }
                std::cerr << std::endl;
                break;  // Stop if an invalid move is encountered.
            }
            board.push_move(boardNum, m);
            moveCount++;
        }
    }
}

void UCI::go(std::istringstream& is) {
    std::string token;
    int moveTime = 0;
    size_t nodes = 0;
    int depth = 0;
    bool infinite = false;
    bool isPonder = false;
    bool isBackground = false;
    int whiteTime = 0, blackTime = 0, whiteIncrement = 0, blackIncrement = 0;
    
    // Parse go parameters
    while (is >> token) {
        if (token == "ponder") {
            isPonder = true;
        } else if (token == "background") {
            isBackground = true;
        } else if (token == "movetime") {
            is >> moveTime;
        } else if (token == "nodes") {
            is >> nodes;
        } else if (token == "depth") {
            is >> depth;
        } else if (token == "infinite") {
            infinite = true;
        } else if (token == "wtime") {
            is >> whiteTime;
        } else if (token == "btime") {
            is >> blackTime;
        } else if (token == "winc") {
            is >> whiteIncrement;
        } else if (token == "binc") {
            is >> blackIncrement;
        }
    }
    
    stop();
    // Background work belongs to the bughouse MCTS tree.
    if (isBackground && (!backgroundSearchEnabled || board.is_single_board())) {
        return;
    }
    if (board.is_single_board()) {
        if (moveTime == 0 && nodes == 0 && !infinite && !isPonder) {
            const bool white = board.side_to_move(BOARD_A) == Stockfish::WHITE;
            const int remaining = white ? whiteTime : blackTime;
            const int increment = white ? whiteIncrement : blackIncrement;
            if (remaining > 0) {
                moveTime = std::clamp(remaining / 30 + increment * 3 / 4,
                                      1, std::max(1, remaining - 50));
            }
        }
        go_single_board(moveTime, nodes, depth, infinite, isPonder);
        return;
    }

    // Ensure that engines have been initialized.
    if (!agent || engines.empty()) {
        std::cerr << "Error: No engines have been initialized!" << std::endl;
        return;
    }

    ongoingSearch.store(true, std::memory_order_release);

    // Build a vector of raw Engine pointers from the unique_ptr collection.
    std::vector<Engine*> enginePtrs;
    enginePtrs.reserve(engines.size());
    for (const auto& eng : engines) {
        enginePtrs.push_back(eng.get());
    }

    // Build search options based on what was specified
    SearchOptions opts;
    if (nodes > 0) {
        opts = SearchOptions::uci(static_cast<int>(nodes), multiPV, isPonder);
        opts.moveTimeMs = 0;  // Node-based search
        opts.targetNodes = nodes;
    } else if (moveTime > 0) {
        opts = SearchOptions::uci(moveTime, multiPV, isPonder);
        opts.maxMoveTimeMs = maxMoveTimeMs;
    } else if (infinite) {
        // "go infinite" thinks until stop. It runs as a node search towards
        // a ceiling no analysis is meant to reach, so a session left running
        // ends with a bestmove instead of filling memory with the tree.
        opts = SearchOptions::uci(0, multiPV, isPonder);
        opts.targetNodes = SearchParams::ANALYSIS_MAX_NODES;
    } else {
        // Default to 1 second if nothing specified
        opts = SearchOptions::uci(1000, multiPV, isPonder);
    }
    opts.enablePonder = ponderEnabled;
    opts.reportIntervalMs = reportIntervalMs;
    opts.search = current_search_config();
    opts.clock = current_clock();
    opts.teamTreeReuse = teamTreeReuse;
    opts.teamTreeReuseToleranceMs = teamTreeReuseToleranceMs;
    opts.planCarry = planCarry;
    opts.planCarryPermille = planCarryPermille;
    opts.hybrid.knobs = hybrid;
    opts.hybrid.clock = current_hybrid_clock();
    // Fix 1 keeps the team's whole tree for the next search. The permanent
    // brain charges our action with v3's advance() and re-roots, which would
    // throw that tree away, so it does not run then. Under fix 3 alone the
    // next durative search drops whatever it retained, so it would be work
    // for nothing.
    const bool keepsTeamTree = (teamTreeReuse || planCarry) && opts.clock
        && opts.clock->is_durative();

    // "go background": the opponents are to move at the current position, so
    // this is the permanent brain, rooted at the position the front end
    // actually reached. It prints no bestmove and runs until the next
    // position, go or stop.
    if (isBackground) {
        agent->clear_stop_request();
        mainSearchThread = new std::thread([this, enginePtrs, opts]() {
            try {
                agent->run_background_search(
                    board, enginePtrs, teamSide, teamHasTimeAdvantage, opts);
            } catch (const std::exception& error) {
                std::cerr << "Background search failed: " << error.what() << std::endl;
            } catch (...) {
                std::cerr << "Background search failed with an unknown exception" << std::endl;
            }
            ongoingSearch.store(false, std::memory_order_release);
        });
        return;
    }

    // Dirichlet noise is applied when a root is first expanded. Discard an
    // already searched/noised retained root during the opening so every move
    // receives an independent sample instead of inheriting the previous one.
    if (opts.search.rootDirichletAlpha > 0.0f
        && opts.search.rootDirichletEpsilon > 0.0f) {
        agent->reset_search_state();
    }
    
    // Launch the search thread
    mainSearchThread = new std::thread([this, enginePtrs, opts, keepsTeamTree]() {
        try {
            const JointActionCandidate played = agent->run_search(
                board, enginePtrs, teamSide, teamHasTimeAdvantage, opts);
            if (!opts.isPonder && !keepsTeamTree) {
                // Nothing else runs between moves. Keep thinking from the
                // position our own move creates, so whichever reply the four
                // asynchronous players produce lands on a subtree that already
                // has work in it. Returns on the next position, go, or stop.
                agent->run_permanent_brain(
                    board, enginePtrs, teamSide, teamHasTimeAdvantage, played,
                    opts);
            }
        } catch (const std::exception& error) {
            std::cerr << "Search failed: " << error.what() << std::endl;
            std::cout << "info string search failed: " << error.what() << std::endl;
            std::cout << "bestmove (none)" << std::endl;
        } catch (...) {
            std::cerr << "Search failed with an unknown exception" << std::endl;
            std::cout << "info string search failed: unknown exception" << std::endl;
            std::cout << "bestmove (none)" << std::endl;
        }
        ongoingSearch.store(false, std::memory_order_release);
    });

    while (ongoingSearch.load(std::memory_order_acquire) && !agent->is_running()) {
        std::this_thread::yield();
    }
}

void UCI::go_single_board(int moveTime, size_t nodes, int depth, bool infinite, bool ponder) {
    if (engines.empty()) {
        std::cerr << "Error: Single-board search requires an ONNX model" << std::endl;
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    single_board::Limits limits;
    limits.nodes = nodes;
    limits.maxDepth = depth > 0 ? depth : 128;
    limits.policyTemperature = searchConfig.policyTemperature;
    limits.moveTimeMs = infinite ? 0 : nodes > 0 && moveTime == 0 ? 0
        : moveTime > 0 ? moveTime : 1000;
    singleStop.store(false);
    singlePonder.store(ponder);
    ongoingSearch.store(true);
    mainSearchThread = new std::thread([this, limits, infinite]() {
        try {
            const auto start = std::chrono::steady_clock::now();
            const auto result = single_board::search(board, *engines.front(), limits, singleStop);
            while (!singleStop.load() && (infinite || singlePonder.load())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            const auto elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count());
            std::ostringstream output;
            output << "info depth " << result.depth << " nodes " << result.nodes
                   << " time " << elapsed << " nps " << result.nodes * 1000 / elapsed;
            if (result.mateInOne) output << " score mate 1";
            else output << " score cp " << static_cast<int>(400 * std::atanh(std::clamp(result.value, -0.999f, 0.999f)));
            output << " pv";
            Board pvBoard(board);
            for (Stockfish::Move move : result.pv) {
                if (!pvBoard.is_legal_move(BOARD_A, move)) break;
                output << ' ' << pvBoard.uci_move(BOARD_A, move);
                pvBoard.push_move(BOARD_A, move);
            }
            output << "\nbestmove " << (result.move == Stockfish::MOVE_NONE
                ? "0000" : board.uci_move(BOARD_A, result.move));
            std::cout << output.str() << std::endl;
        } catch (const std::exception& error) {
            std::cerr << "Single-board search failed: " << error.what() << std::endl;
            std::cout << "info string single-board search failed\nbestmove 0000" << std::endl;
        }
        ongoingSearch.store(false);
    });
}

SearchParams::RuntimeConfig UCI::current_search_config() {
    SearchParams::RuntimeConfig config = searchConfig;
    const int currentPly = std::max(
        board.game_ply(BOARD_A), board.game_ply(BOARD_B));
    if (!openingNoiseEnabled || openingNoisePlies <= 0
        || currentPly >= openingNoisePlies
        || openingNoiseAlphaPermille <= 0
        || openingNoiseEpsilonPermille <= 0) {
        config.rootDirichletAlpha = 0.0f;
        config.rootDirichletEpsilon = 0.0f;
        config.rootNoiseSeed = 0;
        return config;
    }

    config.rootDirichletAlpha =
        static_cast<float>(openingNoiseAlphaPermille) / 1000.0f;
    config.rootDirichletEpsilon =
        static_cast<float>(openingNoiseEpsilonPermille) / 1000.0f;
    config.rootNoiseSeed = openingNoiseGenerator();
    return config;
}

void UCI::ponderhit() {
    singlePonder.store(false);
    if (agent && ongoingSearch.load(std::memory_order_acquire)) {
        agent->ponderhit();
    }
}

void UCI::setoption(std::istringstream& is) {
    stop();

    std::string token;
    is >> token; 
    if (token != "name") return;
    std::string name;
    is >> name;
    is >> token; 
    if (token != "value") return;
    std::string value;
    is >> value;
    if (name == "UCI_Variant") {
        Board::Variant variant;
        if (value == "chess" || value == "standard") variant = Board::Variant::CHESS;
        else if (value == "crazyhouse") variant = Board::Variant::CRAZYHOUSE;
        else if (value == "antichess") variant = Board::Variant::ANTICHESS;
        else if (value == "chess960" || value == "fischerandom") variant = Board::Variant::CHESS960;
        else if (value == "atomic") variant = Board::Variant::ATOMIC;
        else if (value == "3check" || value == "threeCheck" || value == "threecheck") variant = Board::Variant::THREE_CHECK;
        else if (value == "bughouse") variant = Board::Variant::BUGHOUSE;
        else {
            std::cout << "info string unsupported UCI_Variant " << value << std::endl;
            return;
        }
        board.set_variant(variant);
        if (agent) agent->reset_search_state();
    } else if (name == "UCI_Chess960") {
        if (board.variant == Board::Variant::CHESS || board.variant == Board::Variant::CHESS960) {
            board.set_variant(value == "true" ? Board::Variant::CHESS960 : Board::Variant::CHESS);
        }
    } else if (name == "Hash") {
        // Parse hash size in MB (1 - 33554432 MB)
        size_t sizeMB = std::stoull(value);

        // Set hash size via Agent (which owns the transposition table)
        if (agent) {
            agent->setHashSize(sizeMB);
            std::cout << "info string Hash table set to " << sizeMB << " MB" << std::endl;
        }
    } else if (name == "BatchSize") {
        // The batch size is compiled into the TensorRT engine, so this reloads
        // it. A cached engine for the requested size loads in about a second;
        // an uncached one is built from the ONNX first, which takes minutes.
        const int requested = std::stoi(value);
        if (requested < 1 || requested > 1024) {
            std::cout << "info string BatchSize must be between 1 and 1024" << std::endl;
        } else if (requested == batchSize) {
            std::cout << "info string BatchSize already " << batchSize << std::endl;
        } else {
            const int previous = batchSize;
            batchSize = requested;
            if (reload_engines()) {
                std::cout << "info string BatchSize set to " << batchSize << std::endl;
            } else {
                batchSize = previous;
                std::cout << "info string BatchSize " << requested
                          << " failed to load; restoring " << previous << std::endl;
                if (!reload_engines()) {
                    std::cout << "info string no inference engine is loaded" << std::endl;
                }
            }
        }
    } else if (name == "MultiPV") {
        int mpv = std::stoi(value);
        if (mpv >= 1 && mpv <= 500) {
            multiPV = mpv;
            std::cout << "info string MultiPV set to " << multiPV << std::endl;
        }
    } else if (name == "ReportIntervalMs") {
        reportIntervalMs = std::clamp(std::stoi(value), 0, 60000);
        std::cout << "info string ReportIntervalMs set to " << reportIntervalMs
                  << std::endl;
    } else if (name == "Ponder") {
        if (value == "true" || value == "false") {
            ponderEnabled = (value == "true");
            std::cout << "info string Ponder set to " << value << std::endl;
        }
    } else if (name == "BackgroundSearch") {
        if (value == "true" || value == "false") {
            backgroundSearchEnabled = (value == "true");
            std::cout << "info string BackgroundSearch set to " << value << std::endl;
        }
    } else if (name == "OpeningNoise") {
        if (value == "true" || value == "false") {
            openingNoiseEnabled = value == "true";
            std::cout << "info string OpeningNoise set to " << value << std::endl;
        }
    } else if (name == "OpeningNoisePlies") {
        openingNoisePlies = std::clamp(std::stoi(value), 0, 200);
        std::cout << "info string OpeningNoisePlies set to "
                  << openingNoisePlies << std::endl;
    } else if (name == "OpeningNoiseAlphaPermille") {
        openingNoiseAlphaPermille = std::clamp(std::stoi(value), 1, 10000);
        std::cout << "info string OpeningNoiseAlphaPermille set to "
                  << openingNoiseAlphaPermille << std::endl;
    } else if (name == "OpeningNoiseEpsilonPermille") {
        openingNoiseEpsilonPermille = std::clamp(std::stoi(value), 0, 1000);
        std::cout << "info string OpeningNoiseEpsilonPermille set to "
                  << openingNoiseEpsilonPermille << std::endl;
    } else if (name == "DrawContemptPermille") {
        int permille = std::clamp(std::stoi(value), 0, 1000);
        searchConfig.drawContempt = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string DrawContemptPermille set to " << permille << std::endl;
    } else if (name == "MovesLeftDiscountPermille") {
        const int permille = std::clamp(std::stoi(value), 0, 1000);
        searchConfig.movesLeftDiscount = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string MovesLeftDiscountPermille set to " << permille
                  << std::endl;
    } else if (name == "PWCoefficientPermille") {
        int permille = std::clamp(std::stoi(value), 1, 10000);
        searchConfig.pwCoefficient = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string PWCoefficientPermille set to " << permille << std::endl;
    } else if (name == "RootPWCoefficientPermille") {
        int permille = std::clamp(std::stoi(value), 1, 10000);
        searchConfig.rootPwCoefficient = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string RootPWCoefficientPermille set to " << permille << std::endl;
    } else if (name == "PWExponentPermille") {
        int permille = std::clamp(std::stoi(value), 1, 1000);
        searchConfig.pwExponent = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string PWExponentPermille set to " << permille << std::endl;
    } else if (name == "PWMassStartPermille" || name == "RootPWMassStartPermille") {
        const bool root = name == "RootPWMassStartPermille";
        const int permille = std::clamp(std::stoi(value), root ? -1 : 0, 999);
        (root ? searchConfig.rootPwMassStart : searchConfig.pwMassStart) =
            permille < 0 ? -1.0f : static_cast<float>(permille) / 1000.0f;
        std::cout << "info string " << name << " set to " << permille << std::endl;
    } else if (name == "PWMassExponentPermille" || name == "PWMassCapPermille"
               || name == "CPUCTInitPermille") {
        const int permille = std::clamp(std::stoi(value), 1, 10000);
        float& setting = name == "PWMassExponentPermille" ? searchConfig.pwMassExponent
            : name == "PWMassCapPermille" ? searchConfig.pwMassCap : searchConfig.cpuctInit;
        setting = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string " << name << " set to " << permille << std::endl;
    } else if (name == "PWMassNormalize") {
        if (value == "true" || value == "false") {
            searchConfig.pwMassNormalize = value == "true";
            std::cout << "info string PWMassNormalize set to " << value << std::endl;
        }
    } else if (name == "PolicyTemperaturePermille") {
        const int permille = std::clamp(std::stoi(value), 100, 10000);
        const float temperature = static_cast<float>(permille) / 1000.0f;
        // Retained nodes hold priors made at the old temperature.
        if (temperature != searchConfig.policyTemperature && agent) {
            agent->reset_search_state();
        }
        searchConfig.policyTemperature = temperature;
        std::cout << "info string PolicyTemperaturePermille set to " << permille << std::endl;
    } else if (name == "PUCTWidening") {
        if (value == "true" || value == "false") {
            searchConfig.puctWidening = value == "true";
            std::cout << "info string PUCTWidening set to " << value << std::endl;
        }
    } else if (name == "MateProbe") {
        if (value == "true" || value == "false") {
            searchConfig.enableMateProbe = value == "true";
            std::cout << "info string MateProbe set to " << value << std::endl;
        }
    } else if (name == "FrozenRace") {
        if (value == "true" || value == "false") {
            searchConfig.frozenRace = value == "true";
            std::cout << "info string FrozenRace set to " << value << std::endl;
        }
    } else if (name == "ConditionalMateProbe") {
        if (value == "true" || value == "false") {
            searchConfig.conditionalMateProbe = value == "true";
            std::cout << "info string ConditionalMateProbe set to " << value
                      << std::endl;
        }
    } else if (name == "MateThreads") {
        MateProbe::set_threads(std::stoi(value));
        std::cout << "info string MateThreads set to " << MateProbe::threads()
                  << std::endl;
    } else if (name == "CertifySelectedMove") {
        if (value == "true" || value == "false") {
            searchConfig.certifySelectedMove = value == "true";
            std::cout << "info string CertifySelectedMove set to " << value
                      << std::endl;
        }
    } else if (name == "InternalMateProbe") {
        if (value == "off") {
            searchConfig.internalMateProbeMode =
                SearchParams::InternalMateProbeMode::OFF;
        } else if (value == "telemetry") {
            searchConfig.internalMateProbeMode =
                SearchParams::InternalMateProbeMode::TELEMETRY;
        } else if (value == "bias") {
            searchConfig.internalMateProbeMode =
                SearchParams::InternalMateProbeMode::BIAS;
        } else if (value == "certify") {
            searchConfig.internalMateProbeMode =
                SearchParams::InternalMateProbeMode::CERTIFY;
        } else if (value == "certifyonly") {
            searchConfig.internalMateProbeMode =
                SearchParams::InternalMateProbeMode::CERTIFY_ONLY;
        } else {
            return;
        }
        std::cout << "info string InternalMateProbe set to "
                  << value << std::endl;
    } else if (name == "Transpositions") {
        if (value == "true" || value == "false") {
            searchConfig.enableTranspositions = value == "true";
            std::cout << "info string Transpositions set to " << value << std::endl;
        }
    } else if (name == "GumbelRootSearch") {
        if (value == "true" || value == "false") {
            searchConfig.enableGumbelRootSearch = value == "true";
            std::cout << "info string GumbelRootSearch set to " << value << std::endl;
        }
    } else if (name == "RootGumbelPoolSize") {
        searchConfig.rootGumbelPoolSize = std::clamp(
            std::stoi(value), 1, 65536);
        std::cout << "info string RootGumbelPoolSize set to "
                  << searchConfig.rootGumbelPoolSize << std::endl;
    } else if (name == "RootGumbelInitialCandidates") {
        searchConfig.rootGumbelInitialCandidates = std::clamp(
            std::stoi(value), 1, 4096);
        std::cout << "info string RootGumbelInitialCandidates set to "
                  << searchConfig.rootGumbelInitialCandidates << std::endl;
    } else if (name == "RootGumbelReplenishment") {
        searchConfig.rootGumbelReplenishment = std::clamp(
            std::stoi(value), 1, 4096);
        std::cout << "info string RootGumbelReplenishment set to "
                  << searchConfig.rootGumbelReplenishment << std::endl;
    } else if (name == "RootGumbelValueScalePermille") {
        const int permille = std::clamp(std::stoi(value), 0, 10000);
        searchConfig.rootGumbelValueScale =
            static_cast<float>(permille) / 1000.0f;
        std::cout << "info string RootGumbelValueScalePermille set to "
                  << permille << std::endl;
    } else if (name == "RootGumbelMaxRoundVisits") {
        searchConfig.rootGumbelMaxRoundVisits = std::clamp(
            std::stoi(value), 1, 100000);
        std::cout << "info string RootGumbelMaxRoundVisits set to "
                  << searchConfig.rootGumbelMaxRoundVisits << std::endl;
    } else if (name == "Team") {
        if (value == "white") {
            teamSide = Stockfish::WHITE;
        } else if (value == "black") {
            teamSide = Stockfish::BLACK;
        }
    } else if (name == "RequireMoveOn") {
        g_requiredMoveBoard = parseRequiredMoveBoard(value);
        std::cout << "info string RequireMoveOn set to "
                  << (g_requiredMoveBoard == REQUIRE_MOVE_BOARD_A ? "A"
                      : g_requiredMoveBoard == REQUIRE_MOVE_BOARD_B ? "B"
                      : "none")
                  << std::endl;
    } else if (name == "TimeAdvantage") {
        if (value == "true" || value == "false") {
            teamHasTimeAdvantage = value == "true";
            std::cout << "info string TimeAdvantage set to " << value << std::endl;
        }
    } else if (name == "TimeModel") {
        if (value == "flag" || value == "clock") {
            clockModel = value == "clock";
            timeModelSet = true;
            std::cout << "info string TimeModel set to " << value << std::endl;
        }
    } else if (name == "TimeLeadMs") {
        timeLeadMs = std::clamp(std::stoi(value), -TimeState::MAX_LEAD_MS,
                                TimeState::MAX_LEAD_MS);
        std::cout << "info string TimeLeadMs set to " << timeLeadMs << std::endl;
    } else if (name == "TeamMoveTimeMs") {
        teamMoveTimeMs = std::clamp(std::stoi(value), 1, 3600000);
        std::cout << "info string TeamMoveTimeMs set to " << teamMoveTimeMs << std::endl;
    } else if (name == "RootMoveTimeMs") {
        rootMoveTimeMs = std::clamp(std::stoi(value), 0, 3600000);
        std::cout << "info string RootMoveTimeMs set to " << rootMoveTimeMs << std::endl;
    } else if (name == "OpponentMoveTimeMs") {
        opponentMoveTimeMs = std::clamp(std::stoi(value), 1, 3600000);
        std::cout << "info string OpponentMoveTimeMs set to "
                  << opponentMoveTimeMs << std::endl;
    } else if (name == "MaxMoveTimeMs") {
        maxMoveTimeMs = std::clamp(std::stoi(value), 0, 3600000);
        std::cout << "info string MaxMoveTimeMs set to " << maxMoveTimeMs << std::endl;
    } else if (name == "DurativeTree" || name == "SitGate" || name == "FlagLoss"
               || name == "InstantTransfer" || name == "MateRaceByLanding") {
        if (value == "true" || value == "false") {
            const bool on = value == "true";
            if (name == "DurativeTree") durativeTree = on;
            else if (name == "SitGate") sitGate = on;
            else if (name == "FlagLoss") durativeConfig.flagLoss = on;
            else if (name == "InstantTransfer") durativeConfig.instantTransfer = on;
            else durativeConfig.mateRaceByLanding = on;
            std::cout << "info string " << name << " set to " << value << std::endl;
            if (name == "DurativeTree" && on && hybrid.mode != HybridMode::OFF) {
                std::cout << "info string Hybrid acts only with DurativeTree false" << std::endl;
            }
        }
    } else if (name == "TeamTreeReuse") {
        if (value == "true" || value == "false") {
            teamTreeReuse = value == "true";
            std::cout << "info string TeamTreeReuse set to " << value << std::endl;
            if (teamTreeReuse && !durativeTree) {
                std::cout << "info string TeamTreeReuse acts only with DurativeTree true"
                          << std::endl;
            }
        }
    } else if (name == "JointPlans") {
        if (value == "true" || value == "false") {
            durativeConfig.jointPlans = value == "true";
            std::cout << "info string JointPlans set to " << value << std::endl;
            if (durativeConfig.jointPlans && !durativeTree) {
                std::cout << "info string JointPlans acts only with DurativeTree true"
                          << std::endl;
            }
        }
    } else if (name == "PlanCarry") {
        if (value == "true" || value == "false") {
            planCarry = value == "true";
            std::cout << "info string PlanCarry set to " << value << std::endl;
            if (planCarry && !durativeTree) {
                std::cout << "info string PlanCarry acts only with DurativeTree true"
                          << std::endl;
            }
        }
    } else if (name == "PlanCarryPermille") {
        planCarryPermille = std::clamp(std::stoi(value), 0, 1000);
        std::cout << "info string PlanCarryPermille set to " << planCarryPermille << std::endl;
    } else if (name == "OwnPaceBelowRoot" || name == "WaitNodeActingEval") {
        // Diagnosis fixes D1 and D3 (docs/wave2_engine.md, "Diagnosis fixes").
        if (value == "true" || value == "false") {
            const bool on = value == "true";
            if (name == "OwnPaceBelowRoot") durativeConfig.ownPaceBelowRoot = on;
            else durativeConfig.waitNodeActingEval = on;
            std::cout << "info string " << name << " set to " << value << std::endl;
            if (on && !durativeTree) {
                std::cout << "info string " << name << " acts only with DurativeTree true"
                          << std::endl;
            }
        }
    } else if (name == "BlockableCheckNotMate") {
        // Diagnosis fix D5: the arena's mate rule, in either tree.
        if (value == "true" || value == "false") {
            board.set_blockable_check_not_mate(value == "true");
            std::cout << "info string BlockableCheckNotMate set to " << value << std::endl;
        }
    } else if (name == "Continuity") {
        // Preset: the three switches together, with the diagnosis fixes D1
        // and D5; the knobs keep their values.
        if (value == "true" || value == "false") {
            const bool on = value == "true";
            teamTreeReuse = on;
            durativeConfig.jointPlans = on;
            planCarry = on;
            durativeConfig.ownPaceBelowRoot = on;
            board.set_blockable_check_not_mate(on);
            std::cout << "info string Continuity set to " << value
                      << " (TeamTreeReuse, JointPlans, PlanCarry, OwnPaceBelowRoot,"
                      << " BlockableCheckNotMate " << value << ")" << std::endl;
            if (on && !durativeTree) {
                std::cout << "info string Continuity acts only with DurativeTree true"
                          << std::endl;
            }
        }
    } else if (name == "HybridMode" || name == "Hybrid") {
        // Spec v6. Hybrid is the preset: true = HybridMode gate, false = off;
        // the knobs keep their values.
        HybridMode mode = hybrid.mode;
        if (name == "Hybrid" && (value == "true" || value == "false")) {
            mode = value == "true" ? HybridMode::GATE : HybridMode::OFF;
        } else if (name == "Hybrid" || !parse_hybrid_mode(value, mode)) {
            return;
        }
        if (mode == HybridMode::PROBE || mode == HybridMode::PRICED) {
            std::cout << "info string HybridMode " << value
                      << " is not built (spec v6 build order: only if gate shows no gain);"
                      << " HybridMode stays " << hybrid_mode_name(hybrid.mode) << std::endl;
            return;
        }
        hybrid.mode = mode;
        std::cout << "info string " << name << " set to " << value
                  << " (HybridMode " << hybrid_mode_name(hybrid.mode) << ")" << std::endl;
        if (hybrid.mode != HybridMode::OFF && durativeTree) {
            std::cout << "info string Hybrid acts only with DurativeTree false" << std::endl;
        }
    } else if (name == "HybridEventLands" || name == "HybridMaxWaitMs"
               || name == "HybridMinClockMs" || name == "HybridMarginPermille"
               || name == "HybridPassPriorPermille") {
        if (name == "HybridEventLands") {
            if (!parse_hybrid_event_lands(value, hybrid.eventLands)) {
                return;
            }
            std::cout << "info string HybridEventLands set to " << value << std::endl;
        } else {
            int32_t& knob = name == "HybridMaxWaitMs" ? hybrid.maxWaitMs
                : name == "HybridMinClockMs" ? hybrid.minClockMs
                : name == "HybridMarginPermille" ? hybrid.marginPermille
                                                 : hybrid.passPriorPermille;
            const int32_t high = name == "HybridMaxWaitMs" ? 10000
                : name == "HybridMinClockMs" ? 600000 : 1000;
            knob = std::clamp(std::stoi(value), 0, high);
            std::cout << "info string " << name << " set to " << knob << std::endl;
        }
        if (hybrid.mode == HybridMode::OFF) {
            std::cout << "info string " << name << " acts only with HybridMode gate" << std::endl;
        }
    } else if (name == "JointPlanFreePermille") {
        durativeConfig.jointPlanFreePermille = std::clamp(std::stoi(value), 0, 1000);
        std::cout << "info string JointPlanFreePermille set to "
                  << durativeConfig.jointPlanFreePermille << std::endl;
    } else if (name == "TeamTreeReuseToleranceMs") {
        teamTreeReuseToleranceMs = std::clamp(std::stoi(value), 0, 5000);
        std::cout << "info string TeamTreeReuseToleranceMs set to "
                  << teamTreeReuseToleranceMs << std::endl;
    } else if (name == "PassCapK") {
        durativeConfig.passCapK = std::clamp(std::stoi(value), 1, 64);
        std::cout << "info string PassCapK set to " << durativeConfig.passCapK << std::endl;
    } else if (name == "SeatClocksMs" || name == "SeatPaceMs" || name == "SeatElapsedMs"
               || name == "SeatSitting") {
        // Four numbers "<Aw> <Ab> <Bw> <Bb>"; an empty value (or "none")
        // clears the option back to v3's derivation. SeatSitting is 0/1 per
        // seat: an opponent seat on turn that is in a pass, so the sit-off
        // rule knows it will not move (self-play sets this directly).
        std::string rest;
        std::getline(is, rest);
        std::istringstream fields(value + " " + rest);
        std::array<int32_t, 4> seats{0, 0, 0, 0};
        int read = 0;
        long long field = 0;
        while (read < 4 && fields >> field) {
            seats[read++] = static_cast<int32_t>(std::clamp<long long>(
                field, name == "SeatClocksMs" ? -TimeState::MAX_LEAD_MS : 0,
                name == "SeatSitting" ? 1 : TimeState::MAX_LEAD_MS));
        }
        if (read != 0 && read != 4) {
            std::cout << "info string " << name
                      << " needs four values <Aw> <Ab> <Bw> <Bb>" << std::endl;
            return;
        }
        if (name == "SeatSitting") {
            for (int seat = 0; seat < 4; ++seat) {
                durativeConfig.seatSitting[seat] = seats[seat] != 0;
            }
            std::cout << "info string SeatSitting set to " << seats[0] << " " << seats[1]
                      << " " << seats[2] << " " << seats[3] << std::endl;
            return;
        }
        std::array<int32_t, 4>& target = name == "SeatClocksMs"
            ? durativeConfig.seatClocksMs
            : name == "SeatPaceMs" ? durativeConfig.seatPaceMs
                                   : durativeConfig.seatElapsedMs;
        target = seats;
        std::cout << "info string " << name << " set to " << seats[0] << " "
                  << seats[1] << " " << seats[2] << " " << seats[3] << std::endl;
    } else if (name == "SeatArrivalsMs") {
        // Wave 2b: pieces that landed in pockets lately,
        // "<pocket>:<piece>:<ms_ago>,..." with pocket Aw/Ab/Bw/Bb and piece
        // P/N/B/R/Q. Empty clears. Landings 5 s old or more are dropped; past
        // MAX_RECENT_LANDINGS the newest are kept.
        std::string rest;
        std::getline(is, rest);
        std::array<SeatLanding, MAX_RECENT_LANDINGS> landings{};
        std::string error;
        if (!parse_seat_arrivals(value + rest, landings, &error)) {
            std::cout << "info string SeatArrivalsMs needs <pocket>:<piece>:<ms_ago>,... "
                         "(pocket Aw Ab Bw Bb, piece P N B R Q); got " << error << std::endl;
            return;
        }
        durativeConfig.recentLandings = landings;
        const auto count = std::count_if(landings.begin(), landings.end(),
                                         [](const SeatLanding& l) { return l.seat >= 0; });
        std::cout << "info string SeatArrivalsMs set to " << count << " landings" << std::endl;
    } else if (name == "Mode") {
        // Deprecated alias for TimeAdvantage, kept so existing GUI profiles keep
        // working. "sit" meant the team was up on time and could wait; "go"
        // meant it was not. Not advertised in the uci handshake.
        if (value == "sit" || value == "go") {
            teamHasTimeAdvantage = value == "sit";
            std::cout << "info string Mode is deprecated; use "
                      << "'setoption name TimeAdvantage value "
                      << (teamHasTimeAdvantage ? "true" : "false") << "'"
                      << std::endl;
        }
    }
}

bool UCI::clock_model_active() const {
    return clockModel && (timeModelSet || networkReadsClock);
}

std::optional<TimeState> UCI::current_clock() const {
    if (!clock_model_active()) {
        return std::nullopt;
    }
    const TimeState v3 = TimeState::clocked(
        teamSide, timeLeadMs, teamMoveTimeMs, opponentMoveTimeMs, rootMoveTimeMs);
    if (sitGate && !durativeTree) {
        return v3;
    }
    const TimeState gated = v3.with_sit_gate(sitGate);
    if (!durativeTree) {
        // SitGate false without DurativeTree: the gate stays (TimeState::can_sit).
        return gated;
    }
    DurativeConfig config = durativeConfig;
    config.rootMoveTimeMs = rootMoveTimeMs;
    config.wideInputs = networkWide;
    // side_to_move is not const, though it only reads.
    Board& root = const_cast<Board&>(board);
    return TimeState::durative(gated, teamSide, root.side_to_move(BOARD_A),
                               root.side_to_move(BOARD_B), config);
}

std::optional<TimeState> UCI::current_hybrid_clock() const {
    if (hybrid.mode == HybridMode::OFF || durativeTree || !clock_model_active()) {
        return std::nullopt;
    }
    const TimeState v3 = TimeState::clocked(
        teamSide, timeLeadMs, teamMoveTimeMs, opponentMoveTimeMs, rootMoveTimeMs);
    DurativeConfig config = durativeConfig;
    config.rootMoveTimeMs = rootMoveTimeMs;
    config.wideInputs = networkWide;
    Board& root = const_cast<Board&>(board);
    return TimeState::durative(v3.with_sit_gate(false), teamSide, root.side_to_move(BOARD_A),
                               root.side_to_move(BOARD_B), config);
}

bool UCI::team_may_sit() const {
    const std::optional<TimeState> clock = current_clock();
    return clock ? clock->can_sit(teamSide) : teamHasTimeAdvantage;
}

void UCI::send_uci_response() {
    cout << "id name hivemind" << endl;
    cout << "id author aminwoo\n" << endl;
    cout << "option name Hash type spin default 16 min 1 max 33554432" << endl;
    cout << "option name UCI_Variant type combo default bughouse var bughouse var chess var crazyhouse var antichess var chess960 var atomic var 3check" << endl;
    cout << "option name UCI_Chess960 type check default false" << endl;
    cout << "option name BatchSize type spin default " << batchSize
         << " min 1 max 1024" << endl;
    cout << "option name MultiPV type spin default 1 min 1 max 500" << endl;
    cout << "option name Ponder type check default true" << endl;
    cout << "option name ReportIntervalMs type spin default " << reportIntervalMs
         << " min 0 max 60000" << endl;
    cout << "option name BackgroundSearch type check default true" << endl;
    cout << "option name OpeningNoise type check default false" << endl;
    cout << "option name OpeningNoisePlies type spin default 16 min 0 max 200" << endl;
    cout << "option name OpeningNoiseAlphaPermille type spin default 100 min 1 max 10000" << endl;
    cout << "option name OpeningNoiseEpsilonPermille type spin default 600 min 0 max 1000" << endl;
    cout << "option name DrawContemptPermille type spin default 0 min 0 max 1000" << endl;
    cout << "option name MovesLeftDiscountPermille type spin default "
         << static_cast<int>(SearchParams::MOVES_LEFT_DISCOUNT * 1000.0f)
         << " min 0 max 1000" << endl;
    cout << "option name PWCoefficientPermille type spin default "
         << static_cast<int>(SearchParams::PW_COEFFICIENT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name RootPWCoefficientPermille type spin default "
         << static_cast<int>(SearchParams::ROOT_PW_COEFFICIENT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name PWExponentPermille type spin default "
         << static_cast<int>(SearchParams::PW_EXPONENT * 1000.0f) << " min 1 max 1000" << endl;
    cout << "option name PWMassStartPermille type spin default "
         << static_cast<int>(SearchParams::PW_MASS_START * 1000.0f) << " min 0 max 999" << endl;
    cout << "option name RootPWMassStartPermille type spin default -1 min -1 max 999" << endl;
    cout << "option name PWMassExponentPermille type spin default "
         << static_cast<int>(SearchParams::PW_MASS_EXPONENT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name PWMassCapPermille type spin default "
         << static_cast<int>(SearchParams::PW_MASS_CAP * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name PWMassNormalize type check default false" << endl;
    cout << "option name PUCTWidening type check default "
         << (SearchParams::PUCT_WIDENING ? "true" : "false") << endl;
    cout << "option name PolicyTemperaturePermille type spin default "
         << static_cast<int>(SearchParams::POLICY_TEMPERATURE * 1000.0f) << " min 100 max 10000" << endl;
    cout << "option name CPUCTInitPermille type spin default "
         << static_cast<int>(SearchParams::CPUCT_INIT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name MateProbe type check default "
         << (SearchParams::ENABLE_MATE_PROBE ? "true" : "false") << endl;
    cout << "option name FrozenRace type check default "
         << (SearchParams::ENABLE_FROZEN_RACE ? "true" : "false") << endl;
    cout << "option name ConditionalMateProbe type check default "
         << (SearchParams::ROOT_CONDITIONAL_MATE_PROBE ? "true" : "false")
         << endl;
    cout << "option name MateThreads type spin default "
         << MateProbe::DEFAULT_THREADS << " min 1 max "
         << MateProbe::MAX_THREADS << endl;
    cout << "option name CertifySelectedMove type check default "
         << (SearchParams::ENABLE_SELECTED_MOVE_CERTIFICATION ? "true" : "false")
         << endl;
    cout << "option name InternalMateProbe type combo default off"
         << " var off var telemetry var bias var certify var certifyonly"
         << endl;
    cout << "option name Transpositions type check default "
        << (SearchParams::ENABLE_TRANSPOSITIONS ? "true" : "false") << endl;
    cout << "option name GumbelRootSearch type check default "
         << (SearchParams::ENABLE_GUMBEL_ROOT_SEARCH ? "true" : "false") << endl;
    cout << "option name RootGumbelPoolSize type spin default "
         << SearchParams::ROOT_GUMBEL_POOL_SIZE << " min 1 max 65536" << endl;
    cout << "option name RootGumbelInitialCandidates type spin default "
         << SearchParams::ROOT_GUMBEL_INITIAL_CANDIDATES
         << " min 1 max 4096" << endl;
    cout << "option name RootGumbelReplenishment type spin default "
         << SearchParams::ROOT_GUMBEL_REPLENISHMENT
         << " min 1 max 4096" << endl;
    cout << "option name RootGumbelValueScalePermille type spin default "
         << static_cast<int>(SearchParams::ROOT_GUMBEL_VALUE_SCALE * 1000.0f)
         << " min 0 max 10000" << endl;
    cout << "option name RootGumbelMaxRoundVisits type spin default "
         << SearchParams::ROOT_GUMBEL_MAX_ROUND_VISITS
         << " min 1 max 100000" << endl;
    cout << "option name Team type combo default white var white var black" << endl;
    cout << "option name TimeAdvantage type check default false" << endl;
    cout << "option name TimeModel type combo default clock var flag var clock" << endl;
    cout << "option name TimeLeadMs type spin default 0 min "
         << -TimeState::MAX_LEAD_MS << " max " << TimeState::MAX_LEAD_MS << endl;
    cout << "option name TeamMoveTimeMs type spin default "
         << TimeState::DEFAULT_MOVE_TIME_MS << " min 1 max 3600000" << endl;
    cout << "option name OpponentMoveTimeMs type spin default "
         << TimeState::DEFAULT_MOVE_TIME_MS << " min 1 max 3600000" << endl;
    cout << "option name RootMoveTimeMs type spin default 0 min 0 max 3600000" << endl;
    cout << "option name MaxMoveTimeMs type spin default 0 min 0 max 3600000" << endl;
    cout << "option name RequireMoveOn type combo default none var none var A var B" << endl;
    // Wave 2 (docs/wave2_engine.md); every default is stock v3.0.3.
    cout << "option name DurativeTree type check default false" << endl;
    cout << "option name SitGate type check default true" << endl;
    cout << "option name FlagLoss type check default false" << endl;
    cout << "option name InstantTransfer type check default true" << endl;
    cout << "option name MateRaceByLanding type check default false" << endl;
    cout << "option name PassCapK type spin default 8 min 1 max 64" << endl;
    cout << "option name SeatClocksMs type string default <empty>" << endl;
    cout << "option name SeatPaceMs type string default <empty>" << endl;
    cout << "option name SeatElapsedMs type string default <empty>" << endl;
    cout << "option name SeatSitting type string default <empty>" << endl;
    cout << "option name SeatArrivalsMs type string default <empty>" << endl;
    // Wave 2c (docs/custom_engine_spec_v5_continuity.md); off by default.
    cout << "option name TeamTreeReuse type check default false" << endl;
    cout << "option name TeamTreeReuseToleranceMs type spin default 250 min 0 max 5000" << endl;
    cout << "option name JointPlans type check default false" << endl;
    cout << "option name JointPlanFreePermille type spin default 500 min 0 max 1000" << endl;
    cout << "option name PlanCarry type check default false" << endl;
    cout << "option name PlanCarryPermille type spin default 300 min 0 max 1000" << endl;
    cout << "option name Continuity type check default false" << endl;
    // Diagnosis fixes (docs/wave2_engine.md); off by default.
    cout << "option name OwnPaceBelowRoot type check default false" << endl;
    cout << "option name WaitNodeActingEval type check default false" << endl;
    cout << "option name BlockableCheckNotMate type check default false" << endl;
    // Spec v6, the hybrid (docs/custom_engine_spec_v6_hybrid.md); off by default.
    cout << "option name HybridMode type combo default off var off var gate" << endl;
    cout << "option name Hybrid type check default false" << endl;
    cout << "option name HybridEventLands type combo default all var all var sit var none" << endl;
    cout << "option name HybridMaxWaitMs type spin default 1000 min 0 max 10000" << endl;
    cout << "option name HybridMinClockMs type spin default 3000 min 0 max 600000" << endl;
    cout << "option name HybridMarginPermille type spin default 20 min 0 max 1000" << endl;
    cout << "option name HybridPassPriorPermille type spin default 150 min 0 max 1000" << endl;
    cout << "info string CUDA engines " << engines.size()
         << " search workers " << engines.size() * SearchParams::NUM_SEARCH_THREADS
         << " (" << SearchParams::NUM_SEARCH_THREADS << " per engine)" << endl;
    cout << "uciok" << endl;
}

void UCI::mate_oracle(std::istringstream& is) {
    stop();
    int maxAttackerMoves = 8;
    uint64_t maxNodes = 5000000;
    is >> maxAttackerMoves >> maxNodes;
    const std::optional<TimeState> clock = current_clock();
    const TimeState time = clock
        ? *clock : TimeState::legacy(teamSide, teamHasTimeAdvantage);
    const auto start = std::chrono::steady_clock::now();
    const Agent::ExhaustiveMateResult result =
        Agent::prove_joint_mate_exhaustive(
            board, teamSide, time, maxAttackerMoves, maxNodes);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    using Status = Agent::ExhaustiveMateResult::Status;
    std::cout << "info string mate oracle status="
              << (result.status == Status::PROVEN ? "proven"
                  : result.status == Status::NO_WIN ? "no_win" : "unknown")
              << " max_moves=" << maxAttackerMoves
              << " attacker_moves=" << result.attackerMoves
              << " verified=" << (result.verified ? 1 : 0)
              << " nodes=" << result.nodes << " ms=" << elapsed
              << " pv " << result.pv << std::endl;
    std::cout << "oracleok" << std::endl;
}

void UCI::policy() {
    stop();
    if (engines.empty()) {
        cerr << "Error: No engines have been initialized!" << endl;
        return;
    }

    // Allocate inference buffers. runInference reads and writes a full batch,
    // so these must be sized by the loaded engine, not by the compiled default.
    const size_t loadedBatchSize =
        static_cast<size_t>(engines[0]->getBatchSize());
    float* obs = new float[loadedBatchSize * NB_INPUT_VALUES()];
    float* value = new float[loadedBatchSize];
    float* piA = new float[loadedBatchSize * NB_POLICY_VALUES()];
    float* piB = new float[loadedBatchSize * NB_POLICY_VALUES()];
    float* wdl = new float[loadedBatchSize * 3];
    float* movesLeft = new float[loadedBatchSize];

    // Convert board to planes
    const std::optional<TimeState> clock = current_clock();
    const TimeState planeTime = clock ? *clock : TimeState::legacy(teamSide, teamHasTimeAdvantage);
    board_to_planes(board, obs, teamSide, planeTime);
    // Wave 2b: a wide network's extra planes, the same for every batch row.
    Engine* engine = engines[0].get();
    std::vector<float> wideExtras(loadedBatchSize * NB_WIDE_EXTRA_VALUES, 0.0f);
    if (engine->wideInputs()) {
        wide_extra_planes(board, teamSide, planeTime, wideExtras.data());
    }

    // Run inference
    std::vector<float> arrivals(loadedBatchSize * NB_ARRIVALS_VALUES, 0.0f);
    if (!engine->runInference(obs, value, piA, piB, wdl, movesLeft, 0, wideExtras.data(),
                              arrivals.data())) {
        cerr << "Inference failed" << endl;
        delete[] obs;
        delete[] value;
        delete[] piA;
        delete[] piB;
        delete[] wdl;
        delete[] movesLeft;
        return;
    }

    cout << "Value: " << value[0] << endl;
        const float maxWdl = std::max({wdl[0], wdl[1], wdl[2]});
        const float lossExp = std::exp(wdl[0] - maxWdl);
        const float drawExp = std::exp(wdl[1] - maxWdl);
        const float winExp = std::exp(wdl[2] - maxWdl);
        const float wdlTotal = lossExp + drawExp + winExp;
        cout << "WDL: " << winExp / wdlTotal << " " << drawExp / wdlTotal
            << " " << lossExp / wdlTotal << endl;
        cout << "Predicted plies to end: " << movesLeft[0] * 100.0f << endl;
    if (engine->hasArrivals()) {
        // Wave 2b: expected pieces landing in the next 5 s, pockets ours A,
        // theirs A, ours B, theirs B, each P N B R Q.
        cout << "Arrivals:";
        for (int index = 0; index < NB_ARRIVALS_VALUES; ++index) {
            cout << " " << arrivals[index];
        }
        cout << endl;
    }
    cout << endl;

    // Board A policy
    cout << "Board A (" << board.fen(BOARD_A) << "):" << endl;
    if (board.side_to_move(BOARD_A) == teamSide) {
        vector<Stockfish::Move> actionsA = board.legal_moves(BOARD_A);
        actionsA.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsA = get_normalized_probability(
            piA, actionsA, BOARD_A, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsA);
        for (size_t idx : indices) {
            string moveStr = (actionsA[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_A, actionsA[idx]);
            cout << "  " << moveStr << ": " << priorsA[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }
    cout << endl;

    // Board B policy
    cout << "Board B (" << board.fen(BOARD_B) << "):" << endl;
    if (board.side_to_move(BOARD_B) == ~teamSide) {
        vector<Stockfish::Move> actionsB = board.legal_moves(BOARD_B);
        actionsB.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsB = get_normalized_probability(
            piB, actionsB, BOARD_B, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsB);
        for (size_t idx : indices) {
            string moveStr = (actionsB[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_B, actionsB[idx]);
            cout << "  " << moveStr << ": " << priorsB[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }

    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
    delete[] wdl;
    delete[] movesLeft;
}


void UCI::loop() {
    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        try {
            if (token == "uci")             send_uci_response();
            else if (token == "isready")  { cout << "info string livenodes " << Node::live_count() << endl; cout << "readyok" << endl; }
            else if (token == "go")         go(is);
            else if (token == "ponderhit")  ponderhit();
            else if (token == "setoption")  setoption(is);
            else if (token == "position")   position(is);
            else if (token == "ucinewgame") new_game();
            else if (token == "stop")       stop();
            else if (token == "policy")     policy();
            else if (token == "mateoracle") mate_oracle(is);
        } catch (const std::exception& error) {
            stop();
            cerr << "UCI command '" << token << "' failed: " << error.what() << endl;
            cout << "info string " << token << " failed: " << error.what() << endl;
        } catch (...) {
            stop();
            cerr << "UCI command '" << token << "' failed with an unknown exception" << endl;
            cout << "info string " << token << " failed: unknown exception" << endl;
        }

    } while (token != "quit"); // Command line args are one-shot
}
