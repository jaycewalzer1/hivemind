#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>
#include <random>

#include "environment/board.h"
#include "environment/constants.h"
#include "search/searchthread.h"
#include "search/agent.h"
#include "nn/engine.h"

class UCI {
private:
    friend class UCIOpeningNoiseTestPeer;
    friend class UCIClockTestPeer;

    std::thread* mainSearchThread;
    std::unique_ptr<Agent> agent;
    Board board;
    Stockfish::Color teamSide = Stockfish::WHITE;
    // True when our team is ahead on the clocks, which is what makes sitting
    // and double-sitting legal. Set by the TimeAdvantage option.
    bool teamHasTimeAdvantage = false;
    std::vector<std::unique_ptr<Engine>> engines;
    // Retained so the BatchSize option can rebuild the engines in place.
    std::vector<int> deviceIds;
    std::string networkPath;
    int batchSize = SearchParams::BATCH_SIZE;
    std::atomic<bool> ongoingSearch{false};
    int multiPV = 1;  // Number of principal variations to display
    bool ponderEnabled = true;  // Whether to output ponder move and accept ponder search
    SearchParams::RuntimeConfig searchConfig;
    // Optional early-game diversity for normal UCI play. A zero/noise-free
    // RuntimeConfig remains the default so existing users stay deterministic.
    bool openingNoiseEnabled = false;
    int openingNoisePlies = 16;
    int openingNoiseAlphaPermille = 100;
    int openingNoiseEpsilonPermille = 600;
    std::mt19937_64 openingNoiseGenerator;
    // TreeMoveMs option: ms each macro-ply charges inside the search tree
    // when 'position ... clocks' gave clocks. 0 keeps clocks frozen.
    int treeMoveMs = 0;
    // Spec v3 E3 exploration options (all off by default).
    bool rootNoiseAllGame = false;
    int moveTemperaturePermille = 0;
    int moveTemperaturePlies = 0;

    // TimeAdvantage as the search uses it: read from the clocks under tree
    // clocks (spec v3 E1), the TimeAdvantage option otherwise.
    bool effective_time_advantage() const {
        return board.tree_clocks_active()
            ? board.team_time_edge(teamSide) > 0
            : teamHasTimeAdvantage;
    }

    // Prints 'info string sample <base64>' for the finished search
    // ('go ... sample'): one HVM5 TrainingSample record, as selfplay writes it.
    void emit_sample();

    // Prints 'info string think_gain <pa> <pb>' (spec v2 B) for the root
    // position when the network has a think_gain output: the sigmoid of each
    // board's logit, from one extra evaluation after the search (never read
    // by the search). Prints nothing otherwise.
    void emit_think_gain();

    // Rebuilds the engines (and the agent) with the current settings. The batch
    // size is baked into the TensorRT engine, so changing it means reloading.
    bool reload_engines();

    // Returns the per-search config, enabling root noise only while both
    // boards are still inside the configured opening horizon.
    SearchParams::RuntimeConfig current_search_config();

public:
    UCI();
    ~UCI();

    // Initialize engines on the specified GPU devices.
    // For each device ID in deviceIds, a new Engine is constructed.
    // The arguments are retained so reload_engines() can repeat the setup.
    bool initializeEngines(
        const std::vector<int>& deviceIdsToUse,
        const std::string& networkPathToUse = {},
        int batchSizeToUse = SearchParams::BATCH_SIZE);

    void send_uci_response();
    void go(std::istringstream& is);
    void ponderhit();
    void setoption(std::istringstream& is);
    void stop();
    void new_game();
    void position(std::istringstream& is);
    void policy();
    void loop();
};
