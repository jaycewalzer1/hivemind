#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/time_state.h"
#include "interface/uci.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"
#include "search/searchthread.h"

// Wave 2c, fix 3: plan carry (docs/custom_engine_spec_v5_continuity.md,
// acceptance tests 3.1-3.9, the fix-3 half of 1.17, and 4.1). A durative
// search remembers the halves of the pair it chose for the boards our team
// is on turn on; the team's next search mixes a half that is still valid into
// its root prior and expands the pair holding it first.

class AgentContinuityFix3Peer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
    static std::array<CarriedPlan, 2>& plans(Agent& agent) { return agent.plans_; }
    static size_t root_candidates(Agent& agent) { return agent.nextRootCandidates_.size(); }
    static size_t mate_continuations(Agent& agent) { return agent.mateContinuations_.size(); }
    static void plant_mate_continuation(Agent& agent) { agent.mateContinuations_.push_back({}); }
    static std::shared_ptr<Node> team_tree_root(Agent& agent) {
        return agent.teamTree_ ? agent.teamTree_->root : nullptr;
    }
    /** resolve_plans as run_search calls it, after fix 1 reported @p reuse. */
    static void resolve(Agent& agent, Board& board, Stockfish::Color team, const TimeState& clock,
                        const TeamTreeReuseReport& reuse, SearchParams::RuntimeConfig& config,
                        int permille = 300) {
        agent.searchTime_ = clock;
        agent.lastTeamTreeReuse_ = reuse;
        agent.resolve_plans(board, team, permille, config);
    }
};

class UCIContinuityFix3Peer {
public:
    static Agent& agent(UCI& uci) { return *uci.agent; }
    static bool searching(const UCI& uci) { return uci.ongoingSearch.load(); }
    static bool plan_carry(const UCI& uci) { return uci.planCarry; }
    static bool team_tree_reuse(const UCI& uci) { return uci.teamTreeReuse; }
    static bool joint_plans(const UCI& uci) { return uci.durativeConfig.jointPlans; }
    static int plan_carry_permille(const UCI& uci) { return uci.planCarryPermille; }
};

namespace {

using Stockfish::BLACK;
using Stockfish::MOVE_NONE;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

// Few legal moves on each board. White team (Aw, Bb) on turn on both boards
// with kOursA | kOursB; only on A with kOursA | kTheirsB.
constexpr const char* kOursA = "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1";
constexpr const char* kOursB = "4k3/p7/8/8/8/8/P7/4K3[] b - - 0 1";
constexpr const char* kTheirsB = "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1";

Stockfish::Move uci_move(Board& board, int boardNumber, std::string text) {
    return Stockfish::UCI::to_move(*board.pos[boardNumber], text);
}

struct Clocks {
    std::array<int32_t, 4> clocks{60000, 60000, 60000, 60000};
    std::array<int32_t, 4> paces{100, 100, 100, 100};
    std::array<int32_t, 4> elapsed{0, 0, 0, 0};
    std::array<bool, 4> sitting{false, false, false, false};
    int32_t rootMoveMs = 100;
    bool flagLoss = true;
    bool instantTransfer = false;
    bool jointPlans = false;
    std::array<SeatLanding, MAX_RECENT_LANDINGS> landings{};
};

/** A live root as UCI builds it (current_clock): @p team decides now. */
TimeState live_clock(Board& board, Stockfish::Color team, const Clocks& c) {
    DurativeConfig config;
    config.flagLoss = c.flagLoss;
    config.instantTransfer = c.instantTransfer;
    config.rootMoveTimeMs = c.rootMoveMs;
    config.seatClocksMs = c.clocks;
    config.seatPaceMs = c.paces;
    config.seatElapsedMs = c.elapsed;
    config.seatSitting = c.sitting;
    config.recentLandings = c.landings;
    config.jointPlans = c.jointPlans;
    const TimeState base = TimeState::clocked(team, 0, 100, 100).with_sit_gate(false);
    return TimeState::durative(base, team, board.side_to_move(BOARD_A),
                               board.side_to_move(BOARD_B), config);
}

/**
 * The live root at a tree node whose path time is @p path (as test 1.2
 * builds it): the same clocks plus @p offsets, and the other team's seats on
 * turn with the elapsed time that gives them the path's ready time.
 */
TimeState live_at(Board& board, Stockfish::Color team, const TimeState& path,
                  const Clocks& base, std::array<int32_t, 4> offsets = {0, 0, 0, 0}) {
    Clocks c = base;
    for (int seat = 0; seat < 4; ++seat) {
        c.clocks[seat] = path.seat_clock_ms(seat) + offsets[seat];
        c.elapsed[seat] = 0;
        c.sitting[seat] = false;
    }
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        const int seat = TimeState::seat_of(boardNumber, board.side_to_move(boardNumber));
        if (TimeState::team_of_seat(seat) != team) {
            c.elapsed[seat] = c.paces[seat] - path.seat_ready_ms(seat);
            c.sitting[seat] = path.seat_sitting(seat);
        }
    }
    return live_clock(board, team, c);
}

Board make_board(const std::string& fenA, const std::string& fenB) {
    Board board;
    board.set(fenA + " | " + fenB);
    return board;
}

/** The position a FEN-only front end (the arena) sends for @p board. */
Board from_fens(const Board& board) {
    return make_board(board.fen(BOARD_A), board.fen(BOARD_B));
}

CarriedPlan plan_for(Board& board, int boardNumber, Stockfish::Move move,
                     Stockfish::Color team = WHITE) {
    CarriedPlan plan;
    plan.board = boardNumber;
    plan.move = move;
    plan.fen = board.fen(boardNumber);
    plan.team = team;
    return plan;
}

float prior_of(const std::vector<std::pair<Stockfish::Move, float>>& priors,
               Stockfish::Move move) {
    for (const auto& [m, p] : priors) {
        if (m == move) {
            return p;
        }
    }
    return -1.0f;
}

float sum_of(const std::vector<std::pair<Stockfish::Move, float>>& priors) {
    float total = 0.0f;
    for (const auto& entry : priors) {
        total += entry.second;
    }
    return total;
}

bool holds(const JointActionCandidate& action, int boardNumber, Stockfish::Move move) {
    return (boardNumber == BOARD_A ? action.moveA : action.moveB) == move;
}

// ---- 0.1 (fix 3 part), 3.5 without a search ---------------------------------

TEST(ContinuityDefaults, PlanCarryOptionsAdvertisedAndOff) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);
    for (const char* line : {"option name PlanCarry type check default false",
                             "option name PlanCarryPermille type spin default 300 min 0 max 1000",
                             "option name Continuity type check default false"}) {
        EXPECT_NE(output.str().find(line), std::string::npos) << line;
    }
    const SearchOptions defaults;
    EXPECT_FALSE(defaults.planCarry);
    EXPECT_EQ(defaults.planCarryPermille, 300);
    const SearchParams::RuntimeConfig config;
    EXPECT_EQ(config.rootPlanMove[0], MOVE_NONE);
    EXPECT_EQ(config.rootPlanMove[1], MOVE_NONE);
    EXPECT_EQ(config.rootPlanWeight, 0.0f);
}

TEST(ContinuityDefaults, ContinuityPresetSetsTheThreeSwitches) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    std::istringstream knob("name PlanCarryPermille value 450");
    uci.setoption(knob);
    std::istringstream on("name Continuity value true");
    uci.setoption(on);
    std::cout.rdbuf(previous);
    EXPECT_TRUE(UCIContinuityFix3Peer::plan_carry(uci));
    EXPECT_TRUE(UCIContinuityFix3Peer::team_tree_reuse(uci));
    EXPECT_TRUE(UCIContinuityFix3Peer::joint_plans(uci));
    EXPECT_EQ(UCIContinuityFix3Peer::plan_carry_permille(uci), 450);
    // DurativeTree is off: the engine says so once.
    EXPECT_NE(output.str().find("info string Continuity acts only with DurativeTree true"),
              std::string::npos) << output.str();
    // Last write wins: one switch off after the preset, then the preset off.
    std::istringstream carryOff("name PlanCarry value false");
    uci.setoption(carryOff);
    EXPECT_FALSE(UCIContinuityFix3Peer::plan_carry(uci));
    EXPECT_TRUE(UCIContinuityFix3Peer::team_tree_reuse(uci));
    std::istringstream off("name Continuity value false");
    uci.setoption(off);
    EXPECT_FALSE(UCIContinuityFix3Peer::plan_carry(uci));
    EXPECT_FALSE(UCIContinuityFix3Peer::team_tree_reuse(uci));
    EXPECT_FALSE(UCIContinuityFix3Peer::joint_plans(uci));
    EXPECT_EQ(UCIContinuityFix3Peer::plan_carry_permille(uci), 450);
}

TEST(PlanCarryRules, MixKeepsPriorsNormalised) {
    std::vector<float> priors = {0.5f, 0.3f, 0.2f};
    mix_root_plan(priors, 2, 0.3f);
    EXPECT_NEAR(priors[0], 0.35f, 1e-6f);
    EXPECT_NEAR(priors[1], 0.21f, 1e-6f);
    EXPECT_NEAR(priors[2], 0.14f + 0.3f, 1e-6f);
    EXPECT_NEAR(priors[0] + priors[1] + priors[2], 1.0f, 1e-6f);
}

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class PlanCarry : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        engine = std::make_unique<Engine>(0, 1);
        const auto model = std::filesystem::path(__FILE__).parent_path()
            / "data" / "uniform.onnx";
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    }

    static SearchOptions options(const TimeState& clock, size_t nodes, bool carry,
                                 bool teamTree = false, bool verbose = false) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.verbose = verbose;
        opts.clock = clock;
        opts.planCarry = carry;
        opts.teamTreeReuse = teamTree;
        // The time-boxed root scans depend on wall time, not on the tree.
        opts.search.enableRootMateSearch = false;
        opts.search.certifySelectedMove = false;
        opts.search.enableMateProbe = false;
        return opts;
    }

    /** Runs one search, returning what it printed (verbose) or "". */
    std::string search(Agent& agent, Board& board, Stockfish::Color team,
                       const SearchOptions& opts, JointActionCandidate* chosen = nullptr) {
        testing::internal::CaptureStdout();
        const JointActionCandidate action =
            agent.run_search(board, {engine.get()}, team, false, opts);
        if (chosen) {
            *chosen = action;
        }
        return testing::internal::GetCapturedStdout();
    }

    std::unique_ptr<Engine> engine;
};

// ---- 3.3 ---------------------------------------------------------------------

TEST_F(PlanCarry, NoPassPlans) {
    Board board = make_board(kOursA, kOursB);
    Agent agent(1);
    const Stockfish::Move a = uci_move(board, BOARD_A, "a2a3");
    const Stockfish::Move b = uci_move(board, BOARD_B, "a7a6");
    JointActionCandidate action;
    action.moveA = a;
    action.moveB = MOVE_NONE;   // a pass on B: nothing to carry there
    agent.record_plans(board, WHITE, action, {true, true});
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, a);
    EXPECT_EQ(agent.carried_plans()[BOARD_A].team, WHITE);
    EXPECT_EQ(agent.carried_plans()[BOARD_A].fen, board.fen(BOARD_A));
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, MOVE_NONE);
    // Each record replaces both slots; a board outside `boards` is cleared.
    action.moveB = b;
    agent.record_plans(board, WHITE, action, {false, true});
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, b);
    // A pass on both boards records nothing.
    agent.record_plans(board, WHITE, JointActionCandidate(), {true, true});
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, MOVE_NONE);

    // The end of a search: B is the other team's board, so only an A half
    // can be recorded, and only when the chosen A half is a move.
    Board ours = make_board(kOursA, kTheirsB);
    JointActionCandidate chosen;
    search(agent, ours, WHITE, options(live_clock(ours, WHITE, Clocks{}), 64, true), &chosen);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, MOVE_NONE);
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, chosen.moveA);
}

// ---- 3.2 ---------------------------------------------------------------------

TEST_F(PlanCarry, DroppedWhenBoardChangedOrIllegal) {
    struct Case {
        std::string name;
        std::function<CarriedPlan(Board&)> plant;
        std::string reason;
    };
    const std::vector<Case> cases = {
        {"board changed",
         [](Board& live) {
             Board before = make_board(kOursA, "4k3/8/p7/8/8/8/P7/4K3[] b - - 0 1");
             CarriedPlan plan = plan_for(before, BOARD_B, uci_move(before, BOARD_B, "e8d8"));
             (void)live;
             return plan;
         },
         "board changed"},
        {"pocket",
         [](Board& live) {
             // When the plan was made Bb held a knight; now it does not.
             Board before = make_board(kOursA, "4k3/p7/8/8/8/8/P7/4K3[n] b - - 0 1");
             CarriedPlan plan = plan_for(before, BOARD_B, uci_move(live, BOARD_B, "a7a6"));
             return plan;
         },
         "pocket"},
        {"illegal",
         [](Board& live) {
             return plan_for(live, BOARD_B, Stockfish::make_move(Stockfish::SQ_A7, Stockfish::SQ_A2));
         },
         "illegal"},
        {"other team",
         [](Board& live) {
             return plan_for(live, BOARD_B, uci_move(live, BOARD_B, "a7a6"), BLACK);
         },
         "other team"},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        Board live = make_board(kOursA, kOursB);
        Agent agent(1);
        AgentContinuityFix3Peer::plans(agent)[BOARD_B] = c.plant(live);
        const std::string out = search(agent, live, WHITE,
                                       options(live_clock(live, WHITE, Clocks{}), 32, true,
                                               false, true));
        const PlanCarryReport& report = agent.plan_carry();
        EXPECT_TRUE(report.attempted);
        EXPECT_FALSE(report.applied[BOARD_B]);
        EXPECT_EQ(report.reason[BOARD_B], c.reason);
        EXPECT_NE(out.find("dropped (" + c.reason + ")"), std::string::npos) << out;
        EXPECT_EQ(out.find(" weight "), std::string::npos) << out;
    }

    // Not on turn: B now belongs to the other team.
    Board theirs = make_board(kOursA, kTheirsB);
    Agent agent(1);
    Board before = make_board(kOursA, kOursB);
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] =
        plan_for(before, BOARD_B, uci_move(before, BOARD_B, "a7a6"));
    search(agent, theirs, WHITE, options(live_clock(theirs, WHITE, Clocks{}), 32, true));
    EXPECT_EQ(agent.plan_carry().reason[BOARD_B], "not on turn");

    // A pocket that only grew keeps the plan.
    Board grown = make_board(kOursA, "4k3/p7/8/8/8/8/P7/4K3[nP] b - - 0 1");
    Agent keeps(1);
    AgentContinuityFix3Peer::plans(keeps)[BOARD_B] =
        plan_for(before, BOARD_B, uci_move(before, BOARD_B, "a7a6"));
    search(keeps, grown, WHITE, options(live_clock(grown, WHITE, Clocks{}), 32, true));
    EXPECT_TRUE(keeps.plan_carry().applied[BOARD_B]) << keeps.plan_carry().reason[BOARD_B];
}

// ---- 3.7, 3.9 (fresh root) -----------------------------------------------------

TEST_F(PlanCarry, BothBoardsAndHeldBackDrop) {
    Board board = make_board(kOursA, kOursB);
    const TimeState clock = live_clock(board, WHITE, Clocks{});
    // Without a plan: the root's priors as the network gives them.
    Agent plain(1);
    search(plain, board, WHITE, options(clock, 32, true));
    const auto plainA = AgentContinuityFix3Peer::root(plain)->board_priors(BOARD_A);
    const auto plainB = AgentContinuityFix3Peer::root(plain)->board_priors(BOARD_B);
    // Plans on both boards that are not the first actions without them.
    const Stockfish::Move planA = plainA.back().first != MOVE_NONE
        ? plainA.back().first : plainA[plainA.size() - 2].first;
    const Stockfish::Move planB = plainB.back().first != MOVE_NONE
        ? plainB.back().first : plainB[plainB.size() - 2].first;
    ASSERT_NE(planA, plainA.front().first);
    ASSERT_NE(planB, plainB.front().first);

    Agent agent(1);
    AgentContinuityFix3Peer::plans(agent)[BOARD_A] = plan_for(board, BOARD_A, planA);
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] = plan_for(board, BOARD_B, planB);
    const std::string out = search(agent, board, WHITE, options(clock, 32, true, false, true));
    const PlanCarryReport& report = agent.plan_carry();
    EXPECT_TRUE(report.applied[BOARD_A]) << report.reason[BOARD_A];
    EXPECT_TRUE(report.applied[BOARD_B]) << report.reason[BOARD_B];
    EXPECT_NE(out.find("info string Plan carry: A " + board.uci_move(BOARD_A, planA)
                       + " weight 0.3"), std::string::npos) << out;
    EXPECT_NE(out.find("info string Plan carry: B " + board.uci_move(BOARD_B, planB)
                       + " weight 0.3"), std::string::npos) << out;
    const std::shared_ptr<Node> root = AgentContinuityFix3Peer::root(agent);
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        const auto& before = boardNumber == BOARD_A ? plainA : plainB;
        const Stockfish::Move plan = boardNumber == BOARD_A ? planA : planB;
        const auto after = root->board_priors(boardNumber);
        ASSERT_EQ(after.size(), before.size());
        for (const auto& [move, p] : before) {
            const float expected = 0.7f * p + (move == plan ? 0.3f : 0.0f);
            EXPECT_NEAR(prior_of(after, move), expected, 1e-5f);
        }
        EXPECT_NEAR(sum_of(after), 1.0f, 1e-5f);
        // Re-sorted: the carried move now leads its board.
        EXPECT_EQ(after.front().first, plan);
    }
    // 3.9, fresh root: the first child expanded is the pair holding both.
    const JointActionCandidate first = root->get_joint_action(0);
    EXPECT_TRUE(holds(first, BOARD_A, planA) && holds(first, BOARD_B, planB))
        << board.uci_move(BOARD_A, first.moveA) << " " << board.uci_move(BOARD_B, first.moveB);
    // Without plans the first child is the network's best pair.
    const JointActionCandidate plainFirst =
        AgentContinuityFix3Peer::root(plain)->get_joint_action(0);
    EXPECT_FALSE(holds(plainFirst, BOARD_A, planA) && holds(plainFirst, BOARD_B, planB));

    // A planned drop of a piece that landed in Bb's pocket 300 ms into its
    // 600 ms decision (InstantTransfer off). At a live root the root team
    // decides anew (TimeState::durative holds pieces back only for the other
    // team's seats), so the drop is one of the root's actions and the plan
    // applies; the board's priors still sum to 1. The spec's held-back case
    // cannot arise at a root (Build corrections, fix 3).
    Board held = make_board(kOursA, "4k3/p7/8/8/8/8/P7/4K3[n] b - - 0 1");
    Clocks c;
    c.elapsed[BB] = 600;
    c.landings[0] = SeatLanding{static_cast<int8_t>(BB), Stockfish::KNIGHT, 300};
    const TimeState heldClock = live_clock(held, WHITE, c);
    EXPECT_EQ(heldClock.pending_in_hand(BB, Stockfish::KNIGHT), 0);
    Stockfish::Move drop = MOVE_NONE;
    for (Stockfish::Move move : held.legal_moves(BOARD_B)) {
        if (type_of(move) == Stockfish::DROP) {
            drop = move;
            break;
        }
    }
    ASSERT_NE(drop, MOVE_NONE);
    Agent dropper(1);
    AgentContinuityFix3Peer::plans(dropper)[BOARD_B] = plan_for(held, BOARD_B, drop);
    AgentContinuityFix3Peer::plans(dropper)[BOARD_A] = plan_for(held, BOARD_A, planA);
    search(dropper, held, WHITE, options(heldClock, 32, true));
    EXPECT_TRUE(dropper.plan_carry().applied[BOARD_B]) << dropper.plan_carry().reason[BOARD_B];
    EXPECT_TRUE(dropper.plan_carry().applied[BOARD_A]);
    const auto heldB = AgentContinuityFix3Peer::root(dropper)->board_priors(BOARD_B);
    EXPECT_NEAR(sum_of(heldB), 1.0f, 1e-5f);
    EXPECT_EQ(heldB.front().first, drop);
}

TEST(PlanCarryRules, PromotionSurvivesRootConfiguration) {
    // Spec v5 3.9 at the node level, with priors where the carried move is
    // the least likely one and a weight too small to reorder the list: only
    // the promotion can make the pair holding it the first child.
    init_fairy_stockfish();
    Board board = make_board(kOursA, kTheirsB);
    std::vector<Stockfish::Move> actionsA = board.legal_moves(BOARD_A);
    actionsA.push_back(MOVE_NONE);
    const std::vector<Stockfish::Move> actionsB = {MOVE_NONE};
    std::vector<float> priorsA(actionsA.size());
    for (size_t i = 0; i < priorsA.size(); ++i) {
        priorsA[i] = static_cast<float>(priorsA.size() - i);
    }
    const float total = std::accumulate(priorsA.begin(), priorsA.end(), 0.0f);
    for (float& p : priorsA) {
        p /= total;
    }
    const std::vector<float> priorsB = {1.0f};
    const Stockfish::Move plan = actionsA[actionsA.size() - 2];   // last real move
    SearchParams::RuntimeConfig config;
    config.rootPlanMove[BOARD_A] = plan;
    config.rootPlanWeight = 0.001f;

    // Fresh root: mixed and promoted at its expansion, before its first child.
    auto fresh = std::make_shared<Node>(WHITE);
    ASSERT_TRUE(fresh->try_init_and_expand(actionsA, actionsB, priorsA, priorsB, true, true,
                                           false, config));
    EXPECT_NE(fresh->board_priors(BOARD_A).front().first, plan);
    EXPECT_EQ(fresh->get_joint_action(0).moveA, plan);
    // Without the plan the first child is the most likely move.
    auto plain = std::make_shared<Node>(WHITE);
    ASSERT_TRUE(plain->try_init_and_expand(actionsA, actionsB, priorsA, priorsB, true, true,
                                           false, SearchParams::RuntimeConfig{}));
    EXPECT_EQ(plain->get_joint_action(0).moveA, actionsA.front());
    // Below the root (depth > 0) the carried move is ignored.
    auto inner = std::make_shared<Node>(WHITE);
    inner->set_depth(2);
    ASSERT_TRUE(inner->try_init_and_expand(actionsA, actionsB, priorsA, priorsB, true, true,
                                           false, config));
    EXPECT_EQ(inner->get_joint_action(0).moveA, actionsA.front());
    EXPECT_NEAR(prior_of(inner->board_priors(BOARD_A), plan), priorsA[actionsA.size() - 2],
                1e-6f);

    // An expanded root (fix 1's adopted one): a promotion made before
    // configure_root_search is dropped by it; made after, it is the next
    // child expanded.
    const Stockfish::Move second = actionsA[actionsA.size() - 3];
    ASSERT_TRUE(plain->apply_root_plan(BOARD_A, second, 0.001f));
    ASSERT_TRUE(plain->promote_carried_plan(second, MOVE_NONE));
    plain->configure_root_search(SearchParams::RuntimeConfig{});
    EXPECT_FALSE(plain->promoted_candidate());
    ASSERT_TRUE(plain->promote_carried_plan(second, MOVE_NONE));
    ASSERT_TRUE(plain->promoted_candidate());
    EXPECT_EQ(plain->promoted_candidate()->moveA, second);
    JointActionCandidate expanded;
    plain->expand_next_joint_child(nullptr, 0, expanded, SearchParams::RuntimeConfig{});
    EXPECT_EQ(expanded.moveA, second);
    EXPECT_EQ(plain->get_joint_action(1).moveA, second);
    // A pair already expanded is not promoted again; a move off the list
    // is no plan at all.
    EXPECT_FALSE(plain->promote_carried_plan(second, MOVE_NONE));
    EXPECT_FALSE(plain->apply_root_plan(
        BOARD_A, Stockfish::make_move(Stockfish::SQ_H1, Stockfish::SQ_H8), 0.3f));
}

// ---- 3.4, 3.9 (adopted root), and the fix-2 source of a plan ------------------

/**
 * White decides on A (Bw on B lands late); after its move and Ab's reply the
 * game reaches a node of the tree. Returns that node's board, path time and
 * node, as test 1.2 builds them.
 */
struct Reached {
    Board board;
    TimeState time;
    std::shared_ptr<Node> node;
    Clocks clocks;
};
std::optional<Reached> white_then_black_reply(PlanCarry& fixture, Agent& agent, size_t nodes,
                                              bool carry) {
    Reached reached;
    reached.board.set(std::string(kOursA) + " | " + kTheirsB);
    reached.clocks.paces = {100, 100, 5000, 100};
    reached.time = live_clock(reached.board, WHITE, reached.clocks);
    fixture.search(agent, reached.board, WHITE,
                   PlanCarry::options(reached.time, nodes, carry, true));
    reached.node = AgentContinuityFix3Peer::root(agent);
    for (int step = 0; step < 2; ++step) {
        // The most visited unsolved expanded child that moves on A.
        const std::vector<int> visits = reached.node->get_child_visits();
        const auto children = reached.node->get_children();
        int best = -1;
        for (size_t i = 0; i < children.size() && i < visits.size(); ++i) {
            const JointActionCandidate action = reached.node->get_joint_action(static_cast<int>(i));
            if (children[i] && children[i]->is_expanded()
                && children[i]->get_node_type() == NodeType::UNSOLVED
                && action.moveA != MOVE_NONE && action.moveB == MOVE_NONE
                && (best < 0 || visits[i] > visits[static_cast<size_t>(best)])) {
                best = static_cast<int>(i);
            }
        }
        if (best < 0) {
            return std::nullopt;
        }
        const JointActionCandidate action = reached.node->get_joint_action(best);
        reached.time = durative_time_after(reached.time, reached.board,
                                           reached.node->get_team_to_play(), action);
        reached.board.make_moves(action.moveA, action.moveB);
        reached.node = children[static_cast<size_t>(best)];
    }
    return reached;
}

TEST_F(PlanCarry, AdoptedRootReweighted) {
    Agent agent(1);
    std::optional<Reached> reached = white_then_black_reply(*this, agent, 600, true);
    ASSERT_TRUE(reached);
    const std::shared_ptr<Node> expected = reached->node;
    ASSERT_EQ(expected->get_team_to_play(), WHITE);
    // The A half white played was recorded, and A has changed since.
    ASSERT_NE(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    // A move of the reached node that has no expanded edge yet.
    const auto before = expected->board_priors(BOARD_A);
    std::vector<Stockfish::Move> expandedA;
    for (int i = 0; i < static_cast<int>(expected->get_num_generated()); ++i) {
        expandedA.push_back(expected->get_joint_action(i).moveA);
    }
    Stockfish::Move plan = MOVE_NONE;
    for (const auto& [move, p] : before) {
        if (move != MOVE_NONE
            && std::find(expandedA.begin(), expandedA.end(), move) == expandedA.end()) {
            plan = move;
        }
    }
    ASSERT_NE(plan, MOVE_NONE) << "every A move already expanded at the reached node";
    const size_t generatedBefore = expected->get_num_generated();
    // As fix 2's commitment would: a plan for the move the game now stands at.
    Board live = from_fens(reached->board);
    AgentContinuityFix3Peer::plans(agent)[BOARD_A] = plan_for(live, BOARD_A, plan);
    const TimeState clock = live_at(live, WHITE, reached->time, reached->clocks, {3, 5, 0, 2});
    SearchOptions opts = options(clock, 1, true, true, true);
    const std::string out = search(agent, live, WHITE, opts);
    ASSERT_TRUE(agent.team_tree_reuse().adopted) << agent.team_tree_reuse().reason << out;
    ASSERT_EQ(AgentContinuityFix3Peer::root(agent), expected);
    ASSERT_TRUE(agent.plan_carry().applied[BOARD_A]) << agent.plan_carry().reason[BOARD_A];
    EXPECT_NE(out.find("info string Plan carry: A " + live.uci_move(BOARD_A, plan)
                       + " weight 0.3"), std::string::npos) << out;
    // Priors mixed, the list re-sorted, and every expanded edge rescored
    // from the mixed lists (configure_root_search after apply_root_plan).
    const auto after = expected->board_priors(BOARD_A);
    for (const auto& [move, p] : before) {
        EXPECT_NEAR(prior_of(after, move), 0.7f * p + (move == plan ? 0.3f : 0.0f), 1e-5f);
    }
    EXPECT_EQ(after.front().first, plan);
    const auto priorsB = expected->board_priors(BOARD_B);
    for (int i = 0; i < static_cast<int>(generatedBefore); ++i) {
        const JointActionCandidate action = expected->get_joint_action(i);
        EXPECT_NEAR(expected->get_child_prior(i),
                    prior_of(after, action.moveA) * prior_of(priorsB, action.moveB), 1e-5f);
    }
    // 3.9, adopted root: the next new edge holds the plan, whether the one
    // search node opened it or it is still waiting at the front.
    if (expected->get_num_generated() > generatedBefore) {
        EXPECT_EQ(expected->get_joint_action(static_cast<int>(generatedBefore)).moveA, plan);
    } else {
        const std::optional<JointActionCandidate> pending = expected->promoted_candidate();
        ASSERT_TRUE(pending);
        EXPECT_EQ(pending->moveA, plan);
    }
}

TEST_F(PlanCarry, CommittedRefusalBecomesThePlanWhenTheSlotIsEmpty) {
    // Fix 1 refused a board-matching node only for a commitment of one of
    // our seats: that commitment is the tree's plan for that seat (fix 2's
    // second source). The slot written at the end of the last search wins,
    // and the other team's commitments are never recorded.
    Board board = make_board(kOursA, kOursB);
    const TimeState clock = live_clock(board, WHITE, Clocks{});
    const Stockfish::Move committedB = uci_move(board, BOARD_B, "a7a6");
    const Stockfish::Move committedA = uci_move(board, BOARD_A, "e1d1");
    TeamTreeReuseReport refused;
    refused.attempted = true;
    refused.reason = "committed";
    refused.committed[BB] = committedB;     // our seat on B
    refused.committed[AB] = committedA;     // the other team's seat on A: ignored
    {
        Agent agent(1);
        SearchParams::RuntimeConfig config;
        AgentContinuityFix3Peer::resolve(agent, board, WHITE, clock, refused, config);
        EXPECT_TRUE(agent.plan_carry().fromTree[BOARD_B]);
        EXPECT_TRUE(agent.plan_carry().applied[BOARD_B]);
        EXPECT_EQ(config.rootPlanMove[BOARD_B], committedB);
        EXPECT_EQ(config.rootPlanMove[BOARD_A], MOVE_NONE);
        EXPECT_FALSE(agent.plan_carry().fromTree[BOARD_A]);
        EXPECT_NEAR(config.rootPlanWeight, 0.3f, 1e-6f);
    }
    {
        Agent agent(1);
        const Stockfish::Move recorded = uci_move(board, BOARD_B, "e8d8");
        AgentContinuityFix3Peer::plans(agent)[BOARD_B] = plan_for(board, BOARD_B, recorded);
        SearchParams::RuntimeConfig config;
        AgentContinuityFix3Peer::resolve(agent, board, WHITE, clock, refused, config);
        EXPECT_FALSE(agent.plan_carry().fromTree[BOARD_B]);
        EXPECT_EQ(config.rootPlanMove[BOARD_B], recorded);
    }
    {
        // Any other refusal reason is no source.
        Agent agent(1);
        TeamTreeReuseReport shape = refused;
        shape.reason = "shape";
        SearchParams::RuntimeConfig config;
        AgentContinuityFix3Peer::resolve(agent, board, WHITE, clock, shape, config);
        EXPECT_EQ(config.rootPlanMove[BOARD_B], MOVE_NONE);
        EXPECT_EQ(config.rootPlanWeight, 0.0f);
    }
}

// ---- review fixes: proven returns, missed ponders ------------------------------

TEST_F(PlanCarry, ProvenMateRecordsItsPairAndKeepsTheTree) {
    // A root proof returns before the end of the search (return_proven_mate,
    // return_proven_loss): it still records the pair it returns and keeps
    // the tree, so the partner's next search does not carry an older plan.
    // White mates on A (Ra8#); white's Bb is on turn on B.
    Board board = make_board("6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1", kOursB);
    const Stockfish::Move stale = uci_move(board, BOARD_B, "a7a6");
    Agent agent(1);
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] = plan_for(board, BOARD_B, stale);
    SearchOptions opts = options(live_clock(board, WHITE, Clocks{}), 400, true, true);
    opts.search.enableRootMateSearch = true;
    JointActionCandidate chosen;
    search(agent, board, WHITE, opts, &chosen);
    ASSERT_EQ(agent.root_type(), NodeType::WIN);
    ASSERT_EQ(board.uci_move(BOARD_A, chosen.moveA), "a1a8");
    // Both slots are the returned pair's halves (a pass records nothing).
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, chosen.moveA);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, chosen.moveB);
    EXPECT_NE(agent.carried_plans()[BOARD_B].move, stale);
    // The proof's tree is this team's retained tree.
    EXPECT_TRUE(agent.has_team_tree());
    EXPECT_EQ(AgentContinuityFix3Peer::team_tree_root(agent), AgentContinuityFix3Peer::root(agent));
}

TEST_F(PlanCarry, OldRootGetsNoSecondPlanMix) {
    // Fix 1 re-adopts the old root when no move landed since (the partner
    // decides while our move is in flight). Its priors already hold that
    // search's plan mix; the carried halves are not mixed in again on top
    // ("old root"), and no pair is promoted for them.
    Board board = make_board(kOursA, kOursB);
    Clocks clocks;
    Agent agent(1);
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] =
        plan_for(board, BOARD_B, uci_move(board, BOARD_B, "a7a6"));
    search(agent, board, WHITE, options(live_clock(board, WHITE, clocks), 200, true, true));
    ASSERT_TRUE(agent.plan_carry().applied[BOARD_B]) << agent.plan_carry().reason[BOARD_B];
    const std::shared_ptr<Node> oldRoot = AgentContinuityFix3Peer::root(agent);
    ASSERT_TRUE(oldRoot);
    const auto priorsA = oldRoot->board_priors(BOARD_A);
    const auto priorsB = oldRoot->board_priors(BOARD_B);
    // The search recorded its own pair: at least one half to carry.
    ASSERT_TRUE(agent.carried_plans()[BOARD_A].move != MOVE_NONE
                || agent.carried_plans()[BOARD_B].move != MOVE_NONE);
    clocks.clocks[AW] -= 1;
    clocks.clocks[BB] -= 1;
    search(agent, board, WHITE, options(live_clock(board, WHITE, clocks), 64, true, true));
    ASSERT_TRUE(agent.team_tree_reuse().adopted) << agent.team_tree_reuse().reason;
    ASSERT_TRUE(agent.team_tree_reuse().oldRoot);
    EXPECT_FALSE(agent.plan_carry().any_applied());
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        if (agent.plan_carry().move[boardNumber] != MOVE_NONE) {
            EXPECT_EQ(agent.plan_carry().reason[boardNumber], "old root");
        }
    }
    for (const auto& [before, boardNumber] :
         {std::pair{priorsA, BOARD_A}, std::pair{priorsB, BOARD_B}}) {
        const auto after = oldRoot->board_priors(boardNumber);
        ASSERT_EQ(after.size(), before.size());
        for (size_t i = 0; i < after.size(); ++i) {
            EXPECT_EQ(after[i].first, before[i].first);
            EXPECT_FLOAT_EQ(after[i].second, before[i].second);
        }
    }
}

TEST_F(PlanCarry, MissedPonderKeepsTheRealPlan) {
    // A ponder that is stopped guessed the reply wrong: as its team tree,
    // its pair is not recorded, so the real decision's carried half (slot B
    // here) survives it, and no half of the guess is remembered (slot A).
    Board board = make_board(kOursA, kOursB);
    const Stockfish::Move real = uci_move(board, BOARD_B, "a7a6");
    Agent agent(1);
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] = plan_for(board, BOARD_B, real);
    SearchOptions opts = options(live_clock(board, WHITE, Clocks{}), 0, true, true);
    opts.isPonder = true;
    testing::internal::CaptureStdout();
    std::thread ponder([&] { agent.run_search(board, {engine.get()}, WHITE, false, opts); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!agent.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    agent.set_is_running(false);
    ponder.join();
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(agent.plan_carry().applied[BOARD_B]) << agent.plan_carry().reason[BOARD_B];
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, real);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].edgeVisitShare, 0.0f);
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    EXPECT_FALSE(agent.has_team_tree());

    // The same search not pondering records its pair over the planted slot.
    Agent decided(1);
    AgentContinuityFix3Peer::plans(decided)[BOARD_B] = plan_for(board, BOARD_B, real);
    search(decided, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 64, true, true));
    EXPECT_GT(decided.carried_plans()[BOARD_A].edgeVisitShare
                  + decided.carried_plans()[BOARD_B].edgeVisitShare, 0.0f);
    EXPECT_TRUE(decided.has_team_tree());
}

// ---- 3.5, 3.8 (agent level) ----------------------------------------------------

TEST_F(PlanCarry, ClearedByClearPlansNotByResetSearchState) {
    Board board = make_board(kOursA, kOursB);
    Agent agent(1);
    search(agent, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 64, true));
    const bool any = agent.carried_plans()[BOARD_A].move != MOVE_NONE
        || agent.carried_plans()[BOARD_B].move != MOVE_NONE;
    ASSERT_TRUE(any);
    const std::array<CarriedPlan, 2> kept = agent.carried_plans();
    agent.reset_search_state();
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, kept[BOARD_A].move);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, kept[BOARD_B].move);
    agent.clear_plans();
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, MOVE_NONE);
}

TEST_F(PlanCarry, NoLeakAcrossTeams) {
    // The same agent searches for the other team next (a tournament side
    // after the swap): the white team's plan is never applied for black.
    Board board = make_board(kOursA, kOursB);
    Agent agent(1);
    search(agent, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 64, true));
    ASSERT_TRUE(agent.carried_plans()[BOARD_A].move != MOVE_NONE
                || agent.carried_plans()[BOARD_B].move != MOVE_NONE);
    // Black to play on A, and on B (as Bw): the same FENs with the turns flipped.
    Board black = make_board("4k3/p7/8/8/8/8/P7/4K3[] b - - 0 1", kTheirsB);
    const std::array<CarriedPlan, 2> kept = agent.carried_plans();
    search(agent, black, BLACK, options(live_clock(black, BLACK, Clocks{}), 16, true));
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        EXPECT_FALSE(agent.plan_carry().applied[boardNumber]);
        if (kept[boardNumber].move != MOVE_NONE) {
            EXPECT_EQ(agent.plan_carry().reason[boardNumber], "other team");
        }
    }
}

TEST_F(PlanCarry, OptionOffOrNonDurativeDoesNothing) {
    Board board = make_board(kOursA, kOursB);
    Agent agent(1);
    const Stockfish::Move planB = uci_move(board, BOARD_B, "a7a6");
    AgentContinuityFix3Peer::plans(agent)[BOARD_B] = plan_for(board, BOARD_B, planB);
    // Option off: the slot is neither read nor replaced.
    search(agent, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 16, false));
    EXPECT_FALSE(agent.plan_carry().attempted);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, planB);
    // Under a v3 clock the option does nothing either.
    search(agent, board, WHITE, options(TimeState::clocked(WHITE, 500), 16, true));
    EXPECT_FALSE(agent.plan_carry().attempted);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, planB);
}

// ---- 1.17, fix 3 alone ------------------------------------------------------------

TEST_F(PlanCarry, NoOtherStateSurvivesUnderPlanCarryAlone) {
    Board board = make_board(kOursA, kTheirsB);
    Agent agent(1);
    search(agent, board, WHITE, options(live_clock(board, WHITE, Clocks{}), 128, true));
    // The exact-hash candidates are kept as without the option...
    AgentContinuityFix3Peer::plant_mate_continuation(agent);
    const std::shared_ptr<Node> oldRoot = AgentContinuityFix3Peer::root(agent);
    // ...and dropped at the next search: nothing but the plan survives.
    const std::string out = search(agent, board, WHITE,
                                   options(live_clock(board, WHITE, Clocks{}), 32, true, false,
                                           true));
    EXPECT_EQ(AgentContinuityFix3Peer::mate_continuations(agent), 0u);
    EXPECT_NE(AgentContinuityFix3Peer::root(agent), oldRoot);
    EXPECT_EQ(out.find("info string Tree reuse:"), std::string::npos) << out;
    EXPECT_FALSE(AgentContinuityFix3Peer::team_tree_root(agent));
}

// ---- 4.1 (sequenced, Continuity on) -------------------------------------------------

// Spec v5 4.1: v4 acceptance verdicts with every wave 2c switch on, each
// checked on a second search that fix 1 or fix 3 fed.
class ContinuityAcceptance : public PlanCarry {};

/**
 * Spec v4 acceptance test 1 with every wave 2c switch on, searched twice: a
 * first search one opponent landing earlier, and the checked one after it
 * with clocks 3 ms off, where fix 1 adopts the reached node. Aw has 300 ms;
 * the first search's budget is 100 ms, so after its move Aw has 200 left.
 * The second search's budget is 1000 ms: every white root action flags Aw,
 * so the root is a proven loss although the retained subtree was searched
 * under a clock where nothing flagged.
 */
TEST_F(ContinuityAcceptance, Test1_SingleLoserAfterAnAdoptedTree) {
    Agent agent(1);
    Reached reached;
    reached.board.set(std::string(kOursA) + " | " + kTheirsB);
    reached.clocks.clocks = {300, 60000, 60000, 60000};
    reached.clocks.paces = {100, 100, 5000, 100};
    reached.clocks.jointPlans = true;
    reached.time = live_clock(reached.board, WHITE, reached.clocks);
    SearchOptions first = options(reached.time, 600, true, true);
    search(agent, reached.board, WHITE, first);
    reached.node = AgentContinuityFix3Peer::root(agent);
    for (int step = 0; step < 2; ++step) {
        const std::vector<int> visits = reached.node->get_child_visits();
        const auto children = reached.node->get_children();
        int best = -1;
        for (size_t i = 0; i < children.size() && i < visits.size(); ++i) {
            const JointActionCandidate action = reached.node->get_joint_action(static_cast<int>(i));
            if (children[i] && children[i]->is_expanded()
                && children[i]->get_node_type() == NodeType::UNSOLVED
                && action.moveA != MOVE_NONE && action.planned == MOVE_NONE
                && (best < 0 || visits[i] > visits[static_cast<size_t>(best)])) {
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0);
        const JointActionCandidate action = reached.node->get_joint_action(best);
        reached.time = durative_time_after(reached.time, reached.board,
                                           reached.node->get_team_to_play(), action);
        reached.board.make_moves(action.moveA, action.moveB);
        reached.node = children[static_cast<size_t>(best)];
    }
    ASSERT_EQ(reached.time.seat_clock_ms(AW), 200);
    Board live = from_fens(reached.board);
    Clocks budget = reached.clocks;
    budget.rootMoveMs = 1000;
    const TimeState clock = live_at(live, WHITE, reached.time, budget, {3, 3, 3, 3});
    const std::string out = search(agent, live, WHITE, options(clock, 200, true, true, true));
    ASSERT_TRUE(agent.team_tree_reuse().adopted) << agent.team_tree_reuse().reason << out;
    EXPECT_NE(out.find("info string Team tree reuse: "), std::string::npos);
    EXPECT_EQ(agent.root_type(), NodeType::LOSS);
    for (const RootEdgeStats& edge : agent.root_edge_stats()) {
        EXPECT_EQ(edge.childType, NodeType::WIN);   // black's view
    }
}

/**
 * Spec v4 acceptance test 6 with Continuity on, sequenced so that plan carry
 * fires: the black team is on turn on A (Ab) and on B (Bw). The first
 * search records Bw's half; then Ab plays ...Nf6, white replies Ng1 (v4 test
 * 6's position: ...Ng8 would complete a threefold), and the second search (B
 * unchanged) mixes the carried B half in. The durative tree still calls no
 * line a draw by repetition.
 */
TEST_F(ContinuityAcceptance, Test6_NoRepetitionDrawWithACarriedPlan) {
    Board board = make_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
                             "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1");
    for (const char* uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3"}) {
        board.push_move(BOARD_A, uci_move(board, BOARD_A, uci));
    }
    Clocks c;
    c.jointPlans = true;
    Agent agent(1);
    JointActionCandidate chosen;
    search(agent, board, BLACK, options(live_clock(board, BLACK, c), 200, true, true), &chosen);
    ASSERT_NE(chosen.moveB, MOVE_NONE) << "the first search passed on B";
    ASSERT_EQ(agent.carried_plans()[BOARD_B].move, chosen.moveB);
    // Ab plays ...Nf6, Aw replies Ng1: v4 test 6's position, where ...Ng8
    // would complete a threefold.
    board.push_move(BOARD_A, uci_move(board, BOARD_A, "g8f6"));
    board.push_move(BOARD_A, uci_move(board, BOARD_A, "f3g1"));
    const std::string out = search(agent, board, BLACK,
                                   options(live_clock(board, BLACK, c), 2000, true, true, true));
    ASSERT_TRUE(agent.plan_carry().applied[BOARD_B])
        << agent.plan_carry().reason[BOARD_B] << out;
    EXPECT_NE(out.find("info string Plan carry: B "), std::string::npos) << out;
    int draws = 0;
    std::function<void(Node*, int)> visit = [&](Node* node, int depth) {
        if (!node || depth > 40) {
            return;
        }
        draws += node->get_node_type() == NodeType::DRAW ? 1 : 0;
        if (!node->is_expanded()) {
            return;
        }
        for (const std::shared_ptr<Node>& child : node->get_children()) {
            visit(child.get(), depth + 1);
        }
    };
    visit(AgentContinuityFix3Peer::root(agent).get(), 0);
    EXPECT_EQ(draws, 0);
}

// ---- 3.5 (UCI) -------------------------------------------------------------------

class PlanCarryUci : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        const auto model = std::filesystem::path(__FILE__).parent_path()
            / "data" / "uniform.onnx";
        ASSERT_TRUE(uci.initializeEngines({0}, model.string(), 1));
        for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
                 {"TimeModel", "clock"}, {"DurativeTree", "true"}, {"SitGate", "false"},
                 {"FlagLoss", "true"}, {"InstantTransfer", "false"}, {"Team", "white"},
                 {"RootMoveTimeMs", "100"}, {"CertifySelectedMove", "false"},
                 {"PlanCarry", "true"}}) {
            send("setoption name " + name + " value " + value);
        }
    }

    void TearDown() override { uci.stop(); }

    std::string send(const std::string& line) {
        std::ostringstream output;
        std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
        std::istringstream is(line);
        std::string command;
        is >> command;
        if (command == "setoption") uci.setoption(is);
        else if (command == "position") uci.position(is);
        else if (command == "go") {
            uci.go(is);
            while (UCIContinuityFix3Peer::searching(uci)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            uci.stop();
        }
        else if (command == "ucinewgame") uci.new_game();
        std::cout.rdbuf(previous);
        return output.str();
    }

    UCI uci;
};

TEST_F(PlanCarryUci, ClearedByNewGame) {
    send("setoption name SeatClocksMs value 60000 60000 60000 60000");
    send(std::string("position fen ") + kOursA + " | " + kOursB);
    const std::string first = send("go nodes 200");
    ASSERT_NE(first.find("bestmove"), std::string::npos) << first;
    Agent& agent = UCIContinuityFix3Peer::agent(uci);
    ASSERT_TRUE(agent.carried_plans()[BOARD_A].move != MOVE_NONE
                || agent.carried_plans()[BOARD_B].move != MOVE_NONE) << first;
    // The same position again: the plans are applied and printed.
    const std::string second = send("go nodes 64");
    EXPECT_NE(second.find("info string Plan carry: "), std::string::npos) << second;
    EXPECT_NE(second.find(" weight 0.3"), std::string::npos) << second;
    send("ucinewgame");
    EXPECT_EQ(agent.carried_plans()[BOARD_A].move, MOVE_NONE);
    EXPECT_EQ(agent.carried_plans()[BOARD_B].move, MOVE_NONE);
    const std::string third = send("go nodes 32");
    EXPECT_EQ(third.find("info string Plan carry: "), std::string::npos) << third;
}

TEST_F(PlanCarryUci, SetoptionSaysItNeedsTheDurativeTree) {
    send("setoption name DurativeTree value false");
    const std::string out = send("setoption name PlanCarry value true");
    EXPECT_NE(out.find("info string PlanCarry set to true"), std::string::npos) << out;
    EXPECT_NE(out.find("info string PlanCarry acts only with DurativeTree true"),
              std::string::npos) << out;
    const std::string knob = send("setoption name PlanCarryPermille value 1500");
    EXPECT_NE(knob.find("info string PlanCarryPermille set to 1000"), std::string::npos) << knob;
}
// ---- 3.1 (it13) ----------------------------------------------------------------

class PlanCarryIt13 : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        const auto model = std::filesystem::path(__FILE__).parent_path().parent_path()
            / "models" / "twin-s-clock-it13.onnx";
        if (!std::filesystem::exists(model)) {
            GTEST_SKIP() << "needs engine/models/twin-s-clock-it13.onnx";
        }
        engine = std::make_unique<Engine>(0, 1);
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    }

    std::unique_ptr<Engine> engine;

    static SearchOptions opts(const TimeState& clock, size_t nodes, bool carry) {
        return PlanCarry::options(clock, nodes, carry);
    }
    JointActionCandidate run(Agent& agent, Board& board, Stockfish::Color team,
                             const SearchOptions& o) {
        testing::internal::CaptureStdout();
        const JointActionCandidate a = agent.run_search(board, {engine.get()}, team, false, o);
        testing::internal::GetCapturedStdout();
        return a;
    }
};

/**
 * Spec v5 3.1. Board A at the start (Aw to move), board B after 1.e4 (Bb to
 * move): the white team is on turn on both. Search 1 (200 nodes, plan carry
 * on) chooses (d2d4, e7e5). Aw's d2d4 lands and Ab replies d7d5 (black's
 * 64-node choice); B is unchanged, so Bb's half e7e5 is carried into search
 * 2 at 64 nodes. Recorded when the test was written (it13, one worker):
 * with plan carry the chosen B half is e7e5; without it, from a fresh agent
 * at the same position, b8c6. The test also checks the prior mix against
 * the plain root's priors and that the first root child holds e7e5.
 */
TEST_F(PlanCarryIt13, BiasesPartnersNextSearch) {
    Board board = make_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
                             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1");
    Agent agent(1);
    const JointActionCandidate one =
        run(agent, board, WHITE, opts(live_clock(board, WHITE, Clocks{}), 200, true));
    EXPECT_EQ(board.uci_move(BOARD_A, one.moveA), "d2d4");
    ASSERT_EQ(one.moveB, uci_move(board, BOARD_B, "e7e5"));
    ASSERT_EQ(agent.carried_plans()[BOARD_B].move, one.moveB);
    const Stockfish::Move mB = one.moveB;

    Board after(board);
    after.push_move(BOARD_A, one.moveA);
    Agent black(1);
    const JointActionCandidate reply =
        run(black, after, BLACK, opts(live_clock(after, BLACK, Clocks{}), 64, false));
    ASSERT_EQ(after.uci_move(BOARD_A, reply.moveA), "d7d5");
    after.push_move(BOARD_A, reply.moveA);
    Board live = from_fens(after);
    const TimeState clock = live_clock(live, WHITE, Clocks{});

    const JointActionCandidate carried = run(agent, live, WHITE, opts(clock, 64, true));
    ASSERT_TRUE(agent.plan_carry().applied[BOARD_B]) << agent.plan_carry().reason[BOARD_B];
    EXPECT_FALSE(agent.plan_carry().applied[BOARD_A]);   // A was played: dropped
    EXPECT_EQ(agent.plan_carry().reason[BOARD_A], "board changed");
    Agent fresh(1);
    const JointActionCandidate plain = run(fresh, live, WHITE, opts(clock, 64, false));
    EXPECT_EQ(carried.moveB, mB) << live.uci_move(BOARD_B, carried.moveB);
    EXPECT_NE(plain.moveB, mB);
    EXPECT_EQ(live.uci_move(BOARD_B, plain.moveB), "b8c6");

    // The root prior of e7e5 is (1 - w) p + w, every other B action (1 - w) p.
    const auto withPlan = AgentContinuityFix3Peer::root(agent)->board_priors(BOARD_B);
    const auto withoutPlan = AgentContinuityFix3Peer::root(fresh)->board_priors(BOARD_B);
    ASSERT_EQ(withPlan.size(), withoutPlan.size());
    for (const auto& [move, p] : withoutPlan) {
        EXPECT_NEAR(prior_of(withPlan, move), 0.7f * p + (move == mB ? 0.3f : 0.0f), 1e-4f);
    }
    // The pair holding it was expanded first.
    EXPECT_EQ(AgentContinuityFix3Peer::root(agent)->get_joint_action(0).moveB, mB);
}

#endif

}  // namespace
