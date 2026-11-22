#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"
#include "environment/joint_action.h"
#include "environment/time_state.h"
#include "interface/uci.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/hybrid.h"
#include "search/node.h"
#include "tools/seat_clock_game.h"
#include "tools/selfplay_policy.h"

// Spec v6, the hybrid (talisman docs/custom_engine_spec_v6_hybrid.md): the
// v3 joint-action tree chooses the move; the durative clock decides only
// whether the root may wait for a landing on the other board. H3 (HybridMode
// gate) is built: tests 0.2, 0.4, 1.1-1.6, 2.1-2.5 and 2.7. Off by default.

class AgentHybridPeer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
    static TimeState search_time(const Agent& agent) { return agent.searchTime_; }
    static void store_next_roots(Agent& agent, Board& board, const TimeState& time,
                                 const JointActionCandidate* played) {
        agent.store_next_root_candidates(board, time, played);
    }
    /** Whether tree reuse retained @p node (the subtree root it indexes first). */
    static bool retains(const Agent& agent, const Node* node) {
        for (const auto& [hash, candidate] : agent.nextRootCandidates_) {
            if (candidate.node.get() == node) {
                return true;
            }
        }
        return false;
    }
};

class UCIHybridPeer {
public:
    static const HybridKnobs& knobs(const UCI& uci) { return uci.hybrid; }
    static std::optional<TimeState> clock(const UCI& uci) { return uci.current_clock(); }
    static std::optional<TimeState> hybrid_clock(const UCI& uci) {
        return uci.current_hybrid_clock();
    }
};

namespace {

using Stockfish::BLACK;
using Stockfish::MOVE_NONE;
using Stockfish::WHITE;
constexpr int AW = TimeState::SEAT_AW;
constexpr int AB = TimeState::SEAT_AB;
constexpr int BW = TimeState::SEAT_BW;
constexpr int BB = TimeState::SEAT_BB;

// Spec v6 test 2.1: kSitForKnight (test_durative_part2.cc) one ply on. A: a
// knight drop on f7 mates (Aw, our seat, needs a knight). B: Bb (ours) has
// just played Qd4+; Bw (theirs) is in check and thinking. Every reply of Bw
// is a king move (no block, no capture, empty pocket: the knight on h4 can
// neither block the a1-d4 diagonal nor take the queen), and Qxh4 then wins
// the knight for Aw's pocket. White team = Aw + Bb, on turn on A only.
constexpr const char* kKnightA = "6rk/6pp/8/8/8/8/8/6K1[] w - - 0 1";
constexpr const char* kKnightB = "4k3/8/8/8/3q3N/8/8/K7[] w - - 1 2";
// B before Qd4+: our team on turn on both boards (test 1.3).
constexpr const char* kKnightBBefore = "4k3/3q4/8/8/7N/8/8/K7[] b - - 0 1";
// A white king stalemated by a queen (no legal move, empty pocket; test 1.1).
constexpr const char* kStalemateA = "k7/8/8/8/8/8/2q5/K7[] w - - 0 1";

Board boards(const std::string& a, const std::string& b) {
    Board board;
    board.set(a + " | " + b);
    return board;
}

/** The seat options of one decision of the white team (Aw on turn on A, Bw thinking on B). */
struct Seats {
    // Our lead -2 s: Aw - Bw = Bb - Ab = -2000.
    std::array<int32_t, 4> clocks{59000, 61000, 61000, 59000};
    std::array<int32_t, 4> paces{200, 300, 300, 200};
    std::array<int32_t, 4> elapsed{0, 0, 200, 0};
    std::array<bool, 4> sitting{false, false, false, false};
    int32_t rootMs = 200;
    int32_t passCapK = 8;
};

int32_t white_lead(const Seats& s) {
    return ((s.clocks[AW] - s.clocks[BW]) + (s.clocks[BB] - s.clocks[AB])) / 2;
}

/** The v3 clock the search runs on, with the sit gate (nodur). */
TimeState v3_clock(const Seats& s, Stockfish::Color team = WHITE) {
    const int32_t lead = white_lead(s) * (team == WHITE ? 1 : -1);
    return TimeState::clocked(team, lead, 200, 300, s.rootMs);
}

/** The durative clock of the decision, SitGate off: what the trigger reads. */
TimeState durative_clock(Board& board, const Seats& s, Stockfish::Color team = WHITE) {
    DurativeConfig config;
    config.rootMoveTimeMs = s.rootMs;
    config.passCapK = s.passCapK;
    config.seatClocksMs = s.clocks;
    config.seatPaceMs = s.paces;
    config.seatElapsedMs = s.elapsed;
    config.seatSitting = s.sitting;
    return TimeState::durative(v3_clock(s, team).with_sit_gate(false), team,
                               board.side_to_move(BOARD_A), board.side_to_move(BOARD_B), config);
}

HybridTrigger trigger(Board& board, const Seats& s, const HybridKnobs& knobs = HybridKnobs(),
                      Stockfish::Color team = WHITE) {
    return hybrid_sit_boards(board, team, v3_clock(s, team), durative_clock(board, s, team), knobs);
}

std::string capture_setoptions(UCI& uci, std::initializer_list<const char*> texts) {
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    for (const char* text : texts) {
        std::istringstream option(text);
        uci.setoption(option);
    }
    std::cout.rdbuf(previous);
    return output.str();
}

class HybridRules : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

// ---- 0.2 options ------------------------------------------------------------------

TEST_F(HybridRules, OptionsAdvertisedAndOff) {
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);
    for (const char* line : {
             "option name HybridMode type combo default off var off var gate",
             "option name Hybrid type check default false",
             "option name HybridEventLands type combo default all var all var sit var none",
             "option name HybridMaxWaitMs type spin default 1000 min 0 max 10000",
             "option name HybridMinClockMs type spin default 3000 min 0 max 600000",
             "option name HybridMarginPermille type spin default 20 min 0 max 1000",
             "option name HybridPassPriorPermille type spin default 150 min 0 max 1000"}) {
        EXPECT_NE(output.str().find(line), std::string::npos) << line;
    }
    EXPECT_EQ(UCIHybridPeer::knobs(uci), HybridKnobs());
    EXPECT_EQ(HybridKnobs().mode, HybridMode::OFF);
    EXPECT_FALSE(UCIHybridPeer::hybrid_clock(uci).has_value());
    const Wave2GameOptions game;
    EXPECT_EQ(game.hybrid, HybridKnobs());
    EXPECT_FALSE(game.changes_search());
    const SearchParams::RuntimeConfig config;
    EXPECT_EQ(config.hybridSitBoards, 0);
    EXPECT_EQ(config.hybridEventBoard, -1);
}

TEST_F(HybridRules, UciOptionsSetClampAndReport) {
    UCI uci;
    const std::string knobsOff = capture_setoptions(
        uci, {"name HybridMarginPermille value 5000", "name HybridMaxWaitMs value -3",
              "name HybridEventLands value sit"});
    EXPECT_EQ(UCIHybridPeer::knobs(uci).marginPermille, 1000);
    EXPECT_EQ(UCIHybridPeer::knobs(uci).maxWaitMs, 0);
    EXPECT_EQ(UCIHybridPeer::knobs(uci).eventLands, HybridEventLands::SIT);
    EXPECT_NE(knobsOff.find("info string HybridMarginPermille acts only with HybridMode gate"),
              std::string::npos) << knobsOff;
    // The preset sets and clears the mode; the knobs keep their values.
    const std::string preset = capture_setoptions(uci, {"name Hybrid value true"});
    EXPECT_EQ(UCIHybridPeer::knobs(uci).mode, HybridMode::GATE);
    EXPECT_EQ(UCIHybridPeer::knobs(uci).marginPermille, 1000);
    EXPECT_NE(preset.find("info string Hybrid set to true (HybridMode gate)"), std::string::npos)
        << preset;
    capture_setoptions(uci, {"name Hybrid value false"});
    EXPECT_EQ(UCIHybridPeer::knobs(uci).mode, HybridMode::OFF);
    // Modes the round did not build are refused and leave the mode alone.
    const std::string probe = capture_setoptions(
        uci, {"name HybridMode value gate", "name HybridMode value probe"});
    EXPECT_EQ(UCIHybridPeer::knobs(uci).mode, HybridMode::GATE);
    EXPECT_NE(probe.find("HybridMode probe is not built"), std::string::npos) << probe;
    // Under DurativeTree the hybrid is inert and says so.
    const std::string inert = capture_setoptions(uci, {"name DurativeTree value true"});
    EXPECT_NE(inert.find("info string Hybrid acts only with DurativeTree false"),
              std::string::npos) << inert;
    EXPECT_FALSE(UCIHybridPeer::hybrid_clock(uci).has_value());
}

TEST_F(HybridRules, UciHybridClockIsTheDurativeStateWithoutTheGate) {
    UCI uci;
    capture_setoptions(uci, {"name TimeModel value clock", "name Team value white",
                             "name TimeLeadMs value -2000", "name RootMoveTimeMs value 200",
                             "name SeatClocksMs value 59000 61000 61000 59000",
                             "name SeatPaceMs value 200 300 300 200",
                             "name SeatElapsedMs value 0 0 200 0"});
    std::istringstream position("fen " + std::string(kKnightA) + " | " + kKnightB);
    uci.position(position);
    EXPECT_FALSE(UCIHybridPeer::hybrid_clock(uci).has_value());
    capture_setoptions(uci, {"name HybridMode value gate"});
    const std::optional<TimeState> v3 = UCIHybridPeer::clock(uci);
    const std::optional<TimeState> durative = UCIHybridPeer::hybrid_clock(uci);
    ASSERT_TRUE(v3 && durative);
    // The search's own clock is unchanged (v3, gate on): only the trigger
    // reads the durative one.
    EXPECT_FALSE(v3->is_durative());
    EXPECT_TRUE(v3->sit_gate());
    EXPECT_TRUE(durative->is_durative());
    EXPECT_FALSE(durative->sit_gate());
    EXPECT_EQ(durative->seat_ready_ms(BW), 100);
    EXPECT_EQ(durative->seat_clock_ms(AW), 59000);
    Board board = boards(kKnightA, kKnightB);
    const HybridTrigger fired = hybrid_sit_boards(board, WHITE, *v3, *durative, HybridKnobs());
    EXPECT_EQ(fired.mask, 1) << fired.reason;
}

TEST_F(HybridRules, GameOptionsValidate) {
    const auto parsed = [](std::initializer_list<std::pair<const char*, const char*>> flags) {
        Wave2GameOptions options;
        for (const auto& [flag, value] : flags) {
            EXPECT_TRUE(parse_wave2_option(flag, value, options)) << flag;
        }
        return options;
    };
    const Wave2GameOptions ok = parsed({{"--hybrid-mode", "gate"}, {"--base-mix", "30000:1"},
                                        {"--hybrid-margin-permille", "50"}});
    EXPECT_NO_THROW(ok.validate());
    EXPECT_TRUE(ok.changes_search());
    EXPECT_EQ(ok.hybrid.marginPermille, 50);
    // The mode needs the v3 tree and seat clocks.
    EXPECT_THROW(parsed({{"--hybrid-mode", "gate"}}).validate(), std::invalid_argument);
    EXPECT_THROW(parsed({{"--hybrid", "true"}, {"--base-mix", "30000:1"},
                         {"--durative-tree", "true"}}).validate(),
                 std::invalid_argument);
    // A knob without its switch is refused, as are the modes not built.
    EXPECT_THROW(parsed({{"--hybrid-max-wait-ms", "500"}}).validate(), std::invalid_argument);
    EXPECT_THROW(parsed({{"--hybrid-event-lands", "none"}}).validate(), std::invalid_argument);
    EXPECT_THROW(parsed({{"--hybrid-mode", "probe"}, {"--base-mix", "30000:1"}}).validate(),
                 std::invalid_argument);
    EXPECT_THROW(parsed({{"--hybrid-mode", "gate"}, {"--base-mix", "30000:1"},
                         {"--hybrid-margin-permille", "1001"}}).validate(),
                 std::invalid_argument);
    Wave2GameOptions bad;
    EXPECT_THROW(parse_wave2_option("--hybrid-mode", "maybe", bad), std::invalid_argument);
    // The preset sets and clears the mode only.
    Wave2GameOptions preset = parsed({{"--hybrid-margin-permille", "70"}, {"--hybrid", "true"}});
    EXPECT_EQ(preset.hybrid.mode, HybridMode::GATE);
    EXPECT_EQ(preset.hybrid.marginPermille, 70);
    EXPECT_TRUE(parse_wave2_option("--hybrid", "false", preset));
    EXPECT_EQ(preset.hybrid.mode, HybridMode::OFF);
}

// ---- 0.4 state and node size ---------------------------------------------------------

TEST_F(HybridRules, EventMarkIsOneShotAndKeyedOnlyWhileSet) {
    const TimeState plain = TimeState::clocked(WHITE, -2000, 200, 300, 200);
    EXPECT_EQ(plain.hybrid_event_board(), -1);
    // Clearing a state that has no mark is the state itself, key and all.
    EXPECT_EQ(plain.with_hybrid_event(-1), plain);
    EXPECT_EQ(plain.with_hybrid_event(-1).exact_key(), plain.exact_key());
    const TimeState markedA = plain.with_hybrid_event(0);
    const TimeState markedB = plain.with_hybrid_event(1);
    EXPECT_EQ(markedA.hybrid_event_board(), 0);
    EXPECT_EQ(markedB.hybrid_event_board(), 1);
    EXPECT_NE(markedA.exact_key(), plain.exact_key());
    EXPECT_NE(markedB.exact_key(), plain.exact_key());
    EXPECT_NE(markedA.exact_key(), markedB.exact_key());
    EXPECT_FALSE(markedB == plain);
    // The next advance() clears it: the cleared state is the plain one.
    for (const bool both : {false, true}) {
        TimeState next = markedB;
        next.advance(BLACK, both);
        EXPECT_EQ(next.hybrid_event_board(), -1);
        EXPECT_EQ(next, plain.after(BLACK, both));
        EXPECT_EQ(next.exact_key(), plain.after(BLACK, both).exact_key());
    }
    // can_sit is never touched by the mark (classify_terminal_position reads it).
    EXPECT_EQ(markedB.can_sit(BLACK), plain.can_sit(BLACK));
    EXPECT_EQ(markedB.can_sit(WHITE), plain.can_sit(WHITE));
}

TEST_F(HybridRules, NodeAndCandidateSizesUnchanged) {
    // JointActionRules grew from 5 to 7 bytes, inside the generator's
    // padding before plannedBoard: no node grew (1040 since fix 2, f321334, on
    // the Mac's libc++; 928 with libstdc++ on PACE, the same at 72e79eb).
    EXPECT_EQ(sizeof(JointActionRules), 7u);
    EXPECT_EQ(sizeof(JointActionCandidate), 48u);
#if defined(__APPLE__)
    EXPECT_EQ(sizeof(Node), 1040u);
#else
    EXPECT_EQ(sizeof(Node), 928u);
#endif
}

// ---- review fixes ---------------------------------------------------------------------

TEST_F(HybridRules, ReviewFix_SitMarkIsNotKeyedAndOneShot) {
    const TimeState plain = TimeState::clocked(WHITE, -2000, 200, 300, 200);
    const TimeState moved = plain.with_hybrid_event(1);
    const TimeState sat = plain.with_hybrid_event(1, true);
    EXPECT_FALSE(moved.hybrid_event_below_sit());
    EXPECT_TRUE(sat.hybrid_event_below_sit());
    EXPECT_EQ(sat.hybrid_event_board(), 1);
    // Not in the key: the sit's position is no moving child's.
    EXPECT_EQ(sat.exact_key(), moved.exact_key());
    // No mark, no flag; the next advance() clears both.
    EXPECT_FALSE(plain.with_hybrid_event(-1, true).hybrid_event_below_sit());
    TimeState next = sat;
    next.advance(BLACK, false);
    EXPECT_FALSE(next.hybrid_event_below_sit());
    EXPECT_EQ(next, plain.after(BLACK, false));
}

/**
 * Review fix 1: a depth-1 node (Black to play, B alone on turn) whose
 * children are all proven lost for it. Its one move on B is m; the pass on B
 * is what the must-move mask drops.
 */
NodeType all_children_won(bool lead, uint8_t mustMove, bool* provesAll = nullptr,
                          NodeType childType = NodeType::WIN) {
    const Stockfish::Move m = Stockfish::Move(1);
    SearchParams::RuntimeConfig config;
    auto node = std::make_shared<Node>(BLACK);
    EXPECT_TRUE(node->try_init_and_expand({MOVE_NONE}, {m, MOVE_NONE}, {1.0f}, {0.9f, 0.1f},
                                          lead, false, true, config, {}, {}, {}, {}, 0, 0,
                                          -1, {}, {}, mustMove));
    JointActionCandidate expanded;
    while (node->expand_next_joint_child(nullptr, 0, expanded, config)) {
    }
    for (int index = 0; index < static_cast<int>(node->get_num_generated()); ++index) {
        node->update_child_node_type(index, childType);
    }
    if (provesAll) {
        *provesAll = node->proves_from_all_children();
    }
    return node->get_node_type();
}

TEST_F(HybridRules, ReviewFix_MaskedNodeBelowAMoveProvesNoLossOrDraw) {
    bool provesAll = true;
    // Black leads: v3 lets it sit on B, the mask dropped that pass, and the
    // one move left losing proves nothing (the sit may hold).
    EXPECT_EQ(all_children_won(true, 2, &provesAll), NodeType::UNSOLVED);
    EXPECT_FALSE(provesAll);
    EXPECT_EQ(all_children_won(true, 2, nullptr, NodeType::DRAW), NodeType::UNSOLVED);
    // A winning child still proves the node: the mask only removes actions.
    EXPECT_EQ(all_children_won(true, 2, nullptr, NodeType::LOSS), NodeType::WIN);
    // Below the hybrid sit the proof stands ("assumes event", spec v6 10).
    EXPECT_EQ(all_children_won(true, 2 | HYBRID_MUST_MOVE_BELOW_SIT, &provesAll), NodeType::LOSS);
    EXPECT_TRUE(provesAll);
    // Behind, the pass was never v3's to drop: the proof stands.
    EXPECT_EQ(all_children_won(false, 2, &provesAll), NodeType::LOSS);
    EXPECT_TRUE(provesAll);
    // No mask: the stock solver.
    EXPECT_EQ(all_children_won(true, 0, &provesAll), NodeType::LOSS);
    EXPECT_TRUE(provesAll);
    // The generator keeps the board bits only (the expanded rules).
    const Stockfish::Move m = Stockfish::Move(1);
    SearchParams::RuntimeConfig config;
    Node node(BLACK);
    ASSERT_TRUE(node.try_init_and_expand({MOVE_NONE}, {m, MOVE_NONE}, {1.0f}, {0.9f, 0.1f}, true,
                                         false, true, config, {}, {}, {}, {}, 0, 0, -1, {}, {},
                                         2 | HYBRID_MUST_MOVE_BELOW_SIT));
    EXPECT_EQ(node.expanded_rules().mustMoveBoards, 2);
}

RootEdgeStats edge(Stockfish::Move a, Stockfish::Move b, int visits, NodeType type) {
    RootEdgeStats stats;
    stats.action.moveA = a;
    stats.action.moveB = b;
    stats.visits = visits;
    stats.childType = type;
    return stats;
}

TEST_F(HybridRules, ReviewFix_HybridPolicyTargetNeverHoldsTheSit) {
    const Stockfish::Move m1 = Stockfish::Move(1);
    const Stockfish::Move m2 = Stockfish::Move(2);
    const RootEdgeStats sit = edge(MOVE_NONE, MOVE_NONE, 50, NodeType::LOSS);
    const RootEdgeStats a = edge(m1, MOVE_NONE, 30, NodeType::UNSOLVED);
    const RootEdgeStats b = edge(m2, MOVE_NONE, 20, NodeType::UNSOLVED);
    const auto hasSit = [](const std::vector<RootEdgeStats>& edges) {
        return std::any_of(edges.begin(), edges.end(), [](const RootEdgeStats& e) {
            return e.action.moveA == MOVE_NONE && e.action.moveB == MOVE_NONE;
        });
    };
    // The bug: a root WIN only through the sit narrows the stock target to the sit.
    const std::vector<RootEdgeStats> all{sit, a, b};
    const auto stock = selfplay_policy_edges(all, NodeType::WIN, sit.action, false);
    ASSERT_EQ(stock.size(), 1u);
    EXPECT_TRUE(hasSit(stock));
    // Fixed: the moves, read as an unsolved root's.
    auto target = hybrid_policy_edges(all, NodeType::WIN, sit.action, false);
    EXPECT_FALSE(hasSit(target));
    EXPECT_EQ(target.size(), 2u);
    // Certified sit: the same.
    target = hybrid_policy_edges(all, NodeType::WIN, sit.action, true);
    EXPECT_FALSE(hasSit(target));
    EXPECT_EQ(target.size(), 2u);
    // A move proven too: the solved filter keeps that move alone.
    const RootEdgeStats won = edge(m2, MOVE_NONE, 20, NodeType::LOSS);
    target = hybrid_policy_edges({sit, a, won}, NodeType::WIN, sit.action, false);
    ASSERT_EQ(target.size(), 1u);
    EXPECT_EQ(target[0].action.moveA, m2);
    // A lost root whose choice is the sit: every move, never the sit.
    const RootEdgeStats lostSit = edge(MOVE_NONE, MOVE_NONE, 50, NodeType::WIN);
    const RootEdgeStats lostA = edge(m1, MOVE_NONE, 30, NodeType::WIN);
    const RootEdgeStats lostB = edge(m2, MOVE_NONE, 20, NodeType::WIN);
    target = hybrid_policy_edges({lostSit, lostA, lostB}, NodeType::LOSS, lostSit.action, true);
    EXPECT_FALSE(hasSit(target));
    EXPECT_EQ(target.size(), 2u);
    // Every move lost, the sit not: the moves, as an unsolved root with no alternative.
    target = hybrid_policy_edges({edge(MOVE_NONE, MOVE_NONE, 50, NodeType::UNSOLVED), lostA, lostB},
                                 NodeType::UNSOLVED, a.action, false);
    EXPECT_FALSE(hasSit(target));
    EXPECT_EQ(target.size(), 2u);
    // An unsolved root with a move choice: the stock filter without the sit.
    target = hybrid_policy_edges({edge(MOVE_NONE, MOVE_NONE, 50, NodeType::UNSOLVED), a, b},
                                 NodeType::UNSOLVED, a.action, false);
    EXPECT_EQ(target.size(), 2u);
    EXPECT_FALSE(hasSit(target));
}

// ---- 1.x the trigger -----------------------------------------------------------------

TEST_F(HybridRules, Test1_1_AllHoldGivesOneBitAndEachFailureNamesItself) {
    Board board = boards(kKnightA, kKnightB);
    const Seats seats;
    const HybridTrigger fired = trigger(board, seats);
    EXPECT_EQ(fired.mask, 1);
    EXPECT_EQ(fired.sitBoard, BOARD_A);
    EXPECT_EQ(fired.eventBoard, BOARD_B);
    EXPECT_EQ(fired.eventMs, 100);
    EXPECT_STREQ(fired.reason, "");

    // T1: no legal move on our board.
    Board stalemate = boards(kStalemateA, kKnightB);
    EXPECT_EQ(trigger(stalemate, seats).mask, 0);
    EXPECT_STREQ(trigger(stalemate, seats).reason, "no move");
    // T1: RequireMoveOn names our board.
    g_requiredMoveBoard = REQUIRE_MOVE_BOARD_A;
    const HybridTrigger required = trigger(board, seats);
    g_requiredMoveBoard = REQUIRE_MOVE_NONE;
    EXPECT_EQ(required.mask, 0);
    EXPECT_STREQ(required.reason, "required");
    // T2: we lead, the gate allows the sit already.
    Seats ahead = seats;
    ahead.clocks = {61000, 59000, 59000, 61000};
    EXPECT_STREQ(trigger(board, ahead).reason, "gate allows");
    // T3: the opponent on B sits.
    Seats sitter = seats;
    sitter.sitting[BW] = true;
    EXPECT_STREQ(trigger(board, sitter).reason, "opponent sitting");
    // T4: the event is too far away.
    Seats slow = seats;
    slow.paces[BW] = 600;
    slow.elapsed[BW] = 0;
    EXPECT_STREQ(trigger(board, slow).reason, "wait");
    // T5 cannot fail alone: the sit-off rule forces a move only against a
    // sitting opponent, which T3 already refuses (spec: "redundant with T3").
    // T6: our clock is under the floor.
    Seats low = seats;
    low.clocks[AW] = 2000;
    EXPECT_STREQ(trigger(board, low).reason, "clock");
}

TEST_F(HybridRules, Test1_2_GateWaitAndBudgetBounds) {
    Board board = boards(kKnightA, kKnightB);
    Seats seats;
    seats.paces[BW] = 600;
    seats.elapsed[BW] = 500;
    HybridTrigger fired = trigger(board, seats);
    EXPECT_EQ(fired.mask, 1);
    EXPECT_EQ(fired.eventMs, 100);
    // Lead exactly 0 is "no lead": the gate forbids the sit, the trigger fires.
    Seats even = seats;
    even.clocks = {60000, 60000, 60000, 60000};
    EXPECT_EQ(trigger(board, even).mask, 1);
    // Lead > 0: the gate decides.
    Seats ahead = seats;
    ahead.clocks = {60001, 60000, 60000, 60001};
    EXPECT_EQ(trigger(board, ahead).mask, 0);
    // elapsed 0 (event 600) with HybridMaxWaitMs 500: wait.
    Seats fresh = seats;
    fresh.elapsed[BW] = 0;
    HybridKnobs knobs;
    knobs.maxWaitMs = 500;
    EXPECT_STREQ(trigger(board, fresh, knobs).reason, "wait");
    // RootMoveTimeMs 200, event 400, HybridMaxWaitMs 1000: above 1.5 x 200.
    Seats far = seats;
    far.elapsed[BW] = 200;
    EXPECT_EQ(trigger(board, far).eventMs, 0);
    EXPECT_STREQ(trigger(board, far).reason, "wait");
    // ... and 300 is exactly 1.5 x 200: inside the wake.
    far.elapsed[BW] = 300;
    EXPECT_EQ(trigger(board, far).mask, 1);
    // PassCapK 1 caps the wait at 1 x 200.
    Seats capped = seats;
    capped.elapsed[BW] = 350;   // event 250
    EXPECT_EQ(trigger(board, capped).mask, 1);
    capped.passCapK = 1;
    EXPECT_STREQ(trigger(board, capped).reason, "wait");
}

TEST_F(HybridRules, Test1_3_BothBoardsOursIsNeverMasked) {
    Board board = boards(kKnightA, kKnightBBefore);
    for (const int32_t lead : {-2000, 0}) {
        Seats seats;
        seats.clocks = {60000 + lead / 2, 60000 - lead / 2, 60000 - lead / 2, 60000 + lead / 2};
        const HybridTrigger result = trigger(board, seats);
        EXPECT_EQ(result.mask, 0);
        EXPECT_STREQ(result.reason, "both boards");
    }
}

TEST_F(HybridRules, Test1_4And1_5_SittingOpponentAndLowClock) {
    Board board = boards(kKnightA, kKnightB);
    Seats sitter;
    sitter.sitting[BW] = true;
    EXPECT_EQ(trigger(board, sitter).mask, 0);
    EXPECT_STREQ(trigger(board, sitter).reason, "opponent sitting");
    // T5 agrees: our clock (Aw 59 s) is below the sitter's (Bw 61 s).
    const TimeState durative = durative_clock(board, sitter);
    EXPECT_FALSE(durative.can_sit(WHITE));
    Seats low;
    low.clocks = {2000, 61000, 61000, 59000};
    HybridKnobs knobs;
    knobs.minClockMs = 3000;
    EXPECT_STREQ(trigger(board, low, knobs).reason, "clock");
    knobs.minClockMs = 1000;
    EXPECT_EQ(trigger(board, low, knobs).mask, 1);
    // 4 x (event + budget) floors it too: 4 x (100 + 200) = 1200.
    low.clocks[AW] = 1199;
    EXPECT_STREQ(trigger(board, low, knobs).reason, "clock");
}

TEST_F(HybridRules, Test1_6_PassLegalityUnderTheMasks) {
    const auto legal = [](const JointActionRules& rules, bool passA, bool passB,
                          bool captureA = false, bool captureB = false) {
        return is_joint_action_legal(rules, passA ? MOVE_NONE : Stockfish::Move(1),
                                     passB ? MOVE_NONE : Stockfish::Move(1), captureA, captureB);
    };
    // A alone on turn, behind: (pass, none) illegal under v3, legal with A's bit.
    JointActionRules rules{true, false, false, true, false};
    EXPECT_FALSE(legal(rules, true, true));
    rules.hybridSitBoards = 1;
    EXPECT_TRUE(legal(rules, true, true));
    rules.hybridSitBoards = 2;   // the bit of the other board does nothing
    EXPECT_FALSE(legal(rules, true, true));
    // Both on turn, both bits: the double pass stays illegal.
    JointActionRules both{true, true, false, true, true, 3, 0};
    EXPECT_FALSE(legal(both, true, true));
    // mustMoveBoards B: a pass on B is illegal while B can move ...
    JointActionRules must{true, true, true, true, true, 0, 2};
    EXPECT_FALSE(legal(must, false, true));
    EXPECT_FALSE(legal(must, true, true));
    EXPECT_TRUE(legal(must, true, false));   // the pass on A follows v3 (lead: legal)
    EXPECT_TRUE(legal(must, false, false));
    // ... and as legal as v3 makes it when it is forced (B on turn without
    // a legal move): the mask never removes a forced pass.
    for (const bool lead : {false, true}) {
        const JointActionRules plain{false, true, lead, false, false};
        const JointActionRules forced{false, true, lead, false, false, 0, 2};
        EXPECT_EQ(legal(forced, true, true), legal(plain, true, true));
    }
    EXPECT_TRUE(legal(JointActionRules{false, true, true, false, false, 0, 2}, true, true));
    // RequireMoveOn still wins over the hybrid's sit.
    const JointActionRules sitA{true, false, false, true, false, 1, 0};
    g_requiredMoveBoard = REQUIRE_MOVE_BOARD_A;
    EXPECT_FALSE(legal(sitA, true, true));
    g_requiredMoveBoard = REQUIRE_MOVE_NONE;
    EXPECT_TRUE(legal(sitA, true, true));
    // Masks at zero are the old rules: the whole category table agrees.
    for (int bits = 0; bits < 32; ++bits) {
        const JointActionRules old{(bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0,
                                   (bits & 8) != 0, (bits & 16) != 0};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const bool sitA = a == 0;
                const bool sitB = b == 0;
                bool expected = true;
                if (sitA && sitB) {
                    expected = is_double_sit_legal(old.teamHasTimeAdvantage, old.boardAOnTurn,
                                                   old.boardBOnTurn);
                } else if (sitA && old.boardACanMove) {
                    expected = is_single_pass_legal(old.teamHasTimeAdvantage, old.boardAOnTurn,
                                                    old.boardBOnTurn, b == 2);
                } else if (sitB && old.boardBCanMove) {
                    expected = is_single_pass_legal(old.teamHasTimeAdvantage, old.boardAOnTurn,
                                                    old.boardBOnTurn, a == 2);
                }
                EXPECT_EQ(legal(old, sitA, sitB, a == 2, b == 2), expected) << bits << a << b;
            }
        }
    }
}

TEST_F(HybridRules, SelfPlayHybridPassEndsAtTheWake) {
    Wave2GameOptions options;
    options.baseMix = {{30000, 1.0}};
    options.lagMaxMs = 0;
    options.paceJitter = 0.0;
    options.minMoveMs = 1;
    SeatClockGame::Start start;
    start.clocksMs = {30000, 30000, 30000, 30000};
    start.moveTimesMs = {200, 300, 300, 200};
    SeatClockGame game(options, start);
    std::mt19937_64 random(1);
    const std::array<Stockfish::Color, 2> sides{WHITE, WHITE};
    SeatClockGame::Event first = game.next(sides);
    ASSERT_EQ(first.type, SeatClockGame::EventType::DECIDE);
    for (int seat = 0; seat < 4; ++seat) {
        if (first.seats[seat]) {
            game.plan_cost(seat, random);
        }
    }
    // Aw decides first at 0 (white wins the opening tie) and hybrid-sits for
    // an event 100 ms away: the cap is min(8 x 200, 1.5 x 200 + 1, 100 + 100).
    EXPECT_EQ(game.hybrid_pass_cap_ms(AW, 100), 200);
    EXPECT_EQ(game.hybrid_pass_cap_ms(AW, 400), 301);
    game.commit_pass(AW, sides, game.hybrid_pass_cap_ms(AW, 100));
    EXPECT_EQ(game.event_at_ms(AW), 200);
    EXPECT_TRUE(game.sitting(AW));
}

// ---- 2.x searches on it13 --------------------------------------------------------------

class HybridIt13 : public ::testing::Test {
protected:
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
        engine = std::make_unique<Engine>(0, 16);
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
    }

    struct Run {
        JointActionCandidate choice;
        HybridReport report;
        NodeType rootType = NodeType::UNSOLVED;
        std::string output;
    };

    Run search(Agent& agent, const Seats& seats, size_t nodes, HybridKnobs knobs) {
        Board board = boards(kKnightA, kKnightB);
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.verbose = true;
        opts.clock = v3_clock(seats);
        opts.hybrid.knobs = knobs;
        opts.hybrid.clock = durative_clock(board, seats);
        testing::internal::CaptureStdout();
        Run run;
        run.choice = agent.run_search(board, {engine.get()}, WHITE, false, opts);
        run.output = testing::internal::GetCapturedStdout();
        run.report = agent.hybrid_report();
        run.rootType = agent.root_type();
        return run;
    }

    Run search(const Seats& seats, size_t nodes, HybridKnobs knobs) {
        Agent agent(2);
        return search(agent, seats, nodes, knobs);
    }

    static HybridKnobs gate(HybridEventLands lands = HybridEventLands::ALL) {
        HybridKnobs knobs;
        knobs.mode = HybridMode::GATE;
        knobs.eventLands = lands;
        return knobs;
    }

    static bool sits(const JointActionCandidate& action) {
        return action.moveA == MOVE_NONE && action.moveB == MOVE_NONE;
    }

    std::unique_ptr<Engine> engine;
};

TEST_F(HybridIt13, Test2_1_SitsWhenTheKnightIsAboutToLand) {
    const Seats seats;
    // nodur: the sit is not even legal, Aw moves.
    const Run off = search(seats, 2000, HybridKnobs());
    EXPECT_NE(off.choice.moveA, MOVE_NONE);
    EXPECT_FALSE(off.report.active);
    for (const size_t nodes : {2000u, 8000u}) {
        const Run on = search(seats, nodes, gate());
        EXPECT_EQ(on.report.trigger.mask, 1) << on.report.trigger.reason;
        EXPECT_TRUE(sits(on.choice)) << on.output;
        EXPECT_TRUE(on.report.kept);
        EXPECT_TRUE(on.report.proven
                    || on.report.qSit >= on.report.qAct + 0.02f) << on.output;
        EXPECT_NE(on.output.find("info string Hybrid: sit boards A event 100/- mode gate"),
                  std::string::npos) << on.output;
        EXPECT_NE(on.output.find("info string Hybrid: chose sit A"), std::string::npos)
            << on.output;
        if (on.report.proven) {
            // A proof through the sit assumes the event: no mate score for it.
            EXPECT_NE(on.output.find("proven (assumes event)"), std::string::npos);
            EXPECT_EQ(on.output.find("score mate"), std::string::npos) << on.output;
        }
    }
}

TEST_F(HybridIt13, Test2_2_TheEventLandsBetIsVisible) {
    Seats deeper;
    deeper.clocks = {55000, 65000, 65000, 55000};   // lead -10 s
    for (const size_t nodes : {2000u, 8000u}) {
        const Run on = search(deeper, nodes, gate());
        EXPECT_EQ(on.report.trigger.mask, 1);
        EXPECT_TRUE(sits(on.choice)) << on.output;
        Seats sat = deeper;
        sat.sitting[BW] = true;
        const Run sitting = search(sat, nodes, gate());
        EXPECT_EQ(sitting.report.trigger.mask, 0);
        EXPECT_STREQ(sitting.report.trigger.reason, "opponent sitting");
        EXPECT_NE(sitting.choice.moveA, MOVE_NONE);
    }
}

TEST_F(HybridIt13, Test2_3_NoEventNoSit) {
    Seats sat;
    sat.sitting[BW] = true;
    Seats fresh;
    fresh.elapsed[BW] = 0;   // event 300
    HybridKnobs shortWait = gate();
    shortWait.maxWaitMs = 100;
    for (const auto& [seats, knobs] : {std::pair{sat, gate()}, std::pair{fresh, shortWait}}) {
        const Run on = search(seats, 2000, knobs);
        EXPECT_TRUE(on.report.active);
        EXPECT_EQ(on.report.trigger.mask, 0);
        EXPECT_NE(on.choice.moveA, MOVE_NONE);
        // No hybrid choice is reported: the search was nodur's.
        EXPECT_EQ(on.output.find("Hybrid: chose"), std::string::npos) << on.output;
    }
}

TEST_F(HybridIt13, Test2_4_AMarginOf1000KeepsOnlyAProof) {
    HybridKnobs strict = gate();
    strict.marginPermille = 1000;
    const Run on = search(Seats(), 2000, strict);
    EXPECT_EQ(on.report.trigger.mask, 1);
    // q can never beat the best move by 1.0: only a proof keeps the sit.
    if (sits(on.choice)) {
        EXPECT_TRUE(on.report.proven) << on.output;
    } else {
        EXPECT_NE(on.choice.moveA, MOVE_NONE);
    }
}

/** Every node below @p node (each once), with its depth on the first path found. */
void walk(Node* node, int depth, std::unordered_set<const Node*>& seen,
          const std::function<void(Node*, int)>& visit) {
    if (!node || !seen.insert(node).second) {
        return;
    }
    visit(node, depth);
    if (!node->is_expanded()) {
        return;
    }
    for (const std::shared_ptr<Node>& child : node->get_children()) {
        walk(child.get(), depth + 1, seen, visit);
    }
}

TEST_F(HybridIt13, Test2_7_TheFirstDraftWasInertAndDepthOneMustMove) {
    // HybridEventLands none: the opponent may sit back at depth 1, the
    // root's position returns, and the sit never beats the best move.
    const Run none = search(Seats(), 2000, gate(HybridEventLands::NONE));
    EXPECT_EQ(none.report.trigger.mask, 1);
    EXPECT_FALSE(sits(none.choice)) << none.output;
    EXPECT_FALSE(none.report.kept);
    EXPECT_TRUE(none.report.reason == "margin" || none.report.reason == "no sit chosen")
        << none.report.reason;
    EXPECT_FALSE(none.report.proven);
    // All: every depth-1 node expanded forbids the pass on B while B can
    // move; no depth-2 node does.
    Agent agent(2);
    const Run all = search(agent, Seats(), 2000, gate());
    ASSERT_TRUE(sits(all.choice)) << all.output;
    const std::shared_ptr<Node> root = AgentHybridPeer::root(agent);
    ASSERT_TRUE(root && root->is_expanded());
    EXPECT_EQ(root->expanded_rules().hybridSitBoards, 1);
    EXPECT_EQ(root->expanded_rules().mustMoveBoards, 0);
    int depthOne = 0;
    int depthTwo = 0;
    for (const std::shared_ptr<Node>& child : root->get_children()) {
        if (!child || !child->is_expanded()) {
            continue;
        }
        ++depthOne;
        EXPECT_EQ(child->expanded_rules().mustMoveBoards, 2);
        EXPECT_EQ(child->expanded_rules().hybridSitBoards, 0);
        for (const std::shared_ptr<Node>& grandchild : child->get_children()) {
            if (grandchild && grandchild->is_expanded()) {
                ++depthTwo;
                EXPECT_EQ(grandchild->expanded_rules().mustMoveBoards, 0);
                EXPECT_EQ(grandchild->expanded_rules().hybridSitBoards, 0);
            }
        }
    }
    EXPECT_GT(depthOne, 0);
    EXPECT_GT(depthTwo, 0);
}

TEST_F(HybridIt13, Test2_5_NoCycleBelowTheSit) {
    for (const HybridEventLands lands : {HybridEventLands::ALL, HybridEventLands::NONE}) {
        Agent agent(2);
        const Run run = search(agent, Seats(), 20000, gate(lands));
        EXPECT_EQ(run.report.trigger.mask, 1);
        const std::shared_ptr<Node> root = AgentHybridPeer::root(agent);
        ASSERT_TRUE(root);
        const std::vector<std::shared_ptr<Node>> children = root->get_children();
        for (size_t index = 0; index < children.size() && index < root->get_num_generated();
             ++index) {
            if (!sits(root->get_joint_action(static_cast<int>(index))) || !children[index]) {
                continue;
            }
            std::unordered_set<const Node*> seen;
            bool rootBelow = false;
            walk(children[index].get(), 1, seen, [&](Node* node, int) {
                rootBelow = rootBelow || node == root.get();
            });
            EXPECT_FALSE(rootBelow) << hybrid_event_lands_name(lands);
        }
    }
}

TEST_F(HybridIt13, ReviewFix_ProofsBelowMovingChildrenNeedNoEvent) {
    // HybridEventLands all, our lead -2 s: Bw may sit on B under v3, so every
    // depth-1 node below a moving child that the mask touched proves no loss
    // or draw; the sit's node keeps its proofs (assumes event).
    Agent agent(2);
    const Run all = search(agent, Seats(), 4000, gate());
    ASSERT_EQ(all.report.trigger.mask, 1);
    const std::shared_ptr<Node> root = AgentHybridPeer::root(agent);
    ASSERT_TRUE(root && root->is_expanded());
    const std::vector<std::shared_ptr<Node>> children = root->get_children();
    int movers = 0;
    for (size_t index = 0; index < children.size() && index < root->get_num_generated(); ++index) {
        const std::shared_ptr<Node>& child = children[index];
        if (!child || !child->is_expanded()) {
            continue;
        }
        if (sits(root->get_joint_action(static_cast<int>(index)))) {
            EXPECT_TRUE(child->proves_from_all_children());
            continue;
        }
        if (child->expanded_rules().mustMoveBoards == 0) {
            continue;
        }
        ++movers;
        EXPECT_FALSE(child->proves_from_all_children());
        EXPECT_NE(child->get_node_type(), NodeType::LOSS) << all.output;
        EXPECT_NE(child->get_node_type(), NodeType::DRAW) << all.output;
    }
    EXPECT_GT(movers, 0);
    // A proof the root stands on is the sit's or none: no "score mate" for a move.
    if (all.rootType == NodeType::WIN) {
        EXPECT_TRUE(sits(all.choice)) << all.output;
        EXPECT_EQ(all.output.find("score mate"), std::string::npos) << all.output;
    }
}

TEST_F(HybridIt13, ReviewFix_ReuseRetainsThePlayedChild) {
    Agent agent(2);
    const Run on = search(agent, Seats(), 2000, gate());
    ASSERT_EQ(on.report.trigger.mask, 1);
    const std::shared_ptr<Node> root = AgentHybridPeer::root(agent);
    ASSERT_TRUE(root && root->is_expanded());
    const std::vector<std::shared_ptr<Node>> children = root->get_children();
    // The search's own retention is the played action's child.
    int playedIdx = -1;
    int moveIdx = -1;
    for (size_t index = 0; index < children.size() && index < root->get_num_generated(); ++index) {
        const JointActionCandidate action = root->get_joint_action(static_cast<int>(index));
        if (action.moveA == on.choice.moveA && action.moveB == on.choice.moveB) {
            playedIdx = static_cast<int>(index);
        } else if (!sits(action) && children[index] && moveIdx < 0) {
            moveIdx = static_cast<int>(index);
        }
    }
    ASSERT_GE(playedIdx, 0);
    ASSERT_GE(moveIdx, 0);
    EXPECT_TRUE(AgentHybridPeer::retains(agent, children[playedIdx].get()));
    // A move played other than the root's best (a refused sit): its child
    // is retained, not the best one's.
    Board board = boards(kKnightA, kKnightB);
    const JointActionCandidate move = root->get_joint_action(moveIdx);
    AgentHybridPeer::store_next_roots(agent, board, AgentHybridPeer::search_time(agent), &move);
    EXPECT_TRUE(AgentHybridPeer::retains(agent, children[moveIdx].get()));
    if (moveIdx != playedIdx) {
        EXPECT_FALSE(AgentHybridPeer::retains(agent, children[playedIdx].get()));
    }
}

}  // namespace

