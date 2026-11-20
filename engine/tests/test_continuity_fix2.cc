#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/globals.h"
#include "common/utils.h"
#include "durative_key_lines.h"
#include "environment/board.h"
#include "environment/joint_action.h"
#include "environment/time_state.h"
#include "interface/uci.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/node.h"
#include "search/searchthread.h"

// Wave 2c, fix 2: joint team actions in the durative tree
// (docs/custom_engine_spec_v5_continuity.md, acceptance tests 0.2, 0.3, 1.16,
// 2.x and 4.2). Below the root, a team that acts on one board plans its other
// on-turn seat's move in the same joint action; the planned move is the
// seat's committed move and lands at its own ready time.

class AgentContinuityFix2Peer {
public:
    static std::shared_ptr<Node> root(Agent& agent) { return agent.rootNode; }
    static std::string pv_from_child(Agent& agent, Board& board, int childIdx,
                                     Stockfish::Color team, const TimeState& time) {
        return agent.extract_pv_from_child(board, childIdx, 20, team, time);
    }
    static auto find(Agent& agent, Board& board, Stockfish::Color team,
                                     const TimeState& live, int toleranceMs) {
        return agent.find_team_tree_node(board, team, live, toleranceMs);
    }
    static std::shared_ptr<Node> team_tree_root(Agent& agent) {
        return agent.teamTree_ ? agent.teamTree_->root : nullptr;
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

// Few legal moves on each board, so a few thousand nodes reach several plies.
constexpr const char* kSmallWhite = "4k3/p7/8/8/8/8/P7/4K3[] w - - 0 1";
constexpr const char* kSmallBlack = "4k3/p7/8/8/8/8/P7/4K3[] b - - 0 1";

struct Rules {
    bool jointPlans = true;
    int32_t freePermille = 500;
    bool instantTransfer = false;
    bool flagLoss = false;
    bool sitGate = false;
    std::array<int32_t, 4> clocks{60000, 60000, 60000, 60000};
    std::array<int32_t, 4> paces{100, 50, 100, 500};
    std::array<int32_t, 4> elapsed{0, 0, 0, 0};
    std::array<bool, 4> sitting{false, false, false, false};
    int32_t rootMoveMs = 100;
};

TimeState durative_root(Stockfish::Color rootTeam, Stockfish::Color sideA, Stockfish::Color sideB,
                        const Rules& r) {
    DurativeConfig config;
    config.flagLoss = r.flagLoss;
    config.instantTransfer = r.instantTransfer;
    config.rootMoveTimeMs = r.rootMoveMs;
    config.seatClocksMs = r.clocks;
    config.seatPaceMs = r.paces;
    config.seatElapsedMs = r.elapsed;
    config.seatSitting = r.sitting;
    config.jointPlans = r.jointPlans;
    config.jointPlanFreePermille = r.freePermille;
    const TimeState base = TimeState::clocked(rootTeam, 0, 100, 100).with_sit_gate(r.sitGate);
    return TimeState::durative(base, rootTeam, sideA, sideB, config);
}

/**
 * Spec 2.1's state: black's Ab has just moved on A (a black-team root at a
 * 50 ms budget, which every black action below it then costs), so white acts
 * with Aw ready at 100 ms while Bb is on turn on B and ready at 400 (pace 500,
 * 50 ms elapsed, minus the 50 ms root). Boards: A white to move, B black to
 * move.
 */
TimeState aw_ready_bb_planned(Rules r) {
    r.rootMoveMs = 50;
    r.elapsed = {0, 0, 0, 50};
    TimeState time = durative_root(BLACK, BLACK, BLACK, r);
    time.advance_action(BLACK, true, false);
    return time;
}

/**
 * The same with Bb sitting (spec 2.3, 2.8): a white root (Aw and Bb at a
 * budget of rootMoveMs) where Aw moves and Bb passes; Ab lands after its
 * pace and wakes Bb, which passes again until just after Aw's next landing.
 * White then acts on A with Bb sitting, ready 1 ms after Aw. Boards: A white
 * to move, B black to move.
 */
TimeState aw_ready_bb_sitting(const Rules& r) {
    TimeState time = durative_root(WHITE, WHITE, BLACK, r);
    time.advance_action(WHITE, true, false);   // root: Aw moves, Bb sits
    time.advance_action(BLACK, true, false);   // Ab lands, Bb wakes
    time.advance_action(WHITE, false, false);  // Bb sits again
    return time;
}

Board boards(const std::string& a, const std::string& b) {
    Board board;
    board.set(a + " | " + b);
    return board;
}

/** The legal move on @p board_num whose UCI string is @p uci (MOVE_NONE if none). */
Stockfish::Move find_move(Board& board, int boardNum, const std::string& uci) {
    for (Stockfish::Move move : board.legal_moves(boardNum)) {
        if (board.uci_move(boardNum, move) == uci) {
            return move;
        }
    }
    return MOVE_NONE;
}

/** Every node reachable from @p root, visited once, with its board and path time. */
void walk_tree(Node* node, Board& board, const TimeState& time,
               const std::function<void(Node&, Board&, const TimeState&)>& visit,
               std::unordered_set<const Node*>& seen, int depth = 0) {
    if (!node || depth > 40 || !seen.insert(node).second) {
        return;
    }
    visit(*node, board, time);
    if (!node->is_expanded()) {
        return;
    }
    const Stockfish::Color team = node->get_team_to_play();
    const std::vector<std::shared_ptr<Node>> children = node->get_children();
    for (size_t index = 0; index < children.size(); ++index) {
        if (!children[index]) {
            continue;
        }
        const JointActionCandidate action = node->get_joint_action(static_cast<int>(index));
        const TimeState next = durative_time_after(time, board, team, action);
        board.make_moves(action.moveA, action.moveB);
        walk_tree(children[index].get(), board, next, visit, seen, depth + 1);
        board.unmake_moves(action.moveA, action.moveB);
    }
}

void walk_tree(Node* root, Board& board, const TimeState& time,
               const std::function<void(Node&, Board&, const TimeState&)>& visit) {
    std::unordered_set<const Node*> seen;
    walk_tree(root, board, time, visit, seen);
}

// ---- 0.2, 0.3: defaults ---------------------------------------------------

TEST(ContinuityDefaults, JointPlanOptionsAdvertisedAndOff) {
    init_fairy_stockfish();
    init_policy_index();
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);
    EXPECT_NE(output.str().find("option name JointPlans type check default false"),
              std::string::npos);
    EXPECT_NE(output.str().find(
                  "option name JointPlanFreePermille type spin default 500 min 0 max 1000"),
              std::string::npos);
    const DurativeConfig defaults;
    EXPECT_FALSE(defaults.jointPlans);
    EXPECT_EQ(defaults.jointPlanFreePermille, 500);
}

TEST(ContinuityDefaults, CandidateLayoutUnchanged) {
    // 48 bytes at 261081b (measured: seven 4-byte fields, 4 bytes of padding,
    // two size_t); `planned` fills the padding.
    static_assert(sizeof(JointActionCandidate) == 48);
    EXPECT_EQ(JointActionCandidate().planned, MOVE_NONE);
}

TEST(ContinuityDefaults, DurativeKeysUnchanged) {
    const auto path = std::filesystem::path(__FILE__).parent_path() / "data"
        / "durative_keys_261081b.txt";
    std::ifstream file(path);
    ASSERT_TRUE(file) << path;
    std::vector<uint64_t> recorded;
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line[0] != '#') {
            recorded.push_back(std::stoull(line, nullptr, 16));
        }
    }
    ASSERT_EQ(recorded.size(), 2000u);
    // durative_key_lines leaves DurativeConfig::jointPlans at its default
    // (false): the keys of every line equal 261081b's.
    EXPECT_EQ(durative_key_lines(2000), recorded);
}

TEST(ContinuityDefaults, JointPlansOffChangesNoRule) {
    // The same line with the switch off and on, nothing planned: equal
    // states but for the switch, and equal timing.
    Rules off;
    off.jointPlans = false;
    Rules on;
    TimeState a = aw_ready_bb_planned(off);
    TimeState b = aw_ready_bb_planned(on);
    EXPECT_EQ(a.planned_boards(WHITE), (std::array<bool, 2>{false, false}));
    EXPECT_EQ(b.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
    for (int step = 0; step < 6; ++step) {
        const Stockfish::Color team = step % 2 == 0 ? WHITE : BLACK;
        const std::array<bool, 2> acts = a.acting_boards(team);
        ASSERT_EQ(acts, b.acting_boards(team));
        a.advance_action(team, acts[0], acts[1]);
        b.advance_action(team, acts[0], acts[1]);
        for (int seat = 0; seat < 4; ++seat) {
            EXPECT_EQ(a.seat_clock_ms(seat), b.seat_clock_ms(seat));
            EXPECT_EQ(a.seat_ready_ms(seat), b.seat_ready_ms(seat));
        }
        EXPECT_FALSE(a.has_committed());
        EXPECT_FALSE(b.has_committed());
    }
}

// ---- 2.1, 2.3, 2.5, 2.8, 2.9, 4.2: rules ------------------------------------

TEST(JointPlans, PlannedHalfLandsAtItsReadyTime) {
    init_fairy_stockfish();
    Board board = boards(kSmallWhite, kSmallBlack);
    const Stockfish::Move planned = find_move(board, BOARD_B, "e8d8");
    ASSERT_NE(planned, MOVE_NONE);
    TimeState time = aw_ready_bb_planned(Rules{});
    ASSERT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    ASSERT_EQ(time.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
    ASSERT_EQ(time.seat_ready_ms(AW), 100);
    ASSERT_EQ(time.seat_ready_ms(BB), 400);

    // Aw moves at 100 and plans Bb's move: stored, not made, Bb's ready
    // time unchanged (400 - 100 from here).
    time.advance_action(WHITE, true, false, Stockfish::NO_PIECE_TYPE,
                        Stockfish::NO_PIECE_TYPE, MOVE_NONE, planned);
    EXPECT_EQ(time.committed_move(BB), planned);
    EXPECT_EQ(time.seat_ready_ms(BB), 300);
    EXPECT_EQ(time.side_on(BOARD_B), BLACK);
    EXPECT_EQ(time.side_on(BOARD_A), BLACK);
    EXPECT_EQ(time.seat_ready_ms(AB), 50);

    // Ab at 150, Aw at 250 (Bb committed: nothing to plan), Ab at 300.
    ASSERT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
    time.advance_action(BLACK, true, false);
    ASSERT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    EXPECT_EQ(time.planned_boards(WHITE), (std::array<bool, 2>{false, false}));
    time.advance_action(WHITE, true, false);
    ASSERT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
    time.advance_action(BLACK, true, false);

    // At 400 Bb lands its committed move (beside Aw's next action, which
    // falls due at 400 too), a move like any other: B goes to Bw, ready after
    // its action time, and the commitment is spent.
    ASSERT_EQ(time.acting_boards(WHITE), (std::array<bool, 2>{true, true}));
    EXPECT_EQ(time.planned_boards(WHITE), (std::array<bool, 2>{false, false}));
    EXPECT_EQ(time.committed_move(BB), planned);
    time.advance_action(WHITE, true, true);
    EXPECT_EQ(time.side_on(BOARD_B), WHITE);
    EXPECT_EQ(time.committed_move(BB), MOVE_NONE);
    EXPECT_EQ(time.seat_ready_ms(BW), time.action_time_ms(BW));
    EXPECT_FALSE(time.has_committed());
}

TEST(JointPlans, PlannedSeatsAreNeverAtTheRootOrForAWaitingTeam) {
    // A UCI root: our team decides on every board it is on turn (atRoot).
    const TimeState root = durative_root(WHITE, WHITE, BLACK, Rules{});
    EXPECT_EQ(root.planned_boards(WHITE), (std::array<bool, 2>{false, false}));
    // A team that only waits plans nothing.
    const TimeState below = aw_ready_bb_planned(Rules{});
    EXPECT_EQ(below.planned_boards(BLACK), (std::array<bool, 2>{false, false}));
    // A committed seat is not planned again.
    const TimeState committed = below.after_action(
        WHITE, false, false, Stockfish::NO_PIECE_TYPE, Stockfish::NO_PIECE_TYPE,
        MOVE_NONE, Stockfish::Move(1234));
    EXPECT_EQ(committed.committed_move(BB), Stockfish::Move(1234));
    EXPECT_NE(committed.exact_key(), below.after_action(WHITE, false, false).exact_key());
}

TEST(JointPlans, SuppliedDropOnlyWhenTransferAllows) {
    init_fairy_stockfish();
    init_policy_index();
    // A: Aw's queen takes a knight on f6. B: Bb (black, white team) has an
    // empty pocket, so a knight drop exists only if that capture supplies it.
    Board board = boards("4k3/8/5n2/8/8/8/8/4KQ2[] w - - 0 1", "4k3/8/8/8/8/8/8/4K3[] b - - 0 1");
    const Stockfish::Move capture = find_move(board, BOARD_A, "f1f6");
    ASSERT_NE(capture, MOVE_NONE);
    const std::vector<Stockfish::Move> immediate = leaf_board_actions(board, BOARD_A, true,
                                                                      aw_ready_bb_planned(Rules{}));
    const auto knight_drops = [](const PlannedHalf& half) {
        int drops = 0;
        for (size_t i = 0; i < half.actions.size(); ++i) {
            if (half.actions[i] != MOVE_NONE && type_of(half.actions[i]) == Stockfish::DROP) {
                EXPECT_EQ(half.neededTypes[i], Stockfish::KNIGHT);
                ++drops;
            }
        }
        return drops;
    };

    Rules instant;
    instant.instantTransfer = true;
    const TimeState deciding = aw_ready_bb_planned(instant);
    ASSERT_EQ(deciding.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
    const PlannedHalf withInstant = leaf_planned_half(board, BOARD_B, WHITE, deciding,
                                                      BOARD_A, immediate);
    EXPECT_GT(knight_drops(withInstant), 0);
    // The capture is tagged as supplying a knight; quiet moves supply nothing.
    for (size_t i = 0; i < immediate.size(); ++i) {
        EXPECT_EQ(withInstant.suppliedTypes[i],
                  immediate[i] == capture ? Stockfish::KNIGHT : 0);
    }
    EXPECT_EQ(withInstant.actions.back(), MOVE_NONE);

    // Without InstantTransfer a deciding seat cannot drop what lands mid-decision.
    const TimeState delayed = aw_ready_bb_planned(Rules{});
    ASSERT_EQ(delayed.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
    EXPECT_EQ(knight_drops(leaf_planned_half(board, BOARD_B, WHITE, delayed, BOARD_A, immediate)),
              0);

    // A sitting seat decides anew when it wakes, after the capture landed.
    const TimeState sitting = aw_ready_bb_sitting(Rules{});
    ASSERT_TRUE(sitting.seat_sitting(BB));
    ASSERT_EQ(sitting.acting_boards(WHITE), (std::array<bool, 2>{true, false}));
    ASSERT_EQ(sitting.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
    const PlannedHalf whileSitting = leaf_planned_half(board, BOARD_B, WHITE, sitting,
                                                       BOARD_A, immediate);
    EXPECT_GT(knight_drops(whileSitting), 0);
    // The board is left as it was.
    EXPECT_EQ(board.fen(BOARD_B), "4k3/8/8/8/8/8/8/4K3[] b - - 0 1");

    // Pair legality: a planned knight drop comes only with the capture, and
    // the legal prior mass is exactly the sum over legal pairs.
    std::vector<float> priorsA(immediate.size(), 1.0f / immediate.size());
    std::vector<float> priorsB(withInstant.actions.size(), 1.0f / withInstant.actions.size());
    std::vector<uint8_t> capturesA;
    for (Stockfish::Move move : immediate) {
        capturesA.push_back(move != MOVE_NONE && board.is_capture(BOARD_A, move) ? 1 : 0);
    }
    JointCandidateGenerator generator;
    generator.initialize(immediate, withInstant.actions, priorsA, priorsB, true, true, false,
                         capturesA, std::vector<uint8_t>(withInstant.actions.size(), 0), {}, {},
                         0, 8, 1.0f, BOARD_B, withInstant.suppliedTypes, withInstant.neededTypes);
    ASSERT_EQ(generator.planned_board(), BOARD_B);
    double sum = 0.0;
    int drops = 0;
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        EXPECT_EQ(candidate.moveB, MOVE_NONE);
        if (candidate.planned != MOVE_NONE && type_of(candidate.planned) == Stockfish::DROP) {
            EXPECT_EQ(candidate.moveA, capture);
            ++drops;
        }
        sum += candidate.jointPrior;
    }
    EXPECT_EQ(drops, knight_drops(withInstant));
    EXPECT_NEAR(generator.legal_prior_mass(), sum, 1e-5);
}

TEST(JointPlans, DecideLaterKeepsOldBehaviourAvailable) {
    init_fairy_stockfish();
    init_policy_index();
    // The planned half's priors: "decide later" at least the free share,
    // the moves in proportion; a move that mates as it lands raised.
    const std::vector<Stockfish::Move> actions = {Stockfish::Move(11), Stockfish::Move(12),
                                                  Stockfish::Move(13), MOVE_NONE};
    std::vector<float> priors = {0.5f, 0.3f, 0.1f, 0.1f};
    shape_planned_priors(priors, actions, {0, 0, 0, 0}, 0.5f);
    EXPECT_FLOAT_EQ(priors[3], 0.5f);
    EXPECT_FLOAT_EQ(priors[0], 0.5f * 0.5f / 0.9f);
    EXPECT_FLOAT_EQ(priors[0] / priors[1], 0.5f / 0.3f);
    std::vector<float> highPass = {0.1f, 0.1f, 0.1f, 0.7f};
    shape_planned_priors(highPass, actions, {0, 0, 0, 0}, 0.5f);
    EXPECT_FLOAT_EQ(highPass[3], 0.7f);
    std::vector<float> mate = {0.5f, 0.3f, 0.01f, 0.19f};
    shape_planned_priors(mate, actions, {0, 0, 1, 0}, 0.5f);
    float total = 0.0f;
    for (float prior : mate) {
        total += prior;
    }
    EXPECT_NEAR(total, 1.0f, 1e-6);
    EXPECT_GT(mate[2], 0.19f);

    // At a planning node every immediate half has its "decide later" pair,
    // so everything the tree did without planning is still there.
    Board board = boards(kSmallWhite, kSmallBlack);
    const TimeState time = aw_ready_bb_planned(Rules{});
    const std::vector<Stockfish::Move> immediate = leaf_board_actions(board, BOARD_A, true, time);
    PlannedHalf half = leaf_planned_half(board, BOARD_B, WHITE, time, BOARD_A, immediate);
    std::vector<float> plannedPriors(half.actions.size(), 1.0f / half.actions.size());
    shape_planned_priors(plannedPriors, half.actions, half.mates, time.joint_plan_free_share());
    JointCandidateGenerator generator;
    generator.initialize(immediate, half.actions,
                         std::vector<float>(immediate.size(), 1.0f / immediate.size()),
                         plannedPriors, time.can_sit(WHITE), true, false, {},
                         std::vector<uint8_t>(half.actions.size(), 0), {}, {}, 0, 8, 1.0f,
                         BOARD_B, half.suppliedTypes, half.neededTypes);
    std::set<Stockfish::Move> laterPairs;
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        if (candidate.planned == MOVE_NONE) {
            laterPairs.insert(candidate.moveA);
        }
    }
    for (Stockfish::Move move : immediate) {
        EXPECT_TRUE(laterPairs.contains(move)) << board.uci_move(BOARD_A, move);
    }
    EXPECT_GE(generator.boardPrior(BOARD_B, MOVE_NONE) + 1e-6f,
              time.joint_plan_free_share());
}

TEST(JointPlans, CommittedSittingSeatIsNoSitOff) {
    init_fairy_stockfish();
    // aw_ready_bb_sitting with Ab paced 1 ms: after Aw's move at 201 ms, Ab
    // is ready at 202, and Bb, woken by that landing, too. The exact tie
    // goes to black. Ab ran 2 ms on turn, Bb 202: Ab's clock 200 ms lower
    // makes the two equal there.
    Rules r;
    r.paces = {100, 1, 100, 500};
    r.clocks = {60000, 59800, 60000, 60000};
    const Stockfish::Move plan = Stockfish::Move(4321);
    const auto at_tie = [&](bool planned) {
        TimeState time = aw_ready_bb_sitting(r);
        EXPECT_EQ(time.planned_boards(WHITE), (std::array<bool, 2>{false, true}));
        time.advance_action(WHITE, true, false, Stockfish::NO_PIECE_TYPE,
                            Stockfish::NO_PIECE_TYPE, MOVE_NONE, planned ? plan : MOVE_NONE);
        // The committed move lands 1 ms after Aw's landing (the wake-up rule).
        EXPECT_EQ(time.seat_ready_ms(BB), 1);
        EXPECT_TRUE(time.seat_sitting(BB));
        EXPECT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{true, false}));
        return time;
    };
    // Without a plan Bb sits at Ab's clock, and Ab sitting too is the sit-off draw.
    TimeState plain = at_tie(false);
    ASSERT_EQ(plain.seat_clock_ms(AB), plain.seat_clock_ms(BB));
    plain.advance_action(BLACK, false, false);
    EXPECT_EQ(plain.outcome(), TimeState::Outcome::DRAW);
    // With Bb committed its move is certain to land: no draw. (Ab's pass
    // still sends Bb's wake-up, and so its landing, on to Bb's cap, v4's
    // leapfrog rule: a sitting seat keeps its timing.)
    TimeState committed = at_tie(true);
    ASSERT_EQ(committed.seat_clock_ms(AB), committed.seat_clock_ms(BB));
    committed.advance_action(BLACK, false, false);
    EXPECT_EQ(committed.outcome(), TimeState::Outcome::NONE);
    EXPECT_EQ(committed.committed_move(BB), plan);
    EXPECT_TRUE(committed.seat_sitting(BB));
    Stockfish::Color toPlay = WHITE;
    for (int step = 0; step < 400 && committed.committed_move(BB) != MOVE_NONE; ++step) {
        const std::array<bool, 2> acts = committed.acting_boards(toPlay);
        committed.advance_action(toPlay, false, acts[BOARD_B]);
        toPlay = ~toPlay;
    }
    EXPECT_EQ(committed.outcome(), TimeState::Outcome::NONE);
    EXPECT_EQ(committed.committed_move(BB), MOVE_NONE);
    EXPECT_EQ(committed.side_on(BOARD_B), WHITE);

    // sit_off_forces_move: Ab 100 ms lower than a plain sitting Bb must
    // move; a committed Bb is no sitter that could outwait it.
    r.clocks = {60000, 59700, 60000, 60000};
    EXPECT_FALSE(at_tie(false).can_sit(BLACK));
    EXPECT_TRUE(at_tie(true).can_sit(BLACK));
}

TEST(JointPlans, ActionLookupsComparePlanned) {
    // A planning node (A acting, B planned) with the planned moves x and y
    // beside the same immediate move m.
    const Stockfish::Move m = Stockfish::Move(101);
    const Stockfish::Move x = Stockfish::Move(202);
    const Stockfish::Move y = Stockfish::Move(303);
    const std::vector<Stockfish::Move> actionsA = {m, MOVE_NONE};
    const std::vector<Stockfish::Move> actionsB = {x, y, MOVE_NONE};
    const std::vector<float> priorsA = {0.9f, 0.1f};
    const std::vector<float> priorsB = {0.5f, 0.3f, 0.2f};
    for (Stockfish::Move wanted : {x, y}) {
        JointCandidateGenerator generator;
        generator.initialize(actionsA, actionsB, priorsA, priorsB, true, true, false, {}, {},
                             {}, {}, 0, 8, 1.0f, BOARD_B);
        EXPECT_FALSE(generator.promote(m, wanted));  // the move field must be empty
        ASSERT_TRUE(generator.promote(m, MOVE_NONE, wanted));
        EXPECT_EQ(generator.peekNext().planned, wanted);
        EXPECT_EQ(generator.getNext().moveA, m);
    }
    // Without a planned board a planned move is never a match.
    JointCandidateGenerator plain;
    plain.initialize(actionsA, actionsB, priorsA, priorsB, true, true, true);
    EXPECT_FALSE(plain.promote(m, x, y));

    SearchParams::RuntimeConfig config;
    auto node = std::make_shared<Node>(WHITE);
    node->set_depth(1);  // not a root: no root noise or Gumbel order
    node->try_init_and_expand(actionsA, actionsB, priorsA, priorsB, true, true, false, config,
                              {}, {}, {}, {}, 0, 0, BOARD_B);
    ASSERT_EQ(node->planned_board(), BOARD_B);
    JointActionCandidate expanded;
    while (node->expand_next_joint_child(nullptr, 0, expanded, config)) {
    }
    const std::shared_ptr<Node> childX = node->child_for_action(m, MOVE_NONE, x);
    const std::shared_ptr<Node> childY = node->child_for_action(m, MOVE_NONE, y);
    const std::shared_ptr<Node> later = node->child_for_action(m, MOVE_NONE);
    ASSERT_TRUE(childX && childY && later);
    EXPECT_NE(childX, childY);
    EXPECT_NE(childX, later);

    // The mate-hint bias goes to the hinted action, not its planned sibling.
    config.internalMateProbeBias = 100.0f;
    for (Stockfish::Move wanted : {x, y}) {
        JointActionCandidate hint;
        hint.moveA = m;
        hint.planned = wanted;
        const Node::ChildSelection selection =
            node->select_child_and_apply_virtual_loss(config, nullptr, &hint);
        ASSERT_GE(selection.childIdx, 0);
        EXPECT_TRUE(node->get_joint_action(selection.childIdx).same_moves(hint));
        if (selection.hasEvaluationReservation) {
            selection.child->release_evaluation_reservation();
        }
        node->remove_virtual_loss(selection.childIdx);
    }
}

TEST(JointPlans, CommittedMovesStayLegal) {
    init_fairy_stockfish();
    std::mt19937 rng(20261120);
    int landed = 0;
    int planned = 0;
    for (int line = 0; line < 2000; ++line) {
        Board board;
        Rules r;
        r.instantTransfer = rng() % 2 == 0;
        for (int seat = 0; seat < 4; ++seat) {
            r.paces[seat] = 1 + static_cast<int32_t>(rng() % 900);
            r.elapsed[seat] = static_cast<int32_t>(rng() % 300);
        }
        r.rootMoveMs = 1 + static_cast<int32_t>(rng() % 900);
        const Stockfish::Color rootTeam = rng() % 2 ? WHITE : BLACK;
        TimeState time = durative_root(rootTeam, WHITE, WHITE, r);
        Stockfish::Color toPlay = rootTeam;
        for (int ply = 0; ply < 80 && time.outcome() == TimeState::Outcome::NONE; ++ply) {
            ASSERT_EQ(board.side_to_move(BOARD_A), time.side_on(BOARD_A));
            ASSERT_EQ(board.side_to_move(BOARD_B), time.side_on(BOARD_B));
            const std::array<bool, 2> acts = time.acting_boards(toPlay);
            const std::array<bool, 2> plannable = time.planned_boards(toPlay);
            JointActionCandidate action;
            for (int b : {BOARD_A, BOARD_B}) {
                Stockfish::Move& move = b == BOARD_A ? action.moveA : action.moveB;
                if (acts[b]) {
                    const Stockfish::Move committed =
                        time.committed_move(TimeState::seat_of(b, board.side_to_move(b)));
                    if (committed != MOVE_NONE) {
                        ASSERT_TRUE(board.is_legal_move(b, committed))
                            << board.fen(b) << " " << board.uci_move(b, committed);
                        move = committed;
                        ++landed;
                        continue;
                    }
                    const std::vector<Stockfish::Move> legal = board.legal_moves(b);
                    if (!legal.empty() && rng() % 5 != 0) {
                        move = legal[rng() % legal.size()];
                    }
                } else if (plannable[b] && rng() % 2 == 0) {
                    const std::vector<Stockfish::Move> legal = board.legal_moves(b);
                    if (!legal.empty()) {
                        action.planned = legal[rng() % legal.size()];
                        ++planned;
                    }
                }
            }
            const TimeState next = durative_time_after(time, board, toPlay, action);
            board.make_moves(action.moveA, action.moveB);
            time = next;
            toPlay = ~toPlay;
        }
    }
    EXPECT_GT(planned, 1000);
    EXPECT_GT(landed, 500);
}

TEST(ContinuityRules, OutcomeUnchangedByCommitment) {
    // Bb has 350 ms on its clock and would land at 400: it flags first,
    // planned or not, at the same step, and the team loses on the clock.
    Rules r;
    r.flagLoss = true;
    r.clocks = {60000, 60000, 60000, 350};
    const auto play = [&r](Stockfish::Move plan) {
        TimeState time = aw_ready_bb_planned(r);
        time.advance_action(WHITE, true, false, Stockfish::NO_PIECE_TYPE,
                            Stockfish::NO_PIECE_TYPE, MOVE_NONE, plan);
        Stockfish::Color toPlay = BLACK;
        int steps = 1;
        while (time.outcome() == TimeState::Outcome::NONE && steps < 20) {
            const std::array<bool, 2> acts = time.acting_boards(toPlay);
            time.advance_action(toPlay, acts[0], acts[1]);
            toPlay = ~toPlay;
            ++steps;
        }
        return std::make_pair(time.outcome(), steps);
    };
    const auto plain = play(MOVE_NONE);
    const auto committed = play(Stockfish::Move(777));
    EXPECT_EQ(plain.first, TimeState::Outcome::BLACK_TEAM_WINS);
    EXPECT_EQ(committed, plain);

    // v4 acceptance test 3's symmetric sit-off is a draw with JointPlans on
    // too: no seat there holds a commitment.
    Rules sitOff;
    sitOff.clocks = {30000, 60000, 30000, 60000};
    sitOff.paces = {100, 100, 2000, 100};
    sitOff.rootMoveMs = 500;
    TimeState time = durative_root(WHITE, WHITE, WHITE, sitOff);
    time.advance_action(WHITE, false, false);
    ASSERT_EQ(time.acting_boards(BLACK), (std::array<bool, 2>{false, true}));
    EXPECT_EQ(time.planned_boards(BLACK), (std::array<bool, 2>{false, false}));
    time.advance_action(BLACK, false, false);
    EXPECT_EQ(time.outcome(), TimeState::Outcome::DRAW);
}

TEST(ContinuityRules, ASeatWith3sStillMakesAtMost29MovesWithPlans) {
    // v4 acceptance test 4 with plans: Aw has 3 s at the 100 ms minimum, and
    // a planned move counts once it lands.
    constexpr int32_t MIN_MOVE_MS = 100;
    std::mt19937 rng(20261121);
    int longest = 0;
    for (int line = 0; line < 2000; ++line) {
        Rules r;
        r.flagLoss = true;
        r.clocks = {3000, 600000, 600000, 600000};
        r.paces = {MIN_MOVE_MS, MIN_MOVE_MS, MIN_MOVE_MS, MIN_MOVE_MS};
        r.rootMoveMs = MIN_MOVE_MS;
        TimeState time = durative_root(WHITE, WHITE, WHITE, r);
        Stockfish::Color toPlay = WHITE;
        int awMoves = 0;
        for (int ply = 0; ply < 4000 && time.outcome() == TimeState::Outcome::NONE; ++ply) {
            const std::array<bool, 2> acts = time.acting_boards(toPlay);
            const std::array<bool, 2> plannable = time.planned_boards(toPlay);
            bool moved[2] = {false, false};
            Stockfish::Move plan[2] = {MOVE_NONE, MOVE_NONE};
            for (int board = 0; board < 2; ++board) {
                const int seat = TimeState::seat_of(board, time.side_on(board));
                // A committed seat lands its move; the others move or pass.
                moved[board] = acts[board]
                    && (time.committed_move(seat) != MOVE_NONE || line % 2 == 0 || rng() % 4 != 0);
                if (plannable[board] && rng() % 2 == 0) {
                    plan[board] = Stockfish::Move(1);
                }
            }
            const bool awMoving = acts[0] && moved[0] && time.side_on(0) == WHITE;
            time.advance_action(toPlay, moved[0], moved[1], Stockfish::NO_PIECE_TYPE,
                                Stockfish::NO_PIECE_TYPE, plan[0], plan[1]);
            if (awMoving && time.outcome() == TimeState::Outcome::NONE) {
                ++awMoves;
            }
            toPlay = ~toPlay;
        }
        EXPECT_NE(time.outcome(), TimeState::Outcome::NONE);
        longest = std::max(longest, awMoves);
    }
    EXPECT_LE(longest, 3000 / MIN_MOVE_MS - 1);
}

// ---- searches -------------------------------------------------------------

#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
class JointPlansSearch : public ::testing::Test {
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

    static SearchOptions options(const TimeState& clock, size_t nodes, bool teamTree = false) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.clock = clock;
        opts.teamTreeReuse = teamTree;
        opts.search.enableRootMateSearch = false;
        opts.search.certifySelectedMove = false;
        opts.search.enableMateProbe = false;
        return opts;
    }

    std::string search(Agent& agent, Board& board, Stockfish::Color team,
                       const SearchOptions& opts) {
        testing::internal::CaptureStdout();
        agent.run_search(board, {engine.get()}, team, false, opts);
        return testing::internal::GetCapturedStdout();
    }

    std::unique_ptr<Engine> engine;
};

/** Counts per tree: planning nodes, planned edges, forced landings checked. */
struct TreeCounts {
    int planningNodes = 0;
    int plannedEdges = 0;
    int forcedNodes = 0;
};

TreeCounts check_tree(Node* root, Board& board, const TimeState& time) {
    TreeCounts counts;
    walk_tree(root, board, time, [&counts](Node& node, Board& at, const TimeState& t) {
        EXPECT_EQ(at.side_to_move(BOARD_A), t.side_on(BOARD_A));
        EXPECT_EQ(at.side_to_move(BOARD_B), t.side_on(BOARD_B));
        if (!node.is_expanded()) {
            return;
        }
        const Stockfish::Color team = node.get_team_to_play();
        const std::array<bool, 2> plannable = t.planned_boards(team);
        const int planned = node.planned_board();
        if (planned >= 0) {
            ++counts.planningNodes;
            EXPECT_TRUE(plannable[planned]);
        }
        const std::array<bool, 2> acts = t.acting_boards(team);
        std::array<Stockfish::Move, 2> forced{MOVE_NONE, MOVE_NONE};
        for (int b : {BOARD_A, BOARD_B}) {
            if (acts[b]) {
                forced[b] = t.committed_move(TimeState::seat_of(b, at.side_to_move(b)));
            }
        }
        if (forced[0] != MOVE_NONE || forced[1] != MOVE_NONE) {
            ++counts.forcedNodes;
        }
        for (size_t i = 0; i < node.get_num_generated(); ++i) {
            const JointActionCandidate action = node.get_joint_action(static_cast<int>(i));
            if (action.planned != MOVE_NONE) {
                ++counts.plannedEdges;
                EXPECT_GE(planned, 0);
                // The planned board's own move field is empty.
                EXPECT_EQ(planned == BOARD_A ? action.moveA : action.moveB, MOVE_NONE);
            }
            // A seat that lands its committed move has no other action.
            if (forced[BOARD_A] != MOVE_NONE) {
                EXPECT_EQ(action.moveA, forced[BOARD_A]);
            }
            if (forced[BOARD_B] != MOVE_NONE) {
                EXPECT_EQ(action.moveB, forced[BOARD_B]);
            }
        }
    });
    return counts;
}

TEST_F(JointPlansSearch, CandidateLayoutUnchangedWithFixOff) {
    // 0.2: with fix 2 off no candidate carries a plan and no node plans.
    Board board = boards(kSmallWhite, kSmallBlack);
    Rules off;
    off.jointPlans = false;
    const TimeState time = aw_ready_bb_planned(off);
    Agent agent(1);
    search(agent, board, WHITE, options(time, 1500));
    const TreeCounts counts = check_tree(AgentContinuityFix2Peer::root(agent).get(), board, time);
    EXPECT_EQ(counts.planningNodes, 0);
    EXPECT_EQ(counts.plannedEdges, 0);
    EXPECT_EQ(counts.forcedNodes, 0);
}

TEST_F(JointPlansSearch, PlannedHalvesAndForcedLandingsInTheTree) {
    // 2.1 in a search: planning nodes plan only where planned_boards says,
    // and a node where a committed seat lands has that move as its only
    // action on that board.
    Board board = boards(kSmallWhite, kSmallBlack);
    const TimeState time = aw_ready_bb_planned(Rules{});
    Agent agent(1);
    search(agent, board, WHITE, options(time, 3000));
    std::shared_ptr<Node> root = AgentContinuityFix2Peer::root(agent);
    ASSERT_TRUE(root);
    EXPECT_EQ(root->planned_board(), BOARD_B);
    const TreeCounts counts = check_tree(root.get(), board, time);
    EXPECT_GT(counts.planningNodes, 0);
    EXPECT_GT(counts.plannedEdges, 0);
    EXPECT_GT(counts.forcedNodes, 0);
}

TEST_F(JointPlansSearch, CommittedMovesStayLegalInASearch) {
    // 2.2's search half: a 4000-node fix-2 search with the side invariant
    // checked at every leaf, then PV extraction, retain_played_line and a
    // fix-1 adoption walk; the board is unchanged afterwards.
    Board board = boards(kSmallWhite, kSmallBlack);
    const std::string fenA = board.fen(BOARD_A);
    const std::string fenB = board.fen(BOARD_B);
    const TimeState time = aw_ready_bb_planned(Rules{});
    SearchThread::checkDurativeSides = true;
    SearchThread::durativeSideMismatches = 0;
    Agent agent(1);
    search(agent, board, WHITE, options(time, 4000, true));
    EXPECT_EQ(SearchThread::durativeSideMismatches.load(), 0);
    std::shared_ptr<Node> root = AgentContinuityFix2Peer::root(agent);
    ASSERT_TRUE(root);
    EXPECT_GT(check_tree(root.get(), board, time).plannedEdges, 0);
    for (size_t i = 0; i < root->get_num_generated(); ++i) {
        AgentContinuityFix2Peer::pv_from_child(agent, board, static_cast<int>(i), WHITE, time);
    }
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);

    // Play the most visited root action and black's most visited reply,
    // then search again with fix 1 on: the descent and the re-timing walk
    // run over a tree with planned edges.
    const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
    ASSERT_FALSE(edges.empty());
    const RootEdgeStats best = *std::max_element(
        edges.begin(), edges.end(),
        [](const RootEdgeStats& a, const RootEdgeStats& b) { return a.visits < b.visits; });
    std::shared_ptr<Node> reply = root->child_for_action(best.action.moveA, best.action.moveB,
                                                         best.action.planned);
    ASSERT_TRUE(reply);
    TimeState path = durative_time_after(time, board, WHITE, best.action);
    Board live(board);
    live.make_moves(best.action.moveA, best.action.moveB);
    if (reply->is_expanded() && reply->get_num_generated() > 0) {
        const JointActionCandidate answer = reply->get_joint_action(0);
        path = durative_time_after(path, live, BLACK, answer);
        live.make_moves(answer.moveA, answer.moveB);
    }
    EXPECT_GE(agent.retain_played_line(board, {best.action}), 0);
    Rules again;
    for (int seat = 0; seat < 4; ++seat) {
        again.clocks[seat] = path.seat_clock_ms(seat) + 3;
    }
    const TimeState liveTime = durative_root(WHITE, live.side_to_move(BOARD_A),
                                             live.side_to_move(BOARD_B), again);
    search(agent, live, WHITE, options(liveTime, 300, true));
    EXPECT_TRUE(agent.team_tree_reuse().attempted);
    EXPECT_EQ(SearchThread::durativeSideMismatches.load(), 0);
    SearchThread::checkDurativeSides = false;
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
}

TEST_F(JointPlansSearch, RootUnchanged) {
    // 2.6: a UCI root (both our seats decide there) has no planned edge and
    // the same root actions with fix 2 on as off.
    Board board = boards(kSmallWhite, kSmallBlack);
    Rules on;
    Rules off;
    off.jointPlans = false;
    Agent withPlans(1);
    Agent without(1);
    search(withPlans, board, WHITE, options(durative_root(WHITE, WHITE, BLACK, on), 400));
    search(without, board, WHITE, options(durative_root(WHITE, WHITE, BLACK, off), 400));
    std::shared_ptr<Node> a = AgentContinuityFix2Peer::root(withPlans);
    std::shared_ptr<Node> b = AgentContinuityFix2Peer::root(without);
    ASSERT_TRUE(a && b);
    EXPECT_EQ(a->planned_board(), -1);
    for (int boardNum : {BOARD_A, BOARD_B}) {
        std::vector<Stockfish::Move> listA = a->expanded_actions(boardNum);
        std::vector<Stockfish::Move> listB = b->expanded_actions(boardNum);
        std::sort(listA.begin(), listA.end());
        std::sort(listB.begin(), listB.end());
        EXPECT_EQ(listA, listB);
    }
    for (const RootEdgeStats& edge : withPlans.root_edge_stats()) {
        EXPECT_EQ(edge.action.planned, MOVE_NONE);
    }
    const TimeState time = durative_root(WHITE, WHITE, BLACK, on);
    // Below the root, black plans (its planned pairs need visits to be
    // expanded beside "decide later", which holds half the prior).
    EXPECT_GT(check_tree(a.get(), board, time).planningNodes, 0);
}

TEST_F(JointPlansSearch, PvPrintsPlannedHalvesAsPass) {
    // 2.7: a PV through a planned step prints that step's planned board as
    // pass and the move at its landing step; the line parses as the arena
    // reads it.
    Board board = boards(kSmallWhite, kSmallBlack);
    Rules r;
    r.freePermille = 0;  // so most root edges carry a plan
    const TimeState time = aw_ready_bb_planned(r);
    Agent agent(1);
    search(agent, board, WHITE, options(time, 4000));
    std::shared_ptr<Node> root = AgentContinuityFix2Peer::root(agent);
    ASSERT_TRUE(root);
    const std::vector<int> visits = root->get_child_visits();
    int best = -1;
    for (size_t i = 0; i < visits.size(); ++i) {
        const JointActionCandidate action = root->get_joint_action(static_cast<int>(i));
        if (action.planned != MOVE_NONE && (best < 0 || visits[i] > visits[best])) {
            best = static_cast<int>(i);
        }
    }
    ASSERT_GE(best, 0);
    const JointActionCandidate action = root->get_joint_action(best);
    const std::string plannedUci = board.uci_move(BOARD_B, action.planned);
    const std::string pv = AgentContinuityFix2Peer::pv_from_child(agent, board, best, WHITE, time);
    EXPECT_EQ(pv.rfind("(" + board.uci_move(BOARD_A, action.moveA) + ",pass)", 0), 0u) << pv;
    // The landing step shows the planned move on board B.
    EXPECT_NE(pv.find("," + plannedUci + ")"), std::string::npos) << pv << " planned " << plannedUci;
    const std::regex info(
        R"(^info depth (\d+)(?: multipv (\d+))? score (cp|mate) (-?\d+) nodes (\d+))");
    EXPECT_TRUE(std::regex_search("info depth 6 score cp 12 nodes 4000 nps 1 pv " + pv, info));
    const std::regex step(R"(^\((pass|[a-zA-Z0-9@]+),(pass|[a-zA-Z0-9@]+)\)$)");
    std::istringstream tokens(pv);
    for (std::string token; tokens >> token;) {
        EXPECT_TRUE(std::regex_match(token, step)) << token;
    }
}

TEST_F(JointPlansSearch, TeamTreeReuseRefusesCommittedCandidate) {
    // 1.16: the white tree from aw_ready_bb_planned reaches the position
    // after Aw's move and Ab's reply both with Bb committed (planned edge)
    // and without (decide later). The live root there has no commitment, so
    // a committed node is never adopted; the decide-later twin acted on A
    // only and has the wrong shape for a root where both white seats decide,
    // so the furthest refusal is `committed`.
    Board board = boards(kSmallWhite, kSmallBlack);
    const TimeState time = aw_ready_bb_planned(Rules{});
    Agent agent(1);
    search(agent, board, WHITE, options(time, 3000, true));
    std::shared_ptr<Node> root = AgentContinuityFix2Peer::team_tree_root(agent);
    ASSERT_TRUE(root);
    // The most visited planned root edge and the reply below it.
    const std::vector<int> visits = root->get_child_visits();
    int best = -1;
    for (size_t i = 0; i < visits.size(); ++i) {
        if (root->get_joint_action(static_cast<int>(i)).planned != MOVE_NONE
            && (best < 0 || visits[i] > visits[best])) {
            best = static_cast<int>(i);
        }
    }
    ASSERT_GE(best, 0);
    const JointActionCandidate first = root->get_joint_action(best);
    std::shared_ptr<Node> black = root->get_child(best);
    ASSERT_TRUE(black && black->is_expanded());
    TimeState path = durative_time_after(time, board, WHITE, first);
    Board live(board);
    live.make_moves(first.moveA, first.moveB);
    const JointActionCandidate reply = black->get_joint_action(0);
    path = durative_time_after(path, live, BLACK, reply);
    live.make_moves(reply.moveA, reply.moveB);
    ASSERT_EQ(path.committed_move(BB), first.planned);
    // The live root as a front end sends it: the path's clocks, white to
    // decide on A and B.
    Rules liveRules;
    for (int seat = 0; seat < 4; ++seat) {
        liveRules.clocks[seat] = path.seat_clock_ms(seat);
    }
    const TimeState liveTime = durative_root(WHITE, live.side_to_move(BOARD_A),
                                             live.side_to_move(BOARD_B), liveRules);
    const auto match =
        AgentContinuityFix2Peer::find(agent, live, WHITE, liveTime, 5000);
    EXPECT_FALSE(match.node);
    EXPECT_EQ(match.reason, "committed");
    EXPECT_EQ(match.committed[BB], first.planned);

    // The other team's commitment: a white root where Bw (black) is on turn
    // on B and lands late, so black, acting on A, plans Bw. The white node
    // after black's reply holds an opponent commitment and is never adopted.
    Board second = boards(kSmallWhite, kSmallWhite);
    Rules opp;
    opp.freePermille = 0;  // planned pairs expanded early
    opp.paces = {100, 50, 500, 100};
    opp.elapsed = {0, 0, 100, 0};
    const TimeState t0 = durative_root(WHITE, WHITE, WHITE, opp);
    Agent other(1);
    search(other, second, WHITE, options(t0, 4000, true));
    std::shared_ptr<Node> otherRoot = AgentContinuityFix2Peer::team_tree_root(other);
    ASSERT_TRUE(otherRoot);
    int opponentCommitted = 0;
    walk_tree(otherRoot.get(), second, t0,
              [&](Node& node, Board& at, const TimeState& t) {
                  if (node.get_team_to_play() != WHITE || node.is_expanded() == false
                      || t.committed_move(BW) == MOVE_NONE || &node == otherRoot.get()) {
                      return;
                  }
                  ++opponentCommitted;
                  if (opponentCommitted > 5) {
                      return;
                  }
                  Rules liveR = opp;
                  for (int seat = 0; seat < 4; ++seat) {
                      liveR.clocks[seat] = t.seat_clock_ms(seat);
                  }
                  liveR.elapsed = {0, 0, opp.paces[BW] - t.seat_ready_ms(BW), 0};
                  const TimeState l = durative_root(WHITE, at.side_to_move(BOARD_A),
                                                    at.side_to_move(BOARD_B), liveR);
                  Board copy(at);
                  const auto m =
                      AgentContinuityFix2Peer::find(other, copy, WHITE, l, 5000);
                  if (m.node) {
                      // Adopted: a twin without any commitment, never this node.
                      EXPECT_NE(m.node.get(), &node);
                  } else {
                      EXPECT_EQ(m.reason, "committed");
                      EXPECT_NE(m.committed[BW], MOVE_NONE);
                  }
              });
    EXPECT_GT(opponentCommitted, 0);
}

// ---- 2.4a, 2.4b: coordination with a real network --------------------------

class JointPlansIt13 : public ::testing::Test {
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

    /** The most visited root edge of a @p nodes search, as text, and its child's Q. */
    struct Choice {
        JointActionCandidate action;
        std::string text;
        int visits = 0;
        float q = 0.0f;
        NodeType childType = NodeType::UNSOLVED;
    };
    Choice choose(Board& board, Stockfish::Color team, const TimeState& clock, size_t nodes,
                  bool provers = true) {
        Agent agent(1);
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.enablePonder = false;
        opts.clock = clock;
        if (!provers) {
            opts.search.enableRootMateSearch = false;
            opts.search.certifySelectedMove = false;
            opts.search.enableMateProbe = false;
        }
        testing::internal::CaptureStdout();
        const JointActionCandidate chosen = agent.run_search(board, {engine.get()}, team, false, opts);
        testing::internal::GetCapturedStdout();
        lastRootType = static_cast<int>(agent.root_type());
        Choice best;
        best.action = chosen;
        for (const RootEdgeStats& edge : agent.root_edge_stats()) {
            if (edge.action.same_moves(chosen)) {
                best.visits = edge.visits;
                best.q = edge.q;
                best.childType = edge.childType;
            }
        }
        const auto uci = [&board](int b, Stockfish::Move move) {
            return move == MOVE_NONE ? std::string("pass") : board.uci_move(b, move);
        };
        const int plannedBoard = AgentContinuityFix2Peer::root(agent)->planned_board();
        best.text = "(" + uci(BOARD_A, best.action.moveA) + "," + uci(BOARD_B, best.action.moveB)
            + ")";
        if (best.action.planned != MOVE_NONE && plannedBoard >= 0) {
            best.text += " planned " + uci(plannedBoard, best.action.planned);
        }
        return best;
    }

    std::unique_ptr<Engine> engine;
    int lastRootType = 0;
};

/**
 * Spec 2.4a: our team white; B: Bw's king on h1 smothered (Rg1, g2, h2, f2
 * unguarded), Bb (black) to move with an empty pocket, so N@f2# needs a
 * knight; A: Aw's queen takes a knight on f6 that a pawn recaptures. Ab's
 * root move has just landed: Aw is ready at 100 ms, Ab replies after 50, Bb
 * is on turn and ready at 400. InstantTransfer, FlagLoss, no SitGate.
 *
 * As measured (docs/wave2_engine.md, Wave 2c fix 2): the tree without fix 2
 * finds Qxf6 too, at every budget from 64 nodes, with or without the root
 * mate provers, and proves it within a few root visits: after the capture
 * black is on turn on one board while N@f2# waits on the other and survives
 * every reply, which the waiting-board mate rule
 * (has_unavoidable_waiting_board_mate) proves. With fix 2 the tree proves
 * the same move through its "decide later" pair. A harder position that the
 * old tree misses would no longer be a mate that survives every reply, so
 * not the forced win this test needs. The test records that both trees
 * choose and prove the capture.
 */
TEST_F(JointPlansIt13, FindsCaptureFeedingAMatingDropAtLowNodes) {
    struct Case {
        Stockfish::Color team;
        std::string fenA, fenB, capture;
    };
    const std::vector<Case> cases = {
        {WHITE, "4k3/4ppp1/5n2/8/8/5Q2/5PPP/4K3[] w - - 0 1", "k7/8/8/8/8/8/6PP/6RK[] b - - 0 1",
         "f3f6"},
        // The mirror: our team black (Ab, Bw), Bb's king smothered on h8.
        {BLACK, "4k3/5ppp/5q2/8/8/5N2/4PPP1/4K3[] b - - 0 1", "6rk/6pp/8/8/8/8/8/K7[] w - - 0 1",
         "f6f3"},
    };
    for (const Case& c : cases) {
        Board board = boards(c.fenA, c.fenB);
        const Stockfish::Move capture = find_move(board, BOARD_A, c.capture);
        ASSERT_NE(capture, MOVE_NONE);
        for (bool plans : {true, false}) {
            Rules r;
            r.jointPlans = plans;
            r.instantTransfer = true;
            r.flagLoss = true;
            r.rootMoveMs = 50;
            TimeState clock;
            if (c.team == WHITE) {
                r.paces = {100, 50, 100, 500};
                r.elapsed = {0, 0, 0, 50};
                clock = durative_root(BLACK, BLACK, BLACK, r);
            } else {
                r.paces = {50, 100, 500, 100};
                r.elapsed = {0, 0, 50, 0};
                clock = durative_root(WHITE, WHITE, WHITE, r);
            }
            clock.advance_action(~c.team, true, false);
            ASSERT_EQ(clock.planned_boards(c.team),
                      plans ? (std::array<bool, 2>{false, true}) : (std::array<bool, 2>{false, false}));
            for (bool provers : {true, false}) {
                const Choice choice = choose(board, c.team, clock, 64, provers);
                SCOPED_TRACE(std::string(c.team == WHITE ? "white" : "black")
                             + (plans ? " fix 2" : " old tree") + (provers ? " provers" : " tree only")
                             + " chose " + choice.text);
                EXPECT_EQ(choice.action.moveA, capture);
                EXPECT_EQ(choice.childType, NodeType::LOSS);
                EXPECT_EQ(lastRootType, static_cast<int>(NodeType::WIN));
            }
        }
    }
}
#endif

}  // namespace
