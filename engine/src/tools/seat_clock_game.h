#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "environment/board.h"
#include "environment/time_state.h"
#include "search/hybrid.h"
#include "tools/clock_sampling.h"

/**
 * @brief Wave 2 options of the selfplay and tournament commands
 * (docs/wave2_training.md, "Interface the wave 2 engine must provide").
 *
 * The first six are the UCI options of every search (DurativeTree, SitGate,
 * FlagLoss, InstantTransfer, MateRaceByLanding, PassCapK). A base mix turns
 * on real seat clocks: the game is then played event by event on four
 * clocks (SeatClockGame). Every default is stock v3.0.3, and with defaults
 * the commands play exactly as before.
 */
struct Wave2GameOptions {
    bool durativeTree = false;
    bool sitGate = true;
    bool flagLoss = false;
    bool instantTransfer = true;
    bool mateRaceByLanding = false;
    int32_t passCapK = 8;
    // Base clock per game, drawn with these weights; empty: no seat clocks.
    std::vector<std::pair<int32_t, double>> baseMix;
    // Each seat draws its own move time (else each team, as v3).
    bool seatMoveTimes = false;
    // Seat-clock games only (spec v3 D2/D4, arena RULES_V3): every landing
    // costs at least minMoveMs, plus a lag drawn from [0, lagMaxMs], and the
    // seat's move time is scaled by a factor drawn log-uniformly from
    // [1 / (1 + paceJitter), 1 + paceJitter].
    int32_t minMoveMs = 100;
    int32_t lagMaxMs = 150;
    double paceJitter = 0.25;
    // Both seats on turn sitting this long at once ends the game: the lower
    // clock loses, equal clocks draw (arena sitoff_ms).
    int32_t sitoffMs = 5000;
    // Weight of the newest landing in each seat's observed pace (SeatPaceMs).
    double paceEmaAlpha = 0.3;
    // Wave 2c, fix 1 (UCI TeamTreeReuse, docs/custom_engine_spec_v5_continuity.md):
    // each team's agent keeps its tree between its decisions and adopts the
    // node the game reached, re-timed, within this clock tolerance.
    // Seat-clock games with the durative tree only.
    bool teamTreeReuse = false;
    int32_t teamTreeReuseToleranceMs = 250;
    // Wave 2c, fix 2 (UCI JointPlans): below the root a team that acts plans
    // its other on-turn seat's move in the same joint action, with this prior
    // share of "decide later". Durative tree only; inside each search.
    bool jointPlans = false;
    int32_t jointPlanFreePermille = 500;
    // Wave 2c, fix 3 (UCI PlanCarry): the half of a team's chosen action for
    // its on-turn seat that is not deciding (and has no move in flight) is
    // remembered and mixed into that seat's next search at this weight.
    // Seat-clock games with the durative tree only.
    bool planCarry = false;
    int32_t planCarryPermille = 300;
    // Diagnosis fixes (docs/wave2_engine.md, "Diagnosis fixes"). D1 (UCI
    // OwnPaceBelowRoot): below the root our seats act at their own pace.
    // D3 (UCI WaitNodeActingEval): forced waits evaluated from the acting
    // team. Both durative tree only. D5 (UCI BlockableCheckNotMate): a
    // drop-blockable check is never mate, in the game's own end check as in
    // every search (the arena's rule); any tree.
    bool ownPaceBelowRoot = false;
    bool waitNodeActingEval = false;
    bool blockableCheckNotMate = false;
    // Spec v6, the hybrid (UCI HybridMode and knobs; --hybrid-mode,
    // --hybrid, --hybrid-event-lands, --hybrid-max-wait-ms,
    // --hybrid-min-clock-ms, --hybrid-margin-permille,
    // --hybrid-pass-prior-permille): the v3 tree's root may also sit to wait
    // for a landing on the other board. Seat-clock games with the v3 tree.
    HybridKnobs hybrid;

    bool seat_clocks() const { return !baseMix.empty(); }

    /**
     * Whether any search option differs from stock. A wave 2c knob only
     * counts with its switch (validate() refuses it without), so a knob
     * alone never moves a stock run onto the clock time model. D5 is a board
     * rule, not a clock option: alone it does not count either.
     */
    bool changes_search() const {
        return durativeTree || !sitGate || flagLoss || !instantTransfer
            || mateRaceByLanding || passCapK != 8 || teamTreeReuse || jointPlans
            || planCarry || ownPaceBelowRoot || waitNodeActingEval
            || hybrid.mode != HybridMode::OFF;
    }

    /** D5: the game's board (and so every search's copy) plays the arena's mate rule. */
    void apply_board_rules(Board& board) const {
        board.set_blockable_check_not_mate(blockableCheckNotMate);
    }

    void validate() const {
        if (passCapK < 1 || passCapK > 64) {
            throw std::invalid_argument("--pass-cap-k must be in 1..64");
        }
        if (teamTreeReuseToleranceMs < 0 || teamTreeReuseToleranceMs > 5000) {
            throw std::invalid_argument("--team-tree-reuse-tolerance-ms must be in 0..5000");
        }
        if (jointPlanFreePermille < 0 || jointPlanFreePermille > 1000) {
            throw std::invalid_argument("--joint-plan-free-permille must be in 0..1000");
        }
        if (planCarryPermille < 0 || planCarryPermille > 1000) {
            throw std::invalid_argument("--plan-carry-permille must be in 0..1000");
        }
        // A knob without its switch does nothing: refuse it rather than run
        // a setting the caller believes is on.
        if (!teamTreeReuse && teamTreeReuseToleranceMs != 250) {
            throw std::invalid_argument(
                "--team-tree-reuse-tolerance-ms needs --team-tree-reuse true");
        }
        if (!jointPlans && jointPlanFreePermille != 500) {
            throw std::invalid_argument("--joint-plan-free-permille needs --joint-plans true");
        }
        if (!planCarry && planCarryPermille != 300) {
            throw std::invalid_argument("--plan-carry-permille needs --plan-carry true");
        }
        if (planCarry && (!durativeTree || baseMix.empty())) {
            // Only the seat-clock loops play one seat's half and keep the
            // agent until the partner decides.
            throw std::invalid_argument(
                "--plan-carry needs --durative-tree true and seat clocks (--base-mix)");
        }
        if (jointPlans && !durativeTree) {
            throw std::invalid_argument("--joint-plans needs --durative-tree true");
        }
        if (ownPaceBelowRoot && !durativeTree) {
            throw std::invalid_argument("--own-pace-below-root needs --durative-tree true");
        }
        if (waitNodeActingEval && !durativeTree) {
            throw std::invalid_argument("--wait-node-acting-eval needs --durative-tree true");
        }
        if (teamTreeReuse && (!durativeTree || baseMix.empty())) {
            throw std::invalid_argument(
                "--team-tree-reuse needs --durative-tree true and seat clocks (--base-mix)");
        }
        if (hybrid.maxWaitMs < 0 || hybrid.maxWaitMs > 10000) {
            throw std::invalid_argument("--hybrid-max-wait-ms must be in 0..10000");
        }
        if (hybrid.minClockMs < 0 || hybrid.minClockMs > 600000) {
            throw std::invalid_argument("--hybrid-min-clock-ms must be in 0..600000");
        }
        if (hybrid.marginPermille < 0 || hybrid.marginPermille > 1000
            || hybrid.passPriorPermille < 0 || hybrid.passPriorPermille > 1000) {
            throw std::invalid_argument(
                "--hybrid-margin-permille and --hybrid-pass-prior-permille must be in 0..1000");
        }
        if (hybrid.mode == HybridMode::PROBE || hybrid.mode == HybridMode::PRICED) {
            throw std::invalid_argument(std::string("--hybrid-mode ") + hybrid_mode_name(hybrid.mode)
                                        + " is not built (spec v6 builds gate first)");
        }
        if (hybrid.mode == HybridMode::OFF && hybrid.knobs_changed()) {
            throw std::invalid_argument("a --hybrid-* knob needs --hybrid-mode gate");
        }
        if (hybrid.mode != HybridMode::OFF && (durativeTree || baseMix.empty())) {
            // The hybrid adds to the v3 tree, and its trigger reads the seat clocks.
            throw std::invalid_argument(
                "--hybrid-mode needs --durative-tree false and seat clocks (--base-mix)");
        }
        for (const auto& [base, weight] : baseMix) {
            if (base < 1 || !(weight > 0.0)) {
                throw std::invalid_argument("--base-mix entries need base >= 1 ms and weight > 0");
            }
        }
        if (minMoveMs < 1 || lagMaxMs < 0 || paceJitter < 0.0 || sitoffMs < 1
            || !(paceEmaAlpha > 0.0 && paceEmaAlpha <= 1.0)) {
            throw std::invalid_argument("invalid seat-clock timing options");
        }
    }

    /** The tree options of a search; clocks and paces are the caller's. */
    DurativeConfig durative_config() const {
        DurativeConfig config;
        config.flagLoss = flagLoss;
        config.instantTransfer = instantTransfer;
        config.mateRaceByLanding = mateRaceByLanding;
        config.passCapK = passCapK;
        config.jointPlans = jointPlans;
        config.jointPlanFreePermille = jointPlanFreePermille;
        config.ownPaceBelowRoot = ownPaceBelowRoot;
        config.waitNodeActingEval = waitNodeActingEval;
        return config;
    }
};

/** "60000:0.25,120000:0.75" -> {{60000, .25}, {120000, .75}}; "" or "off" -> {}. */
inline std::vector<std::pair<int32_t, double>> parse_base_mix(const std::string& text) {
    std::vector<std::pair<int32_t, double>> mix;
    if (text.empty() || text == "off") {
        return mix;
    }
    std::stringstream items(text);
    for (std::string item; std::getline(items, item, ',');) {
        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            throw std::invalid_argument("--base-mix entry '" + item + "' is not MS:WEIGHT");
        }
        mix.emplace_back(std::stoi(item.substr(0, colon)), std::stod(item.substr(colon + 1)));
    }
    return mix;
}

/**
 * @brief Parses one wave 2 command-line option into @p options; false for any
 * other option.
 */
inline bool parse_wave2_option(const std::string& option, const std::string& value,
                               Wave2GameOptions& options) {
    auto boolean = [&value, &option]() {
        if (value == "true" || value == "1" || value == "on") return true;
        if (value == "false" || value == "0" || value == "off") return false;
        throw std::invalid_argument(option + " expects true/false, got: " + value);
    };
    if (option == "--durative-tree") options.durativeTree = boolean();
    else if (option == "--sit-gate") options.sitGate = boolean();
    else if (option == "--flag-loss") options.flagLoss = boolean();
    else if (option == "--instant-transfer") options.instantTransfer = boolean();
    else if (option == "--mate-race-by-landing") options.mateRaceByLanding = boolean();
    else if (option == "--pass-cap-k") options.passCapK = std::stoi(value);
    else if (option == "--base-mix") options.baseMix = parse_base_mix(value);
    else if (option == "--seat-move-times") options.seatMoveTimes = boolean();
    else if (option == "--min-move-ms") options.minMoveMs = std::stoi(value);
    else if (option == "--lag-max-ms") options.lagMaxMs = std::stoi(value);
    else if (option == "--pace-jitter") options.paceJitter = std::stod(value);
    else if (option == "--sitoff-ms") options.sitoffMs = std::stoi(value);
    else if (option == "--pace-ema-alpha") options.paceEmaAlpha = std::stod(value);
    else if (option == "--team-tree-reuse") options.teamTreeReuse = boolean();
    else if (option == "--team-tree-reuse-tolerance-ms") options.teamTreeReuseToleranceMs = std::stoi(value);
    else if (option == "--joint-plans") options.jointPlans = boolean();
    else if (option == "--joint-plan-free-permille") options.jointPlanFreePermille = std::stoi(value);
    else if (option == "--plan-carry") options.planCarry = boolean();
    else if (option == "--plan-carry-permille") options.planCarryPermille = std::stoi(value);
    else if (option == "--own-pace-below-root") options.ownPaceBelowRoot = boolean();
    else if (option == "--wait-node-acting-eval") options.waitNodeActingEval = boolean();
    else if (option == "--blockable-check-not-mate") options.blockableCheckNotMate = boolean();
    else if (option == "--hybrid-mode") {
        if (!parse_hybrid_mode(value, options.hybrid.mode)) {
            throw std::invalid_argument("--hybrid-mode expects off/gate/probe/priced, got: " + value);
        }
    }
    else if (option == "--hybrid") {
        // Preset: gate or off; the knobs keep their values.
        options.hybrid.mode = boolean() ? HybridMode::GATE : HybridMode::OFF;
    }
    else if (option == "--hybrid-event-lands") {
        if (!parse_hybrid_event_lands(value, options.hybrid.eventLands)) {
            throw std::invalid_argument("--hybrid-event-lands expects all/sit/none, got: " + value);
        }
    }
    else if (option == "--hybrid-max-wait-ms") options.hybrid.maxWaitMs = std::stoi(value);
    else if (option == "--hybrid-min-clock-ms") options.hybrid.minClockMs = std::stoi(value);
    else if (option == "--hybrid-margin-permille") options.hybrid.marginPermille = std::stoi(value);
    else if (option == "--hybrid-pass-prior-permille") options.hybrid.passPriorPermille = std::stoi(value);
    else if (option == "--continuity") {
        // Preset: the three wave 2c switches with the diagnosis fixes D1 and
        // D5; the knobs keep their values.
        const bool on = boolean();
        options.teamTreeReuse = on;
        options.jointPlans = on;
        options.planCarry = on;
        options.ownPaceBelowRoot = on;
        options.blockableCheckNotMate = on;
    }
    else return false;
    return true;
}

/**
 * @brief One bughouse game on four real seat clocks, event by event: the
 * game-side twin of the durative tree, without the board.
 *
 * Each seat on turn is either deciding or has a committed action landing
 * later. A decision is a search at the moment the seat starts thinking (it
 * comes on turn, or a pass of its own ends); a move then lands after its
 * cost (the seat's move time, jittered, floored at minMoveMs, plus lag), a
 * pass lasts until 1 ms after the next landing on the other board, the
 * seat's flag, or PassCapK x its move time. (A seat on the other board that
 * only re-decides is not an event: two seats sitting at once would otherwise
 * leapfrog each other 1 ms at a time.) Its board does not change meanwhile and
 * its pocket only grows, so a committed move stays legal; pieces that reach
 * the pocket during the decision are, as for an engine thinking under UCI,
 * only usable from its next decision. Clocks of both seats on turn run all
 * the time. Events at the same millisecond: landings before decisions, the
 * team alternating between the teams (starting with white), or with
 * MateRaceByLanding the opponent of the last capturer first.
 *
 * The caller owns the board: it asks next() what happens, runs the search for
 * a decision, reports each deciding seat's choice (commit_move/commit_pass),
 * and makes each landing's move before calling landed().
 */
class SeatClockGame {
public:
    enum class EventType { DECIDE, LAND, FLAG, DOUBLE_FLAG, SITOFF };

    struct Event {
        EventType type = EventType::DECIDE;
        int64_t timeMs = 0;
        // The seats deciding or landing now (one team's), or the seat that
        // flagged (FLAG).
        std::array<bool, 4> seats{false, false, false, false};
        Stockfish::Color team = Stockfish::WHITE;
        // FLAG, SITOFF: the losing team; -1 = draw.
        int loser = -1;
    };

    /// Wave 2b: a piece that landed in a seat's pocket (the partner captured it).
    struct Landing {
        int64_t timeMs = 0;
        int seat = 0;
        Stockfish::PieceType type = Stockfish::NO_PIECE_TYPE;
    };

    struct Start {
        int32_t baseMs = 0;
        int32_t whiteLeadMs = 0;
        std::array<int32_t, 4> clocksMs{};
        std::array<int32_t, 4> moveTimesMs{};
    };

    /**
     * Draws a game's start: the base from the mix, the white team's lead
     * (time odds, from @p clock's lead sampler: white seats +lead/2, black
     * seats -lead/2) and the move times, per team from @p clock's range, or
     * per seat with seatMoveTimes.
     */
    static Start sample_start(const Wave2GameOptions& options, const ClockSampling& clock,
                              std::mt19937_64& random) {
        Start start;
        std::vector<double> weights;
        for (const auto& entry : options.baseMix) {
            weights.push_back(entry.second);
        }
        std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
        start.baseMs = options.baseMix.at(pick(random)).first;
        const TimeState teams = sample_clock(clock, random);
        // A lead never takes a seat below 1 s.
        const int32_t limit = std::max(0, start.baseMs - 1000) * 2;
        start.whiteLeadMs = std::clamp(teams.lead_ms(Stockfish::WHITE), -limit, limit);
        const int32_t up = start.whiteLeadMs / 2;
        const int32_t down = start.whiteLeadMs - up;
        for (int seat = 0; seat < 4; ++seat) {
            const bool white = TimeState::team_of_seat(seat) == Stockfish::WHITE;
            start.clocksMs[seat] = white ? start.baseMs + up : start.baseMs - down;
            start.moveTimesMs[seat] = options.seatMoveTimes
                ? sample_log_uniform(clock.moveTimeMinMs, clock.moveTimeMaxMs, random)
                : teams.move_time_ms(TimeState::team_of_seat(seat));
        }
        return start;
    }

    /** @p sides: who is to move on A and B at the start; both decide at 0. */
    SeatClockGame(const Wave2GameOptions& options, const Start& start,
                  const std::array<Stockfish::Color, 2>& sides = {Stockfish::WHITE,
                                                                  Stockfish::WHITE})
        : options_(options), start_(start) {
        for (int seat = 0; seat < 4; ++seat) {
            clockMs_[seat] = start.clocksMs[seat];
            moveTimeMs_[seat] = std::max(1, start.moveTimesMs[seat]);
            paceMs_[seat] = moveTimeMs_[seat];
        }
        come_on_turn(TimeState::seat_of(0, sides[0]));
        come_on_turn(TimeState::seat_of(1, sides[1]));
    }

    /**
     * What happens next with @p sides to move on A and B. Advances every
     * clock on turn to that moment (a flag: to the flag).
     */
    Event next(const std::array<Stockfish::Color, 2>& sides) {
        const int seatA = TimeState::seat_of(0, sides[0]);
        const int seatB = TimeState::seat_of(1, sides[1]);
        Event event;
        int64_t when = std::min(eventAt_[seatA], eventAt_[seatB]);
        // Sit-off: both seats on turn sitting, across the teams.
        int64_t sitoffAt = std::numeric_limits<int64_t>::max();
        if (sitting_[seatA] && sitting_[seatB]
            && TimeState::team_of_seat(seatA) != TimeState::team_of_seat(seatB)) {
            sitoffAt = std::max(sittingSince_[seatA], sittingSince_[seatB]) + options_.sitoffMs;
        }
        if (options_.flagLoss) {
            const int64_t flagA = nowMs_ + clockMs_[seatA];
            const int64_t flagB = nowMs_ + clockMs_[seatB];
            const int64_t flagAt = std::min(flagA, flagB);
            if (flagAt <= std::min(when, sitoffAt)) {
                run_clocks(seatA, seatB, flagAt);
                event.timeMs = flagAt;
                const bool aOut = flagA == flagAt;
                const bool bOut = flagB == flagAt;
                if (aOut && bOut
                    && TimeState::team_of_seat(seatA) != TimeState::team_of_seat(seatB)) {
                    event.type = EventType::DOUBLE_FLAG;
                    event.seats[seatA] = event.seats[seatB] = true;
                    return event;
                }
                const int seat = aOut ? seatA : seatB;
                event.type = EventType::FLAG;
                event.seats[seat] = true;
                event.team = TimeState::team_of_seat(seat);
                event.loser = static_cast<int>(event.team);
                return event;
            }
        }
        if (sitoffAt <= when) {
            run_clocks(seatA, seatB, sitoffAt);
            event.type = EventType::SITOFF;
            event.timeMs = sitoffAt;
            event.seats[seatA] = event.seats[seatB] = true;
            if (clockMs_[seatA] != clockMs_[seatB]) {
                const int lower = clockMs_[seatA] < clockMs_[seatB] ? seatA : seatB;
                event.loser = static_cast<int>(TimeState::team_of_seat(lower));
            }
            return event;
        }
        run_clocks(seatA, seatB, when);
        event.timeMs = when;
        const bool aNow = eventAt_[seatA] == when;
        const bool bNow = eventAt_[seatB] == when;
        int chosen = aNow ? seatA : seatB;
        if (aNow && bNow && kind_[seatA] != kind_[seatB]) {
            chosen = kind_[seatA] == EventType::LAND ? seatA : seatB;
        } else if (aNow && bNow
                   && TimeState::team_of_seat(seatA) != TimeState::team_of_seat(seatB)) {
            const Stockfish::Color first = tie_team();
            chosen = TimeState::team_of_seat(seatA) == first ? seatA : seatB;
        }
        event.type = kind_[chosen];
        event.team = TimeState::team_of_seat(chosen);
        for (int seat : {seatA, seatB}) {
            if (eventAt_[seat] == when && kind_[seat] == kind_[chosen]
                && TimeState::team_of_seat(seat) == event.team) {
                event.seats[seat] = true;
            }
        }
        return event;
    }

    /** Draws what @p seat's next action will cost (call before its search). */
    int32_t plan_cost(int seat, std::mt19937_64& random) {
        double scaled = static_cast<double>(moveTimeMs_[seat]);
        if (options_.paceJitter > 0.0) {
            const double span = std::log1p(options_.paceJitter);
            scaled *= std::exp(std::uniform_real_distribution<double>(-span, span)(random));
        }
        int64_t cost = std::max<int64_t>(options_.minMoveMs, std::llround(scaled));
        if (options_.lagMaxMs > 0) {
            cost += std::uniform_int_distribution<int32_t>(0, options_.lagMaxMs)(random);
        }
        plannedMs_[seat] = static_cast<int32_t>(std::min<int64_t>(cost, TimeState::MAX_LEAD_MS));
        return plannedMs_[seat];
    }

    /**
     * @brief Spec v7 pace mode: each team plays at pace k (@p teamK, white
     * then black): a seat's move time is its clock / k, from the start
     * (replacing the drawn move times, also as the first observed pace).
     */
    void apply_pace(const std::array<double, 2>& teamK) {
        for (int seat = 0; seat < 4; ++seat) {
            const double k = teamK[static_cast<int>(TimeState::team_of_seat(seat))];
            const int32_t paced = paced_ms(clockMs_[seat], k, 1.0);
            moveTimeMs_[seat] = paced;
            paceMs_[seat] = paced;
            start_.moveTimesMs[seat] = paced;
        }
    }

    /**
     * Pace mode's plan_cost: draws the jitter and lag exactly as plan_cost
     * does, then costs @p seat's action at its clock now / @p k x @p scale
     * (see rescale_pace_cost).
     */
    int32_t plan_pace_cost(int seat, double k, double scale, std::mt19937_64& random) {
        double factor = 1.0;
        if (options_.paceJitter > 0.0) {
            const double span = std::log1p(options_.paceJitter);
            factor = std::exp(std::uniform_real_distribution<double>(-span, span)(random));
        }
        int32_t lag = 0;
        if (options_.lagMaxMs > 0) {
            lag = std::uniform_int_distribution<int32_t>(0, options_.lagMaxMs)(random);
        }
        jitter_[seat] = factor;
        lag_[seat] = lag;
        return rescale_pace_cost(seat, k, scale);
    }

    /**
     * Re-costs @p seat's planned action at think scale @p scale with the
     * draws plan_pace_cost made: max(minMoveMs, clock / k x scale x jitter)
     * + lag. The seat's move time becomes the unjittered clock / k x scale,
     * so its pass cap (PassCapK x move time) follows the pace.
     */
    int32_t rescale_pace_cost(int seat, double k, double scale) {
        const double paced = static_cast<double>(clockMs_[seat]) / k * scale;
        moveTimeMs_[seat] = paced_ms(clockMs_[seat], k, scale);
        int64_t cost = std::max<int64_t>(options_.minMoveMs, std::llround(paced * jitter_[seat]));
        cost += lag_[seat];
        plannedMs_[seat] = static_cast<int32_t>(std::min<int64_t>(cost, TimeState::MAX_LEAD_MS));
        return plannedMs_[seat];
    }

    /** The planned cost of @p seat's current decision. */
    int32_t planned_ms(int seat) const { return plannedMs_[seat]; }

    /** @p seat decided to move: it lands after its planned cost. */
    void commit_move(int seat) {
        lastDecisionMs_[seat] = nowMs_;
        kind_[seat] = EventType::LAND;
        eventAt_[seat] = nowMs_ + std::max(1, plannedMs_[seat]);
        sitting_[seat] = false;
        sittingSince_[seat] = -1;
    }

    /**
     * @brief Spec v6: the longest a hybrid sit of @p seat waits: the
     * arena's wake, min(PassCapK, 1.5) x this decision's budget (its planned
     * cost), and the expected event time plus 100 ms. Pass it to commit_pass.
     */
    int64_t hybrid_pass_cap_ms(int seat, int32_t eventMs) const {
        const int64_t budget = std::max(1, plannedMs_[seat]);
        return std::min({static_cast<int64_t>(options_.passCapK) * budget, budget * 3 / 2 + 1,
                         static_cast<int64_t>(eventMs) + 100});
    }

    /**
     * @p seat decided to pass (see the class comment for how long); @p capMs
     * >= 0 also ends it after that many ms (a hybrid sit, hybrid_pass_cap_ms).
     */
    void commit_pass(int seat, const std::array<Stockfish::Color, 2>& sides,
                     int64_t capMs = -1) {
        const int board = seat / 2;
        const int other = TimeState::seat_of(1 - board, sides[1 - board]);
        // Until the other board's next landing; a seat there that is still
        // to decide (or sitting) has none planned, and its landing, if it
        // moves, wakes this one (landed()).
        int64_t wait = std::numeric_limits<int64_t>::max();
        if (kind_[other] == EventType::LAND) {
            wait = std::max<int64_t>(eventAt_[other] - nowMs_, 0) + 1;
        }
        if (clockMs_[seat] > 0) {
            wait = std::min<int64_t>(wait, clockMs_[seat]);
        }
        wait = std::min<int64_t>(wait, static_cast<int64_t>(options_.passCapK) * moveTimeMs_[seat]);
        if (capMs >= 0) {
            wait = std::min<int64_t>(wait, capMs);
        }
        wait = std::max<int64_t>(wait, 1);
        kind_[seat] = EventType::DECIDE;
        eventAt_[seat] = nowMs_ + wait;
        decisionStartMs_[seat] = eventAt_[seat];
        if (!sitting_[seat]) {
            sittingSince_[seat] = nowMs_;
        }
        sitting_[seat] = true;
    }

    /**
     * @p seat's committed move has been made on the board (@p capture: it
     * captured; @p hand: the piece type that goes to the partner's pocket,
     * Board::hand_piece_type, for the wave 2b landing log). Its opponent on
     * that board comes on turn and decides now.
     */
    void landed(int seat, bool capture,
                Stockfish::PieceType hand = Stockfish::NO_PIECE_TYPE) {
        const double observed = static_cast<double>(nowMs_ - lastDecisionMs_[seat]);
        paceMs_[seat] = options_.paceEmaAlpha * observed
            + (1.0 - options_.paceEmaAlpha) * paceMs_[seat];
        kind_[seat] = EventType::DECIDE;
        eventAt_[seat] = std::numeric_limits<int64_t>::max();
        if (capture) {
            lastCapture_ = static_cast<int>(TimeState::team_of_seat(seat));
        }
        const int board = seat / 2;
        lastLandMs_[board] = nowMs_;
        if (hand != Stockfish::NO_PIECE_TYPE) {
            const int partner = TimeState::seat_of(1 - board, ~static_cast<Stockfish::Color>(seat & 1));
            landings_.push_back(Landing{nowMs_, partner, hand});
        }
        come_on_turn(TimeState::seat_of(board, ~static_cast<Stockfish::Color>(seat & 1)));
        // A pass on the other board ends 1 ms after this event.
        for (int waiting : {TimeState::seat_of(1 - board, Stockfish::WHITE),
                            TimeState::seat_of(1 - board, Stockfish::BLACK)}) {
            if (sitting_[waiting] && kind_[waiting] == EventType::DECIDE
                && eventAt_[waiting] > nowMs_ + 1) {
                eventAt_[waiting] = nowMs_ + 1;
                decisionStartMs_[waiting] = eventAt_[waiting];
            }
        }
    }

    /**
     * The clock for @p team's search now, with @p deciding its seats that
     * decide (their planned costs are the root budget): the durative tree
     * (DurativeTree) on the seat clocks, observed paces, the other team's
     * elapsed thinking and sitting; else v3's clocked state at the lead the
     * clocks give, with the teams' mean paces. SitGate as the options say.
     */
    TimeState search_clock(Stockfish::Color team, const std::array<Stockfish::Color, 2>& sides,
                           const std::array<bool, 4>& deciding) const {
        return options_.durativeTree ? durative_clock(team, sides, deciding, options_.sitGate)
                                     : v3_clock(team, sides, deciding);
    }

    /**
     * Spec v6: the durative clock of @p team's decision for the hybrid's
     * trigger (SitGate off), beside the v3 clock the search runs on.
     */
    TimeState hybrid_clock(Stockfish::Color team, const std::array<Stockfish::Color, 2>& sides,
                           const std::array<bool, 4>& deciding) const {
        return durative_clock(team, sides, deciding, false);
    }

    /** search_clock's v3 state: the clocked lead the seat clocks give, the teams' mean paces. */
    TimeState v3_clock(Stockfish::Color team, const std::array<Stockfish::Color, 2>& sides,
                       const std::array<bool, 4>& deciding) const {
        (void)sides;
        return v3_base(team, deciding).with_sit_gate(options_.sitGate);
    }

    /** search_clock's durative state, with the sit gate @p sitGate. */
    TimeState durative_clock(Stockfish::Color team, const std::array<Stockfish::Color, 2>& sides,
                             const std::array<bool, 4>& deciding, bool sitGate) const {
        const TimeState base = v3_base(team, deciding).with_sit_gate(sitGate);
        return TimeState::durative(base, team, sides[0], sides[1],
                                   durative_seats(team, sides, decision_budget(deciding)));
    }

    /** v3's clocked state (no gate set) at the lead the seat clocks give, the teams' mean paces. */
    TimeState v3_base(Stockfish::Color team, const std::array<bool, 4>& deciding) const {
        std::array<int64_t, 2> paceSum{0, 0};
        for (int seat = 0; seat < 4; ++seat) {
            paceSum[TimeState::team_of_seat(seat)] += pace_ms(seat);
        }
        const int32_t teamPace = static_cast<int32_t>(paceSum[team] / 2);
        const int32_t otherPace = static_cast<int32_t>(paceSum[~team] / 2);
        const int32_t lead = white_lead_ms() * (team == Stockfish::WHITE ? 1 : -1);
        return TimeState::clocked(team, lead, teamPace, otherPace, decision_budget(deciding));
    }

    /** The search's budget: the largest planned cost of the seats deciding now. */
    int32_t decision_budget(const std::array<bool, 4>& deciding) const {
        int32_t budget = 0;
        for (int seat = 0; seat < 4; ++seat) {
            if (deciding[seat]) {
                budget = std::max(budget, plannedMs_[seat]);
            }
        }
        return budget;
    }

    /** The durative clock's options for @p team now, on top of decision_budget's root budget. */
    DurativeConfig durative_seats(Stockfish::Color team, const std::array<Stockfish::Color, 2>& sides,
                                  int32_t budget) const {
        DurativeConfig config = options_.durative_config();
        config.rootMoveTimeMs = std::max(1, budget);
        for (int seat = 0; seat < 4; ++seat) {
            // A clock at 0 is still a clock: keep it apart from "not given".
            config.seatClocksMs[seat] = std::max(clockMs_[seat], 1);
            config.seatPaceMs[seat] = pace_ms(seat);
        }
        for (int board = 0; board < 2; ++board) {
            const int seat = TimeState::seat_of(board, sides[board]);
            if (TimeState::team_of_seat(seat) != team) {
                config.seatElapsedMs[seat] = elapsed_ms(seat);
                config.seatSitting[seat] = sitting_[seat];
            }
        }
        if (wideInputs_ || !options_.instantTransfer) {
            // Wave 2b: a wide network also reads each board's thinking time
            // and the pockets' recent landings. Without InstantTransfer the
            // landings also tell the tree what reached an opponent seat's
            // pocket during its current decision (TimeState::durative).
            config.wideInputs = wideInputs_;
            for (int board = 0; board < 2; ++board) {
                config.boardElapsedMs[board] = board_elapsed_ms(board);
            }
            int slot = 0;
            for (auto it = landings_.rbegin(); it != landings_.rend() && slot < MAX_RECENT_LANDINGS; ++it) {
                const int64_t ago = nowMs_ - it->timeMs;
                if (ago >= RECENT_LANDING_WINDOW_MS) {
                    break;
                }
                config.recentLandings[slot++] = SeatLanding{
                    static_cast<int8_t>(it->seat), static_cast<int8_t>(it->type),
                    static_cast<int32_t>(ago)};
            }
        }
        return config;
    }

    /** Wave 2b: searches carry the wide-plane timing (a wide network). */
    void set_wide_inputs(bool wide) { wideInputs_ = wide; }
    /** ms since the last move landed on @p board (since the start before any). */
    int32_t board_elapsed_ms(int board) const {
        return static_cast<int32_t>(std::min<int64_t>(nowMs_ - lastLandMs_[board],
                                                      TimeState::MAX_LEAD_MS));
    }
    /** Every pocket landing so far, in time order. */
    const std::vector<Landing>& landings() const { return landings_; }

    int64_t now_ms() const { return nowMs_; }
    int32_t clock_ms(int seat) const { return clockMs_[seat]; }
    std::array<int32_t, 4> clocks_ms() const { return clockMs_; }
    int32_t move_time_ms(int seat) const { return moveTimeMs_[seat]; }
    int32_t pace_ms(int seat) const {
        return std::max(1, static_cast<int32_t>(std::lround(paceMs_[seat])));
    }
    std::array<int32_t, 4> paces_ms() const {
        return {pace_ms(0), pace_ms(1), pace_ms(2), pace_ms(3)};
    }
    /** How long @p seat (on turn) has been on its current decision. */
    int32_t elapsed_ms(int seat) const {
        return static_cast<int32_t>(std::max<int64_t>(0, nowMs_ - decisionStartMs_[seat]));
    }
    bool sitting(int seat) const { return sitting_[seat]; }
    int64_t event_at_ms(int seat) const { return eventAt_[seat]; }
    /** Whether @p seat (on turn) has a committed move that has not landed yet. */
    bool in_flight(int seat) const { return kind_[seat] == EventType::LAND; }
    const Start& start() const { return start_; }
    /** The white team's lead, as v3 reads it off four clocks. */
    int32_t white_lead_ms() const {
        const int64_t lead = (static_cast<int64_t>(clockMs_[TimeState::SEAT_AW]) - clockMs_[TimeState::SEAT_BW]
            + static_cast<int64_t>(clockMs_[TimeState::SEAT_BB]) - clockMs_[TimeState::SEAT_AB]) / 2;
        return static_cast<int32_t>(lead);
    }

private:
    static int32_t paced_ms(int32_t clockMs, double k, double scale) {
        const double paced = static_cast<double>(clockMs) / k * scale;
        return static_cast<int32_t>(std::clamp<double>(std::llround(paced), 1.0,
                                                       static_cast<double>(TimeState::MAX_LEAD_MS)));
    }

    void come_on_turn(int seat) {
        kind_[seat] = EventType::DECIDE;
        eventAt_[seat] = nowMs_;
        decisionStartMs_[seat] = nowMs_;
        sitting_[seat] = false;
        sittingSince_[seat] = -1;
    }

    void run_clocks(int seatA, int seatB, int64_t until) {
        const int64_t dt = std::max<int64_t>(0, until - nowMs_);
        for (int seat : {seatA, seatB}) {
            clockMs_[seat] = static_cast<int32_t>(std::max<int64_t>(0, clockMs_[seat] - dt));
        }
        nowMs_ = until;
    }

    Stockfish::Color tie_team() {
        if (options_.mateRaceByLanding && lastCapture_ >= 0) {
            return ~static_cast<Stockfish::Color>(lastCapture_);
        }
        const Stockfish::Color team = tieNext_;
        tieNext_ = ~tieNext_;
        return team;
    }

    Wave2GameOptions options_;
    Start start_;
    int64_t nowMs_ = 0;
    std::array<int32_t, 4> clockMs_{};
    std::array<int32_t, 4> moveTimeMs_{};
    std::array<double, 4> paceMs_{};
    std::array<int32_t, 4> plannedMs_{1, 1, 1, 1};
    std::array<EventType, 4> kind_{EventType::DECIDE, EventType::DECIDE, EventType::DECIDE,
                                   EventType::DECIDE};
    std::array<int64_t, 4> eventAt_{std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max()};
    std::array<int64_t, 4> decisionStartMs_{};
    std::array<int64_t, 4> lastDecisionMs_{};
    std::array<bool, 4> sitting_{false, false, false, false};
    std::array<int64_t, 4> sittingSince_{-1, -1, -1, -1};
    Stockfish::Color tieNext_ = Stockfish::WHITE;
    int lastCapture_ = -1;
    bool wideInputs_ = false;
    // Pace mode: the jitter factor and lag plan_pace_cost drew per seat.
    std::array<double, 4> jitter_{1.0, 1.0, 1.0, 1.0};
    std::array<int32_t, 4> lag_{0, 0, 0, 0};
    std::array<int64_t, 2> lastLandMs_{0, 0};
    std::vector<Landing> landings_;
};

/**
 * @brief Spec v6: what the hybrid did in one seat-clock game (self-play or
 * tournament): searches, those with a non-empty trigger mask, hybrid sits
 * the root picked, kept (returned) or refused by the margin or visit rule,
 * and played. `lastTwo[team]`: whether each of that team's last two
 * decisions played a hybrid sit (newest last), to tell a sit-off or flag
 * that followed a hybrid sit.
 */
struct HybridGameStats {
    uint64_t searches = 0;
    uint64_t triggered = 0;
    uint64_t sitsChosen = 0;
    uint64_t kept = 0;
    uint64_t refusedMargin = 0;
    uint64_t refusedVisits = 0;
    uint64_t played = 0;
    std::array<std::array<bool, 2>, 2> lastTwo{};

    /** One search of @p team with the hybrid on; @p playedSit: the game played its hybrid sit. */
    void record(const HybridReport& report, Stockfish::Color team, bool playedSit) {
        if (!report.active) {
            return;
        }
        ++searches;
        triggered += report.trigger.mask != 0 ? 1 : 0;
        sitsChosen += report.sitChosen ? 1 : 0;
        kept += report.kept ? 1 : 0;
        refusedMargin += report.sitChosen && !report.kept && report.reason == "margin" ? 1 : 0;
        refusedVisits += report.sitChosen && !report.kept && report.reason == "visits" ? 1 : 0;
        played += playedSit ? 1 : 0;
        std::array<bool, 2>& recent = lastTwo[static_cast<int>(team)];
        recent = {recent[1], playedSit};
    }
    /** How many of @p team's last two decisions played a hybrid sit. */
    int recent_sits(Stockfish::Color team) const {
        const std::array<bool, 2>& recent = lastTwo[static_cast<int>(team)];
        return (recent[0] ? 1 : 0) + (recent[1] ? 1 : 0);
    }
    void add(const HybridGameStats& other) {
        searches += other.searches;
        triggered += other.triggered;
        sitsChosen += other.sitsChosen;
        kept += other.kept;
        refusedMargin += other.refusedMargin;
        refusedVisits += other.refusedVisits;
        played += other.played;
    }
    /** "<sits> sits in <triggered> triggered of <searches> searches, kept <k> / refused margin <m> visits <v>" */
    std::string summary() const {
        std::ostringstream text;
        text << played << " sits in " << triggered << " triggered of " << searches
             << " searches, kept " << kept << " / refused margin " << refusedMargin
             << " visits " << refusedVisits;
        return text.str();
    }
};

/**
 * @brief Wave 2b: counts of pieces landing in each pocket, in the arrivals
 * head's order from @p team's view: ours on A, theirs on A, ours on B,
 * theirs on B, each P N B R Q. A landing at time t counts when
 * fromMs < t <= toMs. With @p maxLandings only the newest that many count:
 * the recent-landing planes pass MAX_RECENT_LANDINGS, as many as the
 * engine's search keeps (search_clock, TimeState), so a burst of landings
 * reads the same in training as in the search; the arrivals labels count all.
 */
inline std::array<int, 20> pocket_landing_counts(const std::vector<SeatClockGame::Landing>& landings,
                                                 Stockfish::Color team, int64_t fromMs,
                                                 int64_t toMs,
                                                 int maxLandings = std::numeric_limits<int>::max()) {
    const int pockets[4] = {TimeState::seat_of(0, team), TimeState::seat_of(0, ~team),
                            TimeState::seat_of(1, ~team), TimeState::seat_of(1, team)};
    std::array<int, 20> counts{};
    int kept = 0;
    for (auto it = landings.rbegin(); it != landings.rend() && kept < maxLandings; ++it) {
        const SeatClockGame::Landing& landing = *it;
        const int type = static_cast<int>(landing.type);
        if (landing.timeMs <= fromMs || landing.timeMs > toMs || type < 1 || type > 5) {
            continue;
        }
        ++kept;
        for (int pocket = 0; pocket < 4; ++pocket) {
            if (pockets[pocket] == landing.seat) {
                ++counts[pocket * 5 + type - 1];
            }
        }
    }
    return counts;
}
