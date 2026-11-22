#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/time_state.h"

/**
 * @brief Spec v6, the hybrid (docs/custom_engine_spec_v6_hybrid.md in the
 * talisman repo): the v3 joint-action tree chooses the move, the durative
 * clock decides only when to act. Off by default; every mode needs the v3
 * tree (DurativeTree false) and TimeModel clock.
 *
 * H3, HybridMode gate (built): when the trigger below fires, the root may
 * pass on the board it is on turn on although the v3 gate (lead > 0)
 * forbids it, the opponent seat thinking on the other board lands a move at
 * depth 1, and the root keeps the pass only if it beats the best moving
 * action by HybridMarginPermille.
 */
enum class HybridMode : uint8_t { OFF, GATE, PROBE, PRICED };

/** UCI HybridEventLands: below which root children the event seat must move. */
enum class HybridEventLands : uint8_t { ALL, SIT, NONE };

inline const char* hybrid_mode_name(HybridMode mode) {
    switch (mode) {
    case HybridMode::OFF: return "off";
    case HybridMode::GATE: return "gate";
    case HybridMode::PROBE: return "probe";
    case HybridMode::PRICED: return "priced";
    }
    return "off";
}

inline const char* hybrid_event_lands_name(HybridEventLands lands) {
    switch (lands) {
    case HybridEventLands::ALL: return "all";
    case HybridEventLands::SIT: return "sit";
    case HybridEventLands::NONE: return "none";
    }
    return "all";
}

/** "off" / "gate" / ...; false for anything else. */
inline bool parse_hybrid_mode(const std::string& text, HybridMode& mode) {
    for (HybridMode candidate : {HybridMode::OFF, HybridMode::GATE, HybridMode::PROBE,
                                 HybridMode::PRICED}) {
        if (text == hybrid_mode_name(candidate)) {
            mode = candidate;
            return true;
        }
    }
    return false;
}

inline bool parse_hybrid_event_lands(const std::string& text, HybridEventLands& lands) {
    for (HybridEventLands candidate : {HybridEventLands::ALL, HybridEventLands::SIT,
                                       HybridEventLands::NONE}) {
        if (text == hybrid_event_lands_name(candidate)) {
            lands = candidate;
            return true;
        }
    }
    return false;
}

/**
 * @brief The hybrid's mode and knobs (UCI HybridMode, HybridEventLands,
 * HybridMaxWaitMs, HybridMinClockMs, HybridMarginPermille,
 * HybridPassPriorPermille). Every default is the spec's.
 */
struct HybridKnobs {
    HybridMode mode = HybridMode::OFF;
    HybridEventLands eventLands = HybridEventLands::ALL;
    // T4: the longest expected wait for an event.
    int32_t maxWaitMs = 1000;
    // T6: our seat's clock floor.
    int32_t minClockMs = 3000;
    // How much better (q x 1000) a hybrid sit must be than the best move.
    int32_t marginPermille = 20;
    // The pass's root prior floor on a masked board.
    int32_t passPriorPermille = 150;

    bool operator==(const HybridKnobs&) const = default;
    /** Whether a knob differs from the spec's default (the mode aside). */
    bool knobs_changed() const {
        HybridKnobs defaults;
        defaults.mode = mode;
        return !(*this == defaults);
    }
};

/**
 * @brief What the trigger found for one search: the boards the root may sit
 * on (at most one bit), the expected wait for the event on the other board,
 * and, with an empty mask, the first condition that failed.
 */
struct HybridTrigger {
    uint8_t mask = 0;
    // The board the root may sit on (X) and the board of the event (Y), -1 = none.
    int sitBoard = -1;
    int eventBoard = -1;
    // Expected ms until the thinking opponent seat on Y lands (T3).
    int32_t eventMs = 0;
    // Why the mask is empty: "no move", "both boards", "required",
    // "gate allows", "opponent sitting", "wait", "sit-off", "clock"; "" when
    // the mask is set.
    const char* reason = "";
};

/**
 * @brief Spec v6, "The trigger": whether waiting is on the table at this
 * root. A board X gets its bit when all hold:
 *  T1 our @p team is on turn on X and not on the other board Y (an opponent
 *     seat is on turn there), X has a legal move, and RequireMoveOn does not
 *     name X;
 *  T2 the v3 rules forbid the pass on X: !v3.can_sit(team) (lead <= 0);
 *  T3 the opponent seat on Y is thinking, not sitting; eventMs is its ready
 *     time in @p durative, max(1, pace - elapsed);
 *  T4 eventMs <= min(maxWaitMs, 1.5 x RootMoveTimeMs, PassCapK x
 *     RootMoveTimeMs): a short wait that ends inside the arena's wake;
 *  T5 the durative sit-off rule does not force us to move;
 *  T6 our seat on X has clock >= max(minClockMs, 4 x (eventMs +
 *     RootMoveTimeMs)).
 * @p durative is the durative state of this decision (TimeState::durative
 * with SitGate off), @p v3 the search's own clock.
 */
inline HybridTrigger hybrid_sit_boards(Board& board, Stockfish::Color team,
                                       const TimeState& v3, const TimeState& durative,
                                       const HybridKnobs& knobs) {
    HybridTrigger trigger;
    const bool onTurnA = board.side_to_move(BOARD_A) == team;
    const bool onTurnB = board.side_to_move(BOARD_B) == ~team;
    if (onTurnA && onTurnB) {
        trigger.reason = "both boards";
        return trigger;
    }
    if (!onTurnA && !onTurnB) {
        trigger.reason = "no move";
        return trigger;
    }
    const int sitBoard = onTurnA ? BOARD_A : BOARD_B;
    const int eventBoard = 1 - sitBoard;
    if (!board.has_any_legal_move(sitBoard)) {
        trigger.reason = "no move";
        return trigger;
    }
    if ((sitBoard == BOARD_A && g_requiredMoveBoard == REQUIRE_MOVE_BOARD_A)
        || (sitBoard == BOARD_B && g_requiredMoveBoard == REQUIRE_MOVE_BOARD_B)) {
        trigger.reason = "required";
        return trigger;
    }
    if (v3.can_sit(team)) {
        trigger.reason = "gate allows";
        return trigger;
    }
    const int ourSeat = TimeState::seat_of(sitBoard, board.side_to_move(sitBoard));
    const int eventSeat = TimeState::seat_of(eventBoard, board.side_to_move(eventBoard));
    if (durative.seat_sitting(eventSeat)) {
        trigger.reason = "opponent sitting";
        return trigger;
    }
    const int32_t eventMs = std::max<int32_t>(1, durative.seat_ready_ms(eventSeat));
    const int64_t rootMs = std::max<int32_t>(1, durative.action_time_ms(ourSeat, true));
    const int64_t waitLimit = std::min<int64_t>(
        {static_cast<int64_t>(knobs.maxWaitMs), rootMs * 3 / 2,
         static_cast<int64_t>(durative.pass_cap_k()) * rootMs});
    if (eventMs > waitLimit) {
        trigger.reason = "wait";
        return trigger;
    }
    if (!durative.with_sit_gate(false).can_sit(team)) {
        trigger.reason = "sit-off";
        return trigger;
    }
    const int64_t clockFloor = std::max<int64_t>(
        knobs.minClockMs, 4 * (static_cast<int64_t>(eventMs) + rootMs));
    if (durative.seat_clock_ms(ourSeat) < clockFloor) {
        trigger.reason = "clock";
        return trigger;
    }
    trigger.mask = static_cast<uint8_t>(1u << sitBoard);
    trigger.sitBoard = sitBoard;
    trigger.eventBoard = eventBoard;
    trigger.eventMs = eventMs;
    return trigger;
}

/**
 * @brief What the hybrid did in the last search (Agent::hybrid_report):
 * the trigger, and at the root's choice whether a hybrid sit was chosen,
 * kept or refused and why. `reason`: "margin" (kept by it, or refused),
 * "visits" (refused: under 10% of the root's visits), "proven" (a solved
 * win through the sit, kept; the proof assumes the event lands), "no sit
 * chosen" (the root's pick moved anyway).
 */
struct HybridReport {
    bool active = false;           // the mode was on and the search eligible
    HybridTrigger trigger;
    bool sitChosen = false;        // the root's own pick was the hybrid sit
    bool qualifies = false;        // the sit passed the margin rule (may be played)
    bool kept = false;             // the search returned the hybrid sit
    bool proven = false;
    float qSit = 0.0f;
    float qAct = 0.0f;
    int visitsSit = 0;
    int visitsTotal = 0;
    std::string reason;
};
