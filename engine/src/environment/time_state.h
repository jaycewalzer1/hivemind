#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <optional>

#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Wave 2 options for a clocked TimeState (UCI DurativeTree, FlagLoss,
 * InstantTransfer, MateRaceByLanding, PassCapK, SeatClocksMs, SeatPaceMs,
 * SeatElapsedMs, RootMoveTimeMs). Seats are indexed Aw, Ab, Bw, Bb.
 * Every default is stock v3.0.3; see TimeState::durative().
 */
struct DurativeConfig {
    bool flagLoss = false;
    // Read by part 2 (delayed transfer, landing-time mate races); carried in
    // the state so every node sees the search's choice.
    bool instantTransfer = true;
    bool mateRaceByLanding = false;
    int32_t passCapK = 8;
    // This search's budget: what every action of the root team costs.
    // 0 = the root team's usual move time.
    int32_t rootMoveTimeMs = 0;
    // Remaining clock per seat; all 0 = derive from the lead as v3 does.
    std::array<int32_t, 4> seatClocksMs{0, 0, 0, 0};
    // Time per action per seat; 0 = its team's v3 move time.
    std::array<int32_t, 4> seatPaceMs{0, 0, 0, 0};
    // How long each seat on turn has already been thinking.
    std::array<int32_t, 4> seatElapsedMs{0, 0, 0, 0};
};

/**
 * @brief The clock situation of a bughouse match, as the search needs it.
 *
 * Without increment each board always has exactly one clock running, so the
 * two boards' clock sums stay equal and a team's lead over its diagonal
 * opponents is one number: Aw - Bw = Bb - Ab for the white team, and the black
 * team's lead is its negation. Absolute clock values never enter.
 *
 * The lead moves only while one team is on turn on both boards. Both of that
 * team's clocks run and neither of the other team's, so the team pays its own
 * move time for the action it is deciding. In every other configuration one
 * clock of each team runs and the lead holds. A team may sit while its lead is
 * positive; at a lead of exactly zero neither team may.
 *
 * Legacy mode reproduces the TimeAdvantage flag: one team's lead is unbounded,
 * never moves, and the other team never has one.
 */
class TimeState {
public:
    /// Chess.com's fastest possible move, the default for both teams.
    static constexpr int32_t DEFAULT_MOVE_TIME_MS = 100;
    /// Bound on a clocked lead, far beyond any real clock.
    static constexpr int32_t MAX_LEAD_MS = 1 << 30;
    /// Base clock for seats derived from a lead alone: far beyond any flag.
    static constexpr int32_t DERIVED_CLOCK_MS = 1 << 29;

    /// Seat indices: board A white/black, board B white/black.
    enum Seat : int { SEAT_AW = 0, SEAT_AB = 1, SEAT_BW = 2, SEAT_BB = 3 };
    /// The seat on turn on @p board when @p side is to move there.
    static int seat_of(int board, Stockfish::Color side) {
        return board * 2 + static_cast<int>(side);
    }
    /// The team a seat plays for: Aw and Bb are the white team.
    static Stockfish::Color team_of_seat(int seat) {
        const Stockfish::Color color = static_cast<Stockfish::Color>(seat & 1);
        return seat < 2 ? color : ~color;
    }

    /// What a durative state answers about the game at its node.
    enum class Outcome : uint8_t { NONE, WHITE_TEAM_WINS, BLACK_TEAM_WINS, DRAW };

    /** A permanent flag: @p team may always sit if it has the advantage. */
    static TimeState legacy(Stockfish::Color team, bool teamHasTimeAdvantage) {
        TimeState state;
        const bool whiteAhead = (team == Stockfish::WHITE) == teamHasTimeAdvantage;
        state.whiteLeadMs_ = whiteAhead ? MAX_LEAD_MS : -MAX_LEAD_MS;
        return state;
    }

    /**
     * @param team The team the lead and the first move time belong to.
     * @param teamLeadMs Its lead over its diagonal opponents; negative when behind.
     * @param teamMoveTimeMs How long @p team takes for one action.
     * @param opponentMoveTimeMs How long the other team takes for one action.
     * @param teamNextMoveTimeMs What @p team's next action alone costs, when it
     *        is known to differ from its usual move time (0 = the usual): a
     *        front end that has budgeted the move it is about to search.
     */
    static TimeState clocked(Stockfish::Color team,
                             int64_t teamLeadMs,
                             int32_t teamMoveTimeMs = DEFAULT_MOVE_TIME_MS,
                             int32_t opponentMoveTimeMs = DEFAULT_MOVE_TIME_MS,
                             int32_t teamNextMoveTimeMs = 0) {
        TimeState state;
        state.clocked_ = true;
        const int64_t whiteLead = team == Stockfish::WHITE ? teamLeadMs : -teamLeadMs;
        state.whiteLeadMs_ = clamp_lead(whiteLead);
        state.moveTimeMs_[team] = std::max<int32_t>(1, teamMoveTimeMs);
        state.moveTimeMs_[~team] = std::max<int32_t>(1, opponentMoveTimeMs);
        if (teamNextMoveTimeMs > 0) {
            state.nextMoveTeam_ = team;
            state.nextMoveTimeMs_ = teamNextMoveTimeMs;
        }
        return state;
    }

    /**
     * @brief The wave 2 durative clock (UCI DurativeTree): per-seat clocks and
     * ready times instead of alternating whole team turns.
     *
     * Each seat on turn carries a ready time, when its next action lands. A
     * node advances simulated time to the earliest ready time and only the
     * team owning that seat acts, on that board (on both when both of its
     * seats are ready at once). Exact ties between teams alternate, starting
     * with the opponents of @p rootTeam. The root is the exception: the root
     * team decides there, on every board it is on turn, at its own budget.
     * Every clock on turn runs by the time advanced; the lead (and so the
     * clock planes) is read off the four clocks.
     *
     * @param base A clocked state (TimeState::clocked) whose lead and move
     *        times stand in for anything @p config leaves empty.
     * @param sideA, sideB Who is to move on each board at the root.
     */
    static TimeState durative(const TimeState& base, Stockfish::Color rootTeam,
                              Stockfish::Color sideA, Stockfish::Color sideB,
                              const DurativeConfig& config) {
        TimeState state = base.without_next_move_time();
        state.clocked_ = true;
        Durative& d = state.durative_;
        d.active = true;
        d.atRoot = true;
        d.flagLoss = config.flagLoss;
        d.instantTransfer = config.instantTransfer;
        d.mateRaceByLanding = config.mateRaceByLanding;
        d.rootTeam = rootTeam;
        d.tieNext = ~rootTeam;
        d.side = {sideA, sideB};
        const bool clocksGiven = std::any_of(
            config.seatClocksMs.begin(), config.seatClocksMs.end(),
            [](int32_t ms) { return ms != 0; });
        if (clocksGiven) {
            d.clockMs = config.seatClocksMs;
        } else {
            // Aw - Bw = Bb - Ab = the white team's lead, around a base far
            // beyond any flag, as v3's single lead implies.
            const int32_t lead = base.whiteLeadMs_;
            const int32_t low = lead / 2;
            const int32_t high = lead - low;
            d.clockMs = {DERIVED_CLOCK_MS + high, DERIVED_CLOCK_MS - low,
                         DERIVED_CLOCK_MS - low, DERIVED_CLOCK_MS + high};
        }
        for (int seat = 0; seat < 4; ++seat) {
            const int32_t pace = config.seatPaceMs[seat];
            d.paceMs[seat] = pace > 0
                ? pace : base.moveTimeMs_[team_of_seat(seat)];
        }
        if (config.seatPaceMs[SEAT_AW] > 0 || config.seatPaceMs[SEAT_BB] > 0
            || config.seatPaceMs[SEAT_AB] > 0 || config.seatPaceMs[SEAT_BW] > 0) {
            // The speed plane compares the teams' mean paces.
            for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
                int64_t sum = 0;
                for (int seat = 0; seat < 4; ++seat) {
                    if (team_of_seat(seat) == team) {
                        sum += d.paceMs[seat];
                    }
                }
                state.moveTimeMs_[team] = std::max<int32_t>(1, static_cast<int32_t>(sum / 2));
            }
        }
        const int32_t rootMs = config.rootMoveTimeMs > 0
            ? config.rootMoveTimeMs : base.moveTimeMs_[rootTeam];
        d.rootMoveMs = std::max<int32_t>(1, rootMs);
        d.passCapMs = static_cast<int32_t>(std::clamp<int64_t>(
            static_cast<int64_t>(std::max<int32_t>(1, config.passCapK)) * d.rootMoveMs,
            1, MAX_LEAD_MS));
        for (int board = 0; board < 2; ++board) {
            const int seat = seat_of(board, d.side[board]);
            if (team_of_seat(seat) == rootTeam) {
                d.readyMs[seat] = d.rootMoveMs;
            } else {
                d.readyMs[seat] = std::max<int32_t>(
                    1, d.paceMs[seat] - std::max<int32_t>(0, config.seatElapsedMs[seat]));
            }
        }
        state.sync_lead();
        return state;
    }

    bool is_clocked() const { return clocked_; }
    bool is_durative() const { return durative_.active; }

    /**
     * @brief This state with upstream's hard sit gate (can_sit = lead > 0)
     * switched on or off (UCI SitGate). Off, a pass is always legal and the
     * tree prices it; a durative state still applies the sit-off rule. Only
     * a clocked state takes it: a legacy flag is a gate by definition.
     */
    TimeState with_sit_gate(bool gate) const {
        TimeState next = *this;
        next.sitGate_ = gate || !clocked_;
        return next;
    }
    bool sit_gate() const { return sitGate_; }

    /** Upstream's sit right, lead > 0, whatever the gate: the network's sit plane. */
    bool lead_sit_right(Stockfish::Color team) const {
        return lead_ms(team) > 0;
    }

    int32_t lead_ms(Stockfish::Color team) const {
        return team == Stockfish::WHITE ? whiteLeadMs_ : -whiteLeadMs_;
    }

    /** How long @p team usually takes for one action. */
    int32_t move_time_ms(Stockfish::Color team) const {
        return moveTimeMs_[team];
    }

    /** This state with no next-move time: every action at its team's usual time. */
    TimeState without_next_move_time() const {
        TimeState plain = *this;
        plain.nextMoveTeam_ = Stockfish::WHITE;
        plain.nextMoveTimeMs_ = 0;
        return plain;
    }

    /** What @p team's next action costs when it is charged. */
    int32_t next_move_time_ms(Stockfish::Color team) const {
        return nextMoveTimeMs_ > 0 && team == nextMoveTeam_
            ? nextMoveTimeMs_ : moveTimeMs_[team];
    }

    /** Whether @p team may sit: wait on a board instead of moving there. */
    bool can_sit(Stockfish::Color team) const {
        if (sitGate_) {
            return lead_ms(team) > 0;
        }
        return !durative_.active || !sit_off_forces_move(team);
    }

    /**
     * @brief Whether @p team can pay for @p actions more actions on turn on
     * both boards and still not be behind.
     *
     * Proofs that let one team sit a board for a whole line rest on this:
     * while it holds, that team may sit before each of those actions and the
     * other team never gains the right to. A legacy state answers can_sit().
     */
    bool can_sit_through(Stockfish::Color team, int actions) const {
        if (!clocked_ || !sitGate_) {
            return can_sit(team);
        }
        return can_sit(team)
            && static_cast<int64_t>(lead_ms(team))
                   >= next_move_time_ms(team)
                       + static_cast<int64_t>(std::max(actions, 1) - 1) * moveTimeMs_[team];
    }

    /**
     * @brief Charge the action @p actingTeam is about to play.
     *
     * Call it with the position before the action. Only a team on turn on both
     * boards pays; legacy leads never move. A next-move time set for the
     * acting team is used up by this action, whether or not it pays.
     */
    void advance(Stockfish::Color actingTeam, bool actingTeamOnBothBoards) {
        if (durative_.active) {
            // A caller that does not say which boards moved (the mate
            // provers, tree reuse) reasons in v3 alternation: from here on
            // this is v3's clocked state at the current lead.
            durative_ = Durative{};
        }
        const int64_t cost = next_move_time_ms(actingTeam);
        if (actingTeam == nextMoveTeam_) {
            // Back to the plain clock, owner included, so equal states compare equal.
            nextMoveTeam_ = Stockfish::WHITE;
            nextMoveTimeMs_ = 0;
        }
        if (!clocked_ || !actingTeamOnBothBoards) {
            return;
        }
        whiteLeadMs_ = clamp_lead(
            static_cast<int64_t>(whiteLeadMs_)
            + (actingTeam == Stockfish::WHITE ? -cost : cost));
    }

    /** The state after @p actingTeam plays one action; see advance(). */
    TimeState after(Stockfish::Color actingTeam, bool actingTeamOnBothBoards) const {
        TimeState next = *this;
        next.advance(actingTeam, actingTeamOnBothBoards);
        return next;
    }

    /**
     * @brief The boards @p team acts on at this node of a durative tree.
     *
     * Neither, when the other team's seat lands first: @p team then only
     * waits, and the tree gives its node the single action (pass, pass).
     */
    std::array<bool, 2> acting_boards(Stockfish::Color team) const {
        const Turn turn = turn_at_node();
        return {turn.acts[BOARD_A_IDX] && team_of_seat(turn.seat[BOARD_A_IDX]) == team,
                turn.acts[BOARD_B_IDX] && team_of_seat(turn.seat[BOARD_B_IDX]) == team};
    }

    /** Whether @p team acts anywhere at this node; a non-durative team always does. */
    bool team_acts(Stockfish::Color team) const {
        if (!durative_.active) {
            return true;
        }
        const std::array<bool, 2> acts = acting_boards(team);
        return acts[0] || acts[1];
    }

    /**
     * @brief Charge the action @p actingTeam plays at this durative node.
     *
     * @param movedA, movedB Whether it moved (rather than passed) on each
     *        board it acts on; ignored for the others. Time advances to the
     *        landing, every clock on turn runs by that much, a moved seat
     *        hands the board to its opponent (ready after its own action
     *        time), a passing seat waits until just after the next event on
     *        the other board, its flag, or PassCapK x the root budget,
     *        whichever is first. A team that does not act here only waits:
     *        nothing moves. With FlagLoss, a clock that runs out on the way
     *        ends the game (outcome()).
     */
    void advance_action(Stockfish::Color actingTeam, bool movedA, bool movedB) {
        Durative& d = durative_;
        const Turn turn = turn_at_node();
        const bool actA = turn.acts[BOARD_A_IDX]
            && team_of_seat(turn.seat[BOARD_A_IDX]) == actingTeam;
        const bool actB = turn.acts[BOARD_B_IDX]
            && team_of_seat(turn.seat[BOARD_B_IDX]) == actingTeam;
        d.atRoot = false;
        if (!actA && !actB) {
            return;
        }
        if (d.outcome != Outcome::NONE) {
            return;
        }
        const int32_t dt = turn.dt;
        const int seatA = turn.seat[BOARD_A_IDX];
        const int seatB = turn.seat[BOARD_B_IDX];
        if (d.flagLoss) {
            const int32_t first = std::min(d.clockMs[seatA], d.clockMs[seatB]);
            if (first <= dt) {
                const bool aOut = d.clockMs[seatA] == first;
                const bool bOut = d.clockMs[seatB] == first;
                if (aOut && bOut && team_of_seat(seatA) != team_of_seat(seatB)) {
                    d.outcome = Outcome::DRAW;
                } else {
                    const Stockfish::Color loser = team_of_seat(aOut ? seatA : seatB);
                    d.outcome = loser == Stockfish::WHITE
                        ? Outcome::BLACK_TEAM_WINS : Outcome::WHITE_TEAM_WINS;
                }
                d.clockMs[seatA] = std::max(0, d.clockMs[seatA] - dt);
                d.clockMs[seatB] = std::max(0, d.clockMs[seatB] - dt);
                sync_lead();
                return;
            }
        }
        for (int seat : {seatA, seatB}) {
            d.clockMs[seat] = clamp_lead(static_cast<int64_t>(d.clockMs[seat]) - dt);
            d.readyMs[seat] = std::max(0, d.readyMs[seat] - dt);
        }
        if (turn.crossTie) {
            d.tieNext = ~d.tieNext;
        }
        const bool acts[2] = {actA, actB};
        const bool moved[2] = {movedA, movedB};
        for (int board = 0; board < 2; ++board) {
            if (!acts[board] || !moved[board]) {
                continue;
            }
            const int seat = turn.seat[board];
            d.sitting[seat] = false;
            d.readyMs[seat] = 0;
            d.side[board] = ~d.side[board];
            const int next = seat_of(board, d.side[board]);
            d.readyMs[next] = action_time_ms(next);
            d.sitting[next] = false;
        }
        for (int board = 0; board < 2; ++board) {
            if (!acts[board] || moved[board]) {
                continue;
            }
            const int seat = turn.seat[board];
            const int other = seat_of(1 - board, d.side[1 - board]);
            int64_t wait = static_cast<int64_t>(d.readyMs[other]) + 1;
            if (d.clockMs[seat] > 0) {
                wait = std::min<int64_t>(wait, d.clockMs[seat]);
            }
            wait = std::min<int64_t>(wait, d.passCapMs);
            d.readyMs[seat] = static_cast<int32_t>(std::max<int64_t>(wait, 1));
            d.sitting[seat] = true;
        }
        sync_lead();
        {
            // Both seats on turn sitting at equal clocks, across the teams:
            // nothing can change until one moves, and neither has to. The
            // arena adjudicates this sit-off as a draw.
            const int nowA = seat_of(BOARD_A_IDX, d.side[BOARD_A_IDX]);
            const int nowB = seat_of(BOARD_B_IDX, d.side[BOARD_B_IDX]);
            if (d.sitting[nowA] && d.sitting[nowB]
                && team_of_seat(nowA) != team_of_seat(nowB)
                && d.clockMs[nowA] == d.clockMs[nowB]) {
                d.outcome = Outcome::DRAW;
            }
        }
    }

    /** The state after @p actingTeam's action; see advance_action(). */
    TimeState after_action(Stockfish::Color actingTeam, bool movedA, bool movedB) const {
        TimeState next = *this;
        next.advance_action(actingTeam, movedA, movedB);
        return next;
    }

    /** How the game stands on the clock: a flag (FlagLoss) or a sit-off draw. */
    Outcome outcome() const { return durative_.outcome; }

    /** @p team's result from outcome(): +1 win, -1 loss, 0 draw; none while play goes on. */
    std::optional<int> outcome_for(Stockfish::Color team) const {
        switch (durative_.outcome) {
            case Outcome::NONE: return std::nullopt;
            case Outcome::DRAW: return 0;
            case Outcome::WHITE_TEAM_WINS: return team == Stockfish::WHITE ? 1 : -1;
            case Outcome::BLACK_TEAM_WINS: return team == Stockfish::BLACK ? 1 : -1;
        }
        return std::nullopt;
    }

    /** A seat's remaining clock (durative states). */
    int32_t seat_clock_ms(int seat) const { return durative_.clockMs[seat]; }
    /** Time until a seat on turn lands its next action (durative states). */
    int32_t seat_ready_ms(int seat) const { return durative_.readyMs[seat]; }
    /** Whether a seat on turn is in a pass (durative states). */
    bool seat_sitting(int seat) const { return durative_.sitting[seat]; }
    /** Who is to move on @p board, as this durative state has tracked it. */
    Stockfish::Color side_on(int board) const { return durative_.side[board]; }
    bool at_root() const { return durative_.atRoot; }
    bool flag_loss() const { return durative_.flagLoss; }
    bool instant_transfer() const { return !durative_.active || durative_.instantTransfer; }
    bool mate_race_by_landing() const { return durative_.active && durative_.mateRaceByLanding; }
    /** What one action of @p seat costs in this tree. */
    int32_t action_time_ms(int seat) const {
        return team_of_seat(seat) == durative_.rootTeam
            ? durative_.rootMoveMs : durative_.paceMs[seat];
    }

    /** Exact identity of a clocked state: the lead, both move times and any next-move time. */
    uint64_t exact_key() const {
        const uint64_t lead = static_cast<uint32_t>(whiteLeadMs_);
        const uint64_t moveTimes =
            (static_cast<uint64_t>(static_cast<uint32_t>(moveTimeMs_[Stockfish::WHITE])) << 32)
            | static_cast<uint32_t>(moveTimeMs_[Stockfish::BLACK]);
        uint64_t key = (0x434c4f434b000000ULL ^ (lead * 0x9e3779b97f4a7c15ULL))
            ^ (moveTimes * 0xbf58476d1ce4e5b9ULL) ^ (clocked_ ? 1 : 0);
        if (nextMoveTimeMs_ > 0) {
            const uint64_t next = (static_cast<uint64_t>(nextMoveTeam_) << 32)
                | static_cast<uint32_t>(nextMoveTimeMs_);
            key ^= (next + 0x6e657874ULL) * 0x94d049bb133111ebULL;
        }
        if (!sitGate_) {
            key ^= 0x5349544f4646ULL * 0xd6e8feb86659fd93ULL;
        }
        if (durative_.active) {
            key ^= durative_.key();
        }
        return key;
    }

    /**
     * @brief Coarse lead class: which team may sit, and the size of its lead
     * on a log2 scale in its own move times.
     *
     * Nothing in the search keys on it; nodes are identified by the exact
     * clock (exact_key), since leads in one class can differ in sign a few
     * actions later. A front end can use it to judge whether a lead that has
     * drifted is worth a new search.
     */
    uint32_t lead_class(Stockfish::Color team) const {
        const int32_t lead = lead_ms(team);
        if (lead == 0) {
            return 0;
        }
        const Stockfish::Color ahead = lead > 0 ? team : ~team;
        const uint32_t magnitude = static_cast<uint32_t>(std::abs(lead));
        const uint32_t moveTime = static_cast<uint32_t>(moveTimeMs_[ahead]);
        const uint32_t moves = (magnitude + moveTime - 1) / moveTime;
        const uint32_t scale = std::min<uint32_t>(std::bit_width(moves), 31);
        return lead > 0 ? scale : 32 + scale;
    }

    bool operator==(const TimeState&) const = default;

private:
    static constexpr int BOARD_A_IDX = 0;
    static constexpr int BOARD_B_IDX = 1;

    struct Durative {
        bool active = false;
        // The root team decides at the root on every board it is on turn.
        bool atRoot = false;
        bool flagLoss = false;
        bool instantTransfer = true;
        bool mateRaceByLanding = false;
        Outcome outcome = Outcome::NONE;
        Stockfish::Color rootTeam = Stockfish::WHITE;
        // The team that wins the next exact cross-team tie.
        Stockfish::Color tieNext = Stockfish::BLACK;
        std::array<Stockfish::Color, 2> side{Stockfish::WHITE, Stockfish::WHITE};
        std::array<int32_t, 4> clockMs{0, 0, 0, 0};
        // Relative to this node's time; meaningful for the seats on turn.
        std::array<int32_t, 4> readyMs{0, 0, 0, 0};
        std::array<int32_t, 4> paceMs{0, 0, 0, 0};
        std::array<bool, 4> sitting{false, false, false, false};
        int32_t rootMoveMs = 1;
        int32_t passCapMs = 1;

        bool operator==(const Durative&) const = default;

        uint64_t key() const {
            uint64_t h = 0x4455524154495645ULL;
            auto mix = [&h](uint64_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h *= 0xbf58476d1ce4e5b9ULL;
            };
            for (int seat = 0; seat < 4; ++seat) {
                mix(static_cast<uint32_t>(clockMs[seat]));
                mix(static_cast<uint32_t>(readyMs[seat]) | (uint64_t(sitting[seat]) << 32));
                mix(static_cast<uint32_t>(paceMs[seat]));
            }
            mix((uint64_t(side[0]) << 0) | (uint64_t(side[1]) << 1)
                | (uint64_t(tieNext) << 2) | (uint64_t(rootTeam) << 3)
                | (uint64_t(atRoot) << 4) | (uint64_t(flagLoss) << 5)
                | (uint64_t(instantTransfer) << 6) | (uint64_t(mateRaceByLanding) << 7)
                | (uint64_t(outcome) << 8));
            mix((uint64_t(static_cast<uint32_t>(rootMoveMs)) << 32)
                | static_cast<uint32_t>(passCapMs));
            return h;
        }
    };

    // Who acts at a durative node and how far time advances to it.
    struct Turn {
        std::array<int, 2> seat{0, 2};
        std::array<bool, 2> acts{false, false};
        int32_t dt = 0;
        bool crossTie = false;
    };

    Turn turn_at_node() const {
        const Durative& d = durative_;
        Turn turn;
        turn.seat = {seat_of(BOARD_A_IDX, d.side[BOARD_A_IDX]),
                     seat_of(BOARD_B_IDX, d.side[BOARD_B_IDX])};
        if (!d.active || d.outcome != Outcome::NONE) {
            return turn;
        }
        const int32_t readyA = d.readyMs[turn.seat[BOARD_A_IDX]];
        const int32_t readyB = d.readyMs[turn.seat[BOARD_B_IDX]];
        const Stockfish::Color teamA = team_of_seat(turn.seat[BOARD_A_IDX]);
        const Stockfish::Color teamB = team_of_seat(turn.seat[BOARD_B_IDX]);
        if (d.atRoot) {
            // Our decision: every board the root team is on turn on, landing
            // at its budget; the other team's earlier landing (if any) comes
            // right after, at a ready time of 0.
            turn.acts = {teamA == d.rootTeam, teamB == d.rootTeam};
            turn.dt = std::max(turn.acts[0] ? readyA : 0, turn.acts[1] ? readyB : 0);
            return turn;
        }
        turn.dt = std::min(readyA, readyB);
        turn.acts = {readyA == turn.dt, readyB == turn.dt};
        if (turn.acts[0] && turn.acts[1] && teamA != teamB) {
            turn.crossTie = true;
            turn.acts = {teamA == d.tieNext, teamB == d.tieNext};
        }
        return turn;
    }

    // Sit-off: the seat deciding is on turn on one board while the other
    // team's seat on the other board sits. If its clock is lower it would
    // flag first, so it may not sit as well: it must move.
    bool sit_off_forces_move(Stockfish::Color team) const {
        const Durative& d = durative_;
        const Turn turn = turn_at_node();
        for (int board = 0; board < 2; ++board) {
            const int seat = turn.seat[board];
            const int other = turn.seat[1 - board];
            if (turn.acts[board] && !turn.acts[1 - board]
                && team_of_seat(seat) == team
                && team_of_seat(other) != team
                && d.sitting[other]
                && d.clockMs[seat] < d.clockMs[other]) {
                return true;
            }
        }
        return false;
    }

    // The lead, as v3 defines it, read off the four clocks.
    void sync_lead() {
        const Durative& d = durative_;
        const int64_t white = (static_cast<int64_t>(d.clockMs[SEAT_AW]) - d.clockMs[SEAT_BW]
            + static_cast<int64_t>(d.clockMs[SEAT_BB]) - d.clockMs[SEAT_AB]) / 2;
        whiteLeadMs_ = clamp_lead(white);
    }

    static int32_t clamp_lead(int64_t lead) {
        return static_cast<int32_t>(std::clamp<int64_t>(lead, -MAX_LEAD_MS, MAX_LEAD_MS));
    }

    bool clocked_ = false;
    // Legacy default: white may never sit and black always may. Callers build
    // states through legacy() or clocked(); this only keeps the type regular.
    int32_t whiteLeadMs_ = -MAX_LEAD_MS;
    std::array<int32_t, 2> moveTimeMs_{DEFAULT_MOVE_TIME_MS, DEFAULT_MOVE_TIME_MS};
    // The cost of nextMoveTeam_'s next action when it differs from its usual
    // move time; 0 when it does not. Used up by that team's next action.
    Stockfish::Color nextMoveTeam_ = Stockfish::WHITE;
    int32_t nextMoveTimeMs_ = 0;
    // Wave 2. Defaults keep every state stock v3.0.3.
    bool sitGate_ = true;
    Durative durative_;
};
