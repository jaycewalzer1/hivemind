# Wave 2 engine: the durative clock (part 1)

Branch `talisman/wave2`, based on aminwoo/hivemind v3.0.3 (7a4dbdb). Design source:
`docs/custom_engine_spec_v4_durative.md` in the talisman repo (spec v4), written for the old fork and built here into
upstream v3's search. Network: twin-s-clock it13 (`engine/models/twin-s-clock*.onnx`, symlinks to
`third_party/upstream-hivemind/engine/models`; `models/` is git-ignored). The network's inputs do not change: planes
37/38 (lead, speed) are computed from the node's clocks.

Every new option defaults to stock v3.0.3. With defaults the search is the stock search: proven by the unit tests below
and, for the binary, by `engine/tests/stock_parity.sh`.

## UCI contract

| Option | Type, default | Meaning |
|---|---|---|
| `DurativeTree` | check, false | Per-seat clocks and ready times in the tree (below). |
| `SitGate` | check, true | Upstream's hard `can_sit = lead > 0`. Off: a pass is always legal, priced by the tree (durative trees keep the sit-off rule). Applies to v3 clocked states too. |
| `FlagLoss` | check, false | Durative only: a seat whose clock runs out is a terminal loss for its team; two teams out at the same millisecond = draw (arena RULES.md). |
| `InstantTransfer` | check, true | Parsed and carried in every node's state (`TimeState::instant_transfer()`); **behaviour is part 2's** (today a capture is always usable at once). |
| `MateRaceByLanding` | check, false | Parsed and carried (`TimeState::mate_race_by_landing()`); **behaviour is part 2's**. |
| `PassCapK` | spin 1..64, 8 | A pass lasts at most `k x` the root budget. |
| `SeatClocksMs` | string `<Aw> <Ab> <Bw> <Bb>`, empty | The four real clocks. Empty: derived from `TimeLeadMs` around 2^29 ms (no flag can happen), as v3. |
| `SeatPaceMs` | string, empty | Per-seat ms per action (EMA). Empty: `TeamMoveTimeMs` / `OpponentMoveTimeMs` per team, as v3. Given: the speed plane uses each team's mean pace. |
| `SeatElapsedMs` | string, empty | How long each seat on turn has been thinking: an opponent seat's first landing is `pace - elapsed` (at least 1 ms). Ignored for our own seats. |

The string options take four integers; an empty value (`setoption name SeatClocksMs value`) clears them; a count other
than 0 or 4 is refused with an `info string` and keeps the old value. `Team`, `TimeModel`, `TimeLeadMs`,
`TeamMoveTimeMs`, `OpponentMoveTimeMs`, `RootMoveTimeMs` keep their v3 meaning. Every wave 2 option needs
`TimeModel clock` (the default for twin-s-clock); under `TimeModel flag` they do nothing.

Seat order everywhere: Aw, Ab, Bw, Bb (indices 0-3, `TimeState::SEAT_*`). The white team is Aw + Bb.

## The model as built

**State.** `TimeState` (`engine/src/environment/time_state.h`) carries, besides v3's lead and move times, a `sitGate_`
flag and a `Durative` block (`time_state.h:507`): four clocks, four ready times (relative to the node's own time,
meaningful for the two seats on turn), four paces, four "sitting" bits, who is to move on each board, the root team,
whose turn the next cross-team tie is, `atRoot`, the root budget, the pass cap, the flags, and an `outcome`. Built by
`TimeState::durative()` (`time_state.h:129`) from a v3 clocked state plus `DurativeConfig` (`time_state.h:18`).
`exact_key()` mixes all of it in (`time_state.h:473`), so transpositions and node hashes separate clocks; in default
mode the key and `operator==` are v3's.

**Who acts.** `turn_at_node()` (`time_state.h:559`):
- At the root, the root team decides on every board it is on turn, landing at its budget (`RootMoveTimeMs`, else its
  team move time). If an opponent seat would land earlier, its ready time drops to 0 and it acts right after.
- Elsewhere, time advances to the earliest ready time of the two seats on turn; only that seat's team acts, on that
  board, or on both if both of its seats are ready at once. An exact tie between the two teams goes to `tieNext`,
  which starts as the root team's opponents and flips each time a tie is used.

**Waiting keeps the tree alternating.** Upstream's tree assumes teams alternate (child team = `~team`, values negate
each ply, the solver's WIN/LOSS flip). So a node whose team does not act gets exactly one action, (pass, pass), and
time does not move on it. `collect_batch` sets this up (`searchthread.cc:820-857`, `LeafContext::forcedWait`,
`searchthread.h:161`): the on-turn flags passed to the joint-action generator are restricted to the boards the team
acts on; a waiting team is given board A "on turn" with only the pass, and the sit right in `process_batch`
(`searchthread.cc:934`), which makes (pass, pass) its one legal action. Cost: a wait node is a network evaluation, so
lines where one team acts twice in a row spend an extra node. An in-place expansion that skips the evaluation is a
possible optimisation (not done).

**Charging an action.** `advance_action(team, movedA, movedB)` (`time_state.h:333`), called by the tree's
`time_after_action` (`searchthread.cc:1313`) with the action actually played:
1. `dt` = time to the landing. With `FlagLoss`, if either clock on turn is <= `dt`, the lower one flagged first
   (equal across teams: draw) and the state's `outcome` is set; nothing else happens.
2. Both clocks on turn and both ready times run by `dt`.
3. A seat that moved hands its board to the opponent seat, whose ready time is its action time: the root budget for
   the root team's seats, its pace for the others.
4. A seat that passed waits until 1 ms after the next event on the other board (that seat's ready time), or until
   its own flag, or `PassCapK x` the root budget, whichever is first (at least 1 ms), and is marked sitting.
5. The lead is re-read from the clocks: white lead = ((Aw - Bw) + (Bb - Ab)) / 2 (`sync_lead`, `time_state.h:609`),
   so the lead and speed planes (`planes.cc`) see the node's clocks.
6. Sit-off draw: both seats on turn sitting, on different teams, with equal clocks: nothing can change and neither
   has to move; the arena adjudicates that as a draw, so the state's `outcome` is DRAW.

**Sitting.** `can_sit(team)` (`time_state.h:239`): with `SitGate` on, v3's `lead > 0` (the lead from the clocks in a
durative tree). Off: always, except the **sit-off rule** (`sit_off_forces_move`, `time_state.h:591`): a team deciding on
one board while the other team's seat on the other board sits may not sit too if its clock is lower; it must move. At
equal clocks it may sit, which then ends the line as the sit-off draw. A seat that must move but has no legal move is
mated by upstream's own rule (`is_checkmate(team, canSit=false)`), which matches the arena's adjudication (the lower
clock loses). The sit plane the network sees stays upstream's sit right, `lead > 0` (`lead_sit_right`, used at
`planes.cc:218` and `planes.cc:246`), whatever the gate, because that is what twin-s-clock was trained on.

**Terminals.** `collect_batch` asks `outcome_for(team)` before the board's own classification
(`searchthread.cc:754`): a flag or sit-off draw ends the line whatever the board shows. The frozen-board loss, which
assumes the team must act now, is not used in durative trees (`searchthread.cc:769`).

**UCI.** Options parsed in `UCI::setoption` (`uci.cc:674` onward), state in `uci.h` (`durativeTree`, `sitGate`,
`durativeConfig`); `UCI::current_clock()` (`uci.cc:732`) returns v3's `TimeState::clocked(...)` unchanged with
defaults, `.with_sit_gate(false)` for `SitGate false`, and `TimeState::durative(...)` (root sides read off the board)
for `DurativeTree true`. Advertised in the `uci` handshake.

## What still reasons in v3 alternation

Callers that charge an action without saying which boards moved use the old `advance(team, onBoth)`. On a durative
state that call first drops the durative block (`time_state.h:272`), so from there the state is v3's clocked state at
the current lead (with the gate setting kept). This covers:
- the root mate provers and scans in `agent.cc` (mate probe, frozen race, conditional probe, selected-move
  certification, root loss scan) and the waiting-board mate inside `classify_terminal_position`
  (`searchthread.cc` `has_unavoidable_waiting_board_mate`);
- tree reuse (`retain_reuse_candidates`, `agent.cc` around 9139 and 9221) and the permanent-brain clock
  (`agent.cc:8491`): their keys will not match the durative children, so under `DurativeTree` a reused tree is
  simply not found and the search starts fresh (correct, just no reuse; the arena sends `ucinewgame` anyway);
- PV printing, which labels wait steps as `(pass,pass)`.

## Tests

`engine/tests/test_durative.cc` (run: `build-ninja/tests/hivemind_tests --gtest_filter='Durative*'`):
- rules: root decides then earliest-ready acts, tie alternation, pass length and cap, SitGate, SeatElapsedMs,
  lead-derived clocks;
- **acceptance 1**: clocks 100 / 600 ms with a 1000 ms budget have one loser (both orders), equal = double-flag draw,
  no `FlagLoss` = v3 run-down; in a real search every root action of the flagging team is a solved loss;
- **acceptance 3**: sit-off at equal clocks is a draw; 1 s lower must move; 1 s higher may sit and the other then must
  move (the sign flips with 1 s). Rules level only: the uniform test network cannot show a value sign;
- **acceptance 4**: 2000 random lines, a 3 s seat at a 100 ms minimum makes at most 29 landed moves (the 30th lands
  at 0, a flag); in a real search no line gives a 300 ms seat more than 2 moves, and lines end on the clock;
- **acceptance 7**: defaults give v3's states, keys and planes; a durative state with v3's lead gives v3's planes.
`engine/tests/test_uci.cc` (`*Wave2*`): option parsing and that defaults root on `TimeState::clocked(...)` exactly.

`engine/tests/stock_parity.sh` (acceptance 7, binary): builds this engine and stock v3.0.3
(`third_party/upstream-hivemind`, built out of tree, source untouched) with one search worker, then runs `policy` and
`go nodes N` on `engine/tests/data/stock_parity_positions.txt` (8 positions, both teams, leads -15 s..+15 s, unequal
move times, pockets) with default options. It compares the whole output except wall-clock fields (time, nps, ms,
percentages) and the intermediate `info depth` progress lines, which are printed when the reporting thread polls and
so differ between two stock runs too; the last `info depth` line (nodes, score, PV of the finished search), the
`policy` output and `bestmove` are compared exactly. The time-boxed root mate scans are off in both because their
results depend on wall time. Result on 2026-11-19: **8/8 identical**. Build cost: two extra builds
(`engine/build-parity`, `engine/build-parity-stock`, git-ignored), a few minutes at `JOBS=2`.

Upstream's own suite stays green.

## Notes for part 2

- **Delayed transfer (`InstantTransfer false`).** `Board::push_move` (`board.cc:213`) puts a captured piece in the
  partner's pocket at once. The durative state knows when the capture lands (the `advance_action` call that carries
  it) and when the partner seat next decides (its ready time). Suggested: keep a per-seat "arrives after" list of
  pieces in `TimeState::Durative` (or a pocket mask), and filter drop moves of not-yet-arrived pieces in
  `collect_batch`'s `prepare_actions` (`searchthread.cc` after line 830) when `!time.instant_transfer()`. The piece
  becomes droppable from the partner's first decision whose node time is after the landing: i.e. it is usable at the
  partner's next `acting_boards` node, not at a node where the partner was already deciding. The state key must
  include the pending pieces.
- **Mate race by landing (`MateRaceByLanding`).** The mate-race pessimism lives in `classify_terminal_position` and
  `has_unavoidable_waiting_board_mate` (`searchthread.cc` top) and in the agent's provers. In a durative tree a race
  is decided by which mate lands first: compare the ready times of the two seats (`seat_ready_ms`) at the node; on an
  exact tie spec v4 leaves the rule open (opponent of the last capturer, or draw): pick one, document it, test both
  teams.
- **Self-play hooks.** `tools/selfplay.cc` builds `TimeState::clocked` from `sample_clock` (`selfplay.cc:1029`) and
  charges each played action with `time.advance(...)` (`selfplay.cc:1064`, `1207`). For durative self-play it should
  keep four seat clocks per game, build each search's clock with `TimeState::durative(base, team, sideA, sideB, cfg)`
  (sides from the board, `rootMoveTimeMs` = this search's budget, paces from per-seat EMAs), and advance its game
  clocks by real landing times, not through `advance`. The UCI path (arena) already does this per search through
  `SeatClocksMs` / `SeatPaceMs` / `SeatElapsedMs`.
- **Provers.** If the root provers should be landing-aware, give them the action-aware `after_action` instead of
  `after(team, onBoth)`; until then they are v3 reasoners on the root's lead.
- **Wait-node cost.** If the extra evaluation per wait node matters at self-play budgets, expand a forced-wait leaf in
  place in `select_and_expand` and continue to its child.
