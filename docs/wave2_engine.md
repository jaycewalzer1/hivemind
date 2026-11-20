# Wave 2 engine: the durative clock (parts 1 and 2)

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
| `SitGate` | check, true | Upstream's hard `can_sit = lead > 0`. Off: a pass is always legal, priced by the tree (durative trees keep the sit-off rule). Takes effect only with `DurativeTree`: a v3 clocked state keeps the gate (`TimeState::can_sit`), because v3's turn-taking tree has no sit-off rule and with both teams free to pass the search cycled and never answered. |
| `FlagLoss` | check, false | Durative only: a seat whose clock runs out is a terminal loss for its team; two teams out at the same millisecond = draw (arena RULES.md). |
| `InstantTransfer` | check, true | Durative only. Off: a captured piece is in the partner's pocket from the capture's landing but droppable only from the partner's next decision (below). |
| `MateRaceByLanding` | check, false | Durative only. Mate races go to the earlier landing; exact ties to the opponent of the last capturer (below). The root mate provers are off under it. |
| `PassCapK` | spin 1..64, 8 | A pass lasts at most `k x` the root budget. |
| `SeatClocksMs` | string `<Aw> <Ab> <Bw> <Bb>`, empty | The four real clocks. Empty: derived from `TimeLeadMs` around 2^29 ms (no flag can happen), as v3. |
| `SeatPaceMs` | string, empty | Per-seat ms per action (EMA). Empty: `TeamMoveTimeMs` / `OpponentMoveTimeMs` per team, as v3. Given: the speed plane uses each team's mean pace. |
| `SeatElapsedMs` | string, empty | How long each seat on turn has been thinking: an opponent seat's first landing is `pace - elapsed` (at least 1 ms). Ignored for our own seats. |
| `SeatSitting` | string, empty | `"<Aw> <Ab> <Bw> <Bb>"` of 0/1: which opponent seats on turn are in a pass at the root (`DurativeConfig::seatSitting`), so the sit-off rule makes the lower clock move. Self-play sets it directly; an arena infers it. Ignored for our own seats. |

The string options take four integers; an empty value (`setoption name SeatClocksMs value`) clears them; a count other
than 0 or 4 is refused with an `info string` and keeps the old value. `Team`, `TimeModel`, `TimeLeadMs`,
`TeamMoveTimeMs`, `OpponentMoveTimeMs`, `RootMoveTimeMs` keep their v3 meaning. Every wave 2 option needs
`TimeModel clock` (the default for twin-s-clock); under `TimeModel flag` they do nothing.

Seat order everywhere: Aw, Ab, Bw, Bb (indices 0-3, `TimeState::SEAT_*`). The white team is Aw + Bb.

## The model as built

**State.** `TimeState` (`engine/src/environment/time_state.h`) carries, besides v3's lead and move times, a `sitGate_`
flag and a `Durative` block (`time_state.h:598`): four clocks, four ready times (relative to the node's own time,
meaningful for the two seats on turn), four paces, four "sitting" bits, who is to move on each board, the root team,
whose turn the next cross-team tie is, `atRoot`, the root budget, the pass cap, the flags, and an `outcome`. Built by
`TimeState::durative()` (`time_state.h:132`) from a v3 clocked state plus `DurativeConfig` (`time_state.h:18`).
`exact_key()` mixes all of it in (`time_state.h:547`), so transpositions and node hashes separate clocks; in default
mode the key and `operator==` are v3's.

**Who acts.** `turn_at_node()` (`time_state.h:668`):
- At the root, the root team decides on every board it is on turn, landing at its budget (`RootMoveTimeMs`, else its
  team move time). If an opponent seat would land earlier, its ready time drops to 0 and it acts right after. Its
  clock runs only until its landing (not our whole budget), its flag is checked against that, and its action is dated
  back: the seat its move hands the board to has been on turn for the rest of our budget and is charged it (a pass
  likewise runs from the earlier time). Before review fix `d538648` the opponent seat was charged our whole budget,
  which turned every root action of a time scramble into a false flag win.
- Elsewhere, time advances to the earliest ready time of the two seats on turn; only that seat's team acts, on that
  board, or on both if both of its seats are ready at once. An exact tie between the two teams goes to `tieNext`,
  which starts as the root team's opponents and flips each time a tie is used.

**Waiting keeps the tree alternating.** Upstream's tree assumes teams alternate (child team = `~team`, values negate
each ply, the solver's WIN/LOSS flip). So a node whose team does not act gets exactly one action, (pass, pass), and
time does not move on it. `collect_batch` sets this up (`searchthread.cc:913-965`, `LeafContext::forcedWait`,
`searchthread.h:161`): the on-turn flags passed to the joint-action generator are restricted to the boards the team
acts on; a waiting team is given board A "on turn" with only the pass, and the sit right in `process_batch`
(`searchthread.cc:1042`), which makes (pass, pass) its one legal action. Cost: a wait node is a network evaluation, so
lines where one team acts twice in a row spend an extra node. An in-place expansion that skips the evaluation is a
possible optimisation (not done).

**Charging an action.** `advance_action(team, movedA, movedB, handA, handB)` (`time_state.h:338`), called by the tree's
`time_after_action` (`searchthread.cc:1421`) with the action actually played:
(`handA`/`handB`: the piece type each board's move sends to the partner's pocket, `Board::hand_piece_type`.)
1. `dt` = time to the landing. Each clock on turn runs `min(dt, its ready time)` (equal to `dt` except for the root's
   early opponent landing). With `FlagLoss`, if a clock is <= its run, the lower such clock flagged first (equal across
   teams: draw) and the state's `outcome` is set; nothing else happens.
2. Both clocks on turn run, and both ready times by `dt`.
3. A seat that moved hands its board to the opponent seat, whose ready time is its action time: the root budget for
   the root team's seats, its pace for the others.
4. A seat that passed waits until 1 ms after the next landing on the other board, or until its own flag, or
   `PassCapK x` its own action time (the root budget for the root team, its pace for the other team, as the self-play
   game caps a pass at `PassCapK x` the seat's move time), whichever is first (at least 1 ms), and is marked sitting.
   A seat on the other board that is deciding lands at its ready time; one that sits (or passes at this node) lands
   nothing, so it sets no wake-up. When a seat passes, a seat sitting on the other board that was waiting for it to
   land runs on to its own cap; when a seat moves, it wakes a pass on the other board 1 ms later. This is the
   self-play game's rule (passes wake only on landings), and it ends the 1 ms leapfrog of one team passing on both
   boards (review fix `d538648`).
5. The lead is re-read from the clocks: white lead = ((Aw - Bw) + (Bb - Ab)) / 2 (`sync_lead`, `time_state.h:718`),
   so the lead and speed planes (`planes.cc`) see the node's clocks.
6. Sit-off draw: both seats on turn sitting, on different teams, with equal clocks: nothing can change and neither
   has to move; the arena adjudicates that as a draw, so the state's `outcome` is DRAW.

**Sitting.** `can_sit(team)` (`time_state.h:244`): with `SitGate` on, v3's `lead > 0` (the lead from the clocks in a
durative tree). Off: always, except the **sit-off rule** (`sit_off_forces_move`, `time_state.h:700`): a team deciding on
one board while the other team's seat on the other board sits may not sit too if its clock is lower; it must move. At
equal clocks it may sit, which then ends the line as the sit-off draw. A seat that must move but has no legal move is
mated by upstream's own rule (`is_checkmate(team, canSit=false)`), which matches the arena's adjudication (the lower
clock loses). The sit plane the network sees stays upstream's sit right, `lead > 0` (`lead_sit_right`, used at
`planes.cc:218` and `planes.cc:246`), whatever the gate, because that is what twin-s-clock was trained on.

**Terminals.** `collect_batch` asks `outcome_for(team)` before the board's own classification
(`searchthread.cc:847`): a flag or sit-off draw ends the line whatever the board shows. The frozen-board loss, which
assumes the team must act now, is not used in durative trees (`searchthread.cc:862`).

**UCI.** Options parsed in `UCI::setoption` (`uci.cc:674` onward), state in `uci.h` (`durativeTree`, `sitGate`,
`durativeConfig`); `UCI::current_clock()` (`uci.cc:732`) returns v3's `TimeState::clocked(...)` unchanged with
defaults, `.with_sit_gate(false)` for `SitGate false`, and `TimeState::durative(...)` (root sides read off the board)
for `DurativeTree true`. Advertised in the `uci` handshake.

## Part 2: delayed transfer, landing-time mate races, repetition

**Delayed transfer (`InstantTransfer false`).** The board still puts a captured piece in the partner's pocket at once
(`Board::push_move`); the durative state holds, per seat and piece type, how many pieces in that seat's pocket landed
during its current decision (`Durative::pending`, `TimeState::pending_in_hand`, `time_state.h:485`). A seat's decision
starts when its board is handed to it (the opponent moved there: its pending list is cleared) or when a pass of its
own ends (a seat back from a pass sees no pending pieces, and its list is cleared when it acts). So a piece is usable
from the partner's next decision after the capture lands: the "sit for the knight" line works (pass until the capture
lands, then drop), but a seat that was already thinking when the piece arrived cannot use it in that action. A
capture and a drop of the same piece in one joint action never happened anyway: drops come from the pre-action
pocket. `collect_batch` removes drops of held pieces from the seat's actions (`searchthread.cc:930`); the pending
counts are in the node key. The network still sees the whole pocket (its inputs do not change in wave 2).
At the root, an opponent seat on turn (not sitting) starts with the `SeatArrivalsMs` landings into its pocket that are
younger than its `SeatElapsedMs` pending: they arrived mid-decision (review fix `d538648`; before it the root seeded
nothing). Self-play sends its landing log to the tree for this with narrow networks too (`SeatClockGame::search_clock`);
`recentLandings` only reach the wide planes when the network is wide.
Not done: `Board::is_checkmate` still counts held pieces as possible blockers.

**Mate races by landing (`MateRaceByLanding`).** Inside the durative tree, landings are already in time order, so a
mate that lands first ends the line. Two places decided races by rule instead, and now use landing times:
- *The waiting-board mate* (`classify_terminal_position` -> `has_unavoidable_waiting_board_mate`): upstream lets the
  waiting mate win the race exactly when the team to play is the root team (pessimism). Now
  (`waiting_mate_wins_by_landing`, `searchthread.cc:159`) the defenders' reply wins if its seat's ready time is
  earlier, the waiting mate wins if its seat's is earlier.
- *The root.* The root team decides first even when an opponent seat would land before its budget (part 1). The root
  action now records, per opponent seat, how much earlier it would have landed (`mate_race_early_ms`,
  `time_state.h:510`). When the root action mates, and the mated team's early seat has a mate in one of its own, that
  mate landed first: the node is a WIN for the mated team (`early_mate_race`, `searchthread.cc:193`).
- *Exact ties* (`landing_tie_won_by`, `searchthread.cc:145`): **the opponent of the team that captured last wins**.
  The tree tracks the last capturing team (`last_capture_team()`; the root does not know the game's history, so it
  starts unknown). With no capture in the line, upstream's rule stays. Chosen over "draw" because the arena never
  draws a mate race and the rule is cheap to test from both teams.
- The root mate provers (mate probe, root mate scan, selected-move certification) judge races in v3 alternation and
  would overrule the tree, so `Agent::run_search` switches them off for a search whose clock has
  `MateRaceByLanding` (`agent.cc`, top of `run_search`). The tree's solver still finds mates; it is slower at it.

**Repetition.** Under a durative clock a repeated position is not the same state (every action costs time), so the
tree no longer calls a threefold or the fifty-move rule a draw (`searchthread.cc:299`; spec v3 D4: repetition is
non-terminal under tree clocks). The arena still adjudicates FICS threefold; self-play adjudicates `board.is_draw()`
on landing (below).

## Self-play and tournament (`tools/seat_clock_game.h`, `tools/selfplay.cc`, `tools/tournament.cc`)

Flags (both commands, listed in `--help`, contract in `docs/wave2_training.md`): `--durative-tree`, `--sit-gate`,
`--flag-loss`, `--instant-transfer`, `--mate-race-by-landing`, `--pass-cap-k`, `--base-mix MS:W,...`,
`--seat-move-times`, plus the timing knobs `--min-move-ms` (100), `--lag-max-ms` (150), `--pace-jitter` (0.25),
`--sitoff-ms` (5000), `--pace-ema-alpha` (0.3), which only act in seat-clock games. Any wave 2 option implies
`--time-model clock`. With none, both commands are stock: a 2-game self-play run is byte-identical (HVM6 chunks and
PGN) to the stock v3.0.3 binary built the same way.

**Tree options without `--base-mix`:** the v3 game loop (alternating team turns, one lead), with every search on the
durative clock derived from the lead and SitGate as given.

**`--base-mix`: games on four seat clocks** (`SeatClockGame`):
- Start: a base drawn from the mix; the lead sampler (`--level-start-fraction`, `--start-lead-*`) gives time odds,
  white seats +lead/2, black -lead/2 (clamped so no seat starts under 1 s); move times per team from
  `--move-time-*`, or per seat with `--seat-move-times`. Tournament: both games of a pair share start and cost draws;
  `--movetime` stands in for each seat's move time unless `--seat-move-times`.
- Each seat on turn is deciding or has a committed move. A decision is one search by the deciding seat's team (one
  record), rooted then: `SeatClocksMs` = the clocks, `SeatPaceMs` = each seat's observed pace (EMA of its landings),
  `SeatElapsedMs` and a sitting flag (`DurativeConfig::seatSitting`, used by the sit-off rule at the root) for the
  other team's seats, `RootMoveTimeMs` = the deciding seat's drawn cost. Only the deciding seats' boards of the
  chosen action are played: a move lands after its cost (move time x jitter in [1/1.25, 1.25], floor 100 ms, plus
  0-150 ms lag); a pass lasts until 1 ms after the next landing on the other board (a landing wakes it), its flag,
  or `k x` its move time. Pieces that arrive during a decision are not in its search: the game has delayed transfer
  whatever `--instant-transfer` says (that option is the tree's model of it).
- Ends: a seat at 0 with `--flag-loss` -> `flag` (the first to 0 loses; two teams at the same ms -> `double_flag`);
  both seats on turn sitting `--sitoff-ms` at once -> `sitoff_adjudicated` (lower clock loses, equal draws); a mate
  on landing -> `checkmate`; `board.is_draw()` on landing -> `draw`; `--max-macro-plies` decisions ->
  `macro-ply limit`. Ties at one millisecond: landings before decisions, then the teams alternate (or, with
  MateRaceByLanding, the opponent of the last capturer goes first).
- Outputs: HVM6/HDST records unchanged (one per search, planes from the search's clock). The clocks are not in those
  formats, so `seat_clocks.tsv` in the output directory has one row per record (game, record, team, game ms, nodes,
  four clocks, elapsed of the seats on turn, four paces). PGN tags `BaseMs`, `StartClocksMs`, `SeatMoveTimesMs`,
  `FinalClocksMs`, `GameMs`; move comments carry the mover's real clock. The game line adds ` flag-seat <seat>` and
  ` clocks <Aw> <Ab> <Bw> <Bb>`. Raw-policy opening moves work as in v3 (not recorded). Tree reuse is refused
  with `--base-mix`.

Smoke (it13, ONNX Runtime on the Mac, 48 nodes, 15 s base, all wave 2 options): 2 self-play games in 2 s (a flag,
a resignation), 4 tournament games in 5 s.

## What still reasons in v3 alternation

Callers that charge an action without saying which boards moved use the old `advance(team, onBoth)`. On a durative
state that call first drops the durative block (`time_state.h:276`), so from there the state is v3's clocked state at
the current lead (with the gate setting kept). This covers:
- the root mate provers and scans in `agent.cc` (mate probe, frozen race, conditional probe, selected-move
  certification, root loss scan) and the waiting-board mate inside `classify_terminal_position`
  (`searchthread.cc` `has_unavoidable_waiting_board_mate`);
- tree reuse (`retain_reuse_candidates`, `agent.cc` around 9139 and 9221) and the permanent-brain clock
  (`agent.cc:8491`): their keys will not match the durative children, so under `DurativeTree` a reused tree is
  simply not found and the search starts fresh (correct, just no reuse; the arena sends `ucinewgame` anyway);
- PV printing, which labels wait steps as `(pass,pass)`.
- `Board::is_checkmate` and `adjudicate_game` (pocket pieces held by delayed transfer still count as blockers).

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

`engine/tests/test_durative_part2.cc` (part 2):
- delayed transfer: a capture mid-decision is held until the partner's board comes back to it; a seat back from a
  pass may drop it; InstantTransfer and v3 states hold nothing; held pieces separate node keys;
  `hand_piece_type` (promoted pieces go as pawns); in a real search no held knight is ever dropped, while with
  InstantTransfer the same decision drops it;
- **acceptance 2** ("sit for the knight", needs `engine/models/twin-s-clock-it13.onnx`, skipped without it): A needs
  a knight for N@f7#, our partner wins Bw's knight in two moves with a queen check. Wave 2 has no gate, so the
  deficit case is decided by the sit-off rule: with a 10 s diagonal lead Bw (lower clock) must answer the check, the
  root plays (pass, check) and the root is a proven WIN; with a 10 s deficit Bw sits in check, Aw must move first,
  and the root is not a win; with upstream's gate and the deficit the pass beside a quiet move is not legal;
- **acceptance 5** (mirror): both teams have a smothered mate in one on their own board; with Aw landing at 300 and
  Bw at 500 the white root is WIN and the black root LOSS, and mirrored with the paces swapped. Without
  MateRaceByLanding both roots claim the win. Exact ties: both mates captures, so each root's own capture is the
  last one and each root loses (both teams tested); a tie with no capture keeps upstream's rule (the root's mate);
- **acceptance 6** (rules level plus search): a threefold is DRAW in v3's classifier and not in the durative one; a
  search one move from a threefold has DRAW nodes on v3's clock and none on the durative clock. Not covered: the
  "mate in 2 on B evaluates near +1" half (no fixture).
`engine/tests/test_seat_clock_game.cc`: option parsing (the full wave 2 preset, refusals), start sampling (the base
mix weights within 3 points over 4000 draws, lead odds about half the games, per-seat move times), cost floor, lag
and jitter, decisions and landings, elapsed time reaching the search clock, pass length and wake-up, the pass cap,
flag, double flag, no flag without FlagLoss, sit-off adjudication both ways, and that without DurativeTree the search
clock is v3's clocked state at the clocks' lead.
`engine/tests/test_selfplay_wave2.cc`: `run_selfplay` with the preset on the uniform net (3 games, 4 s base): one
`seat_clocks.tsv` row per HVM6 record, game-line sample counts that add up, the PGN tags, a flag game; tree options
without a base mix keep the lead-only game; `run_tournament` plays a seat-clock pair.

`engine/tests/stock_parity.sh` (acceptance 7, binary): builds this engine and stock v3.0.3
(`third_party/upstream-hivemind`, built out of tree, source untouched) with one search worker, then runs `policy` and
`go nodes N` on `engine/tests/data/stock_parity_positions.txt` (8 positions, both teams, leads -15 s..+15 s, unequal
move times, pockets) with default options. It compares the whole output except wall-clock fields (time, nps, ms,
percentages) and the intermediate `info depth` progress lines, which are printed when the reporting thread polls and
so differ between two stock runs too; the last `info depth` line (nodes, score, PV of the finished search), the
`policy` output and `bestmove` are compared exactly. The time-boxed root mate scans are off in both because their
results depend on wall time. Result on 2026-11-19: **8/8 identical**. Build cost: two extra builds
(`engine/build-parity`, `engine/build-parity-stock`, git-ignored), a few minutes at `JOBS=2`.

Upstream's own suite stays green: 498 tests, 497 pass, 1 skipped (upstream's `MeasureFrozenRaceOnTheGame` perf
test, skipped on the stock baseline too). `stock_parity.sh` after part 2: **8/8 identical**.

## Open after part 2

- Strength of wave 2 against stock twin-s-clock is unmeasured: only smoke runs.
- Root provers are off under MateRaceByLanding (they reason in v3 alternation). Landing-aware provers would win back
  their speed on mates.
- At the root, the root team still decides first and its other seat (if on turn) is part of the root action, even
  when in a seat-clock game that seat is mid-decision; self-play plays only the deciding seat's board.
- (Fixed in `d538648`: two seats of one team could pass in 1 ms steps for ever; passes now wake only on landings, as
  in the game model.)
- The root's early opponent landing is dated back for clocks, the board's time since landing and the landing history,
  but the root team's own action is still played first in the tree (mate races aside, `MateRaceByLanding`).
- A wait node costs one network evaluation (part 1 note).
- `seat_clocks.tsv` is not trimmed by `--resume` (checkpointed runs append the replayed games again).

## Wave 2b: wide planes and the arrivals head

Twin-s-clock it13 widened by 26 constant planes per board and an arrivals head, zero-initialised so the widened
network answers exactly as it13 whatever the new planes hold (the network side, `twin-s-clock-wide*`, is
`src/hivemind/wide.py` and `docs/wave2_training.md`). One binary runs both: the engine picks narrow or wide from the
ONNX input width at load (78 or 130 channels; 74, the legacy layout, still works).

**Layout** (`environment/planes.h`, `wide_extra_planes`, `planes.cc:307`). Each board's block is its 39 planes, then
26 more, the same order in both blocks, "our"/"their" from the searching team's view on that board (block B: our
partner's seat and pocket):

| Offset | Value (constant plane) | Source in the tree |
|---|---|---|
| +0, +1 | our seat clock: `min(t/300, 1)`, `exp(-t/10)` (t s) | the node's per-seat clock |
| +2, +3 | same-board opponent's clock, the same | the node's per-seat clock |
| +4, +5 | thinking time of the side on turn there: `min(e/10, 1)`, `1 - exp(-e/2)` | ms since a move last landed on that board |
| +6..+10, +11..+15 | our / their pocket holds >= 1 of P N B R Q | the board's pockets (always set) |
| +16..+20, +21..+25 | our / their pocket: P N B R Q that landed in the last 5 s, `min(n/2, 1)` | the node's landing history |

The clock, thinking and landing planes need a **wide durative state** (`TimeState::wide_timing`,
`time_state.h:612`): `DurativeTree` on, a wide network loaded, and real `SeatClocksMs` (clocks derived from a lead
count as no timing). Otherwise those planes are 0 and only the presence planes are set ("no timing given"). Planes
37/38 (lead, speed) and everything else of the narrow block are unchanged.

**Tree** (`time_state.h`). `DurativeConfig` gains `wideInputs`, `boardElapsedMs[2]` and `recentLandings` (up to
`MAX_RECENT_LANDINGS` = 12 `SeatLanding`s, seat, piece type, ms ago). With `wideInputs` the durative block carries
per-board time since the last landing and the recent landings (`time_state.h:223`): every action ages them by its
`dt` and drops landings 5 s old (`age_landings`, `time_state.h:825`); a move resets its board's time; a capture's
piece is added to the partner's pocket at age 0 (`add_landing`), whatever `InstantTransfer` says (the planes record
arrivals; delayed transfer only governs when the piece may be dropped). A pass is not a landing: its board's thinking
time keeps running. These fields enter the node key only when `wideInputs` is on (`time_state.h:750`), so a narrow
durative tree's states and keys are exactly wave 2's (tested). The v3 `advance()` path (provers, reuse) drops the
durative block as before, so wide timing is 0 there.

**Engine** (`nn/engine.h`). Callers still fill the 78 narrow planes per position and hand the 2 x 26 extra values
beside them (`enqueueInferenceHalf(obs, worker, wideExtras)`, also `runInference`/`runInferenceHalf`); for a wide
network the backend builds the 130-plane input into its staging buffer (`widen_planes`, `constants.h:67`; ONNX
Runtime `engine_ort.cc:446`, TensorRT `engine.cc:945`), a narrow network ignores the extras, so it13's path is
untouched. The search thread fills a `wideExtras` buffer per batch slot only when its engine is wide
(`searchthread.cc:972`, padding rows copied too, `:999`). **Arrivals head**: an output whose name contains
"arrival" (20 values) is bound and copied back (`HalfInferenceOutputs::arrivals`, `Engine::hasArrivals`); the search
does not read it. The UCI `policy` command prints `Arrivals: ...` when the network has it. HVM6/HDST records keep the
narrow 78-plane encoding.

**UCI.** `SeatArrivalsMs` (string, `"<pocket>:<piece>:<ms_ago>,..."`, pocket `Aw Ab Bw Bb`, piece `P N B R Q`
either case, spaces ignored, empty clears; `parse_seat_arrivals`, `planes.cc:341`): recent landings at the root.
Landings 5 s old or more are dropped; past 12 the newest are kept (the `seat_clocks.tsv` `recent` counts that
training reads keep the same newest 12, review fix `79d21fa`; the arrivals labels count every landing); a bad item is refused with an `info string` and
keeps the old value. Board thinking time at the root is the `SeatElapsedMs` of the seat on turn on each board (ours
included, unlike the ready-time use). `networkWide` (`uci.cc:85`) sets `wideInputs` in `current_clock()`
(`uci.cc:780`); a wide network also counts as reading the clock planes (TimeModel clock by default). With it13
`SeatArrivalsMs` is accepted and has no effect.

**Self-play and tournament.** `SeatClockGame` keeps a landing log (`landed(seat, capture, hand)`,
`seat_clock_game.h:369`: the captured piece goes to the mover's partner, a promoted piece as a pawn) and each board's
last landing time. `set_wide_inputs` (from the engine's width; the tournament sets it per side) makes `search_clock`
pass `boardElapsedMs` and the last 5 s of landings into the tree. The log is kept for narrow networks too, so it13
self-play writes the wide sample fields. `seat_clocks.tsv` gains, after `pace_bb`:
- `board_elapsed_a`, `board_elapsed_b`: ms since the last move landed on A / B (since the start before any);
- `recent_<pocket>_<piece>` (20): pieces that landed in the last 5 s before the record's search, `fromMs < t <= now`;
- `arrivals_<pocket>_<piece>` (20): the arrivals label, pieces landing in `(now, now + 5000]`, written at game end
  (a game's end cuts the window short);
with pocket `ours_a, theirs_a, ours_b, theirs_b` from the record team's view and piece `p n b r q`
(`pocket_landing_counts`, `seat_clock_game.h:535`). Old `seat_clocks.tsv` files lack these columns: do not append a
wave 2b run to a wave 2 one. The network side moves them into HDST v3 chunks (`hivemind.distill.timing`).

**Plane dumps.** `hivemind dumpplanes --wide-planes true` writes N x 130 x 8 x 8; a line may add
`;<Aw> <Ab> <Bw> <Bb>;<ea> <eb>;<SeatArrivalsMs>` and is then encoded on a wide durative clock rooted for the team
(lead from the clocks, sit plane 31 = lead > 0 from those clocks, both teams at 100 ms a move).

**Arena** (main repo `arena/players/upstream.py`). The wave2 player sends `SeatArrivalsMs` every search:
`PocketLandingTracker` diffs the pockets in the FENs against its previous search, dates each new piece by the other
board's last landing (`last_move_ms`, else now) and keeps 5 s. It sees only our own decisions, so two captures on the
other board between them both get the later time, and a capture cancelled by a drop of the same piece in between is
missed. Run the wide network with `WAVE2_MODEL=.../engine/models/twin-s-clock-wide-fp32.onnx`; the default is still
it13.

### Wave 2b tests

`engine/tests/test_wave2b.cc` (13 tests; `--gtest_filter='Wave2b*'`):
- plane values on a hand-built position with pockets on both boards, for the white team and the black team (every
  one of the 52 values checked); no timing (v3 clock, legacy flag, narrow durative state, derived clocks): only the
  presence planes; the wide layout is each 39-plane block followed by its 26;
- the tree: a capture lands in the partner's pocket at age 0, landings age and leave at 5 s, a move resets its
  board's thinking time and a pass does not; landings separate wide node keys and never narrow ones (a narrow state
  with landings in its config equals, and keys as, one without, also after an action);
- `pocket_landing_counts` (the head's order, the half-open window) and the seat-clock game's landing log reaching a
  wide search clock;
- engine: `engine/tests/data/wide_probe.onnx` (130 inputs, zero heads, `arrivals` echoes 20 of the extra planes;
  `make_wide_probe.py` writes it) shows the exact extras reach the network per batch row, and zero planes without
  extras; a narrow network's outputs are bit-identical with or without extras (it13 when present); a durative
  search runs on the wide probe;
- **widened it13 = it13 through the engine**: `twin-s-clock-wide-fp32.onnx` vs `twin-s-clock-fp32.onnx` and
  `twin-s-clock-wide.onnx` vs `twin-s-clock-it13.onnx` (skipped if missing), on contract planes and on random
  [0, 1] extras: largest difference over value, policies, WDL and moves left **0**, arrivals exactly 0. (The
  published `twin-s-clock.onnx`/`-it13.onnx` keeps FP16 weights; compare FP32 with FP32.)
- self-play on the wide probe writes `seat_clocks.tsv` with all 59 columns, thinking times within the game, counts
  non-negative (3 games: 199 rows, 381 recent and 576 arriving pieces counted).
`engine/tests/test_uci.cc` `Wave2bSeatArrivalsReachAWideDurativeClockOnly`: parsing, the 5 s window, refusals,
clearing, and no effect with a narrow network.

Checks run 2026-11-19 (ONNX Runtime on the Mac):
- full suite 514 tests: 513 pass, 1 skipped (upstream's perf test, as before);
- `engine/tests/stock_parity.sh`: 8/8 identical to the stock v3.0.3 binary at defaults;
- C++ `dumpplanes --wide-planes` vs Python `board2planes_wide` (`hivemind.domain.wide_planes`) on 300 random
  positions with random pockets (231 with timing and landings): the 52 extra values identical (max difference 0)
  and the board planes 0..36 identical except the sit plane 31 on timed lines, which the C++ dump takes from the
  clocks' lead and the Python call from its flag (the script was a scratch check, not committed);
- UCI: the same wave 2 options and `SeatArrivalsMs` on it13-fp32 and the widened fp32 give the same Value, WDL and
  bestmove at 800 nodes; the widened net prints `Arrivals:` all 0;
- arena: 1 pair at 10 s, rules v3, wave2 on the widened fp32 vs upstream-it13: 2 games, 0 errors.

### Open after 2b

- TensorRT: the widening, the arrivals binding and its copy are written for `engine.cc` but not compiled or run on
  this Mac; build on PACE before relying on it (`cluster/build.sh`). The TensorRT parser accepts an "arrivals"
  output only with exactly 20 values per position.
- A trained wide network has not been searched with yet; with zero weights the new planes cannot change a search
  (except that wide node keys split transpositions differing only in landing history or thinking time).
- The arena's landing times are inferred at our decisions only (see Arena above); the arena's Observation has no
  capture log.
- Thinking time is per board, not per seat: a seat that comes back from a pass keeps the board's time since the last
  landing, as the contract defines `e`.
- The root mate provers and tree reuse still run on the v3 clock, where wide timing is 0.

## Wave 2c: continuity

Spec: talisman `docs/custom_engine_spec_v5_continuity.md`. Three fixes, each a UCI option and a self-play flag, off
by default; with them off the engine is bit for bit the `261081b` one (`engine/tests/stock_parity.sh` and
`engine/tests/wave2_parity.sh`, 8/8 each). Fix 1 is built; fixes 2 (joint plans) and 3 (plan carry) are not yet.

### Fix 1: team tree reuse (`TeamTreeReuse`, `TeamTreeReuseToleranceMs`)

| UCI option | Self-play / tournament flag | Default |
|---|---|---|
| `TeamTreeReuse` (check) | `--team-tree-reuse <bool>` | false |
| `TeamTreeReuseToleranceMs` (spin 0..5000) | `--team-tree-reuse-tolerance-ms <ms>` | 250 |

Only under a durative clock (`DurativeTree true` under `TimeModel clock`). With `DurativeTree` off the option is
accepted and does nothing (`setoption` says so). The self-play and tournament flag needs `--durative-tree true` and
`--base-mix` (seat-clock games); `validate()` refuses it otherwise. Python: `hivemind.wave2.Options.team_tree_reuse`,
`team_tree_reuse_tolerance_ms`; `rl-loop --team-tree-reuse true` passes both to `selfplay` and `tournament`.

What it does (`Agent::run_search`, `agent.cc`):

* **Keep.** A durative search with the option stores its whole tree (`Agent::TeamTree`: root, a copy of the root
  board, both FENs and `game_ply`s, the root clock, the team). It does not index exact-hash candidates and keeps no
  single-board mate continuations: the next search of the team starts with nothing else from this one.
  `ucinewgame`, `reset_search_state()` and a missed ponder drop it; so does a search with the option off.
* **Find** (`find_team_tree_node`). The moves that landed since are the board difference, so the walk follows
  them: an edge is followed while no board has moved more often along the path than in the game (`game_ply`
  offsets from the old root), and a board that has moved as often shows the live FEN (counters left out: the
  arena sends FENs) with a pocket at most the live one. Waits and passes are followed. Each node's time is the
  durative charge the search used (`TimeState::after_action`). Depth-first with make/unmake, at most 8 joint
  plies and 20000 nodes; the board test leaves only the played line and its wait nodes.
* **Adopt** a node with the live boards and our team to play that is expanded and unsolved, in the live turn
  state (`TimeState::same_turn_state`: sides, sitting and held-back pieces on turn, rules), within the tolerance
  (`TimeState::clock_distance`: every seat clock and the other team's on-turn ready times), and expanded as the
  live root would be (its expansion key under the live clock: at the root our team decides on every board it is
  on turn). Smallest clock difference first, then most visits.
* **Re-time** (`retime_team_tree`). A breadth-first count keeps the shallowest 100000 nodes; a depth-first walk
  carries each path's time down from the live clock and replaces by a fresh node (with its path's hash) any child
  beyond the cap, reached again through another parent with another hash (the tree is an MCGS graph: a split),
  solved with a changed hash (its time state moved: `rootMoveMs` and the paces are in the key, so a new budget
  drops every verdict below), or expanded with another expansion key. A replaced child's edge goes back to a new
  edge (no visits, `Q_INIT`) and its visits and value come off the parent's totals (`Node::reset_edge_for_reuse`);
  ancestors keep theirs. Every node kept gets its recomputed hash and goes into the fresh transposition table.
  The adopted root's reverse edges are cleared. The rest of the old tree goes to the GC thread.
* **Expansion key** (`durative_expansion_key`, `searchthread.cc`; stored in `Node::expansionKey_` at expansion):
  the acting boards after the durative filter, a forced wait, the sit right the generator gets, the drops held
  back on the acting boards (InstantTransfer off), and the time inputs of the terminal check (the other team's sit
  right, a clock result, and with MateRaceByLanding the landing order of the seats on turn, the early landings and
  the last capturer). `collect_batch` now builds a leaf's lists through `leaf_turn` and `leaf_board_actions`, which
  the tests use too. `sizeof(Node)` is 1040 bytes before and after (the key sits in padding after `teamToPlay`).
* **Budget.** UCI `go nodes N` searches N new nodes on top of the adopted ones. Self-play tops the root up to its
  budget (`reusedVisitsCount`, at least `--tree-reuse-min-new-fraction` new); tournament likewise (at least a
  quarter new).
* **UCI** does not start the permanent brain after a search that keeps a team tree (it would re-root with v3's
  `advance()`). Opening noise still resets the agent before the search, so it drops the tree.
* **Telemetry.** Verbose UCI prints `info string Team tree reuse: <visits> visits recovered, clock diff <ms> ms,
  <pruned> pruned`, or `... none (<reason>)` with `no tree`, `no node`, `clock`, `shape`, `solved` (followed by
  `, clock diff <ms> ms` when some node had the live boards). The exact path's `Tree reuse:` line is never printed
  under the option. Self-play prints `selfplay team tree reuse: <hit %> of searches (<hits>/<searches>), <mean>
  mean visits recovered, by seat Aw .. Bb .., refused <reason> <count> ...`.

**Hit rate seen so far** (self-play only; the arena gate, `tools/continuity_probe.py reuse`, is still to run):

| Network, nodes, base | Tolerance | Searches adopting | Refused (no node / shape / clock) |
|---|---|---|---|
| uniform test net (`build-ninja`), 256 nodes, 20 s, 4 games | 250 / 5000 ms | 2.3% (5/213) / 3.8% (8/213) | about 87% / 3-4% / under 1% |
| it13 fp32 (`twin-s-clock-it13-fp32.onnx`), 400 nodes, 60 s, 2 games (`build-sp1`) | 250 ms | 28.4% (54/190) | 55 / 56 / 19 |
| same | 1000 ms | 29.9% (38/127) | 38 / 38 / 3 |
| same | 3000 ms | 33.3% (42/126) | 39 / 40 / 0 |

"no node" is mostly structural, not a tolerance question: the tree holds the other team's landings only in the
order their paces predict (a durative node has one timing, not a choice), so when the real order differs no node
has the live boards; and in self-play a seat of ours whose move is in flight acted at the old root (the root acts
on every board our team is on turn), so the tree's line on that board is not the game's.

Tests: `engine/tests/test_continuity_fix1.cc` (spec v5 0.1 and 1.1-1.7, 1.9-1.15, 1.17, plus UCI `ucinewgame` and
option-off cases) and `SelfPlayWave2Test.TeamTreeReuse*` in `test_selfplay_wave2.cc` (1.8). Not yet: 1.16 (needs
fix 2's committed moves).
