# Wave 2 training (talisman fork)

Wave 2 continues **twin-s-clock iteration 13** by self-play on four real seat clocks, with the time model in the tree
(spec v4, the durative tree) and upstream's hard sit gate (`can_sit = lead > 0`) switched off: a pass is always
legal and the search prices it. The network's inputs do not change (twin-s-clock planes 37/38 = lead and speed per
board, computed by the engine from the node's clocks), so it13 plays wave 2 from the first game.

## Start network

| File | Source | SHA-256 |
|---|---|---|
| `twin-s-clock-it13.onnx` | HF `aminwoo/bughouse-twin-s-clock` revision `edaa246cdb86c43bb3f18f0836501912c080e7f1` | `f4ac25e32a993776f26b42d79cc5c1e48b77a16a13d1e2a2db30528f43c9af73` |
| `twin-s-clock-it13.pt` (training checkpoint) | same revision (identical to `twin-s-clock.pt` there) | `5bb43c7f79bcc4cf0af2d2d0a88e7b60cf886312f3617c938f8e5a21ec0730bc` |

Both are pinned in `src/hivemind/network.py` (`clock-it13`, `clock-it13-checkpoint`). The checkpoint is `twin-s` with
`value_diff`, no attention, no ECA, 39 planes per board. Checked 2026-11-19 on this Mac: the checkpoint run in PyTorch
against the it13 FP32 ONNX on 8 real positions (random legal walks, random lead/speed planes) gives max |diff|
value 4.5e-4, policy logits 0.083 (logit range 37), WDL logits 3.5e-3, moves-left 1.9e-4, top-1 8/8: the FP16 rounding of
the published ONNX. So the checkpoint is the network, and it is trainable.

```bash
uv run python -m hivemind.wave2 init            # engine/models/twin-s-clock-w2.onnx (generator),
                                                # engine/models/twin-s-clock-w2-it0.onnx (fixed baseline = it13),
                                                # artifacts/distill/twin-s-clock-w2/best.pt (generator checkpoint)
uv run python -m hivemind.wave2 options         # the preset's engine options
```

`init` downloads with SHA-256 checks and never overwrites a lineage file that exists (a promoted generator stays).

## The loop

```bash
uv run hivemind rl-loop --wave2 --no-upload --keep-selfplay --runs data/distill/runs-w2 \
    --generator engine/models/twin-s-clock-w2.onnx \
    --generator-checkpoint artifacts/distill/twin-s-clock-w2/best.pt \
    --selfplay-engine engine/build-sp1/hivemind --match-engine engine/build-ninja/hivemind
```

`--wave2` sets, on top of upstream's loop (which stays the default without it):

| Setting | Wave 2 | Upstream | Why |
|---|---|---|---|
| engine options | the preset below, to self-play, validation and both matches | none | the rules being trained |
| `--time-model` | clock | flag | the network reads lead/speed |
| `--train-from` | `generator` | `previous` | an unpromoted candidate is not where the next iteration starts (audit: promotion rule, next start = better of the two) |
| `--lr` / `--trunk-lr` | 1e-4 heads / 5e-5 trunk | 1e-4 all | fine-tune of a converged network on shifted rules; theory: trunk <= 1e-4. `distill-train --trunk-lr` puts `policy_head.*` and `value_head.*` in one AdamW group, the rest in the other; warmup and cosine scale both |
| `--promote-above` | 0.5 (strictly better) | score >= 0.5 | "keep the better network": a tie keeps the generator |
| `--label-check` | on | on (`--label-check off` skips it) | value-label hygiene, below |

Gate: each iteration plays `--match-games` (default 1,000) vs the generator and, while the generator is still it13, the
same match counts for the fixed baseline (`-it0` = it13; not replayed). Once a candidate is promoted, the fixed
match vs it13 is played every iteration. Both at `--match-movetime` ms per move for both sides (equal clock), both
networks under the same wave 2 options. The headline numbers vs **stock** v3 (it13 on stock v3.0.3 rules, the arena's
`upstream` kind, at 120 s and 180 s) are the arena's job (`cluster/launch_checks.sh` wave2 group), not the loop's.

Ablations: every option has its own flag (`--durative-tree false`, `--sit-gate true`, `--instant-transfer true`,
`--mate-race-by-landing true` (off in the preset: it switches off the root mate provers and cost drop mates with the gate off, talisman docs/wave2_local_check.md), `--pass-cap-k N`, `--base-mix ...`, `--seat-move-times false`). Without `--wave2` the
same flags apply on top of stock v3, so `--durative-tree true` alone is a one-option run.

`--distributed` is refused with wave 2 options: work orders do not carry them to `hivemind contribute` yet.

## Value-label hygiene (audit training 4)

After each self-play segment (and the validation games), `hivemind.distill.labels.label_report` reads the segment's
chunks against its log's `selfplay game` lines and fails the stage (`StageFailed`, retried, then the loop stops) when:

* the chunk records do not add up to the games' `samples`: something dropped or deduplicated records. There is no
  deduplication anywhere in this training path (upstream has none; the loader keeps every row), and the check keeps it
  that way: a pass followed by a move in the same position is two records, both trained on. The report counts them
  (`repeat_records`);
* a finished game has a record without a one-hot WDL, or a WDL that disagrees with its outcome;
* an unfinished game (`macro-ply limit`) carries a result;
* a `flag` game has a draw outcome: a flag is a loss for the flagged team (and a win for the other);
* a `draw` / `double_flag` game has a decisive outcome.

Counts (games, records, flag games and records, repeated positions, terminations) go into the segment's stage marker
`stages/segNN.done` under `labels`, with the options under `wave2`.

Training targets are unchanged from upstream: policy from root visits, WDL and moves-left from the game result,
`--search-value-weight 0`. So a flag game teaches its result exactly like a mate.

## Interface the wave 2 engine must provide (engine part 2 / integration)

The loop passes these to `hivemind selfplay` **and** `hivemind tournament` (the match command), as `--flag value`.
It reads the engine's `--help` and refuses to start (`--allow-stock-engine` plays stock games instead) unless every
flag it would send is listed there, so **list each one in `printUsage`**. Only options that differ from stock are
sent; with none, the command line is upstream's exactly.

| Flag | Value | Meaning | Stock |
|---|---|---|---|
| `--durative-tree` | `true`/`false` | UCI DurativeTree in every search | false |
| `--sit-gate` | `true`/`false` | UCI SitGate. `false`: pass always legal in the tree and the game | true |
| `--flag-loss` | `true`/`false` | UCI FlagLoss in the tree; in the game, a seat at 0 ends it (below) | false |
| `--instant-transfer` | `true`/`false` | UCI InstantTransfer | true |
| `--mate-race-by-landing` | `true`/`false` | UCI MateRaceByLanding | false |
| `--pass-cap-k` | 1..64 | UCI PassCapK | 8 |
| `--base-mix` | `MS:W,MS:W,...` | per game, draw a base clock with these weights; all four seats start at it (the existing lead sampler `--level-start-fraction`, `--start-lead-*` then shifts it: white team's seats +lead/2, black team's -lead/2, so lead_ms(WHITE) is the drawn lead) | absent: no seat clocks, v3's lead-only clock |
| `--seat-move-times` | `true`/`false` | each seat draws its own move time from `--move-time-min-ms..--move-time-max-ms` (log-uniform), used as its pace (SeatPaceMs) and as what its actions cost on its clock | false: per team, as v3 |

Game rules with `--base-mix`: every action costs the acting seat its move time on its clock (a pass costs it until
its next decision, the durative tree's pass length); the search gets `SeatClocksMs`, `SeatPaceMs` (observed pace, or
the drawn move times) and `SeatElapsedMs` as in the UCI contract. A seat at 0 ends the game at once, termination
**`flag`**, that seat's team loses (`winner` = the other team), so `record.outcome` is -1 for the flagged team's
records and +1 for the other's. Two seats of different teams at 0 in the same step: the one with less time at the
start of the step loses; exactly equal: termination **`double_flag`**, a draw (arena `RULES.md`).

Output the loop depends on (keep it):

* stdout, one line per game: `selfplay game N/M raw R samples S events E termination T` followed optionally by
  ` lead ...`, ` flag-seat <Aw|Ab|Bw|Bb>`, ` clocks <Aw> <Ab> <Bw> <Bb>` (ms at the end), ` resign-check ...`. The
  termination may contain spaces; it ends at one of those keys or the line's end. `S` = the records the game wrote
  (one per search), in game order.
* PGN `Termination "flag"` / `"double_flag"`, and when seat clocks are on: `[BaseMs "..."]`,
  `[SeatMoveTimesMs "<Aw> <Ab> <Bw> <Bb>"]`, `[FinalClocksMs "<Aw> <Ab> <Bw> <Bb>"]` (for analysis; not parsed by the loop).
* HDST v2 chunks unchanged: 78 planes, the clock planes from the node's clocks.

`tournament` takes the same flags (both sides), with `--movetime` as each side's search budget; under `--base-mix`
the movetime also stands in for each seat's move time unless `--seat-move-times true` draws them.

## Local smoke (Mac CPU)

Before the wave 2 engine exists, the smoke runs on a stock v3 engine (`--allow-stock-engine`), so it checks the loop,
labels and training, not the rules:

```bash
uv run python -m hivemind.wave2 init
uv run hivemind rl-loop --wave2 --allow-stock-engine --no-gate --no-upload --keep-selfplay --last 1 \
    --runs /tmp/w2-smoke --games 4 --segment-games 4 --val-games 2 --parallel-games 2 --checkpoint-games 0 \
    --batch-size 256 --pause-for "" \
    --generator engine/models/twin-s-clock-w2.onnx --generator-checkpoint artifacts/distill/twin-s-clock-w2/best.pt \
    --selfplay-engine <engine> --match-engine <engine>
```

With the wave 2 engine built, drop `--allow-stock-engine`. The batch must not exceed the new positions (a few hundred
from 4 games): `distill-train` only draws whole batches and would wait forever, so the loop refuses that case.
`distill-train` runs on the CPU there (no CUDA, `OMP_NUM_THREADS=2`).

Run 2026-11-19 on this Mac (stock v3.0.3 binary `third_party/upstream-hivemind/engine/build-ninja/hivemind`, Core ML,
it13 as generator): 4 + 2 games in 20 s, label check clean (348 + 86 records, 6 checkmates, no flags as no seat
clocks), 3 training steps from it13 (trunk 5e-5, heads 1e-4) in 40 s with search validation loss 3.6366 -> 3.6339,
ONNX exported. Exit 0. Not run: matches (`--no-gate`), any wave 2 option (the engine had none yet).

## Wave 2b: the wide network and the arrivals head

Wave 2b widens it13's inputs and adds an auxiliary head, so the network can see what wave 2's rules put in play
(each seat's own clock, how long the side on turn has been thinking, which pieces just landed) and learn piece flow.
Everything new starts at zero: the widened network computes exactly what it13 computes, whatever the new planes hold,
and the engine plays it exactly as it13 until training moves the new weights.

### Planes (`src/hivemind/domain/wide_planes.py`)

Each board block grows from 39 to 65 planes (block A 0..64, block B 65..129): it13's 39 (37 board planes, lead,
speed), then 26 constant planes in the same order in both blocks, "our"/"their" from the searching team's view on
that board (block B: the partner's seat and pocket):

| Offset | Plane | No timing |
|---|---|---|
| +0, +1 | our seat clock: `min(t/300, 1)`, `exp(-t/10)` (t in s) | 0, 0 |
| +2, +3 | the same-board opponent's clock, same two | 0, 0 |
| +4, +5 | the side on turn on this board has been thinking e s (since the last action landed here): `min(e/10, 1)`, `1 - exp(-e/2)` | 0, 0 |
| +6..+10 | our pocket holds at least one P, N, B, R, Q (0/1) | always set |
| +11..+15 | their pocket, the same | always set |
| +16..+20 | our pocket: P, N, B, R, Q that landed in the last 5 s (`0 <= ms_ago < 5000`), `min(n/2, 1)` | 0 |
| +21..+25 | their pocket, the same | 0 |

Seats from the team's view: the white team (Aw + Bb) has ours/theirs = Aw/Ab on A and Bb/Bw on B; the black team
Ab/Aw and Bw/Bb. `board2planes_wide(board, team, lead, speed, timing)` encodes a `BughouseBoard` (the reference the
C++ `wide_extra_planes` is checked against); `extend_planes` does the same for a batch of narrow plane values plus
the chunk's timing fields. `parse_seat_arrivals` reads the UCI `SeatArrivalsMs` value
(`<Aw|Ab|Bw|Bb>:<P|N|B|R|Q>:<ms_ago>,...`).

Cross-check against the C++ encoder (`hivemind dumpplanes --wide-planes true`, engine `build-ninja` built 11:16 on
2026-11-19), 300 random positions, half with seat clocks, board elapsed times and 0-6 landings aged 0-8 s: the 26
extra planes of all 600 blocks equal (max |diff| 0). `test_python_wide_extras_equal_the_engines` repeats it on 60
positions when an engine is built. (The board planes were not compared: the FEN lines carry no move history, so the
last-move, en passant and sit planes differ by construction.)

### Network (`architectures/twin_board.py`, `src/hivemind/wide.py`)

* `TwinBoardNet(planes_per_board=65, arrivals=True)`. `widen_to_wide_planes` appends 26 zero input columns per block
  to the stem conv; `load_twin_board_checkpoint(..., wide=True)` widens a narrow checkpoint, and a wide checkpoint
  (variant `planes_per_board: 65`) stays wide on every later load.
* Arrivals head: each board's pooled trunk features with the other board's -> Linear(256, 64) -> ReLU -> Linear(64, 10)
  (ours P..Q, theirs P..Q), shared by both boards, so swapping the boards swaps the halves (tested). Output
  `arrivals` [batch, 20] = ours A, theirs A, ours B, theirs B x P, N, B, R, Q: pieces landing within the next 5
  simulated seconds. Last layer zero. It is the sixth PyTorch output and the last ONNX output; the engine matches
  outputs by name and does not read it.
* `python -m hivemind.wave2 widen` makes the files from the pinned it13 checkpoint and records them in
  `engine/models/twin-s-clock-wide.json`:

| File | SHA-256 |
|---|---|
| `artifacts/distill/twin-s-clock-wide/twin-s-clock-wide.pt` (training checkpoint) | `ab834c1e3ae7b4a24ab2fbd23097b50d69597af1278437bc79fe48b7e0e6232d` |
| `engine/models/twin-s-clock-wide.onnx` (FP16 weights, FP32 I/O, as it13 is published) | `42588f4cb00831eff34c7af80c836d5ae497d6a6f1bc006b6bbec2cad41d97ad` |
| `engine/models/twin-s-clock-wide-fp32.onnx` (that file through `engine/scripts/convert_onnx_fp32.py`, as upstream's FP32 copy) | `942606d56449de2e0144c9b5fb446418384017a4e753a200b4c1ad2eb80e3f24` |

  Widened from `twin-s-clock-it13.pt` (`5bb43c7f...`); the .pt is deterministic (seed 0 for the head's hidden
  layer), so a rerun gives the same hashes.
* Parity, 2026-11-19 on this Mac, 300 random legal positions (random walks of up to 80 actions with drops), random
  lead/speed planes, the new planes empty for a third, from random timing for a third and uniform noise in [-2, 2]
  for a third. Max |diff| on value, pi_a, pi_b, wdl_out, moves_left, and arrivals max |value|: **0 on every output,
  all four ways**: PyTorch it13 vs PyTorch wide; it13 vs wide exported to FP32 ONNX by the same exporter (ONNX
  Runtime CPU); the wide FP16 ONNX vs the published `twin-s-clock-it13.onnx`; the wide FP32 ONNX vs the published
  FP32 copy (`efe731c0...`). `tests/test_wave2b_wide.py::test_it13_widened_equals_it13_on_300_positions` repeats
  the PyTorch check (tolerance 1e-5) whenever the checkpoint is downloaded.

### Samples: HDST v3 and `seat_clocks.tsv`

The engine still writes each search as a narrow (78-plane) HDST record; for seat-clock games it writes a
`seat_clocks.tsv` row per record in the same order (`selfplay.cc` `write_seat_rows`), which for wave 2b carries
`board_elapsed_a/b`, `recent_<pocket>_<piece>` (landed in the last 5 s) and `arrivals_<pocket>_<piece>` (the label:
landed in the 5 s after the search started, `from < t <= to`, written at game end), pockets ours_a, theirs_a, ours_b,
theirs_b from the record team's view. `hivemind.distill.timing.attach_timing(log, seat_clocks.tsv, chunks)` moves them
into the chunks as **HDST v3**: the v2 layout with version 3 and a trailing block per record, team-relative in block
order:

| Field | Type | Meaning |
|---|---|---|
| `seatClocksMs` | f32[N x 4] | ours A, theirs A, ours B, theirs B; < 0 = unknown |
| `elapsedMs` | f32[N x 2] | block A, block B: ms since the last action landed there; < 0 = unknown |
| `recentLanded` | u8[N x 20] | landed in the last 5 s, the arrivals order |
| `arrivals` | f32[N x 20] | the label; `arrivals[0] < 0` = no label |

It checks records = tsv rows = the game lines' samples (else the stage fails), keeps the last copy of a game's rows
when a resumed run appended them twice, and gives the records of a game cut at the macro-ply limit no arrivals label
(the window was cut short). `read_chunk(path, layout="wide")` widens a 78-plane chunk from the pocket planes and that
block (a v2 chunk gets presence planes only); a 130-plane v3 chunk's own planes are used as written, and
`layout="narrow"` cuts them back, so a narrow lineage trains on the same chunks. `chunk_problem` also rejects
arrivals labels outside [0, 16].

### Training

* Loss: weighted MSE over the 20 counts, per type P 1, N 2, B 2, R 2, Q 4 (the same in all four pockets), divided by
  the weights' sum (44); only rows with a label; overall weight `distill-train --arrivals-weight` (0.05). The head
  trains at the heads' learning rate (`--lr`), and so do the stem's new columns (the 26 zero-initialised input planes
  per board): they sit inside the trunk's stem tensor, so after each AdamW step `NewInputRate` scales their update
  by `lr / trunk_lr`, which is exactly a step at the heads' rate (AdamW's step, weight decay included, is
  proportional to the rate). `distill-train --new-inputs-lr trunk` restores the old behaviour (they trained at the
  trunk's 5e-5 until review fix, which leaned the narrow/wide A/B toward "no difference").
* `distill-train` reads the layout the network reads (`--init` a wide checkpoint, or `--wide` to widen a narrow one).
  Export adds the `arrivals` output and can write an FP32 copy.
* `rl-loop --wide` (with `--wave2`): the generator must read 130 planes (and a 130-plane generator needs `--wide`);
  after each segment's label check the loop runs `attach_timing` (marker field `timing`) and training gets
  `--arrivals-weight` (`rl-loop --arrivals-weight`, 0.05). Value and flag label hygiene is unchanged.
* `python -m hivemind.wave2 init --wide --name twin-s-clock-w2w`: generator and checkpoint = the widened it13 (made
  first when missing), fixed baseline `-it0` = it13 itself, so the narrow and wide lineages are gated against the same
  network.

### Local smoke, 2026-11-19 (Mac CPU)

`rl-loop --wave2 --wide --no-gate`, 4 + 2 games, batch 128, the wave 2b engine (`engine/build-ninja`, built 11:16 from
the engine agent's commits 86e0029/d386893) for self-play, generator the widened it13: 4 + 2 games (all checkmates),
821 + 364 records, both chunks to v3 with arrivals labels on every record (mean 3.6 pieces landing per record within
5 s; P 1131, N 836, B 525, R 253, Q 186 over 821 records), 10 training steps (search arrivals loss 0.2359 -> 0.2344,
search loss 4.174 -> 4.219 at 10 steps, nothing to read into), exported ONNX with 130 inputs and the `arrivals`
output, checkpoint variant wide. Wall time 2 min 20 s. Not run: matches, any GPU step.

### A/B and cost

`cluster/wave2_iterate.sbatch` (main repo) has `WIDE=1` (a wide lineage) and `AB=1` (narrow and wide lineages,
same settings, same seeds: rl-loop seeds segment k of iteration n with 1000 n + k). Both gate against it13 at equal
movetime under the wave 2 rules; the headline against stock v3 rules is the arena's.

Reading `results.tsv` / `results_ab.tsv`: the `vs_it13` and `elo_vs_it13` columns are each iteration's gate, 400
games (about +-0.05 at 95%) of the candidate against **it13's network inside the wave 2 search and rules**, not
against stock upstream v3. They are not comparable with the arena's `vs_it13_*` rows (stock upstream binary, stock
rules for it13, arena rules v3). Use them to compare iterations and arms with each other only.

Same seeds do not mean same games. Integration 2026-11-19 (talisman `docs/wave2_local_check.md`): with the one-worker
`build-sp1` the narrow and wide arms of a 4-game A/B still played different games (958 vs 555 records). Wide node
keys include landing history and thinking time, so a wide tree does not merge transpositions a narrow tree merges,
and searches diverge even while the wide net's outputs equal it13's. Compare the arms by their gate scores, not game
by game.

No self-play rate has been measured on PACE yet, so the estimate is a formula with a guess:
GPU-hours ~= games / R + iterations x (training + 2 matches), with R games per GPU-hour for 800-node self-play at 6
parallel games. Guess R = 300 (a game is ~200 searches x 800 nodes; the arena's D1 rate of 150 games/GPU-hour is for
real-time 120-180 s games, which self-play does not wait for), training ~0.3 GPU-hour per 20,000-game iteration,
matches 2 x 400 games at 100 ms per move ~1 GPU-hour. Measure R from the first segment's log (`seg00` wall time) and
rescale.

| Run | Settings | Games | GPU-hours at R = 300 |
|---|---|---|---|
| Medium A/B | `AB=1 ITERS=4 GAMES_PER_ITER=10000 MATCH_GAMES=400` | 2 x 4 x 10,500 (with validation) = 84,000 | about **290** (280 self-play + 8 x 1.3) |
| Big run | the A/B's winner, `ITERS=10 GAMES_PER_ITER=50000` | 505,000 | about **1,700** |

At R = 150 both double (580 and 3,400); under 50 games per GPU-hour, halve the iteration count first (audit section 4).
