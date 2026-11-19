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
`--mate-race-by-landing false`, `--pass-cap-k N`, `--base-mix ...`, `--seat-move-times false`). Without `--wave2` the
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
    --batch-size 1024 --pause-for "" \
    --generator engine/models/twin-s-clock-w2.onnx --generator-checkpoint artifacts/distill/twin-s-clock-w2/best.pt \
    --selfplay-engine <engine> --match-engine <engine>
```

With the wave 2 engine built, drop `--allow-stock-engine`. Batch 1024 over a few hundred positions is one training
step. `distill-train` runs on the CPU there (no CUDA); it uses `OMP_NUM_THREADS=2`.
