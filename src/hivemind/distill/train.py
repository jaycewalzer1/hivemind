"""Train a student network on the teacher's policy, WDL and moves-left.

    hivemind distill-train --data data/distill/train --val data/distill/val \\
        --size twin-m --out artifacts/distill/twin-m

Data comes from the engine's `gendistill` command (raw teacher outputs)
and, optionally, `hivemind selfplay --distill-output` (--search-data: MCTS
visit distributions, root Q, and finished-game WDL and moves-left targets). Every
batch mixes the two in a fixed proportion (--search-fraction, by default
their share of all positions). Generate validation sets with their own seeds
so they share no games with the training sets.

An epoch visits every raw position once: chunks are shuffled, loaded a window
at a time, and each window is shuffled and consumed without replacement; the
search pool cycles the same way alongside, and so does --replay-data (older
self-play, a replay buffer) in its own share of each batch. The best
checkpoint by validation loss is exported to ONNX with the same inputs and
outputs as the teacher, so the engine loads it with --model.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hivemind.architectures.twin_board import (
    STEM_WEIGHT,
    TWIN_BOARD_SIZES,
    get_twin_board_model,
    load_twin_board_checkpoint,
)
from hivemind.distill.data import DistillDataset, chunk_paths, chunk_positions, head_chunk, read_chunk
from hivemind.constants import NUM_CLOCKED_CHANNELS_PER_BOARD, NUM_WIDE_CHANNELS_PER_BOARD
from hivemind.paths import PROJECT_ROOT


TERMS = ("policy_a", "policy_b", "wdl", "value", "moves_left", "arrivals", "think_gain")
# Wave 2b arrivals loss: per piece type P, N, B, R, Q, the same in all four
# pockets; a lost queen in a partner's pocket matters most.
ARRIVALS_TYPE_WEIGHTS = (1.0, 2.0, 2.0, 2.0, 4.0)


def arrivals_loss(predicted, target):
    """Per-row weighted mean squared error over the 20 arrival counts."""
    weights = predicted.new_tensor(ARRIVALS_TYPE_WEIGHTS).repeat(4)
    return ((predicted.float() - target) ** 2 * weights).sum(1) / weights.sum()


def think_gain_loss(logits, target, mask):
    """Spec v7 think_gain term: binary cross-entropy on the two logits [A, B],
    labelled boards only. Per row: (mean BCE over its labelled boards, number
    of labelled boards), so the normalized term is the mean over labelled
    boards and a row without labels contributes no loss and no gradient."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    mask = mask.float()
    count = mask.sum(1)
    return (bce * mask).sum(1) / count.clamp_min(1), count


def layout(model):
    """The plane layout a model reads (hivemind.distill.data.read_chunk)."""
    return "wide" if getattr(model, "arrivals_head", None) is not None \
        or model.planes_per_board == NUM_WIDE_CHANNELS_PER_BOARD else "narrow"


def q_wdl(value):
    """A WDL target [loss, draw, win] from a root Q in [-1, 1]: win (1 + q) / 2,
    loss (1 - q) / 2, no draw mass. The engine writes outcome WDL at index
    outcome + 1 (-1 loss, 0 draw, +1 win) and root Q from the same team's
    side, so q = +1 is a certain win."""
    q = value.float().clamp(-1, 1)
    return torch.stack(((1 - q) / 2, torch.zeros_like(q), (1 + q) / 2), dim=1)


def q_blend_rows(batch):
    """Rows whose WDL target --search-q-blend changes: search rows (source > 0)
    with an outcome WDL (not a macro-ply-limit game), a policy on at least one
    board (not a playout-cap fast record: its root Q comes from a smaller
    search) and a finite root Q in [-1, 1]."""
    value = batch["value"]
    source = batch.get("source", torch.zeros_like(value))
    has_policy = batch["policy"][0][2] | batch["policy"][1][2]
    return ((source > 0) & (batch["wdl"].sum(1) > 0) & has_policy
            & torch.isfinite(value) & (value.abs() <= 1))


def wdl_target(batch, q_blend=0.0):
    """The WDL target: the outcome, or with --search-q-blend lambda > 0
    (1 - lambda) * outcome + lambda * q_wdl(root Q) on q_blend_rows. Lambda 0
    returns the outcome tensor itself, untouched."""
    if not q_blend:
        return batch["wdl"]
    outcome = batch["wdl"].float()
    blended = (1 - q_blend) * outcome + q_blend * q_wdl(batch["value"])
    return torch.where(q_blend_rows(batch).unsqueeze(1), blended, outcome)


def _loss_rows(model, batch, q_blend=0.0):
    """Loss numerators and target masks before normalization. `q_blend` mixes
    root Q into the WDL target (wdl_target); 0 is the outcome alone."""
    outputs = model(batch["planes"])
    value, (policy_a, policy_b), _, wdl_logits, plys = outputs[:5]
    per_row = {}
    for name, logits, (target, legal, mask) in (("policy_a", policy_a, batch["policy"][0]),
                                                ("policy_b", policy_b, batch["policy"][1])):
        # The engine softmaxes over legal moves and the pass only; so does the loss.
        log_probs = F.log_softmax(logits.float().masked_fill(~legal, float("-inf")), dim=1)
        per_row[name] = (-torch.where(legal, target * log_probs, 0.0).sum(1), mask.float())
    has_wdl = (batch["wdl"].sum(1) > 0).float()
    per_row["wdl"] = (-(wdl_target(batch, q_blend) * F.log_softmax(wdl_logits.float(), dim=1)).sum(1), has_wdl)
    per_row["value"] = ((value.float().squeeze(1) - batch["value"]) ** 2, torch.ones_like(has_wdl))
    # A resigned game has a result but no length: MOVES_LEFT_UNKNOWN (< 0).
    has_moves_left = has_wdl * (batch["moves_left"] >= 0).float()
    per_row["moves_left"] = ((plys.float().squeeze(1) - batch["moves_left"]) ** 2, has_moves_left)
    if getattr(model, "arrivals_head", None) is not None and "arrivals" in batch:
        per_row["arrivals"] = (arrivals_loss(outputs[5], batch["arrivals"]), batch["has_arrivals"].float())
    if getattr(model, "think_gain_head", None) is not None and "think_gain" in batch:
        # The think_gain head is always the last output.
        per_row["think_gain"] = think_gain_loss(outputs[-1], batch["think_gain"], batch["has_think_gain"])
    return per_row, (value, policy_a, policy_b)


def _source_masks(batch, weights):
    source = batch.get("source", torch.zeros_like(batch["value"]))
    if "raw" in weights:
        return (("raw", source == 0), ("search", source > 0))
    return ((None, torch.ones_like(source, dtype=torch.bool)),)


def losses(model, batch, weights, q_blend=0.0):
    """Normalize each term within its source, then add fixed source weights.

    Flat weights retain a single mean over all applicable rows. Unlabelled
    games contribute to neither WDL nor moves-left. `q_blend` (training only,
    --search-q-blend) blends root Q into the search rows' WDL target; the
    validation metrics (evaluate) keep the outcome target, so runs with
    different lambdas report comparable losses.
    """
    per_row, outputs = _loss_rows(model, batch, q_blend)
    total, terms = 0.0, {}
    for name, (loss, valid) in per_row.items():
        for source, mask in _source_masks(batch, weights):
            active = valid * mask
            weight = (weights[source] if source is not None else weights).get(name, 0.0)
            total = total + weight * (loss * active).sum() / active.sum().clamp_min(1)
        terms[name] = (loss * valid).sum() / valid.sum().clamp_min(1)
    return total, terms, outputs


@torch.no_grad()
def evaluate(model, dataset, weights, batch_size=2048):
    model.eval()
    sums, counts, count = {}, {}, 0
    agree = [0, 0]
    rows = [0, 0]
    value_error = 0.0
    for start in range(0, len(dataset), batch_size):
        index = torch.arange(start, min(start + batch_size, len(dataset)), device=dataset.device)
        batch = dataset.batch(index)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dataset.device.type == "cuda"):
            per_row, (value, policy_a, policy_b) = _loss_rows(model, batch)
        for name, (loss, valid) in per_row.items():
            for source, mask in _source_masks(batch, weights):
                key = (source, name)
                active = valid * mask
                sums[key] = sums.get(key, 0.0) + float((loss * active).sum())
                counts[key] = counts.get(key, 0.0) + float(active.sum())
        count += len(index)
        value_error += float((value.float().squeeze(1) - batch["value"]).abs().sum())
        for board, logits in enumerate((policy_a, policy_b)):
            target, legal, mask = batch["policy"][board]
            # Top-move agreement among the legal moves.
            student_best = logits.float().masked_fill(~legal, float("-inf")).argmax(1)
            teacher_best = target.argmax(1)
            agree[board] += int(((student_best == teacher_best) & mask).sum())
            rows[board] += int(mask.sum())
    model.train()
    metrics = {}
    total = 0.0
    for name in TERMS:
        keys = [key for key in sums if key[1] == name]
        if not keys:
            continue
        metrics[name] = sum(sums[key] for key in keys) / max(1, sum(counts[key] for key in keys))
        for source, term in keys:
            weight = (weights[source] if source is not None else weights).get(term, 0.0)
            total += weight * sums[(source, term)] / max(1, counts[(source, term)])
    metrics["loss"] = total
    metrics["value_mae"] = value_error / count
    metrics["top1_a"] = agree[0] / max(1, rows[0])
    metrics["top1_b"] = agree[1] / max(1, rows[1])
    return metrics


# The arrivals head (wave 2b) and the think_gain head (spec v7) start from
# zero, so they train at the heads' rate.
HEADS = ("policy_head.", "value_head.", "arrivals_head.", "think_gain_head.")


def parameter_groups(model, lr, trunk_lr=None):
    """AdamW parameter groups: one at `lr`, or the trunk at `trunk_lr` and
    the heads at `lr`. The LR schedule scales every group alike."""
    if trunk_lr is None:
        return [{"params": list(model.parameters()), "lr": lr}]
    heads, trunk = [], []
    for name, parameter in model.named_parameters():
        (heads if name.startswith(HEADS) else trunk).append(parameter)
    return [{"params": trunk, "lr": trunk_lr, "name": "trunk"}, {"params": heads, "lr": lr, "name": "heads"}]


class NewInputRate:
    """Wave 2b: the stem weights of a wide network's 26 new planes per board
    start at zero, like the arrivals head, but sit inside the trunk's stem
    tensor, so parameter groups cannot give them the heads' rate. After each
    optimizer step this scales their update by lr / trunk_lr. AdamW's step
    (decoupled weight decay included) is proportional to the group's rate,
    so that is exactly a step at the heads' rate."""

    def __init__(self, model, lr, trunk_lr):
        weight = dict(model.named_parameters()).get(STEM_WEIGHT)
        self.active = (trunk_lr is not None and trunk_lr > 0 and lr != trunk_lr and weight is not None
                       and weight.shape[1] == NUM_WIDE_CHANNELS_PER_BOARD)
        self.weight = weight
        self.ratio = lr / trunk_lr if self.active else 1.0
        self.before = None

    def _slice(self):
        return self.weight.data[:, NUM_CLOCKED_CHANNELS_PER_BOARD:NUM_WIDE_CHANNELS_PER_BOARD]

    def save(self):
        if self.active:
            self.before = self._slice().clone()

    def apply(self):
        if self.active:
            new = self._slice()
            new.copy_(self.before + (new - self.before) * self.ratio)


def concat_batches(first, second):
    """One batch from two (the raw and search streams)."""
    merged = {key: torch.cat((first[key], second[key])) for key in first if key != "policy"}
    merged["policy"] = [tuple(torch.cat(pair) for pair in zip(a, b))
                        for a, b in zip(first["policy"], second["policy"])]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", default=[],
                        help="Teacher (raw) HDST chunks; omit with --search-fraction 1 to train on "
                             "self-play alone")
    parser.add_argument("--val", nargs="+", default=[],
                        help="Teacher validation chunks; optional for self-play-only training")
    parser.add_argument("--search-data", nargs="+", default=[], help="Search (selfplay) HDST chunks")
    parser.add_argument("--search-val", nargs="+", default=[])
    parser.add_argument("--search-fraction", type=float, default=None,
                        help="Share of each batch from --search-data (default: its share of positions)")
    parser.add_argument("--search-window-chunks", type=int, default=2)
    parser.add_argument("--replay-data", nargs="+", default=[],
                        help="Older self-play HDST chunks (a replay buffer), weighted like --search-data")
    parser.add_argument("--replay-fraction", type=float, default=0.0,
                        help="Share of each batch from --replay-data, taken out of the search share")
    parser.add_argument("--replay-window-chunks", type=int, default=2)
    parser.add_argument("--search-policy-weight", type=float, default=1.0)
    parser.add_argument("--search-value-weight", type=float, default=1.0,
                        help="Weight of root-Q value MSE on search rows")
    parser.add_argument("--search-wdl-weight", type=float, default=1.0,
                        help="Weight of game-result WDL on search rows (only finished games carry one)")
    parser.add_argument("--search-moves-left-weight", type=float, default=None,
                        help="Default: --moves-left-weight")
    parser.add_argument("--search-q-blend", type=float, default=0.0, metavar="LAMBDA",
                        help="Search balance: train the WDL head of search rows that carry a policy "
                             "and a game result on (1 - LAMBDA) * outcome + LAMBDA * root-Q WDL "
                             "(win (1 + q) / 2, loss (1 - q) / 2, draw 0). Validation keeps the "
                             "outcome target. 0 (default) is the outcome alone")
    parser.add_argument("--init", type=Path, default=None,
                        help="Start from this checkpoint (its size and variant replace --size and the "
                             "architecture flags)")
    parser.add_argument("--steps", type=int, default=None, help="Train this many steps (overrides --epochs)")
    parser.add_argument("--export", default="best", choices=["best", "last"],
                        help="Export the best checkpoint by validation loss or the last one")
    parser.add_argument("--train-probe", type=int, default=200_000,
                        help="Training positions evaluated like the validation set, for the gap (0 = off)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "distill" / "run")
    parser.add_argument("--size", default="twin-m", choices=sorted(TWIN_BOARD_SIZES))
    parser.add_argument("--mid-attention", action="store_true",
                        help="Add a cross-board attention halfway through the trunk")
    parser.add_argument("--value-diff", action="store_true",
                        help="Feed the value head |A - B| next to A + B")
    parser.add_argument("--blocks", type=int, default=None, help="Override the size's block count")
    parser.add_argument("--channels", type=int, default=None, help="Override the size's trunk width")
    parser.add_argument("--attention-dim", type=int, default=None,
                        help="Override the final attention width (0 = no attention)")
    parser.add_argument("--attention-style", default="full", choices=["full", "lite"])
    parser.add_argument("--block", default="bottleneck", choices=["bottleneck", "dense"])
    parser.add_argument("--se-every", type=int, default=4, help="ECA every this many blocks (0 = none)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048,
                        help="Positions per validation batch; reduce when GPU memory is limited")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trunk-lr", type=float, default=None,
                        help="Learning rate of the trunk (stem, blocks, exchanges, attention); the "
                             "policy and value heads keep --lr. A fine-tune keeps the trunk slow "
                             "(default: --lr for everything)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-chunks", type=int, default=4,
                        help="Chunks held on the device at once")
    parser.add_argument("--eval-every", type=int, default=2000, help="Steps between validations")
    parser.add_argument("--moves-left-weight", type=float, default=0.1)
    parser.add_argument("--new-inputs-lr", choices=("heads", "trunk"), default="heads",
                        help="Wave 2b, with --trunk-lr: the rate of the stem weights of the 26 new "
                             "(zero-initialised) planes per board of a wide network: the heads' --lr "
                             "(default, like the arrivals head) or the trunk's")
    parser.add_argument("--arrivals-weight", type=float, default=0.05,
                        help="Weight of the wave 2b arrivals loss (wide networks, rows with a label)")
    parser.add_argument("--wide", action="store_true",
                        help="Wave 2b: widen the --init network to the wide planes and add the arrivals "
                             "head (zero-initialised; a wide checkpoint stays wide without this)")
    parser.add_argument("--think-weight", type=float, default=0.0,
                        help="Spec v7: weight of the think_gain loss (masked BCE on rows labelled by "
                             "hivemind.think labels, think_labels.npz next to the chunks). Above 0, a "
                             "network without the head gets a fresh one (p = 1/3). 0 (default): off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile of the residual blocks (about 1.5x slower)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("--batch-size and --eval-batch-size must be positive")
    if not 0 <= args.search_q_blend <= 1:
        parser.error("--search-q-blend must be in [0, 1]")
    if args.think_weight < 0:
        parser.error("--think-weight must be >= 0")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 1.0,
                   "moves_left": args.moves_left_weight, "arrivals": args.arrivals_weight,
                   "think_gain": args.think_weight}
    search_moves_left = (args.moves_left_weight if args.search_moves_left_weight is None
                         else args.search_moves_left_weight)
    search_weights = {"policy_a": args.search_policy_weight, "policy_b": args.search_policy_weight,
                      "wdl": args.search_wdl_weight, "value": args.search_value_weight,
                      "moves_left": search_moves_left, "arrivals": args.arrivals_weight,
                      "think_gain": args.think_weight}
    weights = {"raw": raw_weights, "search": search_weights}

    pool = chunk_paths(args.data)
    positions = sum(chunk_positions(path) for path in pool)
    rotation = np.random.default_rng(args.seed)
    window = min(args.window_chunks, len(pool))

    search_pool = chunk_paths(args.search_data)
    search_positions = sum(chunk_positions(path) for path in search_pool)
    fraction = args.search_fraction
    if fraction is None:
        fraction = search_positions / (positions + search_positions)
    search_rows = round(args.batch_size * fraction) if search_pool else 0
    raw_rows = args.batch_size - search_rows
    if raw_rows < 0 or (raw_rows == 0) != (not pool) or (not pool and not search_pool):
        raise SystemExit("--data is needed unless --search-fraction is 1 with --search-data, "
                         "and is unused then")
    if raw_rows and not args.val:
        raise SystemExit("--val is required when training on teacher data")
    if not raw_rows and not args.search_val:
        raise SystemExit("--search-val is required for self-play-only training")
    replay_pool = chunk_paths(args.replay_data)
    replay_rows = round(args.batch_size * args.replay_fraction) if replay_pool else 0
    if bool(replay_pool) != (args.replay_fraction > 0):
        raise SystemExit("--replay-data and --replay-fraction go together")
    if replay_pool and not 0 < replay_rows < search_rows:
        raise SystemExit("--replay-fraction must leave part of the search share for --search-data")
    search_rows -= replay_rows

    variant = {"mid_attention": args.mid_attention, "value_diff": args.value_diff,
               "blocks": args.blocks, "channels": args.channels, "attention_dim": args.attention_dim,
               "attention_style": args.attention_style, "block": args.block, "se_every": args.se_every}
    size = args.size
    if args.init:
        # A network from before the clock planes starts reading them with
        # zero weights, computing exactly what it did until training moves them.
        # --think-weight > 0 adds a fresh think_gain head to a network without one.
        model, size, variant = load_twin_board_checkpoint(args.init, wide=True if args.wide else None,
                                                          think_gain=True if args.think_weight > 0 else None)
        print(f"starting from {args.init} ({size}, {variant})", flush=True)
    else:
        if args.wide:
            variant.update(planes_per_board=NUM_WIDE_CHANNELS_PER_BOARD, arrivals=True)
        if args.think_weight > 0:
            variant.update(think_gain=True)
        model = get_twin_board_model(size, **variant)
    planes_layout = layout(model)

    def read(path):
        return read_chunk(path, layout=planes_layout)

    def batches(paths, window, rows, source):
        """Endless batches of `rows` positions; every position once per pass."""
        while True:
            order = rotation.permutation(len(paths))
            for start in range(0, len(order), window):
                data = None  # free the previous window before loading the next
                data = DistillDataset([read(paths[i]) for i in order[start:start + window]], device,
                                      source=source)
                shuffled = torch.randperm(len(data), device=device)
                for first in range(0, len(data) - rows + 1, rows):
                    yield data.batch(shuffled[first:first + rows])

    raw_stream = batches(pool, window, raw_rows, source=0) if raw_rows else None
    search_stream = (batches(search_pool, min(args.search_window_chunks, len(search_pool)), search_rows,
                             source=1)
                     if search_rows else None)
    replay_stream = (batches(replay_pool, min(args.replay_window_chunks, len(replay_pool)), replay_rows,
                             source=1)
                     if replay_rows else None)

    def next_batch():
        parts = [next(stream) for stream in (raw_stream, search_stream, replay_stream) if stream is not None]
        batch = parts[0]
        for part in parts[1:]:
            batch = concat_batches(batch, part)
        return batch

    val = (DistillDataset([read(p) for p in chunk_paths(args.val)], device)
           if args.val else None)
    # A fixed slice of the training data, scored like the validation set: the
    # gap between the two says whether more data would help.
    train_probe = (DistillDataset([head_chunk(read((pool or search_pool)[0]), args.train_probe)], device,
                                  source=0 if pool else 1)
                   if args.train_probe else None)
    search_val = (DistillDataset([read(p) for p in chunk_paths(args.search_val)], device, source=1)
                  if args.search_val else None)
    print(f"train {len(pool)} chunks ({positions} positions, window {window}), "
          f"val {len(val) if val is not None else 0}", flush=True)
    if search_pool:
        print(f"search {len(search_pool)} chunks ({search_positions} positions), "
              f"{search_rows}/{args.batch_size} of each batch, val {len(search_val) if search_val else 0}",
              flush=True)
    if args.search_q_blend:
        print(f"search q blend {args.search_q_blend}: WDL target of policy-carrying search rows = "
              f"(1 - lambda) outcome + lambda root-Q WDL", flush=True)
    if replay_pool:
        print(f"replay {len(replay_pool)} chunks "
              f"({sum(chunk_positions(path) for path in replay_pool)} positions), "
              f"{replay_rows}/{args.batch_size} of each batch", flush=True)

    model = model.to(device)
    if device.type == "cuda" and not args.no_compile:
        # In place, so state_dict keys are unchanged. The whole model does not
        # compile (its heads return views of shared tensors).
        torch.backends.cudnn.benchmark = True
        for block in model.blocks:
            block.compile()
    print(f"{size}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr, args.trunk_lr), lr=args.lr,
                                  weight_decay=args.weight_decay)
    new_inputs = NewInputRate(model, args.lr, args.trunk_lr if args.new_inputs_lr == "heads" else None)
    if args.trunk_lr is not None:
        print(f"learning rates: trunk {args.trunk_lr}, heads {args.lr}"
              + (", wide new-plane stem weights at the heads' rate" if new_inputs.active else ""), flush=True)
    # An epoch is one pass over the raw data, or over the search data without it.
    steps_per_epoch = max(1, positions // raw_rows if raw_rows else search_positions // search_rows)
    total_steps = args.steps if args.steps else int(args.epochs * steps_per_epoch)
    warmup = min(1000, total_steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup))
        * (0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)) * 0.99 + 0.01))

    best, history, started = math.inf, [], time.time()
    for step in range(1, total_steps + 1):
        batch = next_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Lambda 0 makes the same call as before the flag existed.
            total, _, _ = (losses(model, batch, weights, q_blend=args.search_q_blend)
                           if args.search_q_blend else losses(model, batch, weights))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        new_inputs.save()
        optimizer.step()
        new_inputs.apply()
        schedule.step()

        if step % args.eval_every == 0 or step == total_steps:
            metrics = (evaluate(model, val, weights, batch_size=args.eval_batch_size)
                       if val is not None else {})
            if train_probe is not None:
                probe = evaluate(model, train_probe, weights, batch_size=args.eval_batch_size)
                metrics.update(train_loss=probe["loss"], train_top1_a=probe["top1_a"],
                               train_value_mae=probe["value_mae"])
            if search_val is not None:
                search_metrics = evaluate(model, search_val, weights, batch_size=args.eval_batch_size)
                metrics.update({f"search_{k}": v for k, v in search_metrics.items()})
                # Self-play-only runs select exclusively on their own targets.
                # Mixed runs retain the fixed sum of source validation losses.
                metrics["selection_loss"] = (search_metrics["loss"] if not raw_rows else
                                             metrics["loss"] +
                                             (search_metrics["loss"] if search_rows else 0.0))
            metrics.update(epoch=round(step / steps_per_epoch, 2), lr=schedule.get_last_lr()[-1],
                           seconds=round(time.time() - started))
            history.append(metrics)
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}),
                  flush=True)
            selection = metrics["selection_loss"] if "selection_loss" in metrics else metrics["loss"]
            if selection < best:
                best = selection
                torch.save({"model": model.state_dict(), "size": size, "variant": variant,
                            "step": step, "selection_loss": selection},
                           args.out / "best.pt")

    torch.save({"model": model.state_dict(), "size": size, "variant": variant}, args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    export(args.out, size, checkpoint_name=f"{args.export}.pt")


OUTPUT_NAMES = ("value", "pi_a", "pi_b", "auxiliary_out", "wdl_out", "moves_left")
# Wave 2b: wide networks add this output last; the engine matches outputs by
# name and does not read it.
ARRIVALS_OUTPUT = "arrivals"
# Spec v7: a network with the think_gain head adds this output last (after
# arrivals): [batch, 2] logits, boards A and B.
THINK_GAIN_OUTPUT = "think_gain"


def export(out, size, checkpoint_name="best.pt", path=None, fp32_path=None):
    """ONNX export with the teacher's inputs and outputs (value, pi_a, pi_b,
    wdl_out, moves_left, arrivals for a wide network, think_gain for a network
    with that head); FP16 weights with
    FP32 inputs and outputs, plus an all-FP32 copy at `fp32_path` if given."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxsim import simplify

    from hivemind.inference.onnx_graph import topologically_sort_graph

    model, size, _ = load_twin_board_checkpoint(out / checkpoint_name)
    model.eval()
    names = (OUTPUT_NAMES + ((ARRIVALS_OUTPUT,) if model.arrivals_head is not None else ())
             + ((THINK_GAIN_OUTPUT,) if getattr(model, "think_gain_head", None) is not None else ()))
    path = Path(path) if path else out / f"distill-{size}-v3.0.onnx"
    torch.onnx.export(model, (torch.ones(1, 2 * model.planes_per_board, 8, 8),), str(path),
                      input_names=["data"], output_names=list(names),
                      dynamic_axes={name: {0: "batch"} for name in ("data", *names)},
                      dynamo=False, export_params=True, opset_version=18,
                      do_constant_folding=True, keep_initializers_as_inputs=False)
    simplified, check = simplify(onnx.load(str(path)))
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    if fp32_path is not None:
        fp32 = onnx.load_from_string(simplified.SerializeToString())
        kept = [output for output in fp32.graph.output if output.name != "auxiliary_out"]
        del fp32.graph.output[:]
        fp32.graph.output.extend(kept)
        onnx.checker.check_model(fp32)
        onnx.save(fp32, str(fp32_path))
    fp16 = convert_float_to_float16(simplified, keep_io_types=True)
    topologically_sort_graph(fp16)
    onnx.checker.check_model(fp16)
    # The engine reads value, both policies, WDL and moves-left; drop the unused auxiliary head.
    outputs = [output for output in fp16.graph.output if output.name != "auxiliary_out"]
    del fp16.graph.output[:]
    fp16.graph.output.extend(outputs)
    onnx.save(fp16, str(path))
    print(f"exported {out}")

if __name__ == "__main__":
    main()
