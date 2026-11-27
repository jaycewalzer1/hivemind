"""Spec v7 section 6 (talisman docs/custom_engine_spec_v7_league.md): the think_gain head's labels, the
headed-network tool and rl-loop's hooks for the learned time policy.

The twin-board network may carry a think_gain head (hivemind.architectures.twin_board._ThinkGainHead): two
logits [A, B], the chance that searching that board's decision 4x longer changes the move. The engine turns
p into a budget scale s = clamp(0.5 + 1.5 p, 0.5, 2.0) (engine/src/tools/think_policy.h); a fresh head says
p = 1/3, s = 1, so it changes nothing until it trains.

  python -m hivemind.think labels --it RUN/runs/rl-itN --iteration N --model GEN.onnx --engine HIVEMIND
      For every finished non-val segment with a state dump (selfplay --record-state-output, added by
      selfplay_args when --think-labels is on), a hashed fraction of its searched records is searched again
      with the engine's reanalyse command at the recorded budget and at --factor x it (same generator, the
      segment's wave 2 options, no noise). Under --think-policy the recorded budget is the state line's
      think_ref_nodes, the budget before the generator's head scaled it, so the labels do not feed back on it. Per board on turn: label 1 when the two searches' most-visited
      entries differ (a pass counts as an entry), else 0; boards not on turn are masked. A repeat of the base
      search on 5% of them gives noise_rate (how often the base search disagrees with itself). Writes
      think_labels.npz into the segment's chunk dir (hivemind.distill.data reads it) and prints one JSON line
      {records, labelled, rate_a, rate_b, noise_rate, gpu_s, segments}.

  python -m hivemind.think add-head --checkpoint X.pt --out DIR
      X with a fresh think_gain head: DIR/<X>-think.pt and DIR/<X>-think.onnx (exported like distill-train's
      networks), checked against X: every other output equal (PyTorch exactly, ONNX within fp16 rounding),
      think_gain = -ln 2 everywhere. Writes DIR/<X>-think.json and prints it as one line.

rl-loop hooks (spec 3.5; WP1 calls them, all off unless --think-labels / --think-policy):
add_arguments, validate, selfplay_args, label_command, trainer_args.
"""

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

LABEL_SALT = "think"
REPEAT_SALT = "think-repeat"
REPEAT_FRACTION = 0.05
DEFAULT_FRACTION = 0.02
DEFAULT_FACTOR = 4
DEFAULT_WEIGHT = 0.1
STATES_NAME = "states.jsonl"
WORK_NAME = "think"
INITIAL_LOGIT = -math.log(2.0)


# ---------------------------------------------------------------- rl-loop hooks

def add_arguments(parser):
    """rl-loop's think options (spec 6.4). All off by default."""
    group = parser.add_argument_group("think_gain head (spec v7)")
    group.add_argument("--think-labels", action="store_true",
                       help="Dump self-play states and label a share of the records (hivemind.think labels) "
                            "after self-play; train the think_gain head on them (--think-weight)")
    group.add_argument("--think-label-fraction", type=float, default=DEFAULT_FRACTION,
                       help=f"Share of searched records labelled (default {DEFAULT_FRACTION})")
    group.add_argument("--think-factor", type=int, default=DEFAULT_FACTOR,
                       help=f"The label's longer search, in multiples of the recorded budget "
                            f"(default {DEFAULT_FACTOR})")
    group.add_argument("--think-policy", action="store_true",
                       help="Self-play budgets scaled by the generator's think_gain head "
                            "(engine --think-policy true)")
    group.add_argument("--think-weight", type=float, default=None,
                       help=f"Trainer weight of the think_gain loss (default {DEFAULT_WEIGHT}; "
                            f"only with --think-labels)")


def labels_on(args):
    return bool(getattr(args, "think_labels", False))


def policy_on(args):
    return bool(getattr(args, "think_policy", False))


def validate(args, parser):
    """parser.error on bad combinations."""
    fraction = getattr(args, "think_label_fraction", DEFAULT_FRACTION)
    factor = getattr(args, "think_factor", DEFAULT_FACTOR)
    weight = getattr(args, "think_weight", None)
    if not 0 < fraction <= 1:
        parser.error("--think-label-fraction must be in (0, 1]")
    if factor < 2:
        parser.error("--think-factor must be at least 2")
    if weight is not None and not labels_on(args):
        parser.error("--think-weight needs --think-labels")
    if weight is not None and weight < 0:
        parser.error("--think-weight must be >= 0")


def selfplay_args(args):
    """Engine args for non-val self-play segments ({segment_dir} is the segment's records dir)."""
    extra = []
    if labels_on(args):
        extra += ["--record-state-output", "{segment_dir}/" + STATES_NAME]
    if policy_on(args):
        extra += ["--think-policy", "true"]
    return extra


def label_command(args, n, it, generator, engine):
    """The labels stage's command (prints one JSON line), or None when labels are off."""
    if not labels_on(args):
        return None
    return [sys.executable, "-m", "hivemind.think", "labels",
            "--it", str(it), "--iteration", str(n),
            "--model", str(generator), "--engine", str(engine),
            "--fraction", str(getattr(args, "think_label_fraction", DEFAULT_FRACTION)),
            "--factor", str(getattr(args, "think_factor", DEFAULT_FACTOR)),
            "--parallel-engines", str(getattr(args, "parallel_games", 6))]


def trainer_args(args):
    """distill-train args: the think_gain loss weight when labels are on, else []."""
    if not labels_on(args):
        return []
    weight = getattr(args, "think_weight", None)
    return ["--think-weight", str(DEFAULT_WEIGHT if weight is None else weight)]


# ---------------------------------------------------------------- labels

def top_entries(chunk):
    """Per row, per board: the policy index with the most visits (the largest probability; the pass is an
    entry like any move), or None for a board with no policy (not on turn)."""
    import numpy as np
    counts = chunk["policy_count"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts.ravel())])
    tops = []
    for row in range(len(counts)):
        boards = []
        for board in range(2):
            a, z = starts[2 * row + board], starts[2 * row + board + 1]
            if z == a:
                boards.append(None)
                continue
            probabilities = chunk["policy_probability"][a:z].astype(np.float32)
            boards.append(int(chunk["policy_index"][a + int(np.argmax(probabilities))]))
        tops.append(tuple(boards))
    return tops


def make_labels(base, deep):
    """(record_index [M], label [M, 2] float32, mask [M, 2] bool) from {record_index: (top_a, top_b)} of the
    base and deep searches: 1 where the most-visited entries differ, masked where either has no policy."""
    import numpy as np
    indices = sorted(base)
    label = np.zeros((len(indices), 2), np.float32)
    mask = np.zeros((len(indices), 2), bool)
    for row, index in enumerate(indices):
        for board in range(2):
            a, b = base[index][board], deep[index][board]
            if a is None or b is None:
                continue
            mask[row, board] = True
            label[row, board] = float(a != b)
    return np.array(indices, np.int64), label, mask


def disagreement(base, repeat):
    """(boards compared, boards whose most-visited entry differs) between two searches of the same records."""
    compared = differ = 0
    for index, tops in repeat.items():
        for board in range(2):
            a, b = base[index][board], tops[board]
            if a is None or b is None:
                continue
            compared += 1
            differ += int(a != b)
    return compared, differ


def reference_dump(dump):
    """A copy of a reanalyse StateDump whose chosen record lines ask for their think_ref_nodes (self-play under
    --think-policy: the budget at think scale 1) instead of their target_nodes, so a label measures the gain of
    4x the clock's budget, not of 4x the budget the generator's own head chose. Lines without think_ref_nodes
    (no --think-policy) are kept byte-exact."""
    import copy
    import re
    target = re.compile(r'("target_nodes":\s*)(\d+)')
    reference = re.compile(r'"think_ref_nodes":\s*(\d+)')
    out = copy.copy(dump)
    out.records = {}
    for index, line in dump.records.items():
        found = reference.search(line)
        if found:
            line, count = target.subn(lambda m: m.group(1) + found.group(1), line, count=1)
            if count != 1:
                raise RuntimeError(f"record {index}: no target_nodes in its state line")
        out.records[index] = line
    return out


def scaled_dump(dump, factor):
    """A copy of a reanalyse StateDump whose chosen record lines ask for factor x their target_nodes (the
    engine's reanalyse with --nodes 0 searches each record at its own target_nodes)."""
    import copy
    import re
    pattern = re.compile(r'("target_nodes":\s*)(\d+)')
    scaled = copy.copy(dump)
    scaled.records = {}
    for index, line in dump.records.items():
        new, count = pattern.subn(lambda m: m.group(1) + str(int(m.group(2)) * factor), line, count=1)
        if count != 1:
            raise RuntimeError(f"record {index}: no target_nodes in its state line")
        scaled.records[index] = new
    return scaled


def search(job, dump, indices, offsets, wave2_args, root, prefix):
    """Reanalyse these records (hivemind reanalyse's run_batch: the engine, then every output record checked
    against its source record) and read back each one's most-visited entries.
    Returns ({record_index: (top_a, top_b)}, the batch's stats)."""
    from hivemind.cli import reanalyse
    from hivemind.distill.data import chunk_paths, read_chunk
    out = root / "out"
    stats = reanalyse.run_batch(job, dump, indices, offsets, wave2_args, root / "work", out, prefix)
    paths = chunk_paths([out])
    layout = reanalyse._layout(paths)
    tops = []
    for path in paths:
        tops += top_entries(read_chunk(path, layout=layout))
    ordered = sorted(indices)
    if len(tops) != len(ordered):
        raise RuntimeError(f"{len(tops)} reanalysed records for {len(ordered)} states")
    return dict(zip(ordered, tops)), stats


def write_labels(path, record_index, label, mask):
    """think_labels.npz, atomically (hivemind.distill.data reads it next to the chunks)."""
    import numpy as np
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, record_index=record_index, label=label, mask=mask)
    temporary.replace(path)


def label_segment(args, it, name, job, log):
    """Labels one segment; returns its summary dict, or None when it has no state dump."""
    import numpy as np
    from hivemind.cli import reanalyse
    from hivemind.distill.data import THINK_LABELS_NAME
    data, records = reanalyse.segment_dirs(it, name)
    states = records / args.states_name
    target = data / THINK_LABELS_NAME
    if not states.is_file():
        return None
    offsets, total = reanalyse.chunk_offsets(data)
    if target.is_file() and not args.force:
        with np.load(target) as saved:
            label, mask = saved["label"], saved["mask"]
        log(f"{name}: labels exist, kept ({target})")
        return {"segment": name, "records": total, "labelled": int(mask.any(1).sum()),
                "positives": [float((label[:, b] * mask[:, b]).sum()) for b in range(2)],
                "boards": [int(mask[:, b].sum()) for b in range(2)], "compared": 0, "differ": 0,
                "gpu_s": 0.0, "kept": True}
    marker = it / "stages" / f"{name}.done"
    wave2_args = reanalyse.wave2_engine_args(json.loads(marker.read_text())) if marker.is_file() else []
    every = reanalyse.read_states(states, total, wanted=set())
    chosen = reanalyse.select_records(every.nodes, every.nodes, args.fraction, LABEL_SALT, args.run_seed,
                                      args.iteration, name, args.max_states)
    if not chosen:
        log(f"{name}: no records chosen")
        return {"segment": name, "records": total, "labelled": 0, "positives": [0.0, 0.0], "boards": [0, 0],
                "compared": 0, "differ": 0, "gpu_s": 0.0}
    dump = reference_dump(reanalyse.read_states(states, total, wanted=set(chosen)))
    work = it / "search" / WORK_NAME
    reanalyse.aside(work / name, work / "aborted")
    base, base_stats = search(job, dump, chosen, offsets, wave2_args, work / name / "base", "base")
    deep, deep_stats = search(job, scaled_dump(dump, args.factor), chosen, offsets, wave2_args,
                              work / name / "deep", "deep")
    repeat_indices = [index for index in chosen
                      if reanalyse.selection_value(REPEAT_SALT, args.run_seed, args.iteration, name, index)
                      < REPEAT_FRACTION]
    compared = differ = 0
    gpu_s = base_stats["gpu_s"] + deep_stats["gpu_s"]
    if repeat_indices:
        repeat, repeat_stats = search(job, dump, repeat_indices, offsets, wave2_args,
                                      work / name / "repeat", "repeat")
        compared, differ = disagreement(base, repeat)
        gpu_s += repeat_stats["gpu_s"]
    record_index, label, mask = make_labels(base, deep)
    write_labels(target, record_index, label, mask)
    summary = {"segment": name, "records": total, "labelled": int(mask.any(1).sum()),
               "positives": [float((label[:, b] * mask[:, b]).sum()) for b in range(2)],
               "boards": [int(mask[:, b].sum()) for b in range(2)], "compared": compared, "differ": differ,
               "gpu_s": round(gpu_s, 3)}
    log(f"{name}: {summary['labelled']} of {total} records labelled, boards {summary['boards']}, "
        f"positives {summary['positives']}, repeat {differ}/{compared} -> {target}")
    return summary


def labels(args):
    from hivemind.cli import reanalyse
    it = Path(args.it)

    def log(message):
        print(f"think labels: {message}", file=sys.stderr, flush=True)

    names = args.segments or reanalyse.finished_segments(it)
    names = [name for name in names if name != "val"]
    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"no model at {model}")
    sha = reanalyse.sha256(model)
    model_copy = reanalyse.private_model_copy(model, it / "search" / WORK_NAME / "model", sha)
    job = SimpleNamespace(engine=Path(args.engine), model_copy=model_copy, nodes=0,
                          parallel_engines=args.parallel_engines, dirichlet=False, replay_noise=False,
                          engine_extra=shlex.split(args.engine_extra), timeout=args.timeout, gpus=args.gpus)
    segments, skipped = [], []
    for name in names:
        summary = label_segment(args, it, name, job, log)
        if summary is None:
            skipped.append(name)
        else:
            segments.append(summary)
    if skipped:
        log(f"no {args.states_name} (skipped): {' '.join(skipped)}")
    boards = [sum(s["boards"][b] for s in segments) for b in range(2)]
    positives = [sum(s["positives"][b] for s in segments) for b in range(2)]
    compared = sum(s["compared"] for s in segments)
    result = {
        "records": sum(s["records"] for s in segments),
        "labelled": sum(s["labelled"] for s in segments),
        "rate_a": round(positives[0] / boards[0], 6) if boards[0] else None,
        "rate_b": round(positives[1] / boards[1], 6) if boards[1] else None,
        "noise_rate": round(sum(s["differ"] for s in segments) / compared, 6) if compared else None,
        "gpu_s": round(sum(s["gpu_s"] for s in segments), 3),
        "segments": len(segments),
    }
    print(json.dumps(result), flush=True)
    return 0


# ---------------------------------------------------------------- add-head

PARITY_OUTPUTS = ("value", "pi_a", "pi_b", "wdl_out", "moves_left")
TORCH_TOLERANCE = 0.0
ONNX_TOLERANCE = 1e-4
LOGIT_TOLERANCE = 1e-3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_outputs(model, planes):
    import torch
    with torch.no_grad():
        outputs = model(planes)
    value, (pi_a, pi_b), _, wdl, plys = outputs[:5]
    named = dict(zip(PARITY_OUTPUTS, (value, pi_a, pi_b, wdl, plys)))
    if getattr(model, "arrivals_head", None) is not None:
        named["arrivals"] = outputs[5]
    if getattr(model, "think_gain_head", None) is not None:
        named["think_gain"] = outputs[-1]
    return {name: tensor.numpy() for name, tensor in named.items()}


def _onnx_outputs(path, planes):
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(None, {"data": planes.numpy()})))


def _max_differences(reference, ours):
    """Max |difference| per output but think_gain, over entries finite in both; inf when the two are not
    finite in the same places (an FP16 network overflows to the same inf on random planes)."""
    import numpy as np
    report = {}
    for name in reference:
        if name == "think_gain":
            continue
        a, b = reference[name].astype(np.float64), ours[name].astype(np.float64)
        finite = np.isfinite(a)
        if not np.array_equal(finite, np.isfinite(b)) or not np.array_equal(a[~finite], b[~finite]) \
                and not (np.isnan(a[~finite]) & np.isnan(b[~finite])).all():
            report[name] = math.inf
        else:
            report[name] = float(np.abs(a[finite] - b[finite]).max(initial=0.0))
    return report


def add_head(checkpoint, out, positions=64, seed=0):
    """A headed copy of a twin-board checkpoint, exported and parity checked (module docstring). Returns the
    record (also written next to the ONNX); raises when parity fails."""
    import tempfile

    import numpy as np
    import torch

    from hivemind.architectures.twin_board import load_twin_board_checkpoint
    from hivemind.distill.train import export

    checkpoint, out = Path(checkpoint), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = torch.load(checkpoint, map_location="cpu")
    plain, size, plain_variant = load_twin_board_checkpoint(source, think_gain=False)
    torch.manual_seed(seed)  # the head's hidden layer
    headed, _, headed_variant = load_twin_board_checkpoint(source, think_gain=True)
    plain.eval()
    headed.eval()

    stem = f"{checkpoint.stem}-think"
    pt, onnx_path = out / f"{stem}.pt", out / f"{stem}.onnx"
    torch.save({"model": headed.state_dict(), "size": size, "variant": headed_variant,
                "headed_from": {"path": str(checkpoint), "sha256": sha256(checkpoint)}}, pt)

    # Random legal positions (hivemind.wide's parity inputs), narrowed for a clocked network.
    from hivemind.constants import NUM_WIDE_CHANNELS_PER_BOARD
    from hivemind.wide import narrow_inputs, parity_inputs
    planes = parity_inputs(positions, seed)
    if headed.planes_per_board != NUM_WIDE_CHANNELS_PER_BOARD:
        planes = narrow_inputs(planes)
    reference, ours = _torch_outputs(plain, planes), _torch_outputs(headed, planes)
    report = {"torch": _max_differences(reference, ours)}
    report["torch"]["think_gain_logit_error"] = float(np.abs(ours["think_gain"] - INITIAL_LOGIT).max())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        export(out, size, checkpoint_name=pt.name, path=onnx_path)
        # The source exported by the same exporter, as the ONNX reference.
        torch.save({"model": plain.state_dict(), "size": size, "variant": plain_variant}, tmp / "plain.pt")
        export(tmp, size, checkpoint_name="plain.pt", path=tmp / "plain.onnx")
        reference, ours = _onnx_outputs(tmp / "plain.onnx", planes), _onnx_outputs(onnx_path, planes)
    if "think_gain" not in ours:
        raise RuntimeError(f"{onnx_path} has no think_gain output")
    report["onnx"] = _max_differences(reference, ours)
    report["onnx"]["think_gain_logit_error"] = float(np.abs(ours["think_gain"] - INITIAL_LOGIT).max())
    report["onnx"]["think_gain_shape"] = list(ours["think_gain"].shape)
    record = {"network": stem, "headed_from": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
              "size": size, "variant": headed_variant, "positions": positions, "seed": seed,
              "parity": report, "files": {path.name: sha256(path) for path in (pt, onnx_path)}}
    (out / f"{stem}.json").write_text(json.dumps(record, indent=1) + "\n")
    torch_worst = max(v for k, v in report["torch"].items() if k in PARITY_OUTPUTS or k == "arrivals")
    onnx_worst = max(v for k, v in report["onnx"].items() if k in PARITY_OUTPUTS or k == "arrivals")
    # Written as "not within" so a NaN fails.
    if not (torch_worst <= TORCH_TOLERANCE and onnx_worst <= ONNX_TOLERANCE
            and report["torch"]["think_gain_logit_error"] <= 1e-6
            and report["onnx"]["think_gain_logit_error"] <= LOGIT_TOLERANCE
            and report["onnx"]["think_gain_shape"] == [positions, 2]):
        raise RuntimeError(f"headed network fails parity: {report}")
    return record


# ---------------------------------------------------------------- CLI

def parser():
    top = argparse.ArgumentParser(prog="python -m hivemind.think", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("labels", help="Label a share of an iteration's records for the think_gain head")
    lab.add_argument("--it", type=Path, required=True, help="The iteration dir (runs/rl-itN)")
    lab.add_argument("--iteration", type=int, required=True)
    lab.add_argument("--model", type=Path, required=True, help="The iteration's generator ONNX")
    lab.add_argument("--engine", type=Path, required=True, help="Engine with the reanalyse command")
    lab.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    lab.add_argument("--factor", type=int, default=DEFAULT_FACTOR)
    lab.add_argument("--segments", nargs="+", help="segNN names (default: every finished segment)")
    lab.add_argument("--max-states", type=int, help="At most this many records per segment")
    lab.add_argument("--run-seed", type=int, default=0)
    lab.add_argument("--parallel-engines", type=int, default=6)
    lab.add_argument("--engine-extra", default="", help="More engine options, one shell-quoted string")
    lab.add_argument("--timeout", type=float, default=None, help="Seconds per engine batch")
    lab.add_argument("--gpus", type=int, default=1, help="GPUs the engine uses, for GPU-second accounting")
    lab.add_argument("--states-name", default=STATES_NAME)
    lab.add_argument("--force", action="store_true", help="Relabel segments that already have labels")
    a = sub.add_parser("add-head", help="A checkpoint with a fresh think_gain head, exported and checked")
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--positions", type=int, default=64)
    a.add_argument("--seed", type=int, default=0)
    return top


def main(argv=None):
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "labels":
        if not 0 < args.fraction <= 1 or args.factor < 2:
            raise SystemExit("--fraction must be in (0, 1] and --factor at least 2")
        from hivemind.cli.reanalyse import ReanalyseError
        try:
            return labels(args)
        except ReanalyseError as error:
            print(f"think labels: {error}", file=sys.stderr, flush=True)
            return 2
    record = add_head(args.checkpoint, args.out, args.positions, args.seed)
    print(json.dumps({"network": record["network"], "files": record["files"], "parity": record["parity"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
