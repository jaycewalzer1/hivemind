#!/usr/bin/env python3
"""Search balance stages (talisman docs/search_balance.md sections 3, 4, 5 and 9): the parts of the controller that
need torch, the engine or the run's files, next to the pure controller in talisman's cluster/search_balance.py.

    hivemind balance minitrain  RUN N [--steps S] [--batch-size B] [--lr LR] [--trunk-lr T] [--init PT] [--dry-run]
    hivemind balance score      RUN N [--probe PATH] [--swarm-probe DIR] [--alpha 1.0] [--device cpu]
    hivemind balance probe      RUN N [--engine E] [--model M] [--games 150] [--parallel-games 6] [--dry-run]
    hivemind balance accounting RUN N [--gpus 1]
    hivemind balance plan|decide|realise|status|simulate ...   (passed to cluster/search_balance.py unchanged)

RUN is the run root (runs/full/<RUN>: iterations in RUN/runs/rl-itN, controller files in RUN/balance). Every stage
reads the iteration's plan, RUN/balance/itN.plan.json (search_balance.py plan), and does nothing but write its
marker when the plan has nothing for it (no challenger, no probe refresh). Nothing here runs unless called: the
rl-loop's default path never calls it.

minitrain  The four compute-matched mini-trains of section 4: for arm in (inc, chal) and seed index s, one
           `hivemind distill-train` from the iteration-start generator checkpoint (the checkpoint whose sha256 is
           rl-itN/stages/train.done "init"; --init overrides) on the arm's segment(s), plan.minitrain.steps steps,
           seed plan.minitrain.seeds[s] (the same for both arms), the same replay slice for all four (the train
           segments of the previous --replay-window iterations, --replay-fraction 0.5), and the arm's
           trainer_args (--search-q-blend for a lambda arm). Seed s trains only on the games whose index has
           game_index % 2 == plan.minitrain.data_half.seed_half[s] (game_index = seat_clocks.tsv "game" - 1, the
           engine's 0-based game number); X and T share seeds, so the halves stay paired. The halves are written
           once as HDST chunks under RUN/balance/itN/mini/data/<seg>_h<half>/ (byte-exact record subsets).
           Out: RUN/balance/itN/mini/<arm>_s<s>/{last.pt, distill-*.onnx, history.json}, <arm>_s<s>.log,
           <arm>_s<s>.done (per-run resume unit), and rl-itN/stages/balance_minitrain.done.
score      Each mini net (last.pt) on the probe's deep targets (plan.probe.path, RUN/balance/probe/deep_vV.dst):
             score = CE_policy (both boards, policy-carrying records) + alpha * WDL log-loss (decided records)
                     + 0.1 * moves-left loss  [+ 0.5 * alpha * WDL log-loss on the swarm probe RUN/probe/ if present]
           (lower is better), the trainer's own loss terms (hivemind.distill.train.evaluate). The previous probe
           version is scored too when its file exists ("previous", for chaining versions). Out:
           RUN/balance/itN/score.json  {"schema": 1, "iteration": N, "probe_version": V, "scores": {"inc_s0",
           "chal_s0", "inc_s1", "chal_s1"}, "d": ..., "sigma_n": ..., "terms": {...}, "nets": {name: sha256},
           "probe": {...}, "swarm_probe": {...} | null, "previous": {...} | null, "alpha": 1.0, "alpha2": 0.5}
           (search_balance.py decide reads "scores"; d_s = inc_s - chal_s), and rl-itN/stages/balance_score.done.
           Never rewritten once written (decide checks its sha256); --force rescores.
probe      Probe-mode self-play (section 3) when plan.probe.refresh: --games (150) games with the iteration's
           generator (the current best network while iteration N trains), seed 1000 N + 998, the segments' wave 2
           and time-model options, plus --deep-fraction 0.125 --deep-multiplier 16 --deep-dirichlet false
           --fast-policy-mask true (only the 12,800-node deep records keep a policy; every record keeps its outcome
           WDL). Out: RUN/balance/probe/deep_vV.dst (all chunks merged into one HDST file, readable by
           hivemind.distill.data.read_chunk, by distill-train and by cluster/probe_net.py build --sources),
           RUN/balance/probe/deep_vV.json (sidecar: model + sha256, seed, settings, records, deep/masked counts,
           GPU seconds), RUN/balance/probe/deep_vV.work/ (engine output, log; resumable with --checkpoint-games),
           and rl-itN/stages/balance_probe.done. Never recomputed once the sidecar exists.
accounting GPU seconds per self-play segment from the segment markers' own timings ("gpu_s", or "wall_s" x --gpus)
           when they carry one, else from RUN/runs/rl-loop.log: the segment's start line ("itN self-play segNN:") to
           its "itN segNN labels:" line (the marker's mtime without one), x --gpus (GPUs per self-play job, 1 on
           PACE). A segment resumed or restarted (more than one start line, or "resuming") is left out, as is one
           with no start line (a distributed contributor): no number is better than a guess, and realise then takes
           no gamma sample. Out: RUN/balance/itN/accounting.json {"segNN": gpu_s, ..., "excluded": {segNN: why},
           "sources": {segNN: how}, "stages": {name: gpu_s}} (search_balance.py realise reads the segNN keys) and
           RUN/balance/itN/overhead.json {"gpu_s": {"minitrain", "score", "probe", "minigate"}} from the balance
           stage markers (decide's overhead ledger). Rerunnable; run it before `realise`, then `decide`.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time

from hivemind.paths import PROJECT_ROOT

SCHEMA = 1
PROBE_F, PROBE_K, PROBE_GAMES = 0.125, 16, 150
PROBE_SEED_OFFSET = 998                  # 1000 N + 998; 999 is validation
ALPHA, ALPHA2_SHARE, MOVES_LEFT_WEIGHT = 1.0, 0.5, 0.1
REPLAY_FRACTION = 0.5
ARMS = ("inc", "chal")
TALISMAN_ROOT = PROJECT_ROOT.parents[1] if len(PROJECT_ROOT.parents) > 1 else PROJECT_ROOT
CONTROLLER = TALISMAN_ROOT / "cluster" / "search_balance.py"
CONTROLLER_COMMANDS = ("plan", "decide", "realise", "status", "simulate")
DEFAULT_ENGINE = PROJECT_ROOT / "engine" / "build-sp1" / "hivemind"
ENGINE_DIR = PROJECT_ROOT / "engine"


class BalanceError(RuntimeError):
    pass


# ---------------------------------------------------------------- small helpers

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sig(x, digits=6):
    """Floats to 6 significant digits (search_balance.py's rounding), so reruns give the same bytes."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, digits) for v in x]
    return x


def write_json(path, data):
    """Atomic (tmp + rename), sorted keys, rounded floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_text(json.dumps(sig(data), indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def soft_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def aside(path, aborted):
    """Move an unfinished output out of the way (never deleted)."""
    path = Path(path)
    if path.exists():
        aborted.mkdir(parents=True, exist_ok=True)
        base = f"{path.name}-{datetime.now():%Y%m%d-%H%M%S}"
        target, k = aborted / base, 1
        while target.exists():
            target, k = aborted / f"{base}-{k}", k + 1
        path.rename(target)


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


class Paths:
    def __init__(self, run, n):
        self.run = Path(run).resolve()
        self.n = n
        self.it = self.run / "runs" / f"rl-it{n}"
        self.stages = self.it / "stages"
        self.balance = self.run / "balance"
        self.bit = self.balance / f"it{n}"
        self.plan = self.balance / f"it{n}.plan.json"
        self.mini = self.bit / "mini"
        self.probe_dir = self.balance / "probe"

    def segment_data(self, seg, n=None):
        it = self.it if n is None else self.run / "runs" / f"rl-it{n}"
        return it / "search" / "train" / seg

    def segment_records(self, seg):
        return self.it / "search" / "train_run" / seg

    def marker(self, stage):
        return self.stages / f"{stage}.done"


def load_plan(p):
    if not p.plan.is_file():
        raise BalanceError(f"no plan at {p.plan} (run `search_balance.py plan` first)")
    plan = read_json(p.plan)
    if plan.get("iteration") != p.n:
        raise BalanceError(f"{p.plan} is for iteration {plan.get('iteration')}, not {p.n}")
    return plan


def mark(p, stage, value):
    p.stages.mkdir(parents=True, exist_ok=True)
    write_json(p.marker(stage), dict(value, finished=now()))


# ---------------------------------------------------------------- HDST chunks, byte-exact

PER_RECORD = (("plane_bits", "<i8", "planes"), ("plane_values", "<f2", "planes"), ("value", "<f4", 1),
              ("wdl", "<f4", 3), ("moves_left", "<f2", 1), ("ply", "<u2", 1), ("outcome", "<i1", 1),
              ("policy_count", "<u2", 2))
TIMING = (("seat_clocks_ms", "<f4", 4), ("elapsed_ms", "<f4", 2), ("recent_landed", "<u1", 20),
          ("arrivals", "<f4", 20))
HEADER = 28


def read_raw(path):
    """A chunk's arrays exactly as stored (no widening): (version, planes, arrays)."""
    import numpy as np
    raw = Path(path).read_bytes()
    if raw[:4] != b"HDST":
        raise BalanceError(f"{path}: not an HDST chunk")
    version, planes = struct.unpack_from("<II", raw, 4)
    n, entries = struct.unpack_from("<QQ", raw, 12)
    offset, out = HEADER, {}

    def take(name, dtype, width, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count * width, offset)
        offset += array.nbytes
        out[name] = array.reshape(count, width)

    for name, dtype, width in PER_RECORD:
        take(name, dtype, planes if width == "planes" else width, n)
    take("policy_index", "<u2", 1, entries)
    take("policy_probability", "<f2", 1, entries)
    if version == 3:
        for name, dtype, width in TIMING:
            take(name, dtype, width, n)
    if offset != len(raw) or int(out["policy_count"].astype(np.int64).sum()) != entries:
        raise BalanceError(f"{path}: truncated or inconsistent chunk")
    return version, planes, out


def take_rows(arrays, rows):
    """The given records of a chunk's arrays, their policy entries included."""
    import numpy as np
    rows = np.asarray(rows, np.int64)
    totals = arrays["policy_count"].astype(np.int64).sum(1)
    starts = np.cumsum(totals) - totals
    take = totals[rows]
    within = np.arange(int(take.sum())) - np.repeat(np.cumsum(take) - take, take)
    entries = np.repeat(starts[rows], take) + within
    return {k: (v[entries] if k in ("policy_index", "policy_probability") else v[rows]) for k, v in arrays.items()}


def write_raw(path, version, planes, parts):
    """One HDST chunk from record subsets of chunks with the same version and planes."""
    import numpy as np
    names = [name for name, _, _ in PER_RECORD] + (["policy_index", "policy_probability"])
    timing = [name for name, _, _ in TIMING] if version == 3 else []
    cat = {k: np.concatenate([part[k] for part in parts]) for k in names + timing}
    n, entries = len(cat["value"]), len(cat["policy_index"])
    body = [b"HDST", struct.pack("<II", version, planes), struct.pack("<QQ", n, entries)]
    body += [np.ascontiguousarray(cat[k]).tobytes() for k in names + timing]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(b"".join(body))
    temporary.replace(path)
    return n


def merge_chunks(paths, out):
    """Every record of `paths` (in order) into one chunk at `out`; returns (records, policy records)."""
    parts, kind = [], None
    for path in paths:
        version, planes, arrays = read_raw(path)
        if kind is not None and kind != (version, planes):
            raise BalanceError(f"{path}: version/planes {version}/{planes} differ from {kind}")
        kind = (version, planes)
        parts.append(arrays)
    if not parts:
        raise BalanceError("no chunks to merge")
    n = write_raw(out, kind[0], kind[1], parts)
    policy = sum(int((part["policy_count"].astype("int64").sum(1) > 0).sum()) for part in parts)
    return n, policy


def record_games(seat_clocks):
    """The 0-based game index of every record of a segment, in record order (seat_clocks.tsv rows sorted by
    (game, record), a resumed run's repeated rows counted once: hivemind.distill.timing.read_seat_rows)."""
    keys = {}
    with open(seat_clocks, newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            keys[(int(row["game"]), int(row["record"]))] = None
    return [game - 1 for game, _ in sorted(keys)]


def half_subset(src_dir, seat_clocks, half, out_dir):
    """The records of the games with game_index % 2 == half, chunk by chunk, into out_dir (written once; a
    .done file says it is complete). Returns {"records", "games", "source_records"}."""
    from hivemind.distill.data import chunk_paths
    done = out_dir / "subset.done"
    if done.is_file():
        return read_json(done)
    aside(out_dir, out_dir.parent / "aborted")
    if not seat_clocks.is_file():
        raise BalanceError(f"no {seat_clocks}: cannot split {src_dir} by game (seat-clock self-play only)")
    games = record_games(seat_clocks)
    paths = chunk_paths([src_dir])
    offset, kept, chosen_games = 0, 0, set()
    out_dir.mkdir(parents=True)
    for k, path in enumerate(paths):
        version, planes, arrays = read_raw(path)
        count = len(arrays["value"])
        if offset + count > len(games):
            raise BalanceError(f"{src_dir}: more records than {seat_clocks} rows ({len(games)})")
        rows = [i for i in range(count) if games[offset + i] % 2 == half]
        chosen_games.update(games[offset + i] for i in rows)
        if rows:
            kept += write_raw(out_dir / f"half{half}_{k:05d}.dst", version, planes, [take_rows(arrays, rows)])
        offset += count
    if offset != len(games):
        raise BalanceError(f"{src_dir}: {offset} records but {len(games)} rows in {seat_clocks}")
    info = {"source": str(src_dir), "half": half, "records": kept, "source_records": offset,
            "games": len(chosen_games)}
    write_json(done, info)
    return info


# ---------------------------------------------------------------- minitrain

def iteration_init(p, explicit=None):
    """(path, sha256) of the checkpoint iteration N's main training started from (train.done "init")."""
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise BalanceError(f"--init {path} not found")
        return path, sha256(path)
    done = soft_json(p.marker("train")) or {}
    want = done.get("init")
    if not want:
        raise BalanceError(f"no init sha256 in {p.marker('train')} (minitrain runs after train); pass --init")
    candidates = [p.run / "runs" / f"rl-it{p.n - 1}" / "arms" / "C" / "last.pt"]
    candidates += sorted((p.run / "artifacts").rglob("*.pt")) if (p.run / "artifacts").is_dir() else []
    from hivemind.cli import train as train_cli
    candidates.append(Path(train_cli.GENERATOR_CHECKPOINT))
    for path in candidates:
        if path.is_file() and sha256(path) == want:
            return path.resolve(), want
    raise BalanceError(f"no checkpoint with train.done's init sha256 {want[:12]} (looked in rl-it{p.n - 1}/arms/C, "
                       f"{p.run / 'artifacts'}, the generator checkpoint); pass --init")


def replay_dirs(p, window):
    """The fixed old slice shared by all four mini-trains: the train segments of the previous `window`
    iterations (none for iteration 1)."""
    out = []
    for m in range(max(1, p.n - window), p.n):
        stages = p.run / "runs" / f"rl-it{m}" / "stages"
        out += [p.segment_data(marker.stem, m) for marker in sorted(stages.glob("seg*.done"))]
    return [d for d in out if d.is_dir()]


def run_command(command, log, env=None, cwd=None):
    """Run one stage process, its output to `log`; returns (exit code, wall seconds)."""
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as stream:
        stream.write(" ".join(shlex.quote(str(c)) for c in command) + "\n")
        stream.flush()
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=cwd,
                              env={**os.environ, **(env or {})}).returncode
    return code, time.monotonic() - started


def minitrain(args):
    from hivemind.cli import train as train_cli
    from hivemind.distill.data import chunk_paths, chunk_positions
    p = Paths(args.run, args.iteration)
    plan = load_plan(p)
    mini = plan.get("minitrain")
    if not plan.get("challenger") or not mini:
        if not args.dry_run:
            mark(p, "balance_minitrain", {"skipped": "no challenger in the plan", "gpu_s": 0.0})
        print(f"it{p.n}: no challenger, no mini-trains")
        return 0
    steps = args.steps or int(mini["steps"])
    seeds = [int(s) for s in mini["seeds"]]
    halves = (mini.get("data_half") or {}).get("seed_half") or [None] * len(seeds)
    if len(halves) != len(seeds):
        raise BalanceError(f"plan.minitrain: {len(seeds)} seeds but data_half.seed_half {halves}")
    init, init_sha = iteration_init(p, args.init)
    val = p.it / "search" / "val"
    if not chunk_paths([val]):
        raise BalanceError(f"no validation chunks in {val}")
    replay = replay_dirs(p, args.replay_window)
    fraction = REPLAY_FRACTION if replay else 0.0
    search_rows = round(args.batch_size * (1 - fraction))
    extra_user = shlex.split(args.extra or "")
    runs, total_gpu = {}, 0.0
    for arm in ARMS:
        spec = mini["arms"][arm]
        for s, (seed, half) in enumerate(zip(seeds, halves)):
            name = f"{arm}_s{s}"
            data = []
            for seg in spec["data"]:
                src = p.segment_data(seg)
                if not (p.stages / f"{seg}.done").is_file() or not chunk_paths([src]):
                    raise BalanceError(f"{seg}: not finished or no chunks in {src}")
                if half is None:
                    data.append(src)
                elif args.dry_run:
                    data.append(p.mini / "data" / f"{seg}_h{half}")
                else:
                    half_subset(src, p.segment_records(seg) / "seat_clocks.tsv", int(half),
                                p.mini / "data" / f"{seg}_h{half}")
                    data.append(p.mini / "data" / f"{seg}_h{half}")
            out, log, done = p.mini / name, p.mini / f"{name}.log", p.mini / f"{name}.done"
            extra = [*spec.get("trainer_args", []), "--eval-every", str(steps), "--train-probe", "0", *extra_user]
            command = train_cli.distill_command(init, data, [val], replay, steps, fraction, args.batch_size, seed,
                                                out, extra=extra, lr=args.lr, trunk_lr=args.trunk_lr)
            settings = {"init_sha256": init_sha, "data": [str(d) for d in data], "half": half, "seed": seed,
                        "steps": steps, "batch_size": args.batch_size, "lr": str(args.lr),
                        "trunk_lr": None if args.trunk_lr is None else str(args.trunk_lr),
                        "replay": [str(d) for d in replay], "replay_fraction": fraction,
                        "trainer_args": list(spec.get("trainer_args", [])), "extra": extra_user}
            if args.dry_run:
                print(" ".join(shlex.quote(str(c)) for c in command))
                continue
            positions = sum(chunk_positions(path) for path in chunk_paths(data))
            if positions < search_rows:
                raise BalanceError(f"{name}: {positions} positions in {[str(d) for d in data]}, fewer than the "
                                   f"{search_rows} search rows of one batch; lower --batch-size")
            if done.is_file():
                old = read_json(done)
                if old.get("settings") == sig(settings):
                    runs[name] = old
                    total_gpu += old.get("gpu_s", 0.0)
                    continue
                raise BalanceError(f"{done} was made with other settings; move balance/it{p.n}/mini aside to redo")
            aside(out, p.mini / "aborted")
            code, wall = run_command(command, log, env=train_cli.TRAINING_ENV, cwd=PROJECT_ROOT)
            onnx = next(out.glob("distill-*.onnx"), None) if out.is_dir() else None
            if code or not (out / "last.pt").is_file() or onnx is None:
                raise BalanceError(f"{name}: distill-train failed (exit {code}; see {log})")
            result = {"settings": settings, "positions": positions, "wall_s": round(wall, 3),
                      "gpu_s": round(wall * args.gpus, 3), "last_sha256": sha256(out / "last.pt"),
                      "onnx": str(onnx), "command": [str(c) for c in command]}
            write_json(done, result)
            runs[name] = read_json(done)
            total_gpu += result["gpu_s"]
            print(f"it{p.n} mini-train {name}: {steps} steps on {positions} positions in {wall:.0f}s", flush=True)
    if args.dry_run:
        return 0
    mark(p, "balance_minitrain", {"runs": {k: {x: v.get(x) for x in ("gpu_s", "positions", "last_sha256")}
                                           for k, v in runs.items()},
                                  "gpu_s": round(total_gpu, 3), "steps": steps, "init_sha256": init_sha})
    return 0


# ---------------------------------------------------------------- score

def load_model(path, device):
    import torch
    from hivemind.architectures.twin_board import load_twin_board_checkpoint
    model, _, _ = load_twin_board_checkpoint(Path(path))
    return model.to(device).eval()


def chunk_dataset(paths, model, device):
    from hivemind.distill import train
    from hivemind.distill.data import DistillDataset, read_chunk
    layout = train.layout(model)
    return DistillDataset([read_chunk(path, layout=layout) for path in paths], device, source=1)


def swarm_dataset(probe_dir, model, device):
    """The swarm's frozen probe set (cluster/probe_net.py build: probe/manifest.json + probe.npz)."""
    import numpy as np
    from hivemind.distill import train
    from hivemind.distill.data import DistillDataset, narrow_planes, widen_planes
    manifest = read_json(probe_dir / "manifest.json")
    path = probe_dir.parent / manifest["path"] if not Path(manifest["path"]).is_absolute() else Path(manifest["path"])
    if not path.is_file():
        path = probe_dir / Path(manifest["path"]).name
    with np.load(path, allow_pickle=False) as data:
        chunk = {k: data[k] for k in data.files if k not in ("category", "source", "source_kind", "fen")}
    wide = train.layout(model) == "wide"
    stored_wide = chunk["plane_bits"].shape[1] == 130
    if wide and not stored_wide:
        chunk["plane_bits"], chunk["plane_values"] = widen_planes(chunk)
    elif stored_wide and not wide:
        chunk["plane_bits"], chunk["plane_values"] = narrow_planes(chunk["plane_bits"]), \
            narrow_planes(chunk["plane_values"])
    n = len(chunk["value"])
    chunk.setdefault("arrivals", np.zeros((n, 20), np.float32))
    chunk.setdefault("has_arrivals", np.zeros(n, bool))
    return DistillDataset([chunk], device, source=1), {"path": str(path), "sha256": manifest.get("sha256"),
                                                       "positions": n}


def evaluate_net(net, probe_paths, swarm_dir, alpha, device, batch):
    """{"score", terms...} of one mini net (see the module doc for the formula)."""
    from hivemind.distill import train
    model = load_model(net, device)
    weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": alpha, "value": 0.0, "moves_left": MOVES_LEFT_WEIGHT,
               "arrivals": 0.0}
    deep = train.evaluate(model, chunk_dataset(probe_paths, model, device), weights, batch_size=batch)
    out = {"deep_loss": deep["loss"], "policy_a": deep.get("policy_a"), "policy_b": deep.get("policy_b"),
           "wdl": deep.get("wdl"), "moves_left": deep.get("moves_left"), "top1_a": deep.get("top1_a"),
           "top1_b": deep.get("top1_b")}
    score = deep["loss"]
    if swarm_dir is not None:
        data, _ = swarm_dataset(swarm_dir, model, device)
        swarm = train.evaluate(model, data, {"wdl": 1.0}, batch_size=batch)
        out["swarm_wdl"] = swarm["wdl"]
        score += ALPHA2_SHARE * alpha * swarm["wdl"]
    out["score"] = score
    return out


def probe_info(path):
    from hivemind.distill.data import chunk_paths
    records = policy = 0
    for chunk in chunk_paths([path]):
        _, _, arrays = read_raw(chunk)
        records += len(arrays["value"])
        policy += int((arrays["policy_count"].astype("int64").sum(1) > 0).sum())
    return {"path": str(path), "sha256": sha256(path) if Path(path).is_file() else None, "records": records,
            "policy_records": policy}


def score_diffs(scores, seeds):
    ds = [scores[f"inc_s{s}"] - scores[f"chal_s{s}"] for s in range(seeds)]
    d = sum(ds) / len(ds)
    return d, (abs(ds[0] - ds[1]) / math.sqrt(2) if len(ds) >= 2 else None)


def score(args):
    import torch
    from hivemind.distill.data import chunk_paths
    p = Paths(args.run, args.iteration)
    plan = load_plan(p)
    out = p.bit / "score.json"
    if not plan.get("challenger") or not plan.get("minitrain"):
        mark(p, "balance_score", {"skipped": "no challenger in the plan", "gpu_s": 0.0})
        print(f"it{p.n}: no challenger, nothing to score")
        return 0
    if out.is_file() and not args.force:
        print(f"{out} exists; kept (decide checks its sha256; --force rescores)")
        if not p.marker("balance_score").is_file():
            mark(p, "balance_score", {"path": str(out), "sha256": sha256(out), "gpu_s": 0.0, "kept": True})
        return 0
    version = (plan.get("probe") or {}).get("version")
    probe = Path(args.probe) if args.probe else (p.run / plan["probe"]["path"] if plan["probe"].get("path") else None)
    if probe is None or not probe.exists() or not chunk_paths([probe]):
        raise BalanceError(f"no probe deep targets at {probe} (run `hivemind balance probe` for version {version})")
    previous = None
    if not args.probe and version and version > 1:
        prev = p.probe_dir / f"deep_v{version - 1}.dst"
        previous = prev if prev.exists() and chunk_paths([prev]) else None
    swarm = Path(args.swarm_probe) if args.swarm_probe else p.run / "probe"
    if not (swarm / "manifest.json").is_file():
        if args.swarm_probe:
            raise BalanceError(f"no swarm probe at {swarm}/manifest.json")
        swarm = None
    seeds = len(plan["minitrain"]["seeds"])
    nets = {f"{arm}_s{s}": p.mini / f"{arm}_s{s}" / "last.pt" for arm in ARMS for s in range(seeds)}
    missing = [str(path) for path in nets.values() if not path.is_file()]
    if missing:
        raise BalanceError(f"mini nets missing (run minitrain): {missing}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    started = time.monotonic()
    terms, scores, prev_scores = {}, {}, {}
    for name, net in nets.items():
        terms[name] = evaluate_net(net, chunk_paths([probe]), swarm, args.alpha, device, args.batch)
        scores[name] = terms[name]["score"]
        if previous is not None:
            prev_scores[name] = evaluate_net(net, chunk_paths([previous]), swarm, args.alpha, device,
                                             args.batch)["score"]
        print(f"it{p.n} score {name}: {scores[name]:.6g}", flush=True)
    wall = time.monotonic() - started
    d, sigma_n = score_diffs(scores, seeds)
    result = {"schema": SCHEMA, "iteration": p.n, "probe_version": version, "alpha": args.alpha,
              "alpha2": ALPHA2_SHARE * args.alpha if swarm is not None else 0.0, "scores": scores, "d": d,
              "sigma_n": sigma_n, "terms": terms, "nets": {k: sha256(v) for k, v in nets.items()},
              "probe": probe_info(probe),
              "swarm_probe": swarm_dataset_info(swarm) if swarm is not None else None,
              "previous": ({"probe_version": version - 1, "scores": prev_scores,
                            "d": score_diffs(prev_scores, seeds)[0]} if previous is not None else None),
              "device": str(device)}
    write_json(out, result)
    gpu = round(wall * (args.gpus if device.type == "cuda" else 0.0), 3)
    mark(p, "balance_score", {"path": str(out), "sha256": sha256(out), "wall_s": round(wall, 3), "gpu_s": gpu,
                              "d": d, "sigma_n": sigma_n})
    print(out)
    return 0


def swarm_dataset_info(swarm):
    manifest = read_json(swarm / "manifest.json")
    return {"path": str(swarm), "sha256": manifest.get("sha256"), "positions": manifest.get("positions")}


# ---------------------------------------------------------------- probe

def engine_help(engine):
    try:
        result = subprocess.run([str(engine), "--help"], capture_output=True, text=True, timeout=60,
                                cwd=ENGINE_DIR if ENGINE_DIR.is_dir() else None)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout + result.stderr


def probe_engine_args(k=PROBE_K, f=PROBE_F):
    return ["--deep-fraction", f"{f:g}", "--deep-multiplier", str(k), "--deep-dirichlet", "false",
            "--fast-policy-mask", "true"]


def segment_wave2(p):
    """The wave 2 options the iteration's segments were played with (seg00's marker, else val's)."""
    from hivemind.cli.reanalyse import wave2_engine_args
    for name in ("seg00", "val"):
        marker = soft_json(p.marker(name))
        if marker:
            return wave2_engine_args(marker)
    for marker in sorted(p.stages.glob("seg*.done")):
        return wave2_engine_args(read_json(marker))
    return []


def probe(args):
    from hivemind.cli import selfplay
    from hivemind.cli.reanalyse import find_model, private_model_copy
    from hivemind.distill.data import chunk_paths
    p = Paths(args.run, args.iteration)
    plan = load_plan(p)
    pr = plan.get("probe") or {}
    version = args.version or pr.get("version")
    if not args.version and not pr.get("refresh"):
        if not args.dry_run:
            mark(p, "balance_probe", {"skipped": "no probe refresh in the plan", "version": version, "gpu_s": 0.0})
        print(f"it{p.n}: no probe refresh planned (version {version})")
        return 0
    if not version:
        raise BalanceError("no probe version (plan.probe.version or --version)")
    name = f"deep_v{version}"
    out, sidecar, work = p.probe_dir / f"{name}.dst", p.probe_dir / f"{name}.json", p.probe_dir / f"{name}.work"
    if sidecar.is_file() and out.is_file():
        info = read_json(sidecar)
        print(f"{sidecar} exists; kept (never recomputed)")
        if not args.dry_run and not p.marker("balance_probe").is_file():
            mark(p, "balance_probe", {"version": version, "path": str(out), "sha256": info.get("sha256"),
                                      "gpu_s": 0.0, "kept": True})
        return 0
    generator = (p.stages / "generator").read_text().strip() if (p.stages / "generator").is_file() else None
    model = Path(args.model) if args.model else (find_model(p.run, generator) if generator else None)
    if model is None or not model.is_file():
        raise BalanceError(f"no model: pass --model (the iteration's generator {str(generator)[:12]} was not found "
                           f"under {p.run}/models)")
    model_sha = sha256(model)
    seed = args.seed if args.seed is not None else 1000 * p.n + PROBE_SEED_OFFSET
    games = args.games
    cap = probe_engine_args(args.multiplier, args.fraction)
    data, records, log = work / "data", work / "records", work / "selfplay.log"
    settings = {"version": version, "model_sha256": model_sha, "seed": seed, "games": games,
                "f": args.fraction, "k": args.multiplier, "b": 800, "nodes": 800, "deep_dirichlet": False,
                "fast_policy_mask": True, "time_model": args.time_model, "engine_extra": args.engine_extra or ""}
    if args.dry_run:
        extra = [*selfplay.time_model_options(args.time_model), *segment_wave2(p), *cap,
                 *shlex.split(args.engine_extra or "")]
        print(" ".join(shlex.quote(str(c)) for c in selfplay.engine_command(
            args.engine, model, games, seed, data, records, args.parallel_games, extra)))
        return 0
    if not Path(args.engine).is_file():
        raise BalanceError(f"engine not found at {args.engine}")
    help_text = engine_help(args.engine)
    if "--deep-fraction" not in help_text and not args.skip_engine_check:
        raise BalanceError(f"{args.engine} has no --deep-fraction (build the search-balance engine)")
    plan_file = work / "plan.json"
    if plan_file.is_file() and read_json(plan_file) != sig(settings):
        raise BalanceError(f"{plan_file} was written with other settings; pass another --version")
    progress = records / "checkpoint" / "progress"
    checkpointing = args.checkpoint_games > 0 and "--checkpoint-games" in help_text
    resuming = checkpointing and progress.is_file()
    if not resuming:
        aside(work, p.probe_dir / "aborted")
    write_json(plan_file, settings)
    model_copy = private_model_copy(model, work / "models", model_sha)
    extra = [*selfplay.time_model_options(args.time_model), *segment_wave2(p), *cap,
             *shlex.split(args.engine_extra or "")]
    if checkpointing:
        extra += ["--checkpoint-games", str(args.checkpoint_games), "--resume", "true"]
    command = selfplay.engine_command(args.engine, model_copy, games, seed, data.resolve(), records.resolve(),
                                      args.parallel_games, extra)
    records.mkdir(parents=True, exist_ok=True)
    print(f"it{p.n} probe v{version}: {'resuming ' if resuming else ''}{games} games, seed {seed}", flush=True)
    started = time.monotonic()
    with open(log, "a" if resuming else "w") as stream:
        stream.write(" ".join(shlex.quote(str(c)) for c in command) + "\n")
        stream.flush()
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                              cwd=ENGINE_DIR if ENGINE_DIR.is_dir() else None).returncode
    wall = time.monotonic() - started
    previous_wall = num((soft_json(work / "wall.json") or {}).get("wall_s")) or 0.0 if resuming else 0.0
    total_wall = previous_wall + wall
    write_json(work / "wall.json", {"wall_s": total_wall})      # every attempt's time counts
    if code:
        raise BalanceError(f"probe self-play failed with exit code {code} (see {log})")
    lines = log.read_text(errors="replace").splitlines()
    if checkpointing:
        if f"selfplay complete {games} games" not in lines:
            raise BalanceError(f"no completion line in {log}")
    elif (count := sum(line.startswith("selfplay game ") for line in lines)) != games:
        raise BalanceError(f"expected {games} games in {log}, got {count}")
    if args.wide and (records / "seat_clocks.tsv").is_file():
        from hivemind.distill import timing
        timing.attach_timing(log, records / "seat_clocks.tsv", data)
    chunks = chunk_paths([data])
    n, policy = merge_chunks(chunks, out)
    cap_counts = soft_json(data / "cap_counts.json") or {}
    info = {"schema": SCHEMA, "version": version, "iteration": p.n, "path": str(out.relative_to(p.run)),
            "sha256": sha256(out), "model": str(model), "model_sha256": model_sha, "seed": seed, "games": games,
            "settings": settings, "engine_args": cap, "records": n, "policy_records": policy,
            "cap_counts": cap_counts, "chunks": [c.name for c in chunks], "wall_s": round(total_wall, 3),
            "gpu_s": round(total_wall * args.gpus, 3), "created": now()}
    write_json(sidecar, info)
    mark(p, "balance_probe", {"version": version, "path": str(out), "sha256": info["sha256"],
                              "records": n, "policy_records": policy, "wall_s": info["wall_s"],
                              "gpu_s": info["gpu_s"]})
    print(f"probe v{version}: {n} records, {policy} with a deep policy -> {out}")
    return 0


# ---------------------------------------------------------------- accounting

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")


def log_time(stamp):
    return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))


def loop_log_events(run, n):
    """{name: {"starts": [(t, resuming)], "end": t}} for iteration n's self-play segments and val, plus
    "train"/"match_generator" start times, from RUN/runs/rl-loop.log."""
    path = Path(run) / "runs" / "rl-loop.log"
    events = {}
    if not path.is_file():
        return events
    start = re.compile(rf"^it{n} self-play (seg\d+|val): (resuming )?\d+ games, seed \d+")
    end = re.compile(rf"^it{n} (seg\d+|val) (labels|timing): ")
    stage = {"train": re.compile(rf"^it{n} training: "), "match_generator": re.compile(rf"^it{n} match vs generator")}
    # the line that ends a stage (the next stage's first line): its timestamp is the stage's end
    stage_end = {"train": re.compile(rf"^it{n}(:? no matches| match vs generator)"),
                 "match_generator": re.compile(rf"^it{n} vs generator: ")}
    with open(path, errors="replace") as stream:
        for line in stream:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            t, text = log_time(m.group(1)), m.group(2)
            if s := start.match(text):
                events.setdefault(s.group(1), {"starts": [], "end": None})["starts"].append((t, bool(s.group(2))))
            elif e := end.match(text):
                ev = events.setdefault(e.group(1), {"starts": [], "end": None})
                if ev["end"] is None:          # the labels line comes before the timing line
                    ev["end"] = t
            else:
                for name, pattern in stage.items():
                    ev = events.get(name)
                    if ev and ev["starts"] and ev["end"] is None and stage_end[name].match(text):
                        ev["end"] = t
                    if pattern.match(text):
                        events.setdefault(name, {"starts": [], "end": None})["starts"].append((t, False))
    return events


def segment_seconds(p, name, events, gpus):
    """(gpu_s, how) or (None, why) for one finished segment."""
    marker_path = p.marker(name)
    marker = soft_json(marker_path)
    if marker is None:
        return None, "not finished"
    if num(marker.get("gpu_s")) is not None:
        return float(marker["gpu_s"]), "marker gpu_s"
    for k in ("wall_s", "seconds"):
        if num(marker.get(k)) is not None:
            return float(marker[k]) * gpus, f"marker {k} x {gpus} GPU"
    ev = events.get(name)
    if not ev or not ev["starts"]:
        return None, "no start line in rl-loop.log (distributed contributor or other log)"
    if len(ev["starts"]) > 1 or ev["starts"][0][1]:
        return None, f"resumed or restarted ({len(ev['starts'])} start lines): partial time unknown"
    t0 = ev["starts"][0][0]
    t1, how = ev["end"], "rl-loop.log start to labels line"
    if t1 is None:
        t1, how = marker_path.stat().st_mtime, "rl-loop.log start to marker mtime"
    if t1 <= t0:
        return None, "end before start"
    return (t1 - t0) * gpus, f"{how} x {gpus} GPU"


def stage_gpu(p, stage):
    marker = soft_json(p.marker(stage))
    if not marker:
        return None
    if num(marker.get("gpu_s")) is not None:
        return float(marker["gpu_s"])
    return None


def accounting(args):
    p = Paths(args.run, args.iteration)
    plan = soft_json(p.plan) or {}
    names = sorted(set(plan.get("segments", {})) | {m.stem for m in p.stages.glob("seg*.done")}) \
        if p.stages.is_dir() else sorted(plan.get("segments", {}))
    events = loop_log_events(p.run, p.n)
    out, sources, excluded = {}, {}, {}
    for name in names:
        gpu, how = segment_seconds(p, name, events, args.gpus)
        if gpu is None:
            excluded[name] = how
        else:
            out[name], sources[name] = round(gpu, 3), how
    stages = {}
    val, how = segment_seconds(p, "val", events, args.gpus)
    if val is not None:
        stages["val"] = round(val, 3)
    for stage, marker in (("train", "train"), ("match_generator", "match_generator")):
        ev = events.get(stage)
        if ev and len(ev["starts"]) == 1 and p.marker(marker).is_file():
            end = ev["end"] if ev["end"] is not None else p.marker(marker).stat().st_mtime
            seconds = end - ev["starts"][0][0]
            if seconds > 0:
                stages[stage] = round(seconds * args.gpus, 3)
    overhead = {}
    for item in ("minitrain", "score", "probe", "minigate"):
        gpu = stage_gpu(p, f"balance_{item}")
        if gpu is not None:
            overhead[item] = gpu
            stages[f"balance_{item}"] = gpu
    acc = dict(out, sources=sources, excluded=excluded, stages=stages, gpus_per_job=args.gpus, iteration=p.n)
    write_json(p.bit / "accounting.json", acc)
    if overhead:
        write_json(p.bit / "overhead.json", {"gpu_s": overhead})
    print(p.bit / "accounting.json")
    return 0


# ---------------------------------------------------------------- controller passthrough

def controller(argv):
    path = Path(os.environ.get("HIVEMIND_SEARCH_BALANCE", CONTROLLER))
    if not path.is_file():
        raise BalanceError(f"controller not found at {path} (set HIVEMIND_SEARCH_BALANCE)")
    spec = importlib.util.spec_from_file_location("search_balance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


# ---------------------------------------------------------------- CLI

def parser():
    top = argparse.ArgumentParser(prog="hivemind balance", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("run", type=Path, help="Run root (runs/full/<RUN>)")
        sp.add_argument("iteration", type=int)
        sp.add_argument("--gpus", type=float, default=1.0, help="GPUs a stage holds, for GPU-second accounting")

    m = sub.add_parser("minitrain", help="The compute-matched mini-trains of the plan's two arms")
    common(m)
    m.add_argument("--steps", type=int, help="Override plan.minitrain.steps (local tests)")
    m.add_argument("--batch-size", type=int, default=1024, help="The loop's --batch-size")
    m.add_argument("--lr", default="1e-4", help="The loop's --lr")
    m.add_argument("--trunk-lr", default=None, help="The loop's --trunk-lr (5e-5 on the wave 2 runs)")
    m.add_argument("--init", type=Path, help="Start checkpoint (default: the one train.done's init sha256 names)")
    m.add_argument("--replay-window", type=int, default=1, help="Iterations of old segments in the replay slice")
    m.add_argument("--extra", default="", help="More distill-train options, one shell-quoted string "
                                               "(e.g. '--arrivals-weight 0.05' for wide runs)")
    m.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("score", help="Score the mini nets on the probe's deep targets -> balance/itN/score.json")
    common(s)
    s.add_argument("--probe", help="Deep-target chunk file or dir (default: the plan's probe path)")
    s.add_argument("--swarm-probe", help="Swarm probe dir (default RUN/probe when it has a manifest)")
    s.add_argument("--alpha", type=float, default=ALPHA)
    s.add_argument("--device", default="cpu")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--batch", type=int, default=1024)
    s.add_argument("--force", action="store_true", help="Rescore although score.json exists")

    pr = sub.add_parser("probe", help="Probe-mode self-play -> balance/probe/deep_vV.dst")
    common(pr)
    pr.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    pr.add_argument("--model", type=Path, help="Default: the run's ONNX with the iteration's generator sha256")
    pr.add_argument("--version", type=int, help="Write this version although the plan has no refresh")
    pr.add_argument("--games", type=int, default=PROBE_GAMES)
    pr.add_argument("--seed", type=int, help="Default 1000 N + 998")
    pr.add_argument("--fraction", type=float, default=PROBE_F)
    pr.add_argument("--multiplier", type=int, default=PROBE_K)
    pr.add_argument("--parallel-games", type=int, default=6)
    pr.add_argument("--time-model", default="clock", help="The loop's --time-model")
    pr.add_argument("--checkpoint-games", type=int, default=10, help="Resume unit (0 = off)")
    pr.add_argument("--engine-extra", default="", help="More engine options, one shell-quoted string")
    pr.add_argument("--wide", action="store_true", help="Attach seat timing (v3 chunks), like the loop's --wide")
    pr.add_argument("--skip-engine-check", action="store_true")
    pr.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("accounting", help="GPU seconds per segment and balance stage -> balance/itN/accounting.json")
    common(a)
    return top


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in CONTROLLER_COMMANDS:
        try:
            return controller(argv)
        except BalanceError as error:
            print(f"balance: {error}", file=sys.stderr, flush=True)
            return 2
    args = parser().parse_args(argv)
    try:
        return {"minitrain": minitrain, "score": score, "probe": probe, "accounting": accounting}[args.command](args)
    except BalanceError as error:
        print(f"balance: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
