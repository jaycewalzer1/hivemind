#!/usr/bin/env python3
"""Deep search targets by reanalysis (talisman docs/search_balance.md, sections 6
and 9; the src = reanalyse arm).

Self-play played with `--record-state-output` leaves one JSON line per training
record (rl-itN/search/train_run/segNN/states.jsonl): the exact board pair, clocks
and hybrid state the record was searched in. This tool picks a deterministic,
hash-chosen fraction F of those records and has the engine search them again
at K x 800 nodes with the iteration's generator (the engine's `reanalyse`
command), then writes the new targets as ordinary HDST chunks the trainer reads:

  rl-itN/search/deep/<arm>/segNN/deep_sKKK_NNNNN.dst   the deep targets
  rl-itN/search/deep/<arm>/segNN/selected.tsv          which records (record_index, shard) were reanalysed,
                                                       for the trainer's --dedupe-reanalysed
  rl-itN/search/deep/<arm>/plan.json                   the arm's settings, fixed once written
  rl-itN/search/deep/<arm>/deep.json                   totals, written when every segment is done
  rl-itN/stages/deep_<arm>_segNN.done, deep_<arm>.done markers, like rl-loop's

Subcommands:

  run     reanalyse one iteration's segments for one arm. Resumable: work is split in shards of
          --shard-states states, each verified and then marked done (deep/<arm>/work/segNN/sKKK.done); a rerun
          skips finished shards and segments, moves an unfinished shard's leftovers to deep/<arm>/aborted, and
          refuses to continue when the arm's settings changed (plan.json).
          A segment played without a state dump can be replayed with --replay-missing-states: its self-play is
          played again (same seed, games, generator and wave 2 options, plus --record-state-output) into
          deep/<arm>/replay/segNN, and that replay becomes the segment's source (its chunks are compared with the
          original's and the marker says whether they are byte-identical).
  probe   deep targets for the frozen probe set of cluster/probe_net.py (runs/full/<RUN>/probe/): every probe
          position whose segment has a state dump is searched at --nodes (default 16 x 800) with --model, into
          <RUN>/balance/probe/frozen_deep_vV/ and its sidecar frozen_deep_vV.json. Never recomputed once written.
  status  print the JSON summary of an arm (markers, totals) for the loop, the watchdog and the panel.

Every output is checked before it is marked done: the engine must write exactly one record per state, and the
multiset of (input planes, WDL, outcome, moves-left) of its records must equal that of the selected source
records (the engine asserts plane equality per record too; this catches a misaligned state dump), every record
must carry a policy, and every chunk must pass hivemind.distill.data.chunk_problem.

The engine command (spec section 6, coded against the spec while the engine side is built):

  hivemind reanalyse --model M.onnx --states SHARD.jsonl --nodes N --dirichlet false
                     --replay-noise false --parallel-engines P --output OUT_DIR <wave 2 engine options>

The state dump is the engine's own format (selfplay --record-state-output): a header line, then per game one
"game" line (its moves) followed by one "record" line per training record. A record's record_index is its
position among all record lines of the dump, which is its position among the segment's HDST records.

No --select is passed: the selection is made here, so it is known (and recorded) before any GPU time is spent.
Each shard's states file is the dump's header, the game lines of the games it touches and its chosen record
lines, in dump order. --source-dst is not passed: the engine numbers source records by their order in the file
it is given, so on a subset it would compare against the wrong records. Instead this tool checks every output
record, in order, against its own source record (planes, WDL, outcome, moves-left; and the engine's
reanalysed.tsv (game, record) against the dump), and counts policy/value equality with the source (the
fidelity figures; meaningful with --replay-noise true on a segment played with --root-scans false).
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from hivemind.paths import PROJECT_ROOT

SCHEMA = 1
BASE_NODES = 800
PROBE_MULTIPLIER = 16                 # spec section 3: probe depth 16 x 800, >= 2 x the largest arm k
V_GPU_S_PER_VISIT = 4.24e-5           # spec section 2, measured on hybrid1 it1 (RTX6000, sp1, TRT fp16)
STATES_NAME = "states.jsonl"
DEFAULT_SALT = "reanalyse"
DEFAULT_ENGINE = PROJECT_ROOT / "engine" / "build-sp1" / "hivemind"
ENGINE_DIR = PROJECT_ROOT / "engine"


class ReanalyseError(RuntimeError):
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


def write_json(path, data):
    """Atomic (tmp + rename), sorted keys: the same data always gives the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def aside(path, aborted):
    """Move an unfinished stage's leftovers out of the way (never deleted)."""
    path = Path(path)
    if path.exists():
        aborted.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = aborted / f"{path.name}-{stamp}"
        k = 1
        while target.exists():
            target, k = aborted / f"{path.name}-{stamp}-{k}", k + 1
        path.rename(target)


def selection_value(salt, run_seed, iteration, segment, record_index):
    """A uniform number in [0, 1) fixed by the record's identity alone: the same records are chosen whatever
    order they are read in, on any machine, and the cap coin of self-play (another salt) is independent of it."""
    key = f"{salt}|{run_seed}|{iteration}|{segment}|{record_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2.0 ** 64


def wave2_engine_args(marker):
    """The wave 2 engine options a segment was played with, from its rl-loop marker (stock when absent)."""
    from hivemind import wave2
    saved = marker.get("wave2")
    if not saved:
        return []
    saved = dict(saved)
    saved["base_mix"] = wave2.parse_base_mix(saved["base_mix"]) if saved.get("base_mix") else ()
    return wave2.Options(**saved).engine_args()


def engine_help(engine):
    try:
        result = subprocess.run([str(engine), "--help"], capture_output=True, text=True, timeout=60,
                                cwd=ENGINE_DIR if ENGINE_DIR.is_dir() else None)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout + result.stderr


def engine_command(engine, model, states, source_dst, nodes, output, wave2_args=(), parallel_engines=6,
                   dirichlet=False, replay_noise=False, extra=()):
    """The engine's reanalyse command line (spec section 6)."""
    return [
        str(engine), "reanalyse",
        "--model", str(model),
        "--states", str(states),
        *(["--source-dst", str(source_dst)] if source_dst is not None else []),
        "--nodes", str(nodes),
        "--dirichlet", "true" if dirichlet else "false",
        "--replay-noise", "true" if replay_noise else "false",
        "--parallel-engines", str(parallel_engines),
        "--output", str(output),
        *wave2_args,
        *extra,
    ]


# ---------------------------------------------------------------- run layout

def iteration_dir(run, iteration):
    return Path(run) / "runs" / f"rl-it{iteration}"


def segment_dirs(it, name):
    """(chunk dir, engine records dir) of a segment named as rl-loop names it: segNN or val."""
    if name == "val":
        return it / "search" / "val", it / "search" / "val_run"
    return it / "search" / "train" / name, it / "search" / "train_run" / name


def records_dir_of(data_dir):
    """The engine records dir (games.pgn, seat_clocks.tsv, states.jsonl) of a chunk dir."""
    data_dir = Path(data_dir)
    if data_dir.name == "val" and data_dir.parent.name == "search":
        return data_dir.parent / "val_run"
    if data_dir.parent.name == "train":
        return data_dir.parent.parent / "train_run" / data_dir.name
    return data_dir.parent / f"{data_dir.name}_run"


def finished_segments(it):
    return sorted(marker.stem for marker in (it / "stages").glob("seg*.done"))


def chunk_offsets(data_dir):
    """[(chunk path, first record_index, positions)] in the order the engine wrote them, and the total."""
    from hivemind.distill.data import chunk_paths, chunk_positions
    out, offset = [], 0
    for path in chunk_paths([data_dir]):
        count = chunk_positions(path)
        out.append((path, offset, count))
        offset += count
    return out, offset


def find_model(run, sha):
    """An ONNX file of the run with this SHA-256 (the iteration's generator), preferring a non -sp1 copy."""
    run = Path(run)
    candidates = sorted(run.glob("models/*.onnx")) + sorted(run.glob("runs/rl-it*/models/*.onnx"))
    candidates.sort(key=lambda p: p.stem.endswith("-sp1"))
    for path in candidates:
        if sha256(path) == sha:
            return path
    return None


def private_model_copy(model, directory, sha):
    """The -sp1 copy the one-worker engine loads, kept in our own directory so nothing in the run's models/ is
    written (TensorRT caches are named after the ONNX file and land next to it)."""
    copy = Path(directory) / f"generator-{sha[:12]}-sp1.onnx"
    if not (copy.is_file() and sha256(copy) == sha):
        copy.parent.mkdir(parents=True, exist_ok=True)
        temporary = copy.with_name(copy.name + ".tmp")
        shutil.copyfile(model, temporary)
        temporary.replace(copy)
    return copy


# ---------------------------------------------------------------- state dumps

class StateDump:
    """A parsed engine state dump: the header line, game lines by game number, and the record lines by
    record_index (their order among the dump's record lines), all byte-exact."""

    def __init__(self, header):
        self.header = header
        self.games = {}       # game number -> its "game" line
        self.records = {}     # record_index -> its "record" line (only the wanted ones)
        self.game_of = {}     # record_index -> (game number, record number within the game)
        self.nodes = {}       # record_index -> target_nodes (every record, for the selection)
        self.deep = {}        # record_index -> whether self-play already searched it deep (playout cap)

    def shard_text(self, record_indices):
        """The states file the engine reanalyses these records from: header, each touched game's line, then
        the chosen record lines, in dump order."""
        out, last_game = [self.header], None
        for index in sorted(record_indices):
            game = self.game_of[index][0]
            if game != last_game:
                out.append(self.games[game])
                last_game = game
            out.append(self.records[index])
        return "".join(line + "\n" for line in out)


def read_states(path, total_records, wanted=None):
    """The engine's state dump (header, game and record lines), checked against the segment's chunks: one
    record line per HDST record, numbered 0 to total_records - 1 in dump order, every record after its game's
    line with record numbers 0, 1, ... within the game. Keeps the record lines of `wanted` indices (all when
    None) and the game lines they need; lines stay byte-exact. Returns a StateDump."""
    dump = None
    index = 0
    current, expected_record = None, 0
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                state = json.loads(line)
                kind = state["type"]
            except (ValueError, KeyError, TypeError) as error:
                raise ReanalyseError(f"{path}:{number}: not a state line ({error})")
            if dump is None:
                if kind != "header":
                    raise ReanalyseError(f"{path}:{number}: the dump does not start with its header line")
                dump = StateDump(line)
                continue
            if kind == "game":
                current, expected_record = int(state["game"]), 0
                if current in dump.games:
                    raise ReanalyseError(f"{path}:{number}: game {current} appears twice")
                dump.games[current] = line
                continue
            if kind != "record":
                raise ReanalyseError(f"{path}:{number}: unknown state line type {kind!r}")
            try:
                game, record = int(state["game"]), int(state["record"])
                target = int(state["target_nodes"])
            except (KeyError, TypeError, ValueError) as error:
                raise ReanalyseError(f"{path}:{number}: not a state record line ({error})")
            if current is None or game != current:
                raise ReanalyseError(f"{path}:{number}: record of game {game} not after its game line")
            if record != expected_record:
                raise ReanalyseError(f"{path}:{number}: game {game} record {record} where {expected_record} "
                                     f"was due (record lines out of order or duplicated)")
            expected_record += 1
            if index >= total_records:
                raise ReanalyseError(f"{path}: more state record lines than the segment's {total_records} "
                                     f"records")
            dump.game_of[index] = (game, record)
            dump.nodes[index] = target
            dump.deep[index] = bool(state.get("deep", 0))
            if wanted is None or index in wanted:
                dump.records[index] = line
            index += 1
    if dump is None:
        raise ReanalyseError(f"{path}: empty state dump")
    if index != total_records:
        raise ReanalyseError(f"{path}: {index} state lines for {total_records} records: the dump does not match "
                             f"the chunks (an unfinished or resumed self-play?)")
    needed = {dump.game_of[i][0] for i in dump.records}
    dump.games = {g: l for g, l in dump.games.items() if g in needed}
    return dump


def select_records(lines, nodes, fraction, salt, run_seed, iteration, segment, max_states=None):
    """Record indices to reanalyse: hash value below `fraction`, searched decisions only (a record with
    target_nodes 0 was a raw-policy move), at most `max_states` (the lowest hash values), in record order."""
    chosen = []
    for index in lines:
        if nodes.get(index) == 0:
            continue
        value = selection_value(salt, run_seed, iteration, segment, index)
        if value < fraction:
            chosen.append((value, index))
    chosen.sort()
    if max_states is not None:
        chosen = chosen[:max_states]
    return sorted(index for _, index in chosen)


# ---------------------------------------------------------------- verification

def _layout(paths):
    from hivemind.distill.data import HEADER_BYTES, WIDE_PLANES, _header
    planes = set()
    for path in paths:
        with open(path, "rb") as stream:
            planes.add(_header(stream.read(HEADER_BYTES), path)[2])
    return "wide" if planes == {WIDE_PLANES} else "narrow"


def _keys(chunk, rows=None):
    import numpy as np
    rows = range(len(chunk["value"])) if rows is None else rows
    return [(np.ascontiguousarray(chunk["plane_bits"][r]).tobytes()
             + np.ascontiguousarray(chunk["plane_values"][r]).tobytes(),
             chunk["wdl"][r].tobytes(), int(chunk["outcome"][r]), chunk["moves_left"][r].tobytes())
            for r in rows]


def _targets(chunk, rows=None):
    """(value bytes, policy bytes per board) of these rows: the search targets, for the fidelity counts."""
    import numpy as np
    counts = chunk["policy_count"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts.ravel())])
    rows = range(len(chunk["value"])) if rows is None else rows
    out = []
    for r in rows:
        boards = []
        for b in range(2):
            a, z = starts[2 * r + b], starts[2 * r + b + 1]
            boards.append(chunk["policy_index"][a:z].tobytes() + chunk["policy_probability"][a:z].tobytes())
        out.append((chunk["value"][r].tobytes(), tuple(boards)))
    return out


def source_records(offsets, record_indices, layout):
    """(keys, targets) of the source records with these indices, in index order."""
    from hivemind.distill.data import read_chunk
    keys, targets, wanted = [], [], sorted(record_indices)
    position = 0
    for path, first, count in offsets:
        rows = []
        while position < len(wanted) and wanted[position] < first + count:
            rows.append(wanted[position] - first)
            position += 1
        if rows:
            chunk = read_chunk(path, layout=layout)
            keys += _keys(chunk, rows)
            targets += _targets(chunk, rows)
    if position != len(wanted):
        raise ReanalyseError(f"record indices beyond the source chunks: {wanted[position:][:5]}")
    return keys, targets


def source_keys(offsets, record_indices, layout):
    """(planes, wdl, outcome, moves-left) keys of the source records with these indices."""
    return source_records(offsets, record_indices, layout)[0]


def read_reanalysed_index(path):
    """The engine's reanalysed.tsv: [(game, record)] in output order, or None when it wrote none."""
    if not Path(path).is_file():
        return None
    rows = Path(path).read_text().splitlines()
    return [tuple(int(x) for x in row.split("\t")[:2]) for row in rows[1:] if row.strip()]


def verify_output(out_files, offsets, record_indices, expected_ids=None, written_ids=None):
    """Checks the engine's output chunks record by record, in order, against the source records they
    reanalysed (and, when given, the engine's (game, record) ids against the dump's); returns their totals and
    the fidelity counts (policy/value equal to the source record's)."""
    import numpy as np
    from hivemind.distill.data import chunk_problem, read_chunk
    if not out_files:
        raise ReanalyseError("the engine wrote no chunks")
    for path in out_files:
        problem = chunk_problem(path)
        if problem:
            raise ReanalyseError(f"engine output unfit to train on: {problem}")
    layout = _layout([*out_files, *(p for p, _, _ in offsets)])
    produced, targets, records, entries, values, no_policy = [], [], 0, 0, 0.0, 0
    for path in out_files:
        chunk = read_chunk(path, layout=layout)
        produced += _keys(chunk)
        targets += _targets(chunk)
        records += len(chunk["value"])
        entries += int(chunk["policy_count"].astype(np.int64).sum())
        values += float(chunk["value"].astype(np.float64).sum())
        no_policy += int((chunk["policy_count"].astype(np.int64).sum(1) == 0).sum())
    if records != len(record_indices):
        raise ReanalyseError(f"the engine wrote {records} records for {len(record_indices)} states")
    if no_policy:
        raise ReanalyseError(f"{no_policy} reanalysed records carry no policy")
    if written_ids is not None and expected_ids is not None and list(written_ids) != list(expected_ids):
        raise ReanalyseError("the engine's reanalysed.tsv (game, record) ids differ from the shard's states")
    expected, source_targets = source_records(offsets, record_indices, layout)
    wrong = sum(a != b for a, b in zip(produced, expected))
    if wrong:
        raise ReanalyseError(f"{wrong} of {records} reanalysed records do not match their source record's "
                             f"planes and result: the state dump is misaligned with the chunks")
    value_equal = sum(a[0] == b[0] for a, b in zip(targets, source_targets))
    policy_equal = sum(a[1] == b[1] for a, b in zip(targets, source_targets))
    both_equal = sum(a == b for a, b in zip(targets, source_targets))
    return {"records": records, "policy_entries": entries,
            "mean_value": round(values / records, 6) if records else None,
            "fidelity": {"compared": records, "value_equal": value_equal, "policy_equal": policy_equal,
                         "policy_and_value_equal": both_equal}}


# ---------------------------------------------------------------- one engine batch

def run_batch(job, dump, record_indices, offsets, wave2_args, work, out_dir, prefix):
    """States -> engine -> verified chunks moved to out_dir as <prefix>_NNNNN.dst. Returns the batch's stats."""
    work.mkdir(parents=True, exist_ok=True)
    states = work / STATES_NAME
    write_text(states, dump.shard_text(record_indices))
    output = work / "out"
    output.mkdir()
    command = engine_command(job.engine, job.model_copy, states, None, job.nodes, output, wave2_args,
                             job.parallel_engines, job.dirichlet, job.replay_noise, job.engine_extra)
    started = time.monotonic()
    with (work / "engine.log").open("w") as log:
        log.write(" ".join(shlex.quote(c) for c in command) + "\n")
        log.flush()
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=job.timeout,
                                  cwd=ENGINE_DIR if ENGINE_DIR.is_dir() else None).returncode
        except subprocess.TimeoutExpired:
            raise ReanalyseError(f"engine timed out after {job.timeout}s (see {work / 'engine.log'})")
    seconds = time.monotonic() - started
    if code:
        raise ReanalyseError(f"engine reanalyse failed with exit code {code} (see {work / 'engine.log'})")
    from hivemind.distill.data import chunk_paths
    produced = chunk_paths([output])
    stats = verify_output(produced, offsets, record_indices,
                          [dump.game_of[i] for i in sorted(record_indices)],
                          read_reanalysed_index(output / "reanalysed.tsv"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, path in enumerate(produced):
        path.replace(out_dir / f"{prefix}_{k:05d}.dst")
    stats.update({"states": len(record_indices), "chunks": len(produced), "wall_s": round(seconds, 3),
                  "gpu_s": round(seconds * job.gpus, 3), "nodes": job.nodes,
                  "nominal_visits": len(record_indices) * job.nodes})
    return stats


class Job:
    """Settings shared by every batch of one invocation."""

    def __init__(self, args, nodes, model_copy):
        self.engine = Path(args.engine)
        self.model_copy = model_copy
        self.nodes = nodes
        self.parallel_engines = args.parallel_engines
        self.dirichlet = args.dirichlet == "true"
        self.replay_noise = args.replay_noise == "true"
        self.engine_extra = shlex.split(args.engine_extra or "")
        self.timeout = args.timeout
        self.gpus = args.gpus


# ---------------------------------------------------------------- run

def arm_paths(args):
    it = iteration_dir(args.run, args.iteration)
    default = it / "search" / "deep"
    deep_root = Path(args.deep_root).resolve() if args.deep_root else default
    stages = it / "stages" if deep_root == default.resolve() or deep_root == default else deep_root / "stages"
    return it, deep_root, deep_root / args.arm, stages


def arm_plan(args, generator):
    return {"schema": SCHEMA, "iteration": args.iteration, "arm": args.arm, "multiplier": args.multiplier,
            "base_nodes": args.base_nodes, "nodes": args.multiplier * args.base_nodes, "fraction": args.fraction,
            "salt": args.salt, "run_seed": args.run_seed, "max_states": args.max_states,
            "shard_states": args.shard_states, "dirichlet": args.dirichlet, "replay_noise": args.replay_noise,
            "engine_extra": args.engine_extra or "", "generator": generator}


def segment_source(args, it, arm_dir, stages, name, job, log):
    """(chunk dir, states file, marker) the segment is reanalysed from: its own when it has a state dump, else
    its replay (played here with --replay-missing-states)."""
    data, records = segment_dirs(it, name)
    marker = read_json(it / "stages" / f"{name}.done")
    states = records / args.states_name
    if states.is_file():
        return data, states, {"source": str(data), "replayed": False}
    replay_marker = stages / f"deep_{args.arm}_replay_{name}.done"
    replay = arm_dir / "replay" / name
    if replay_marker.is_file():
        done = read_json(replay_marker)
        return Path(done["source"]), Path(done["states"]), done
    if not args.replay_missing_states:
        raise ReanalyseError(f"{name}: no state dump at {states} (self-play ran without --record-state-output); "
                             f"pass --replay-missing-states to play the segment again with one")
    return replay_segment(args, name, marker, data, replay, replay_marker, job, log)


def replay_segment(args, name, marker, original, replay, replay_marker, job, log):
    from hivemind.cli import selfplay
    aside(replay, replay.parent.parent / "aborted")
    data, records = replay / "data", replay / "records"
    states = records / args.states_name
    games = args.replay_games or marker["games"]
    extra = [*selfplay.time_model_options(args.time_model), *wave2_engine_args(marker),
             *shlex.split(args.replay_extra or ""), "--record-state-output", str(states)]
    command = selfplay.engine_command(args.engine, job.model_copy, games, marker["seed"], data.resolve(),
                                      records.resolve(), args.parallel_games, extra)
    records.mkdir(parents=True, exist_ok=True)
    log(f"{name}: replaying {games} games (seed {marker['seed']}) with a state dump")
    started = time.monotonic()
    with (replay / "selfplay.log").open("w") as stream:
        stream.write(" ".join(shlex.quote(c) for c in command) + "\n")
        stream.flush()
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                              cwd=ENGINE_DIR if ENGINE_DIR.is_dir() else None).returncode
    seconds = time.monotonic() - started
    if code:
        raise ReanalyseError(f"{name}: replay self-play failed with exit code {code} (see {replay / 'selfplay.log'})")
    if not states.is_file():
        raise ReanalyseError(f"{name}: the replay wrote no state dump at {states} (engine without "
                             f"--record-state-output?)")
    from hivemind.distill.data import chunk_paths
    mine, theirs = chunk_paths([data]), chunk_paths([original])
    identical = (games == marker["games"] and [p.name for p in mine] == [p.name for p in theirs]
                 and all(sha256(a) == sha256(b) for a, b in zip(mine, theirs)))
    done = {"source": str(data), "states": str(states), "replayed": True, "games": games, "seed": marker["seed"],
            "identical_to_original": identical, "wall_s": round(seconds, 3), "gpu_s": round(seconds * args.gpus, 3)}
    write_json(replay_marker, done)
    log(f"{name}: replay done in {seconds:.0f}s, chunks {'identical to' if identical else 'differ from'} the "
        f"original's")
    return data, states, done


def run(args):
    it, deep_root, arm_dir, stages = arm_paths(args)
    log = lambda message: print(f"{datetime.now():%Y-%m-%d %H:%M:%S} reanalyse it{args.iteration} {args.arm}: "
                                f"{message}", flush=True)
    if not (it / "stages" / "generator").is_file():
        raise ReanalyseError(f"{it}: no stages/generator (not an rl-loop iteration)")
    generator = (it / "stages" / "generator").read_text().strip()
    segments = args.segments if args.segments and args.segments != ["all"] else finished_segments(it)
    for name in segments:
        if not (it / "stages" / f"{name}.done").is_file():
            raise ReanalyseError(f"{name}: not finished (no {it / 'stages' / (name + '.done')})")
    if not segments:
        raise ReanalyseError(f"{it}: no finished segments")
    model = Path(args.model) if args.model else find_model(args.run, generator)
    if model is None or not model.is_file():
        raise ReanalyseError(f"no model with the iteration's generator sha256 {generator[:12]} under {args.run}; "
                             f"pass --model")
    if sha256(model) != generator and not args.any_model:
        raise ReanalyseError(f"{model} is not it{args.iteration}'s generator ({generator[:12]}); deep targets "
                             f"come from the generator (--any-model overrides)")
    nodes = args.multiplier * args.base_nodes

    if args.dry_run:
        return dry_run(args, it, segments, model, nodes, generator)

    if not Path(args.engine).is_file():
        raise ReanalyseError(f"engine not found at {args.engine}")
    help_text = engine_help(args.engine)
    if "reanalyse" not in help_text and not args.skip_engine_check:
        raise ReanalyseError(f"{args.engine} has no reanalyse command (build the search-balance engine)")

    plan = arm_plan(args, sha256(model))
    plan_path = arm_dir / "plan.json"
    if plan_path.is_file():
        recorded = read_json(plan_path)
        if recorded != plan:
            changed = sorted(k for k in set(plan) | set(recorded) if plan.get(k) != recorded.get(k))
            raise ReanalyseError(f"{plan_path} was written with other settings ({', '.join(changed)}); "
                                 f"use another --arm")
    else:
        write_json(plan_path, plan)
    job = Job(args, nodes, private_model_copy(model, deep_root / "models", plan["generator"]))

    for name in segments:
        marker = stages / f"deep_{args.arm}_{name}.done"
        if marker.is_file():
            log(f"{name}: done")
            continue
        source, states, origin = segment_source(args, it, arm_dir, stages, name, job, log)
        seg_marker = read_json(it / "stages" / f"{name}.done")
        offsets, total = chunk_offsets(source)
        if not total:
            raise ReanalyseError(f"{name}: no chunks under {source} (archived? run `hivemind restore` first)")
        dump = read_states(states, total)
        chosen = select_records(dump.records, dump.nodes, args.fraction, args.salt, args.run_seed, args.iteration,
                                name, args.max_states)
        shards = [chosen[k:k + args.shard_states] for k in range(0, len(chosen), args.shard_states)]
        out_dir = arm_dir / name
        selection = "record_index\tshard\tgame\trecord\n" + "".join(
            f"{index}\t{s}\t{dump.game_of[index][0]}\t{dump.game_of[index][1]}\n"
            for s, shard in enumerate(shards) for index in shard)
        selected_path = out_dir / "selected.tsv"
        if selected_path.is_file() and selected_path.read_text() != selection:
            raise ReanalyseError(f"{selected_path} differs from this run's selection (the source changed)")
        write_text(selected_path, selection)
        wave2_args = wave2_engine_args(seg_marker)
        work_root = arm_dir / "work" / name
        totals = Counter()
        log(f"{name}: {len(chosen)} of {total} records at {nodes} nodes in {len(shards)} shards")
        for s, shard in enumerate(shards):
            shard_marker = work_root / f"s{s:03d}.done"
            if shard_marker.is_file():
                stats = read_json(shard_marker)
            else:
                prefix = f"deep_s{s:03d}"
                for stale in sorted(out_dir.glob(f"{prefix}_*.dst")):
                    aside(stale, arm_dir / "aborted")
                aside(work_root / f"s{s:03d}", arm_dir / "aborted")
                stats = run_batch(job, dump, shard, offsets, wave2_args, work_root / f"s{s:03d}", out_dir,
                                  prefix)
                write_json(shard_marker, stats)
                log(f"{name} shard {s + 1}/{len(shards)}: {stats['records']} records in {stats['wall_s']:.0f}s")
            for key in ("states", "records", "chunks", "policy_entries", "nominal_visits"):
                totals[key] += stats[key]
            totals["gpu_s"] += stats["gpu_s"]
            for key, value in stats.get("fidelity", {}).items():
                totals["fidelity_" + key] += value
        result = {"segment": name, "source": str(source), "states_file": str(states),
                  "states_sha256": sha256(states), "source_records": total, "selected": len(chosen),
                  "shards": len(shards), "deep": str(out_dir), "selected_tsv": str(selected_path),
                  "nodes": nodes, "fraction": args.fraction, "multiplier": args.multiplier,
                  "generator": plan["generator"], "replay": origin,
                  "records": totals["records"], "chunks": totals["chunks"],
                  "policy_entries": totals["policy_entries"], "nominal_visits": totals["nominal_visits"],
                  "gpu_s": round(totals["gpu_s"], 3),
                  "fidelity": {k[len("fidelity_"):]: v for k, v in totals.items() if k.startswith("fidelity_")},
                  # self-play with time-budgeted root scans (the default) is not reproducible, so the fidelity
                  # figures only test the engine when both the source and this pass ran without them
                  "fidelity_valid": fidelity_valid(args, origin)}
        write_json(marker, result)
    summary = arm_summary(args, stages, arm_dir, segments)
    write_json(arm_dir / "deep.json", summary)
    if args.segments in (None, [], ["all"]):
        write_json(stages / f"deep_{args.arm}.done", summary)
    log(f"done: {summary['records']} deep records, {summary['gpu_s'] / 3600:.2f} GPU-h "
        f"(+{summary['replay_gpu_s'] / 3600:.2f} replay)")
    return 0


def _root_scans_off(options):
    words = shlex.split(options or "")
    return any(a == "--root-scans" and b == "false" for a, b in zip(words, words[1:]))


def fidelity_valid(args, origin):
    """Whether this arm's policy/value equality counts can pass the fidelity gate: recorded noise replayed, root
    scans off in the reanalysis, and the source played with root scans off (only known for a replay made here)."""
    return bool(args.replay_noise == "true" and _root_scans_off(args.engine_extra) and origin.get("replayed")
                and _root_scans_off(getattr(args, "replay_extra", "")))


def arm_summary(args, stages, arm_dir, segments):
    per = {name: read_json(stages / f"deep_{args.arm}_{name}.done") for name in segments
           if (stages / f"deep_{args.arm}_{name}.done").is_file()}
    replay_gpu = sum(p["replay"].get("gpu_s", 0) for p in per.values())
    return {"schema": SCHEMA, "arm": args.arm, "iteration": args.iteration, "plan": read_json(arm_dir / "plan.json"),
            "segments": per, "records": sum(p["records"] for p in per.values()),
            "source_records": sum(p["source_records"] for p in per.values()),
            "gpu_s": round(sum(p["gpu_s"] for p in per.values()), 3), "replay_gpu_s": round(replay_gpu, 3),
            "search_data": [p["source"] for p in per.values()], "deep_data": [p["deep"] for p in per.values()]}


def dry_run(args, it, segments, model, nodes, generator):
    report = {"iteration": args.iteration, "arm": args.arm, "nodes": nodes, "model": str(model), "segments": {}}
    for name in segments:
        data, records = segment_dirs(it, name)
        states = records / args.states_name
        offsets, total = chunk_offsets(data)
        entry = {"records": total, "states": str(states) if states.is_file() else None}
        if states.is_file():
            dump = read_states(states, total)
            chosen = select_records(dump.records, dump.nodes, args.fraction, args.salt, args.run_seed,
                                    args.iteration, name, args.max_states)
            entry["selected"] = len(chosen)
            entry["estimated_gpu_h"] = round(len(chosen) * nodes * V_GPU_S_PER_VISIT / args.eta / 3600, 4)
        else:
            entry["selected"] = None
            entry["note"] = "no state dump" + ("; would replay" if args.replay_missing_states else "")
        entry["command"] = engine_command(args.engine, "<model-sp1 copy>", "<shard states>", None, nodes, "<out>",
                                          wave2_engine_args(read_json(it / "stages" / f"{name}.done")),
                                          args.parallel_engines, args.dirichlet == "true",
                                          args.replay_noise == "true", shlex.split(args.engine_extra or ""))
        report["segments"][name] = entry
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def status(args):
    it, deep_root, arm_dir, stages = arm_paths(args)
    markers = sorted(stages.glob(f"deep_{args.arm}_seg*.done")) + sorted(stages.glob(f"deep_{args.arm}_val.done"))
    segments = [m.stem[len(f"deep_{args.arm}_"):] for m in markers]
    out = arm_summary(args, stages, arm_dir, segments) if (arm_dir / "plan.json").is_file() else {"arm": args.arm}
    out["done"] = (stages / f"deep_{args.arm}.done").is_file()
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


# ---------------------------------------------------------------- probe

def probe(args):
    import numpy as np
    root = Path(args.run).resolve()
    manifest_path = root / "probe" / "manifest.json"
    if not manifest_path.is_file():
        raise ReanalyseError(f"no probe set at {manifest_path} (cluster/probe_net.py build first)")
    manifest = read_json(manifest_path)
    probe_path = root / manifest["path"]
    if sha256(probe_path) != manifest["sha256"]:
        raise ReanalyseError(f"{probe_path} does not match its manifest sha256")
    out = Path(args.out).resolve() if args.out else root / "balance" / "probe"
    version = args.version
    if version is None:
        version = 1
        while (out / f"frozen_deep_v{version}.json").is_file():
            version += 1
    name = f"frozen_deep_v{version}"
    sidecar = out / f"{name}.json"
    if sidecar.is_file():
        print(f"{sidecar} exists; kept (never recomputed)")
        return 0
    model = Path(args.model)
    if not model.is_file():
        raise ReanalyseError(f"model not found at {model}")
    model_sha = sha256(model)
    nodes = args.nodes

    with np.load(probe_path, allow_pickle=False) as data:
        sources = [str(s) for s in data["source"]]
        categories = [str(c) for c in data["category"]] if "category" in data else ["all"] * len(sources)
    groups = {}
    for row, source in enumerate(sources):
        chunk, _, index = source.rpartition("#")
        chunk_path = (root / chunk).resolve()
        groups.setdefault(chunk_path.parent, []).append((row, chunk_path, int(index)))

    work_root = out / f"{name}.work"
    if not args.dry_run:
        if not Path(args.engine).is_file():
            raise ReanalyseError(f"engine not found at {args.engine}")
        if "reanalyse" not in engine_help(args.engine) and not args.skip_engine_check:
            raise ReanalyseError(f"{args.engine} has no reanalyse command (build the search-balance engine)")
        plan = {"schema": SCHEMA, "version": version, "nodes": nodes, "model_sha256": model_sha,
                "probe_sha256": manifest["sha256"], "dirichlet": args.dirichlet, "replay_noise": args.replay_noise,
                "engine_extra": args.engine_extra or ""}
        if (work_root / "plan.json").is_file() and read_json(work_root / "plan.json") != plan:
            raise ReanalyseError(f"{work_root / 'plan.json'} was written with other settings; pass another --version")
        write_json(work_root / "plan.json", plan)
    job = None
    covered_rows, group_stats, uncovered = [], {}, Counter()
    for gk, (data_dir, rows) in enumerate(sorted(groups.items())):
        key = str(data_dir.relative_to(root)) if data_dir.is_relative_to(root) else str(data_dir)
        offsets, total = chunk_offsets(data_dir)
        first = {path.resolve(): offset for path, offset, _ in offsets}
        missing_chunk = [r for r, chunk, _ in rows if chunk not in first]
        states = records_dir_of(data_dir) / args.states_name
        if missing_chunk or not states.is_file():
            why = "chunk missing" if missing_chunk else "no state dump"
            uncovered[why] += len(rows)
            group_stats[key] = {"positions": len(rows), "reanalysed": 0, "skipped": why}
            continue
        index_of = {r: first[chunk] + i for r, chunk, i in rows}
        wanted = set(index_of.values())
        dump = read_states(states, total, wanted)
        record_indices = sorted(wanted)
        it_dir = data_dir.parent.parent
        seg_name = "val" if data_dir.name == "val" else data_dir.name
        seg_marker = it_dir / "stages" / f"{seg_name}.done"
        wave2_args = wave2_engine_args(read_json(seg_marker)) if seg_marker.is_file() else []
        covered_rows += [r for r, _, _ in rows]
        if args.dry_run:
            group_stats[key] = {"positions": len(rows), "would_reanalyse": len(record_indices)}
            continue
        if job is None:
            job = Job(args, nodes, private_model_copy(model, work_root / "models", model_sha))
        marker = work_root / f"g{gk:03d}.done"
        prefix = f"g{gk:03d}"
        if marker.is_file():
            stats = read_json(marker)
        else:
            for stale in sorted((out / name).glob(f"{prefix}_*.dst")):
                aside(stale, work_root / "aborted")
            aside(work_root / prefix, work_root / "aborted")
            stats = run_batch(job, dump, record_indices, offsets, wave2_args, work_root / prefix, out / name,
                              prefix)
            stats.update({"group": key, "record_indices": record_indices,
                          "probe_rows": sorted(r for r, _, _ in rows)})
            write_json(marker, stats)
        group_stats[key] = {"positions": len(rows), "reanalysed": stats["records"], "gpu_s": stats["gpu_s"],
                            "prefix": prefix}
    report = {"schema": SCHEMA, "version": version, "path": str((out / name).relative_to(root))
              if (out / name).is_relative_to(root) else str(out / name),
              "nodes": nodes, "model": str(model), "model_sha256": model_sha,
              "probe": {"path": manifest["path"], "sha256": manifest["sha256"], "positions": len(sources)},
              "covered": len(covered_rows), "uncovered": dict(uncovered),
              "by_category": dict(Counter(categories[r] for r in covered_rows)), "groups": group_stats,
              "dirichlet": args.dirichlet, "created": now()}
    if args.dry_run:
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    if not covered_rows:
        print(json.dumps(report, indent=1, sort_keys=True))
        raise ReanalyseError("no probe position has a state dump (the run predates --record-state-output): "
                             "nothing reanalysed; use probe-mode self-play (spec section 3) instead")
    report["gpu_s"] = round(sum(g.get("gpu_s", 0) for g in group_stats.values()), 3)
    report["records"] = sum(g.get("reanalysed", 0) for g in group_stats.values())
    write_json(sidecar, report)
    print(f"{report['records']} of {len(sources)} probe positions reanalysed at {nodes} nodes -> {out / name}")
    return 0


# ---------------------------------------------------------------- CLI

def add_engine_options(parser):
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                        help="Engine with the reanalyse command (and --record-state-output for replays)")
    parser.add_argument("--parallel-engines", type=int, default=6)
    parser.add_argument("--dirichlet", choices=("true", "false"), default="false",
                        help="Root Dirichlet noise in the deep searches (default false: clean targets)")
    parser.add_argument("--replay-noise", choices=("true", "false"), default="false",
                        help="Replay the recorded root noise (the fidelity test's mode)")
    parser.add_argument("--engine-extra", default="", help="More engine options, one shell-quoted string")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds per engine batch")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs the engine uses, for GPU-second accounting")
    parser.add_argument("--states-name", default=STATES_NAME, help="State dump file name in a records dir")
    parser.add_argument("--skip-engine-check", action="store_true",
                        help="Do not require 'reanalyse' in the engine's --help")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run and exit")


def parser():
    top = argparse.ArgumentParser(prog="hivemind reanalyse", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Reanalyse one iteration's segments for one arm")
    r.add_argument("--run", type=Path, required=True, help="Run root (runs/full/<RUN>: holds runs/rl-itN)")
    r.add_argument("--iteration", type=int, required=True)
    r.add_argument("--arm", default="reanalyse", help="Arm name: output in search/deep/<arm>")
    r.add_argument("--multiplier", "-k", type=int, default=4, help="Deep budget = k x --base-nodes (default 4)")
    r.add_argument("--fraction", "-f", type=float, default=0.125, help="Share of records reanalysed (1/8)")
    r.add_argument("--base-nodes", type=int, default=BASE_NODES)
    r.add_argument("--segments", nargs="+", help="segNN names (default: every finished segment; val is never "
                                                  "reanalysed unless named)")
    r.add_argument("--max-states", type=int, help="At most this many records per segment (lowest hash values)")
    r.add_argument("--shard-states", type=int, default=4096, help="States per engine batch (resume unit)")
    r.add_argument("--salt", default=DEFAULT_SALT)
    r.add_argument("--run-seed", type=int, default=0)
    r.add_argument("--model", type=Path, help="Generator ONNX (default: found in the run by the iteration's "
                                               "stages/generator sha256)")
    r.add_argument("--any-model", action="store_true", help="Allow a model other than the generator")
    r.add_argument("--deep-root", type=Path, help="Write here instead of rl-itN/search/deep (markers then go to "
                                                   "<deep-root>/stages): a scratch dir leaves the run untouched")
    r.add_argument("--replay-missing-states", action="store_true",
                   help="Replay a segment without a state dump (same seed and settings) to get one")
    r.add_argument("--replay-games", type=int, help="Replay only this many games (default: the segment's)")
    r.add_argument("--parallel-games", type=int, default=6, help="Replay self-play parallel games")
    r.add_argument("--time-model", default="clock", help="Replay self-play time model (rl-loop's --time-model)")
    r.add_argument("--replay-extra", default="", help="More replay self-play options (e.g. '--tree-reuse true')")
    r.add_argument("--eta", type=float, default=1.0, help="Batching speed-up over self-play nps (dry-run estimate)")
    add_engine_options(r)

    p = sub.add_parser("probe", help="Deep targets for the frozen probe set (runs/full/<RUN>/probe)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="The current best network's ONNX")
    p.add_argument("--nodes", type=int, default=PROBE_MULTIPLIER * BASE_NODES)
    p.add_argument("--version", type=int, help="Output version V (default: the next free)")
    p.add_argument("--out", type=Path, help="Output dir (default <RUN>/balance/probe)")
    add_engine_options(p)

    s = sub.add_parser("status", help="JSON summary of an arm")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--iteration", type=int, required=True)
    s.add_argument("--arm", default="reanalyse")
    s.add_argument("--deep-root", type=Path)
    return top


def main(argv=None):
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        if not 0 < args.fraction <= 1 or args.multiplier < 1 or args.shard_states < 1:
            raise SystemExit("--fraction must be in (0, 1], --multiplier and --shard-states positive")
        if not args.arm.replace("-", "").replace("_", "").isalnum():
            raise SystemExit("--arm: letters, digits, - and _ only")
    try:
        return {"run": run, "probe": probe, "status": status}[args.command](args)
    except ReanalyseError as error:
        print(f"reanalyse: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
