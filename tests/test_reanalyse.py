"""hivemind.cli.reanalyse against a stub engine that reads and writes the real engine's formats: the state dump
of selfplay --record-state-output (header, game and record lines, see write_state_lines in selfplay.cc) and the
reanalyse command's outputs (chunks in states-file order, reanalysed.tsv). Like the engine, the stub numbers a
--source-dst record by its order in the states file it is given."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys

import numpy as np
import pytest

from hivemind import wave2
from hivemind.cli import reanalyse as ra
from hivemind.distill.data import DistillDataset, chunk_paths, read_chunk

TALISMAN = Path(__file__).resolve().parents[3]
FIXTURE = TALISMAN / "runs" / "wave2" / "int-shared"

STUB = r'''#!PYTHON
"""Stub engine: `reanalyse` copies each state's source record (planes, WDL, outcome, moves-left; found through
the stub-only "src" field the test's dump carries, as the real engine recomputes them from the moves) and writes
a one-hot policy and a halved value, in states-file order, plus reanalysed.tsv; `selfplay` copies
STUB_REPLAY_SOURCE's chunks and writes a state dump. STUB_FAIL_ON=N fails the Nth reanalyse call; STUB_CORRUPT=1
flips a plane value. Like the engine it refuses a states file without its header or with a record before its
game line."""
import json, os, struct, sys
from pathlib import Path
import numpy as np
from hivemind.distill.data import read_chunk, chunk_paths

argv = sys.argv[1:]
log = os.environ.get("STUB_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps(argv) + "\n")
def write_dump(path, n, src_dir, raw_every=0, per_game=50):
    """The engine's dump layout: header, then per game a game line and its record lines (record 0, 1, ...).
    "src" (stub only) names the record's source as <data dir>#<record_index>."""
    with open(path, "w") as f:
        f.write(json.dumps({"type": "header", "version": 1, "seed": 1000, "nodes": 800, "max_macro_plies": 400,
                            "seat_clocks": True, "input_channels": 78, "time_state_bytes": 304}) + "\n")
        for k in range(n):
            if k % per_game == 0:
                f.write(json.dumps({"type": "game", "game": k // per_game, "winner": 1,
                                    "termination": "checkmate", "moves": [[0, 593], [1, 3364]]}) + "\n")
            nodes = 0 if raw_every and k % raw_every == 0 else 800
            f.write(json.dumps({"type": "record", "game": k // per_game, "record": k % per_game, "prefix": 0,
                                "ply": k % per_game, "team": 0, "base_nodes": 800, "target_nodes": nodes,
                                "deep": 0, "masked": 0, "outcome": 0, "wdl": [0, 1, 0], "moves_left": 0,
                                "src": f"{src_dir}#{k}"}) + "\n")

if argv == ["--help"]:
    print("commands: selfplay reanalyse tournament; --record-state-output PATH")
    sys.exit(0)
cmd, opts = argv[0], {}
i = 1
while i < len(argv):
    opts[argv[i]] = argv[i + 1]
    i += 2

def write(path, recs, planes):
    n = len(recs)
    counts = np.array([r["counts"] for r in recs], np.uint16).reshape(n, 2)
    idx = np.array([e for r in recs for e in r["index"]], np.uint16)
    prob = np.array([e for r in recs for e in r["prob"]], np.float16)
    with open(path, "wb") as s:
        s.write(b"HDST"); s.write(struct.pack("<II", 2, planes)); s.write(struct.pack("<QQ", n, len(idx)))
        s.write(np.stack([r["bits"] for r in recs]).astype(np.int64).tobytes())
        s.write(np.stack([r["values"] for r in recs]).astype(np.float16).tobytes())
        s.write(np.array([r["value"] for r in recs], np.float32).tobytes())
        s.write(np.stack([r["wdl"] for r in recs]).astype(np.float32).tobytes())
        s.write(np.array([r["ml"] for r in recs], np.float16).tobytes())
        s.write(np.array([r["ply"] for r in recs], np.uint16).tobytes())
        s.write(np.array([r["outcome"] for r in recs], np.int8).tobytes())
        s.write(counts.tobytes()); s.write(idx.tobytes()); s.write(prob.tobytes())

if cmd == "reanalyse":
    counter = Path(os.environ["STUB_COUNTER"]) if os.environ.get("STUB_COUNTER") else None
    calls = (int(counter.read_text()) if counter and counter.exists() else 0) + 1
    if counter:
        counter.write_text(str(calls))
    if os.environ.get("STUB_FAIL_ON") == str(calls):
        sys.exit(7)
    lines = [json.loads(l) for l in open(opts["--states"]) if l.strip()]
    if not lines or lines[0].get("type") != "header":
        print("has no state header", file=sys.stderr); sys.exit(1)
    states, game = [], None
    for line in lines[1:]:
        if line["type"] == "game":
            game = line["game"]; continue
        if game is None or line["game"] != game:
            print("State record before its game line", file=sys.stderr); sys.exit(1)
        states.append(line)
    rows = []
    for p in chunk_paths([os.environ["STUB_SOURCE_ROOT"] + "/" + states[0]["src"].split("#")[0]]) if states else []:
        c = read_chunk(p)
        totals = c["policy_count"].astype(np.int64).sum(1)
        starts = np.cumsum(totals) - totals
        for r in range(len(c["value"])):
            rows.append((c, r, starts[r]))
    recs = []
    for st in states:
        c, r, start = rows[int(st["src"].split("#")[1])]
        counts = [int(x) for x in c["policy_count"][r]]
        index, prob, out_counts, at = [], [], [], start
        for k in counts:
            if k:
                seg = c["policy_index"][at:at + k]
                index.append(int(seg[0])); prob.append(1.0); out_counts.append(1)
            else:
                out_counts.append(0)
            at += k
        if not any(out_counts):
            index.append(0); prob.append(1.0); out_counts = [1, 0]
        values = c["plane_values"][r].copy()
        if os.environ.get("STUB_CORRUPT"):
            values[0] = values[0] + np.float16(0.5)
        recs.append(dict(bits=c["plane_bits"][r], values=values, value=float(c["value"][r]) / 2, wdl=c["wdl"][r],
                         ml=c["moves_left"][r], ply=int(c["ply"][r]), outcome=int(c["outcome"][r]),
                         counts=out_counts, index=index, prob=prob))
    out = Path(opts["--output"])
    with open(out / "reanalysed.tsv", "w") as f:
        f.write("game\trecord\tsource_index\n")
        f.writelines(f"{st['game']}\t{st['record']}\t{k}\n" for k, st in enumerate(states))
    planes = recs[0]["bits"].shape[0]
    half = (len(recs) + 1) // 2
    write(out / "reanalyse_00000.dst", recs[:half], planes)
    if recs[half:]:
        write(out / "reanalyse_00001.dst", recs[half:], planes)
    sys.exit(0)

if cmd == "selfplay":
    src = Path(os.environ["STUB_REPLAY_SOURCE"])
    data, records = Path(opts["--distill-output"]), Path(opts["--output"])
    data.mkdir(parents=True, exist_ok=True); records.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in chunk_paths([src]):
        (data / p.name).write_bytes(p.read_bytes())
        n += len(read_chunk(p)["value"])
    write_dump(opts["--record-state-output"], n, os.path.relpath(data, os.environ["STUB_SOURCE_ROOT"]))
    print("selfplay complete %s games" % opts["--games"])
    sys.exit(0)
sys.exit(3)
'''


def write_chunk(path, rng, n, planes=78):
    """A v2 chunk with distinct planes per record (bits are random bitboards, values set), some records with
    policies on one or both boards."""
    bits = rng.integers(1, 2 ** 62, (n, planes), dtype=np.int64)
    values = np.ones((n, planes), np.float16)
    counts = rng.integers(1, 5, (n, 2)).astype(np.uint16)
    counts[::3, 1] = 0
    entries = int(counts.sum())
    index = rng.integers(0, 4672, entries).astype(np.uint16)
    probability = np.zeros(entries, np.float16)
    at = 0
    for c in counts.ravel():
        if c:
            probability[at:at + c] = np.float16(1.0 / c)
            probability[at] = np.float16(1.0 - float(probability[at + 1:at + c].astype(np.float32).sum()))
        at += c
    wdl = np.zeros((n, 3), np.float32)
    wdl[np.arange(n), rng.integers(0, 3, n)] = 1
    with open(path, "wb") as s:
        s.write(b"HDST")
        s.write(struct.pack("<II", 2, planes))
        s.write(struct.pack("<QQ", n, entries))
        s.write(bits.tobytes())
        s.write(values.tobytes())
        s.write(np.tanh(rng.normal(size=n)).astype(np.float32).tobytes())
        s.write(wdl.tobytes())
        s.write(rng.random(n).astype(np.float16).tobytes())
        s.write(np.arange(n, dtype=np.uint16).tobytes())
        s.write(np.zeros(n, np.int8).tobytes())
        s.write(counts.tobytes())
        s.write(index.tobytes())
        s.write(probability.tobytes())


def write_states(path, n, src_dir, raw_every=0):
    """A dump in the engine's layout (see the stub's write_dump; "src" is relative to STUB_SOURCE_ROOT)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    namespace = {}
    exec(STUB[STUB.index("def write_dump"):STUB.index('if argv == ["--help"]')], {"json": json}, namespace)
    namespace["write_dump"](path, n, src_dir, raw_every)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def run(tmp_path):
    """A one-iteration run: seg00 with a state dump (two chunks), seg01 without one, val with one."""
    rng = np.random.default_rng(0)
    root = tmp_path / "run"
    it = root / "runs" / "rl-it1"
    (root / "models").mkdir(parents=True)
    model = root / "models" / "gen.onnx"
    model.write_bytes(b"fake onnx generator")
    (it / "stages").mkdir(parents=True)
    (it / "stages" / "generator").write_text(sha(model) + "\n")
    for name, sizes, states in (("seg00", (300, 220), True), ("seg01", (150,), False), ("val", (90,), True)):
        data, records = ra.segment_dirs(it, name)
        data.mkdir(parents=True)
        seed = 1999 if name == "val" else 1000 + int(name[3:])
        for k, n in enumerate(sizes):
            write_chunk(data / f"search_{seed}_{k:05d}.dst", rng, n)
        if states:
            write_states(records / "states.jsonl", sum(sizes), str(data.relative_to(tmp_path)), raw_every=10)
        else:
            records.mkdir(parents=True)
        (it / "stages" / f"{name}.done").write_text(json.dumps(
            {"games": 3, "seed": seed, "positions": sum(sizes), "generator": sha(model),
             "wave2": wave2.WAVE2.as_dict()}) + "\n")
    return root


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "stub_engine"
    path.write_text(STUB.replace("PYTHON", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("STUB_COUNTER", str(tmp_path / "calls"))
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "argv.jsonl"))
    for name in ("STUB_FAIL_ON", "STUB_CORRUPT", "STUB_REPLAY_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def run_args(root, engine, *extra):
    return ["run", "--run", str(root), "--iteration", "1", "--engine", str(engine), "--shard-states", "25", *extra]


def calls(env):
    path = env / "calls"
    return int(path.read_text()) if path.exists() else 0


def test_selection_is_deterministic_and_fraction_sized():
    lines = {i: "" for i in range(20000)}
    nodes = {i: (0 if i % 10 == 0 else 800) for i in lines}
    a = ra.select_records(lines, nodes, 0.125, "reanalyse", 0, 1, "seg00")
    b = ra.select_records(dict(reversed(list(lines.items()))), nodes, 0.125, "reanalyse", 0, 1, "seg00")
    assert a == b == sorted(a)
    assert all(i % 10 for i in a)                       # raw-policy (nodes 0) records are never chosen
    expected = 18000 * 0.125
    assert abs(len(a) - expected) < 4 * (18000 * 0.125 * 0.875) ** 0.5
    assert a != ra.select_records(lines, nodes, 0.125, "reanalyse", 0, 1, "seg01")   # salted by segment
    capped = ra.select_records(lines, nodes, 0.125, "reanalyse", 0, 1, "seg00", max_states=10)
    assert len(capped) == 10 and set(capped) <= set(a)


def test_engine_command_follows_the_spec():
    command = ra.engine_command("E", "M.onnx", "S.jsonl", None, 3200, "out", ["--sit-gate", "true"], 6)
    assert command[:2] == ["E", "reanalyse"]
    pairs = dict(zip(command[2::2], command[3::2]))
    assert pairs == {"--model": "M.onnx", "--states": "S.jsonl", "--nodes": "3200",
                     "--dirichlet": "false", "--replay-noise": "false", "--parallel-engines": "6",
                     "--output": "out", "--sit-gate": "true"}
    assert "--select" not in command
    # A shard's states file is a subset: the engine would number --source-dst records by their order in it.
    assert "--source-dst" not in command
    assert "--source-dst" in ra.engine_command("E", "M", "S", "src", 1, "out")


def test_read_states_takes_the_engine_dump_and_shards_keep_header_and_game_lines(tmp_path):
    path = tmp_path / "states.jsonl"
    write_states(path, 120, "d", raw_every=10)
    dump = ra.read_states(path, 120)
    assert sorted(dump.records) == list(range(120)) and dump.game_of[51] == (1, 1)
    assert dump.nodes[0] == 0 and dump.nodes[1] == 800
    text = dump.shard_text([3, 51, 52, 119])
    kinds = [(json.loads(l)["type"], json.loads(l).get("game"), json.loads(l).get("record"))
             for l in text.splitlines()]
    assert kinds == [("header", None, None), ("game", 0, None), ("record", 0, 3), ("game", 1, None),
                     ("record", 1, 1), ("record", 1, 2), ("game", 2, None), ("record", 2, 19)]
    lines = path.read_text().splitlines()
    assert all(l in lines for l in text.splitlines())       # byte-exact lines
    wanted = ra.read_states(path, 120, {51})
    assert list(wanted.records) == [51] and list(wanted.games) == [1]


def test_run_writes_verified_deep_chunks_the_trainer_reads(run, engine, env):
    assert ra.main(run_args(run, engine, "--segments", "seg00", "--fraction", "0.25", "-k", "4")) == 0
    it = run / "runs" / "rl-it1"
    deep = it / "search" / "deep" / "reanalyse" / "seg00"
    marker = json.loads((it / "stages" / "deep_reanalyse_seg00.done").read_text())
    selected = [int(l.split("\t")[0]) for l in (deep / "selected.tsv").read_text().splitlines()[1:]]
    assert marker["selected"] == len(selected) == marker["records"] > 50
    assert marker["source_records"] == 520 and marker["nodes"] == 3200
    assert all(i % 10 for i in selected)
    files = chunk_paths([deep])
    assert files and all(p.name.startswith("deep_s") for p in files)
    chunks = [read_chunk(p) for p in files]
    assert sum(len(c["value"]) for c in chunks) == len(selected)
    import torch
    dataset = DistillDataset(chunks, torch.device("cpu"), source=1)
    assert len(dataset) == len(selected)
    _, _, mask = dataset.policy_targets(torch.arange(len(dataset)), 0)
    assert mask.any()
    # Engine got the segment's wave 2 options and the deep budget; the run's models/ was not written.
    argv = [json.loads(l) for l in (env / "argv.jsonl").read_text().splitlines()]
    reanalyse = [a for a in argv if a[0] == "reanalyse"]
    assert len(reanalyse) == marker["shards"] == calls(env)
    assert "--durative-tree" in reanalyse[0] and reanalyse[0][reanalyse[0].index("--nodes") + 1] == "3200"
    assert sorted(p.name for p in (run / "models").iterdir()) == ["gen.onnx"]
    # Only seg00 asked for: no arm marker; status reports it.
    assert not (it / "stages" / "deep_reanalyse.done").exists()
    summary = json.loads((it / "search" / "deep" / "reanalyse" / "deep.json").read_text())
    assert summary["records"] == len(selected) and summary["deep_data"] == [str(deep)]


def test_rerun_is_a_no_op_and_resume_redoes_only_the_failed_shard(run, engine, env, monkeypatch, tmp_path):
    clean = tmp_path / "clean"
    args = ("--segments", "seg00", "--fraction", "0.25")
    assert ra.main(run_args(run, engine, *args, "--deep-root", str(clean))) == 0
    clean_calls = calls(env)
    assert clean_calls >= 3
    assert ra.main(run_args(run, engine, *args, "--deep-root", str(clean))) == 0
    assert calls(env) == clean_calls                     # finished segment: nothing reruns

    (env / "calls").unlink()
    monkeypatch.setenv("STUB_FAIL_ON", "3")
    resumed = tmp_path / "resumed"
    assert ra.main(run_args(run, engine, *args, "--deep-root", str(resumed))) == 2
    assert not (resumed / "stages" / "deep_reanalyse_seg00.done").exists()
    assert len(list((resumed / "reanalyse" / "work" / "seg00").glob("s*.done"))) == 2
    monkeypatch.delenv("STUB_FAIL_ON")
    assert ra.main(run_args(run, engine, *args, "--deep-root", str(resumed))) == 0
    assert calls(env) == clean_calls + 1                 # two shards kept, the failed one and the rest played
    a = {p.name: sha(p) for p in chunk_paths([clean / "reanalyse" / "seg00"])}
    b = {p.name: sha(p) for p in chunk_paths([resumed / "reanalyse" / "seg00"])}
    assert a == b
    assert (resumed / "reanalyse" / "aborted").is_dir()  # the failed shard's leftovers, moved not deleted
    # The run's own dirs were not touched by --deep-root.
    assert not (run / "runs" / "rl-it1" / "search" / "deep").exists()


def test_misaligned_output_is_refused(run, engine, env, monkeypatch):
    monkeypatch.setenv("STUB_CORRUPT", "1")
    assert ra.main(run_args(run, engine, "--segments", "seg00")) == 2
    stages = run / "runs" / "rl-it1" / "stages"
    assert not (stages / "deep_reanalyse_seg00.done").exists()
    assert not list((run / "runs" / "rl-it1" / "search" / "deep" / "reanalyse" / "seg00").glob("*.dst"))


def test_changed_settings_need_a_new_arm(run, engine, env, capsys):
    assert ra.main(run_args(run, engine, "--segments", "seg00", "--max-states", "30")) == 0
    assert ra.main(run_args(run, engine, "--segments", "seg00", "--max-states", "30", "-k", "2")) == 2
    assert "multiplier" in capsys.readouterr().err
    assert ra.main(run_args(run, engine, "--segments", "seg00", "--max-states", "30", "-k", "2",
                            "--arm", "k2")) == 0


def test_state_dump_must_match_the_chunks(run, engine, env, capsys):
    states = run / "runs" / "rl-it1" / "search" / "train_run" / "seg00" / "states.jsonl"
    lines = states.read_text().splitlines()
    states.write_text("\n".join(lines[:-1]) + "\n")
    assert ra.main(run_args(run, engine, "--segments", "seg00")) == 2
    assert "519 state lines for 520 records" in capsys.readouterr().err
    states.write_text("\n".join(lines[:-1] + [lines[-2]]) + "\n")
    assert ra.main(run_args(run, engine, "--segments", "seg00")) == 2
    assert "out of order or duplicated" in capsys.readouterr().err
    states.write_text("\n".join(lines[1:]) + "\n")
    assert ra.main(run_args(run, engine, "--segments", "seg00")) == 2
    assert "header" in capsys.readouterr().err


def test_missing_state_dump_and_replay(run, engine, env, monkeypatch, capsys):
    it = run / "runs" / "rl-it1"
    assert ra.main(run_args(run, engine, "--segments", "seg01")) == 2
    assert "--replay-missing-states" in capsys.readouterr().err
    monkeypatch.setenv("STUB_REPLAY_SOURCE", str(it / "search" / "train" / "seg01"))
    assert ra.main(run_args(run, engine, "--segments", "seg01", "--replay-missing-states")) == 0
    replay = json.loads((it / "stages" / "deep_reanalyse_replay_seg01.done").read_text())
    assert replay["identical_to_original"] is True and replay["seed"] == 1001
    argv = [json.loads(l) for l in (env / "argv.jsonl").read_text().splitlines()]
    sp = next(a for a in argv if a[0] == "selfplay")
    assert sp[sp.index("--seed") + 1] == "1001" and "--record-state-output" in sp and "--time-model" in sp
    marker = json.loads((it / "stages" / "deep_reanalyse_seg01.done").read_text())
    assert marker["replay"]["replayed"] and marker["source"].endswith("replay/seg01/data")


def test_all_segments_write_the_arm_marker(run, engine, env, monkeypatch):
    it = run / "runs" / "rl-it1"
    monkeypatch.setenv("STUB_REPLAY_SOURCE", str(it / "search" / "train" / "seg01"))
    assert ra.main(run_args(run, engine, "--replay-missing-states", "--max-states", "20")) == 0
    done = json.loads((it / "stages" / "deep_reanalyse.done").read_text())
    assert sorted(done["segments"]) == ["seg00", "seg01"]       # val is never reanalysed unless named
    assert done["records"] == 40


def test_dry_run_writes_nothing(run, engine, env, capsys):
    assert ra.main(run_args(run, engine, "--dry-run")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["segments"]["seg00"]["selected"] > 0 and report["segments"]["seg01"]["selected"] is None
    assert report["segments"]["seg00"]["estimated_gpu_h"] > 0
    assert not (run / "runs" / "rl-it1" / "search" / "deep").exists() and calls(env) == 0


def test_wrong_model_is_refused(run, engine, env, tmp_path, capsys):
    other = tmp_path / "other.onnx"
    other.write_bytes(b"another net")
    assert ra.main(run_args(run, engine, "--segments", "seg00", "--model", str(other))) == 2
    assert "generator" in capsys.readouterr().err


def make_probe(root, rows):
    probe = root / "probe"
    probe.mkdir()
    path = probe / "probe.npz"
    np.savez_compressed(path, source=np.array(rows, dtype=str),
                        category=np.array(["opening"] * len(rows), dtype=str))
    (probe / "manifest.json").write_text(json.dumps({"path": "probe/probe.npz", "sha256": sha(path),
                                                     "positions": len(rows)}))


def test_probe_deep_targets(run, engine, env):
    rows = ["runs/rl-it1/search/train/seg00/search_1000_00001.dst#5",      # record_index 305
            "runs/rl-it1/search/train/seg00/search_1000_00000.dst#7",
            "runs/rl-it1/search/val/search_1999_00000.dst#3",
            "runs/rl-it1/search/train/seg01/search_1001_00000.dst#1"]      # no state dump
    make_probe(run, rows)
    model = run / "models" / "gen.onnx"
    args = ["probe", "--run", str(run), "--model", str(model), "--engine", str(engine)]
    assert ra.main(args) == 0
    sidecar = json.loads((run / "balance" / "probe" / "frozen_deep_v1.json").read_text())
    assert sidecar["covered"] == 3 and sidecar["records"] == 3 and sidecar["nodes"] == 12800
    assert sidecar["uncovered"] == {"no state dump": 1}
    chunks = [read_chunk(p) for p in chunk_paths([run / "balance" / "probe" / "frozen_deep_v1"])]
    assert sum(len(c["value"]) for c in chunks) == 3
    groups = sorted((run / "balance" / "probe" / "frozen_deep_v1.work").glob("g*.done"))
    indices = sorted(i for g in groups for i in json.loads(g.read_text())["record_indices"])
    assert indices == [3, 7, 305]
    before = calls(env)
    assert ra.main(args + ["--version", "1"]) == 0 and calls(env) == before   # kept, never recomputed


def test_probe_without_any_state_dump_fails_cleanly(run, engine, env, capsys):
    make_probe(run, ["runs/rl-it1/search/train/seg01/search_1001_00000.dst#1"])
    assert ra.main(["probe", "--run", str(run), "--model", str(run / "models" / "gen.onnx"),
                    "--engine", str(engine)]) == 2
    assert "predates --record-state-output" in capsys.readouterr().err
    assert not (run / "balance" / "probe" / "frozen_deep_v1.json").exists()


def test_cluster_wrappers():
    wrapper = TALISMAN / "cluster" / "probe_deep_targets.py"
    result = subprocess.run([sys.executable, str(wrapper), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--model" in result.stdout
    assert subprocess.run(["bash", "-n", str(TALISMAN / "cluster" / "reanalyse.sbatch")]).returncode == 0


@pytest.mark.skipif(not (FIXTURE / "runs" / "rl-it1" / "stages" / "seg00.done").is_file(),
                    reason="needs the talisman int-shared fixture")
def test_on_a_copy_of_the_real_fixture(tmp_path, engine, env):
    root = tmp_path / "int-shared"
    it = root / "runs" / "rl-it1"
    (it / "stages").mkdir(parents=True)
    (root / "models").mkdir()
    for name in ("generator", "seg00.done", "seg01.done"):
        shutil.copy(FIXTURE / "runs" / "rl-it1" / "stages" / name, it / "stages" / name)
    for name in ("seg00", "seg01"):
        shutil.copytree(FIXTURE / "runs" / "rl-it1" / "search" / "train" / name, it / "search" / "train" / name)
    for model in (FIXTURE / "models").glob("*.onnx"):
        shutil.copy(model, root / "models" / model.name)
    total = sum(len(read_chunk(p)["value"]) for p in chunk_paths([it / "search" / "train" / "seg00"]))
    write_states(it / "search" / "train_run" / "seg00" / "states.jsonl", total,
                 str((it / "search" / "train" / "seg00").relative_to(tmp_path)))
    assert ra.main(run_args(root, engine, "--segments", "seg00", "--fraction", "0.125", "-k", "2")) == 0
    marker = json.loads((it / "stages" / "deep_reanalyse_seg00.done").read_text())
    assert marker["source_records"] == total and marker["records"] == marker["selected"] > 0
    reanalyse = [json.loads(l) for l in (env / "argv.jsonl").read_text().splitlines() if '"reanalyse"' in l]
    assert "--hybrid-mode" in reanalyse[0]               # hybrid1-style wave 2 options from the marker
