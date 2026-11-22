"""hivemind balance (search balance stages: minitrain, score, probe, accounting) against stubs: a fake run tree in the
rl-loop's layout, plans written by talisman's own controller (cluster/search_balance.py), a stub trainer (the
distill-train command is recorded, not run), a stub scorer and a stub engine that writes real HDST chunks. The real
end-to-end run (engine/build-balance, CPU mini-trains, torch scoring) is in the module doc of hivemind.cli.balance's
report, not here: it needs the int-shared fixture and a minute of CPU."""

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import textwrap

import numpy as np
import pytest

from hivemind.cli import COMMANDS, balance
from hivemind.distill.data import read_chunk

CONTROLLER = balance.CONTROLLER
pytestmark = pytest.mark.skipif(not CONTROLLER.is_file(), reason="needs talisman's cluster/search_balance.py")


def controller():
    spec = importlib.util.spec_from_file_location("search_balance_under_test", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fixtures

def arrays(rng, n, planes=78, version=2, masked_every=0):
    counts = rng.integers(1, 4, size=(n, 2)).astype(np.uint16)
    if masked_every:
        counts[::masked_every] = 0
    entries = int(counts.astype(np.int64).sum())
    probs = np.zeros(entries, np.float16)
    start = 0
    for c in counts.ravel():
        if c:
            probs[start:start + c] = 1.0 / c
            start += c
    out = {"plane_bits": rng.integers(0, 2 ** 62, size=(n, planes)).astype("<i8"),
           "plane_values": np.ones((n, planes), "<f2"),
           "value": rng.uniform(-1, 1, size=(n, 1)).astype("<f4"),
           "wdl": np.tile(np.array([0, 0, 1], "<f4"), (n, 1)),
           "moves_left": np.full((n, 1), 0.5, "<f2"), "ply": np.arange(n, dtype="<u2").reshape(n, 1),
           "outcome": np.ones((n, 1), "<i1"), "policy_count": counts,
           "policy_index": rng.integers(0, 4672, size=(entries, 1)).astype("<u2"),
           "policy_probability": probs.reshape(entries, 1)}
    if version == 3:
        out.update(seat_clocks_ms=np.ones((n, 4), "<f4"), elapsed_ms=np.zeros((n, 2), "<f4"),
                   recent_landed=np.zeros((n, 20), "<u1"), arrivals=np.zeros((n, 20), "<f4"))
    return out


def write_segment(it, name, games, rng, per_game=(7, 5)):
    """A finished segment: chunks (two, splitting a game), seat_clocks.tsv rows (game, record) in record order,
    the rl-loop marker."""
    sizes = [per_game[g % 2] for g in range(games)]
    total = sum(sizes)
    data = it / "search" / "train" / name
    run = it / "search" / "train_run" / name
    data.mkdir(parents=True)
    run.mkdir(parents=True)
    a = arrays(rng, total)
    a["ply"] = np.arange(total, dtype="<u2").reshape(total, 1)      # record id, to check subsets
    cut = total // 2 + 1
    balance.write_raw(data / "search_1_00000.dst", 2, 78, [balance.take_rows(a, range(cut))])
    balance.write_raw(data / "search_1_00001.dst", 2, 78, [balance.take_rows(a, range(cut, total))])
    rows = ["game\trecord\tteam"] + [f"{g + 1}\t{r}\t0" for g, size in enumerate(sizes) for r in range(size)]
    (run / "seat_clocks.tsv").write_text("\n".join(rows) + "\n")
    (it / "stages").mkdir(parents=True, exist_ok=True)
    (it / "stages" / f"{name}.done").write_text(json.dumps({"games": games, "seed": 1, "positions": total}) + "\n")
    return sizes


@pytest.fixture
def run(tmp_path):
    """RUN with rl-it1 (replay source) and rl-it2 (5 segments, val, train.done whose init is artifacts/x/best.pt)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "RUN"
    for n in (1, 2):
        it = root / "runs" / f"rl-it{n}"
        for k in range(5):
            write_segment(it, f"seg{k:02d}", 6, rng)
        val = it / "search" / "val"
        val.mkdir(parents=True)
        balance.write_raw(val / "search_999_00000.dst", 2, 78, [arrays(rng, 20)])
    init = root / "artifacts" / "x" / "best.pt"
    init.parent.mkdir(parents=True)
    init.write_bytes(b"iteration-start checkpoint")
    (root / "runs" / "rl-it1" / "arms" / "C").mkdir(parents=True)
    (root / "runs" / "rl-it1" / "arms" / "C" / "last.pt").write_bytes(b"not the init")
    (root / "runs" / "rl-it2" / "stages" / "train.done").write_text(json.dumps({"init": balance.sha256(init)}))
    (root / "runs" / "rl-it2" / "stages" / "generator").write_text("f" * 64 + "\n")
    return root


def write_plan(root, n, prior, state=None):
    sb = controller()
    state = state or sb.init_state(prior)
    plan = sb.plan_core(n, state, {"prior": prior, "segment_games": 6, "segments": 5, "mini_steps": 8})
    (root / "balance").mkdir(exist_ok=True)
    (root / "balance" / f"it{n}.plan.json").write_text(sb.dumps(plan))
    return plan


DEEP_PRIOR = {"score_stage": True, "probe_bias_measured": True}


@pytest.fixture
def trainer(monkeypatch):
    """Record distill-train commands and leave what a real run leaves (last.pt, ONNX)."""
    calls = []

    def fake(command, log, env=None, cwd=None):
        calls.append([str(c) for c in command])
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "last.pt").write_bytes(" ".join(map(str, command)).encode())
        (out / "distill-twin-s-v3.0.onnx").write_bytes(b"onnx")
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("ok\n")
        return 0, 2.0

    monkeypatch.setattr(balance, "run_command", fake)
    return calls


def opt(command, name):
    i = command.index(name)
    out = []
    for c in command[i + 1:]:
        if c.startswith("--"):
            break
        out.append(c)
    return out


def last_opt(command, name):
    return command[len(command) - 1 - command[::-1].index(name) + 1]


# ---------------------------------------------------------------- registration and chunks

def test_commands_are_registered():
    assert COMMANDS["balance"][0] == "hivemind.cli.balance"
    assert COMMANDS["reanalyse"][0] == "hivemind.cli.reanalyse"


@pytest.mark.parametrize("version,planes", [(2, 78), (3, 78), (3, 130)])
def test_raw_chunks_round_trip_byte_exact(tmp_path, version, planes):
    rng = np.random.default_rng(1)
    a = arrays(rng, 9, planes, version, masked_every=3)
    first = tmp_path / "a.dst"
    balance.write_raw(first, version, planes, [a])
    v, p, back = balance.read_raw(first)
    assert (v, p) == (version, planes)
    second = tmp_path / "b.dst"
    balance.write_raw(second, v, p, [balance.take_rows(back, range(9))])
    assert first.read_bytes() == second.read_bytes()
    rows = [0, 3, 4, 8]
    balance.write_raw(tmp_path / "c.dst", v, p, [balance.take_rows(back, rows)])
    full, part = read_chunk(first, "wide" if planes == 130 else "narrow"), read_chunk(tmp_path / "c.dst",
                                                                                     "wide" if planes == 130 else "narrow")
    assert (part["ply"] == full["ply"][rows]).all() and (part["policy_count"] == full["policy_count"][rows]).all()
    starts = np.concatenate([[0], np.cumsum(full["policy_count"].astype(int).sum(1))])
    want = np.concatenate([full["policy_index"][starts[r]:starts[r + 1]] for r in rows])
    assert (part["policy_index"] == want).all()


def test_merge_chunks_counts_policy_records(tmp_path):
    rng = np.random.default_rng(2)
    paths = []
    for k in range(3):
        paths.append(tmp_path / f"{k}.dst")
        balance.write_raw(paths[-1], 2, 78, [arrays(rng, 8, masked_every=2)])
    n, policy = balance.merge_chunks(paths, tmp_path / "all.dst")
    assert (n, policy) == (24, 12)
    assert len(read_chunk(tmp_path / "all.dst")["value"]) == 24
    balance.write_raw(tmp_path / "v3.dst", 3, 78, [arrays(rng, 2, version=3)])
    with pytest.raises(balance.BalanceError):
        balance.merge_chunks(paths + [tmp_path / "v3.dst"], tmp_path / "bad.dst")


def test_halves_split_by_game_index_across_chunks(run):
    it = run / "runs" / "rl-it2"
    sizes = [7, 5, 7, 5, 7, 5]       # games 0..5 (0-based); records numbered by ply
    first = [sum(sizes[:g]) for g in range(6)]
    for half in (0, 1):
        out = run / "h" / str(half)
        info = balance.half_subset(it / "search/train/seg00", it / "search/train_run/seg00/seat_clocks.tsv", half, out)
        plies = np.concatenate([read_chunk(p)["ply"] for p in sorted(out.glob("*.dst"))])
        want = [first[g] + r for g in range(6) if g % 2 == half for r in range(sizes[g])]
        assert plies.tolist() == want
        assert info["games"] == 3 and info["records"] == len(want) and info["source_records"] == 36
    tsv = it / "search/train_run/seg01/seat_clocks.tsv"
    tsv.write_text("\n".join(tsv.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(balance.BalanceError, match="rows"):
        balance.half_subset(it / "search/train/seg01", tsv, 0, run / "h" / "bad")


# ---------------------------------------------------------------- minitrain

def test_minitrain_follows_the_plan(run, trainer):
    plan = write_plan(run, 2, DEEP_PRIOR)
    assert plan["challenger"]["f"] == 0.125 and plan["challenger_segment"] == "seg04"
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8", "--trunk-lr", "5e-5"]) == 0
    assert len(trainer) == 4
    by = {Path(opt(c, "--out")[0]).name: c for c in trainer}
    assert sorted(by) == ["chal_s0", "chal_s1", "inc_s0", "inc_s1"]
    mini = run / "balance" / "it2" / "mini"
    for s, seed in enumerate(plan["minitrain"]["seeds"]):
        inc, chal = by[f"inc_s{s}"], by[f"chal_s{s}"]
        assert opt(inc, "--seed") == opt(chal, "--seed") == [str(seed)]
        assert opt(inc, "--search-data") == [str(mini / "data" / f"seg00_h{s}")]
        assert opt(chal, "--search-data") == [str(mini / "data" / f"seg04_h{s}")]
        for c in (inc, chal):
            assert opt(c, "--init") == [str((run / "artifacts/x/best.pt").resolve())]
            assert opt(c, "--steps") == ["8"] and last_opt(c, "--eval-every") == "8"
            assert opt(c, "--replay-fraction") == ["0.5"]
            assert opt(c, "--replay-data") == [str(run.resolve() / f"runs/rl-it1/search/train/seg{k:02d}")
                                              for k in range(5)]
            assert opt(c, "--search-val") == [str(run.resolve() / "runs/rl-it2/search/val")]
            assert opt(c, "--trunk-lr") == ["5e-5"] and "--search-q-blend" not in c
            for flag, value in (("--search-value-weight", "0"), ("--search-wdl-weight", "1"),
                                ("--search-moves-left-weight", "0.1"), ("--export", "last")):
                assert opt(c, flag) == [value]
    done = json.loads((run / "runs/rl-it2/stages/balance_minitrain.done").read_text())
    assert done["gpu_s"] == 8.0 and done["steps"] == 8 and set(done["runs"]) == set(by)
    # a rerun is a no-op; other settings are refused rather than mixed
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8", "--trunk-lr", "5e-5"]) == 0
    assert len(trainer) == 4
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8"]) == 2


def test_lambda_challenger_trains_both_arms_on_the_twin(run, trainer):
    sb = controller()
    prior = {"score_stage": True, "search_q_blend": True}
    state = sb.init_state(prior)
    state["dir_idx"] = sb.DIRS.index("l+")
    state["step"] = 1
    plan = write_plan(run, 2, prior, state)
    assert plan["challenger_kind"] == "lambda" and plan["challenger_segment"] is None
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8"]) == 0
    by = {Path(opt(c, "--out")[0]).name: c for c in trainer}
    for s in (0, 1):
        assert opt(by[f"inc_s{s}"], "--search-data") == opt(by[f"chal_s{s}"], "--search-data")
        assert opt(by[f"chal_s{s}"], "--search-q-blend") == ["0.25"]
        assert "--search-q-blend" not in by[f"inc_s{s}"]


def test_no_challenger_writes_skip_markers_only(run, trainer, capsys):
    plan = write_plan(run, 2, {})          # the safe state: no score_stage -> frozen at theta0
    assert plan["challenger"] is None and plan["mode"] == "frozen"
    for stage in ("minitrain", "score", "probe"):
        assert balance.main([stage, str(run), "2"]) == 0
        marker = json.loads((run / f"runs/rl-it2/stages/balance_{stage}.done").read_text())
        assert marker["gpu_s"] == 0.0 and "skipped" in marker
    assert trainer == [] and not (run / "balance/it2").exists()


def test_init_must_match_train_done(run, trainer, capsys):
    write_plan(run, 2, DEEP_PRIOR)
    (run / "artifacts/x/best.pt").write_bytes(b"changed")
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8"]) == 2
    assert "init sha256" in capsys.readouterr().err
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8",
                         "--init", str(run / "artifacts/x/best.pt")]) == 0


def test_too_little_data_for_a_batch_is_refused(run, trainer, capsys):
    write_plan(run, 2, DEEP_PRIOR)
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "1024"]) == 2
    assert "fewer than" in capsys.readouterr().err and trainer == []


# ---------------------------------------------------------------- score

@pytest.fixture
def scorer(monkeypatch):
    seen = []

    def fake(net, probe_paths, swarm_dir, alpha, device, batch):
        seen.append((Path(net).parent.name, [Path(p).name for p in probe_paths], swarm_dir))
        base = {"inc_s0": 2.10, "chal_s0": 2.05, "inc_s1": 2.12, "chal_s1": 2.09}[Path(net).parent.name]
        if "deep_v1" in probe_paths[0].name:
            base += 0.5
        return {"score": base, "deep_loss": base, "wdl": 0.7}

    monkeypatch.setattr(balance, "evaluate_net", fake)
    return seen


def test_score_writes_what_decide_reads(run, trainer, scorer):
    sb = controller()
    plan = write_plan(run, 2, DEEP_PRIOR)
    assert balance.main(["minitrain", str(run), "2", "--batch-size", "8"]) == 0
    assert balance.main(["score", str(run), "2"]) == 2          # no probe deep targets yet
    probe = run / plan["probe"]["path"]
    probe.parent.mkdir(parents=True)
    balance.write_raw(probe, 2, 78, [arrays(np.random.default_rng(3), 10, masked_every=2)])
    assert balance.main(["score", str(run), "2"]) == 0
    out = json.loads((run / "balance/it2/score.json").read_text())
    assert out["probe_version"] == plan["probe"]["version"] == 1      # no earlier version: nothing chained
    assert set(out["scores"]) == {"inc_s0", "chal_s0", "inc_s1", "chal_s1"}
    d, sigma_n, seeds = sb.score_diffs(out["scores"])
    assert seeds == 2 and d == pytest.approx(out["d"], rel=1e-5) and sigma_n == pytest.approx(out["sigma_n"], rel=1e-5)
    assert out["probe"]["records"] == 10 and out["probe"]["policy_records"] == 5
    assert out["swarm_probe"] is None and all(s[2] is None for s in scorer)
    sha = balance.sha256(run / "balance/it2/score.json")
    assert balance.main(["score", str(run), "2"]) == 0                # kept: decide checks its sha256
    assert balance.sha256(run / "balance/it2/score.json") == sha and len(scorer) == 4
    # decide reads it
    assert sb.main(["decide", str(run), "2"]) == 0
    result = json.loads((run / "balance/it2.json").read_text())
    assert result["results"]["d"] == pytest.approx(out["d"], rel=1e-5) and result["decision"]["action"] == "continue"


def test_score_chains_the_previous_probe_version_and_uses_the_swarm_probe(run, trainer, scorer):
    sb = controller()
    state = sb.init_state(DEEP_PRIOR)
    state["probe"] = {"version": 1, "iter": 0}
    plan = write_plan(run, 2, DEEP_PRIOR, state)
    assert plan["probe"]["version"] == 2 and plan["probe"]["refresh"]
    balance.main(["minitrain", str(run), "2", "--batch-size", "8"])
    pdir = run / "balance" / "probe"
    pdir.mkdir(parents=True)
    for v in (1, 2):
        balance.write_raw(pdir / f"deep_v{v}.dst", 2, 78, [arrays(np.random.default_rng(v), 6)])
    (run / "probe").mkdir()
    (run / "probe" / "manifest.json").write_text(json.dumps({"path": "probe/probe.npz", "sha256": "s",
                                                             "positions": 3}))
    assert balance.main(["score", str(run), "2"]) == 0
    out = json.loads((run / "balance/it2/score.json").read_text())
    assert out["previous"]["probe_version"] == 1
    assert out["previous"]["scores"]["inc_s0"] == pytest.approx(out["scores"]["inc_s0"] + 0.5)
    assert out["alpha2"] == 0.5 and out["swarm_probe"]["sha256"] == "s"
    assert all(s[2] == run.resolve() / "probe" for s in scorer)


# ---------------------------------------------------------------- probe

STUB_ENGINE = r'''#!{python}
import json, struct, sys
from pathlib import Path
argv = sys.argv[1:]
if argv == ["--help"]:
    print("selfplay --deep-fraction <f> --deep-multiplier <k> --checkpoint-games <n>")
    sys.exit(0)
Path({calls!r}).open("a").write(json.dumps(argv) + "\n")
opt = lambda name: argv[argv.index(name) + 1]
games, out, records = int(opt("--games")), Path(opt("--distill-output")), Path(opt("--output"))
out.mkdir(parents=True, exist_ok=True); records.mkdir(parents=True, exist_ok=True)
n = 3 * games
counts = [(1, 1) if i % 3 == 0 else (0, 0) for i in range(n)]
entries = sum(a + b for a, b in counts)
body = [b"HDST", struct.pack("<II", 2, 78), struct.pack("<QQ", n, entries), bytes(8 * 78 * n),
        struct.pack("<e", 1.0) * (78 * n), bytes(4 * n), struct.pack("<fff", 0, 0, 1) * n, bytes(2 * n), bytes(2 * n),
        bytes(n), b"".join(struct.pack("<HH", a, b) for a, b in counts), bytes(2 * entries),
        struct.pack("<e", 1.0) * entries]
(out / "search_{{}}_00000.dst".format(opt("--seed"))).write_bytes(b"".join(body))
(out / "cap_counts.json").write_text(json.dumps({{"records": n, "deep": games, "masked": n - games}}))
for g in range(games):
    print(f"selfplay game {{g + 1}}/{{games}} raw 0 samples 3 termination checkmate")
print(f"selfplay complete {{games}} games")
'''


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "stub_engine"
    path.write_text(STUB_ENGINE.format(python=sys.executable, calls=str(tmp_path / "engine_calls.jsonl")))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_probe_mode_selfplay_writes_one_deep_chunk_and_sidecar(run, engine, tmp_path):
    plan = write_plan(run, 2, DEEP_PRIOR)
    assert plan["probe"]["refresh"] and plan["probe"]["path"] == "balance/probe/deep_v1.dst"
    model = run / "models" / "gen.onnx"
    model.parent.mkdir()
    model.write_bytes(b"generator")
    (run / "runs/rl-it2/stages/generator").write_text(balance.sha256(model) + "\n")
    assert balance.main(["probe", str(run), "2", "--engine", str(engine), "--games", "4"]) == 0
    calls = [json.loads(line) for line in (tmp_path / "engine_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    argv = calls[0]
    for flag, value in (("--deep-fraction", "0.125"), ("--deep-multiplier", "16"), ("--deep-dirichlet", "false"),
                        ("--fast-policy-mask", "true"), ("--seed", "2998"), ("--games", "4"), ("--nodes", "800"),
                        ("--fairy-stockfish-mate-nodes", "0"), ("--checkpoint-games", "10")):
        assert argv[argv.index(flag) + 1] == value
    out = run / plan["probe"]["path"]
    chunk = read_chunk(out)
    assert len(chunk["value"]) == 12 and int((chunk["policy_count"].sum(1) > 0).sum()) == 4
    side = json.loads((run / "balance/probe/deep_v1.json").read_text())
    assert side["sha256"] == balance.sha256(out) and side["model_sha256"] == balance.sha256(model)
    assert side["records"] == 12 and side["policy_records"] == 4 and side["cap_counts"]["deep"] == 4
    assert side["settings"]["k"] == 16 and side["seed"] == 2998
    marker = json.loads((run / "runs/rl-it2/stages/balance_probe.done").read_text())
    assert marker["version"] == 1 and marker["gpu_s"] >= 0
    # never recomputed
    assert balance.main(["probe", str(run), "2", "--engine", str(engine), "--games", "4"]) == 0
    assert len((tmp_path / "engine_calls.jsonl").read_text().splitlines()) == 1


def test_probe_refuses_an_engine_without_the_cap(run, tmp_path, capsys):
    write_plan(run, 2, DEEP_PRIOR)
    old = tmp_path / "old_engine"
    old.write_text("#!/bin/sh\necho selfplay\n")
    old.chmod(0o755)
    assert balance.main(["probe", str(run), "2", "--engine", str(old), "--model", str(old)]) == 2
    assert "--deep-fraction" in capsys.readouterr().err


# ---------------------------------------------------------------- accounting

def test_accounting_times_segments_from_the_loop_log_and_feeds_realise(run):
    sb = controller()
    write_plan(run, 2, DEEP_PRIOR)
    stages = run / "runs/rl-it2/stages"
    marker = json.loads((stages / "seg03.done").read_text())
    (stages / "seg03.done").write_text(json.dumps(dict(marker, gpu_s=1234.5)))
    (stages / "balance_minitrain.done").write_text(json.dumps({"gpu_s": 600.0}))
    (stages / "balance_probe.done").write_text(json.dumps({"gpu_s": 1800.0}))
    (stages / "train.done").write_text(json.dumps({"init": "x"}))
    (stages / "match_generator.done").write_text(json.dumps({"elo": 0}))
    log = textwrap.dedent("""\
        2026-11-22 10:00:00 it1 self-play seg00: 2000 games, seed 1000
        2026-11-22 10:00:00 it2 self-play seg00: 2000 games, seed 2000
        2026-11-22 10:00:05 it2 self-play seg01: 2000 games, seed 2001 (host:1:job-)
        2026-11-22 12:00:00 it2 seg00 labels: 2000 games, 290000 records, 1 flag games
        2026-11-22 12:00:00 it2 seg00 timing: ignored, the labels line came first
        2026-11-22 12:10:05 it2 seg01 labels: 2000 games
        2026-11-22 12:10:05 it2 self-play seg02: 2000 games, seed 2002
        2026-11-22 13:00:00 it2 self-play seg02: resuming 2000 games, seed 2002
        2026-11-22 14:00:00 it2 seg02 labels: 2000 games
        2026-11-22 14:00:00 it2 training: 2000 steps
        2026-11-22 15:00:00 it2 match vs generator (x)
        2026-11-22 15:30:00 it2 vs generator: {}
        """)
    (run / "runs" / "rl-loop.log").write_text(log)
    assert balance.main(["accounting", str(run), "2"]) == 0
    acc = json.loads((run / "balance/it2/accounting.json").read_text())
    assert acc["seg00"] == 7200.0 and acc["seg01"] == 7800.0 and acc["seg03"] == 1234.5
    assert "seg02" in acc["excluded"] and "resumed" in acc["excluded"]["seg02"]
    assert "seg04" in acc["excluded"] and "no start line" in acc["excluded"]["seg04"]
    assert acc["stages"]["train"] == 3600.0 and acc["stages"]["match_generator"] == 1800.0
    oh = json.loads((run / "balance/it2/overhead.json").read_text())
    assert oh == {"gpu_s": {"minitrain": 600.0, "probe": 1800.0}}
    # realise takes the segment numbers (and nothing it was not given)
    assert sb.main(["realise", str(run), "2"]) == 0
    realised = json.loads((run / "balance/it2/realised.json").read_text())
    assert realised["seg00"]["gpu_s"] == 7200.0 and "gpu_s" not in realised["seg04"]
    assert balance.main(["accounting", str(run), "2", "--gpus", "2"]) == 0
    assert json.loads((run / "balance/it2/accounting.json").read_text())["seg00"] == 14400.0


def test_controller_commands_pass_through(run, capsys):
    write_plan(run, 2, {})
    assert balance.main(["status", str(run), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["iteration"] == 2 and rows[0]["mode"] == "frozen"
