"""rl-loop --gate2 and the spec v7 hooks (talisman docs/custom_engine_spec_v7_league.md 3.4, 3.5).

The stub engines and trainer of test_rl_loop_search_balance.World play self-play, training and the 100 ms match; a
fake cluster/gate2_match.sh stands in for the PACE matches (it writes each network's net dir: config.json, per arm
pairs.tsv and summary.json, done), and stub hivemind.league / hivemind.think modules (sys.modules) stand in for WP3's
and WP4's hooks."""

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import pytest

from hivemind.cli import rl_loop
from test_rl_loop_search_balance import World, arg, head, snapshot  # noqa: F401  (head is a fixture)

FAKE_GATE = r'''#!/bin/bash
# Fake gate2_match.sh: RUN_ROOT LABEL ONNX. Behaviour from $RUN_ROOT/fake_gate.json:
#   {"mode": "finish" | "never" | "fail_first", "pairs": {"candidate"|"incumbent": {"equal": [...], "free": [...]}}}
set -e
exec python3 - "$@" <<'EOF'
import hashlib, json, os, sys
root, label, onnx = sys.argv[1:]
cfg = json.load(open(os.path.join(root, "fake_gate.json")))
with open(os.path.join(root, "fake_gate_calls.jsonl"), "a") as f:
    f.write(json.dumps({"label": label, "onnx": onnx, "cwd": os.getcwd(),
                        "env": {k: v for k, v in os.environ.items() if k.startswith("GATE2_")}}) + "\n")
calls = sum(1 for _ in open(os.path.join(root, "fake_gate_calls.jsonl")))
if cfg["mode"] == "fail_first" and calls == 1:
    print("gate2: boom", file=sys.stderr)
    sys.exit(3)
sha = hashlib.sha256(open(onnx, "rb").read()).hexdigest()
net = os.path.join(root, "gate2", "nets", sha[:16])
os.makedirs(net, exist_ok=True)
if not os.path.exists(os.path.join(net, "config.json")):
    json.dump({"label": label, "sha256": sha}, open(os.path.join(net, "config.json"), "w"))
print(f"gate2 {label}: called")
if cfg["mode"] == "never":
    sys.exit(0)
role = "candidate" if label.endswith("candidate") else "incumbent"
for arm, scores in cfg["pairs"][role].items():
    os.makedirs(os.path.join(net, arm), exist_ok=True)
    with open(os.path.join(net, arm, "pairs.tsv"), "w") as f:
        f.write("part\tpair_index\topening_id\tcontender_team\tscore_pair\tgame_ids\n")
        for i, s in enumerate(scores):
            f.write(f"{1 + i % 4}\t{i // 4}\t{i}\tT1,T2\t{s}\tp1,p2\n")
    json.dump({"arm": arm, "score": sum(scores) / len(scores), "pairs": len(scores)},
              open(os.path.join(net, arm, "summary.json"), "w"))
open(os.path.join(net, "done"), "w").write("x\n")
EOF
'''

GOOD = {"equal": [1.0] * 200, "free": [1.0] * 200}
EVEN = {"equal": [0.5] * 200, "free": [1.0] * 200}
WORSE = {"equal": [0.0] * 40 + [0.5] * 160, "free": [1.0] * 200}


def gate_script(tmp_path):
    script = tmp_path / "fakerepo" / "cluster" / "gate2_match.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(FAKE_GATE)
    return script


def gate_world(tmp_path, monkeypatch, mode="finish", candidate=GOOD, incumbent=EVEN, module=rl_loop, name="run",
               **options):
    script = gate_script(tmp_path)
    world = World(tmp_path / name, module, monkeypatch,
                  **{"gate2": True, "gate2_script": script, "gate2_poll_s": 0, **options})
    (world.root / "fake_gate.json").write_text(json.dumps({"mode": mode, "pairs": {"candidate": candidate,
                                                                                   "incumbent": incumbent}}))
    return world


def calls(world):
    path = world.root / "fake_gate_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def stages(world, n=1):
    return world.runs / f"rl-it{n}" / "stages"


def done(world, stage, n=1):
    return json.loads((stages(world, n) / f"{stage}.done").read_text())


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- the gate decides

def test_a_better_candidate_is_promoted_by_gate2(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch)
    world.loop().run(1, 2)
    gate = done(world, "gate2")
    assert gate["promoted"] is True and gate["reason"].startswith("promoted: d sum +0.500")
    assert gate["arms"]["equal"] == {"pairs": 200, "d": 0.5, "se": 0.0, "drop": False, "complete": True}
    assert gate["candidate"]["sha16"] == sha16(world.runs / "rl-it1" / "models" / "C.onnx")
    assert gate["incumbent"]["sha16"] == hashlib.sha256(b"generator").hexdigest()[:16]
    assert gate["candidate"]["equal"]["pairs"] == 200 and gate["incumbent"]["label"] == "it1-incumbent"
    assert gate["settings"] == {"games": 400, "parts": 4, "base_ms": 60000, "free_k": 160, "equal_k": 110,
                                "seed_base": 7000, "opp_sit": "observed", "arms": ["equal", "free"],
                                "drop_margin": 0.02, "min_pairs": 150, "timeout_h": 10.0}
    promote = done(world, "promote")
    assert promote["promoted"] is True and promote["score"] == 0.75  # the 100 ms score stays
    assert promote["gate2"] == {"promoted": True, "arms": gate["arms"], "reason": gate["reason"]}
    assert world.generator.read_bytes() == (world.runs / "rl-it2" / "models" / "C.onnx").read_bytes()
    # The 100 ms match still played, unchanged.
    assert (stages(world) / "match_generator.done").is_file()
    assert (stages(world) / "gate2.started").is_file()
    # it1: both networks; it2: only the candidate (its incumbent is it1's candidate, already done).
    labels = [c["label"] for c in calls(world)]
    assert labels == ["it1-candidate", "it1-incumbent", "it2-candidate"]
    call = calls(world)[0]
    assert call["env"] == {"GATE2_GAMES": "400", "GATE2_PARTS": "4", "GATE2_BASE_MS": "60000",
                           "GATE2_FREE_K": "160", "GATE2_EQUAL_K": "110", "GATE2_SEED_BASE": "7000",
                           "GATE2_OPP_SIT": "observed", "GATE2_ARMS": "equal free"}
    assert call["cwd"] == str((tmp_path / "fakerepo").resolve())
    assert done(world, "gate2", 2)["incumbent"]["sha16"] == gate["candidate"]["sha16"]
    root = world.root / "gate2"
    assert json.loads((root / "wave2.json").read_text()) == rl_loop.wave2.STOCK.as_dict()
    assert json.loads((root / "settings.json").read_text())["arms"] == ["equal", "free"]
    log = (world.runs / "rl-loop.log").read_text()
    assert "it1 gate2: promote (promoted: d sum" in log and "it1 gate2: gate2 it1-candidate: called" in log
    assert "real-clock gate 400 games a network and arm at 60000 ms" in log


def test_a_drop_is_not_promoted_whatever_the_100ms_score(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch, candidate=WORSE, incumbent=EVEN)
    world.loop().run(1, 1)
    gate = done(world, "gate2")
    assert gate["arms"]["equal"]["drop"] is True and gate["promoted"] is False
    promote = done(world, "promote")
    assert promote["promoted"] is False and promote["score"] == 0.75
    assert promote["gate2"]["reason"].startswith("drop in equal")
    assert world.generator.read_bytes() == b"generator"
    assert "it1 not promoted (score 0.75, gate2 drop in equal" in (world.runs / "rl-loop.log").read_text()


def test_an_unfinished_gate_times_out_unpromoted(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch, mode="never", gate2_timeout_h=1e-9)
    world.loop().run(1, 1)
    gate = done(world, "gate2")
    assert gate["promoted"] is False
    assert gate["reason"].startswith("incomplete: not done after 1e-09 h (candidate, incumbent); incomplete:")
    assert done(world, "promote")["promoted"] is False
    assert len(calls(world)) == 2  # one poll, then the timeout


def test_a_resumed_gate_keeps_its_clock(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch, mode="never", gate2_timeout_h=1.0, gate2_poll_s=3600)
    stages(world).mkdir(parents=True)
    (stages(world) / "gate2.started").write_text(f"{time.time() - 3601}\n")
    slept = []
    monkeypatch.setattr(rl_loop.time, "sleep", lambda s: slept.append(s))
    world.loop().run(1, 1)
    assert 3600 not in slept and done(world, "gate2")["reason"].startswith("incomplete: not done after 1 h")


def test_a_failing_script_is_called_again_at_the_next_poll(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch, mode="fail_first")
    world.loop().run(1, 1)
    assert [c["label"] for c in calls(world)] == ["it1-candidate", "it1-incumbent", "it1-candidate"]
    assert done(world, "gate2")["promoted"] is True
    log = (world.runs / "rl-loop.log").read_text()
    assert "it1 gate2: gate2: boom" in log
    assert "it1 gate2: gate2_match.sh it1-candidate failed (exit 3): trying again at the next poll" in log


def test_other_settings_in_the_run_are_refused(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch)
    (world.root / "gate2").mkdir()
    (world.root / "gate2" / "settings.json").write_text(json.dumps({"games": 200}))
    with pytest.raises(SystemExit):
        world.loop().run(1, 1)
    assert "holds {'games': 200}, not this run's" in (world.runs / "rl-loop.log").read_text()
    assert not (stages(world) / "promote.done").exists()


def test_a_finished_gate_is_not_run_again(tmp_path, monkeypatch):
    world = gate_world(tmp_path, monkeypatch)
    world.loop().run(1, 1)
    for name in ("iteration", "promote"):
        (stages(world) / f"{name}.done").unlink()
    world.loop().run(1, 1)
    assert len(calls(world)) == 2


# ---------------------------------------------------------------- defaults unchanged

def test_the_v7_defaults_are_off(head):  # noqa: F811
    assert rl_loop.WAVE2_DEFAULTS == head.WAVE2_DEFAULTS or \
        {k: v for k, v in rl_loop.WAVE2_DEFAULTS.items() if not k.startswith("search_balance")} == \
        {k: v for k, v in head.WAVE2_DEFAULTS.items() if not k.startswith("search_balance")}
    assert rl_loop.V7_DEFAULTS["gate2"] is False
    assert rl_loop.V7_DEFAULTS["league_on"] is False and rl_loop.V7_DEFAULTS["think_on"] is False


def test_off_the_commands_markers_and_log_are_heads(tmp_path, monkeypatch, head):  # noqa: F811
    """With every spec v7 option at its default, as main() sets them, nothing changes."""
    off = {name: value for name, value in rl_loop.V7_DEFAULTS.items()}
    old = World(tmp_path / "head", head, monkeypatch)
    old.loop().run(1, 2)
    new = World(tmp_path / "new", rl_loop, monkeypatch, **off)
    new.loop().run(1, 2)
    assert new.commands == old.commands
    assert snapshot(new) == snapshot(old)
    assert not (new.root / "gate2").exists()


@pytest.mark.parametrize("argv, message", [
    (["--gate2-games", "200"], "need --gate2"),
    (["--gate2", "--no-gate"], "does not go with --no-gate"),
    (["--gate2", "--gate2-games", "402"], "split into colour-swapped pairs"),
    (["--gate2", "--gate2-min-pairs", "300"], "more pairs than"),
    (["--gate2", "--gate2-script", "/no/such.sh"], "no matches script"),
])
def test_the_gate2_flags_are_checked(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, "argv", ["rl-loop", *argv])
    with pytest.raises(SystemExit) as raised:
        rl_loop.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_the_gate2_script_is_found_above_the_fork(monkeypatch, tmp_path):
    monkeypatch.delenv("HIVEMIND_GATE2_SCRIPT", raising=False)
    found = rl_loop.gate2_script()
    assert found is None or found == rl_loop.PROJECT_ROOT.parents[1] / "cluster" / "gate2_match.sh"
    script = gate_script(tmp_path)
    monkeypatch.setenv("HIVEMIND_GATE2_SCRIPT", str(script))
    assert rl_loop.gate2_script() == script.resolve()
    assert rl_loop.gate2_script(tmp_path / "missing.sh") is None


# ---------------------------------------------------------------- the league and think hooks (stub modules)

def league_stub(monkeypatch, opponent, fail=False):
    calls = []

    def add_arguments(parser):
        parser.add_argument("--league", action="store_true")
        parser.add_argument("--pace", action="store_true")

    def plan(args, runs, n, generator, fixed, segment_games):
        calls.append(("plan", n, Path(generator).name, Path(fixed).name, segment_games))
        if fail:
            raise RuntimeError("no past networks")
        pace = ["--pace-k-min", "80", "--pace-k-max", "400"]
        return {"summary": "self 1, fixed 1", "segments": {
            "seg00": {"theta": "self", "seed": 1000 * n, "games": 2, "engine_args": pace, "opponent_model": None},
            "seg01": {"theta": "fixed:k110", "seed": 1000 * n + 1, "games": 2, "opponent_model": str(opponent),
                      "engine_args": [*pace, "--opponent-model", "{opponent_model}", "--opponent-k", "110",
                                      "--seat-rows", "{segment_dir}/rows.tsv"], "league": {"k": 110}}}}

    def segment_summary(log, spec):
        calls.append(("summary", Path(log).name, spec["theta"]))
        return {"theta": spec["theta"], "learner_wdl": [1, 0, 1]}

    module = types.ModuleType("hivemind.league")
    module.add_arguments, module.plan, module.segment_summary = add_arguments, plan, segment_summary
    module.validate = lambda args, parser: None
    module.active = lambda args: bool(args.league or args.pace)
    monkeypatch.setitem(sys.modules, "hivemind.league", module)
    return calls


def think_stub(monkeypatch, tmp_path, labels_exit=0):
    script = tmp_path / "labels.py"
    script.write_text(f"import sys\nprint('working')\nprint('{{\"records\": 3, \"labelled\": 2}}')\n"
                      f"sys.exit({labels_exit})\n")
    module = types.ModuleType("hivemind.think")
    module.add_arguments = lambda parser: parser.add_argument("--think-labels", action="store_true")
    module.validate = lambda args, parser: None
    module.selfplay_args = lambda args: ["--think-policy", "true", "--record-state-output",
                                         "{segment_dir}/states.jsonl"]
    module.label_command = lambda args, n, it, generator, engine: [sys.executable, script, n, it.name, generator.name]
    module.trainer_args = lambda args: ["--think-weight", "0.1"]
    monkeypatch.setitem(sys.modules, "hivemind.think", module)
    return module


def selfplay_commands(world, n=1):
    return {arg(c, "--distill-output").rsplit("/", 1)[1]: c for c in world.commands
            if c[1] == "selfplay" and f"rl-it{n}/" in arg(c, "--distill-output")}


def test_a_league_plan_drives_the_segments_and_validation_stays_stock(tmp_path, monkeypatch):
    opponent = tmp_path / "it13.onnx"
    opponent.write_bytes(b"it13 network")
    log = league_stub(monkeypatch, opponent)
    world = World(tmp_path / "run", rl_loop, monkeypatch, league_on=True)
    world.loop().run(1, 1)
    marker = done(world, "league_plan")
    assert marker["summary"] == "self 1, fixed 1" and sorted(marker["segments"]) == ["seg00", "seg01"]
    assert marker["segments"]["seg01"]["opponent_model"] == str(opponent)
    assert log[0] == ("plan", 1, "net.onnx", "net-it0.onnx", 2)
    played = selfplay_commands(world)
    assert played["seg00"][-4:] == ["--pace-k-min", "80", "--pace-k-max", "400"]
    copy = world.runs / "league_opponents" / f"opponent-{sha16(opponent)}-sp1.onnx"
    assert arg(played["seg01"], "--opponent-model") == str(copy).replace(str(world.root), "<RUN>")
    assert copy.read_bytes() == b"it13 network"
    assert arg(played["seg01"], "--seat-rows") == "<RUN>/runs/rl-it1/search/train_run/seg01/rows.tsv"
    assert not any(flag in played["val"] for flag in ("--pace-k-min", "--opponent-model"))
    assert done(world, "seg01")["league"] == {"theta": "fixed:k110", "learner_wdl": [1, 0, 1]}
    assert "balance" not in done(world, "seg01") and "league" not in done(world, "val")
    assert ("summary", "seg01.log", "fixed:k110") in log
    assert "it1 self-play seg01: 2 games, seed 1001, league fixed:k110 --pace-k-min" in (
        world.runs / "rl-loop.log").read_text()


def test_a_worker_claims_the_league_table_and_uses_its_own_opponent_copy(tmp_path, monkeypatch):
    opponent = tmp_path / "it13.onnx"
    opponent.write_bytes(b"it13 network")
    league_stub(monkeypatch, opponent)
    world = World(tmp_path / "run", rl_loop, monkeypatch, league_on=True, shared_selfplay=True, selfplay_hook=None)
    main = world.loop()
    main.n, main.it, main.arms = 1, world.runs / "rl-it1", world.runs / "rl-it1" / "arms"
    for directory in ("stages", "search", "arms", "models"):
        (main.it / directory).mkdir(parents=True)
    main.league_plan()
    main.mark("selfplay_mode", {"shared": True, "segment_games": 2})
    (main.it / "stages" / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    # A worker: the loop's flags; its own directory for the generator and opponent copies.
    worker = world.loop()
    own = world.runs / "workers" / "w1"
    worker.opponent_dir = own
    assert worker.work() == 2
    assert world.claims["seg01"]["balance"]["theta"] == "fixed:k110"
    assert "{opponent_model}" in world.claims["seg01"]["balance"]["engine_args"]
    played = selfplay_commands(world)
    assert arg(played["seg01"], "--opponent-model") == f"<RUN>/runs/workers/w1/opponent-{sha16(opponent)}-sp1.onnx"
    assert not main.selfplay_left()


def test_a_failed_league_plan_plays_plain_segments(tmp_path, monkeypatch):
    log = league_stub(monkeypatch, tmp_path / "missing.onnx", fail=True)
    old = World(tmp_path / "plain", rl_loop, monkeypatch)
    old.loop().run(1, 1)
    world = World(tmp_path / "run", rl_loop, monkeypatch, league_on=True)
    world.loop().run(1, 1)
    marker = done(world, "league_plan")
    assert marker["failsafe"] == "RuntimeError: no past networks"
    assert [s["games"] for s in marker["segments"].values()] == [2, 2]
    assert [c for c in world.commands] == [c for c in old.commands]  # plain self-play, the same commands
    assert all(entry[0] == "plan" for entry in log)  # no segment summaries of a failsafe table
    assert "league" not in done(world, "seg00")
    assert "league plan failed (RuntimeError: no past networks): failsafe" in (world.runs / "rl-loop.log").read_text()


def test_an_iteration_begun_without_the_league_finishes_as_it_began(tmp_path, monkeypatch):
    league_stub(monkeypatch, tmp_path / "x.onnx")
    world = World(tmp_path / "run", rl_loop, monkeypatch, league_on=True)
    stages(world).mkdir(parents=True)
    (stages(world) / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    world.loop().run(1, 1)
    assert done(world, "league_plan") == {"skipped": "the iteration started without --league/--pace"}
    assert "--pace-k-min" not in selfplay_commands(world)["seg00"]


def test_think_hooks_reach_selfplay_labels_and_training(tmp_path, monkeypatch):
    think_stub(monkeypatch, tmp_path)
    world = World(tmp_path / "run", rl_loop, monkeypatch, think_on=True)
    world.loop().run(1, 1)
    played = selfplay_commands(world)
    assert played["seg00"][-4:] == ["--think-policy", "true", "--record-state-output",
                                    "<RUN>/runs/rl-it1/search/train_run/seg00/states.jsonl"]
    assert "--think-policy" not in played["val"] and "--record-state-output" not in played["val"]
    assert done(world, "think_labels") == {"records": 3, "labelled": 2}
    labels = next(c for c in world.commands if c[0] == "<PYTHON>" and c[1].endswith("labels.py"))
    assert labels[2:] == ["1", "rl-it1", "net.onnx"]
    order = [("labels" if c[1].endswith("labels.py") else "train" if "distill-train" in c else None)
             for c in world.commands]
    assert order.index("labels") < order.index("train")
    trainer = next(c for c in world.commands if "distill-train" in c)
    assert trainer[-2:] == ["--think-weight", "0.1"]
    assert done(world, "train")["think_trainer_args"] == ["--think-weight", "0.1"]


def test_failing_labels_are_not_fatal(tmp_path, monkeypatch):
    think_stub(monkeypatch, tmp_path, labels_exit=1)
    world = World(tmp_path / "run", rl_loop, monkeypatch, think_on=True)
    world.loop().run(1, 1)
    assert done(world, "think_labels") == {"failed": "exit 1"}
    assert (stages(world) / "train.done").is_file() and (stages(world) / "iteration.done").is_file()
    log = (world.runs / "rl-loop.log").read_text()
    assert log.count("think labels failed (exit 1;") == 3
    assert "think labels skipped after 3 attempts: training without them" in log


def test_validation_with_league_flags_is_refused(tmp_path, monkeypatch):
    world = World(tmp_path / "run", rl_loop, monkeypatch)
    loop = world.loop()
    loop.n, loop.it, loop.arms = 1, world.runs / "rl-it1", world.runs / "rl-it1" / "arms"
    for directory in ("stages", "search", "arms", "models"):
        (loop.it / directory).mkdir(parents=True)
    (loop.it / "stages" / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    monkeypatch.setattr(rl_loop.selfplay, "time_model_options", lambda model: ("--opponent-model", "x"))
    with pytest.raises(rl_loop.StageFailed, match="validation must play without league"):
        loop.selfplay("val", 2, 1999, loop.it / "search" / "val", loop.it / "search" / "val_run")


@pytest.mark.parametrize("argv, message", [
    (["--league", "--search-balance"], "pick one"),
    (["--league"], "need --wave2 and a base mix"),
    (["--pace", "--distributed"], "without --distributed"),
])
def test_the_league_flags_are_checked(monkeypatch, capsys, tmp_path, argv, message):
    league_stub(monkeypatch, tmp_path / "x.onnx")
    monkeypatch.setattr(sys, "argv", ["rl-loop", *argv])
    with pytest.raises(SystemExit) as raised:
        rl_loop.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_hook_modules_are_imported_only_for_their_flags(monkeypatch):
    monkeypatch.setitem(sys.modules, "hivemind.league", None)  # import fails
    assert rl_loop.hook_modules(["--wave2", "--gate2"]) == {"league": None, "think": None}
    assert rl_loop.hook_modules(["--help"])["league"] is None
    with pytest.raises(SystemExit, match="the league flags need hivemind.league"):
        rl_loop.hook_modules(["--pace"])
    think = types.ModuleType("hivemind.think")
    monkeypatch.setitem(sys.modules, "hivemind.think", think)
    assert rl_loop.hook_modules(["--think-labels"])["think"] is think


def test_think_on_follows_the_think_flags(monkeypatch, tmp_path, capsys):
    """main() sets think_on only when a think flag is given (seen through the parse, before the generator check)."""
    think_stub(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(rl_loop, "onnx_input_planes", lambda path: (_ for _ in ()).throw(SystemExit(7)))
    real = rl_loop.wave2.from_args

    def spy(args):
        seen["args"] = args
        return real(args)
    monkeypatch.setattr(rl_loop.wave2, "from_args", spy)
    generator = tmp_path / "g.onnx"
    generator.write_bytes(b"g")
    for argv, on in ((["--think-labels"], True), ([], False)):
        monkeypatch.setattr(sys, "argv", ["rl-loop", "--generator", str(generator), *argv])
        with pytest.raises(SystemExit):
            rl_loop.main()
        assert seen["args"].think_on is on and seen["args"].league_on is False

