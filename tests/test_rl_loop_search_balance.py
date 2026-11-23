"""rl-loop --search-balance: the controller wired into the loop (talisman docs/search_balance.md section 9).

A stub engine (self-play, tournaments) and a stub trainer stand in for the real ones; the controller
(talisman cluster/search_balance.py) and `hivemind balance accounting` run for real, and the torch/engine stages
(`hivemind balance probe|minitrain|score`) are stubbed with the files and markers they write. Without the flag the
loop must give the same commands, markers and log as the committed rl_loop.py (git HEAD)."""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hivemind.cli import rl_loop, train
from hivemind.paths import PROJECT_ROOT

REAL_RUN = subprocess.run
REAL_POPEN = subprocess.Popen
CONTROLLER = rl_loop.search_balance_controller()
CHALLENGER_PRIOR = {"score_stage": True, "probe_bias_measured": True, "search_q_blend": True,
                    "policy_positions": True}

pytestmark = pytest.mark.skipif(CONTROLLER is None, reason="talisman cluster/search_balance.py not found")


@pytest.fixture(scope="module")
def head():
    """The committed rl_loop.py, loaded as its own module."""
    source = REAL_RUN(["git", "show", "HEAD:src/hivemind/cli/rl_loop.py"], cwd=PROJECT_ROOT, capture_output=True,
                      text=True)
    if source.returncode:
        pytest.skip("no git HEAD for rl_loop.py")
    path = Path(__file__).parent / "__pycache__" / "rl_loop_head_snapshot.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(source.stdout)
    spec = importlib.util.spec_from_file_location("rl_loop_head_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arg(command, flag):
    return command[command.index(flag) + 1]


class World:
    """A run root RUN (iterations in RUN/runs) with stub engines and trainer; every command is recorded."""

    def __init__(self, root, module, monkeypatch, prior=None, stages=None, **options):
        self.root, self.module = root, module
        self.runs = root / "runs"
        self.runs.mkdir(parents=True)
        (root / "models").mkdir()
        self.generator = root / "models" / "net.onnx"
        self.generator.write_bytes(b"generator")
        self.checkpoint = root / "artifacts" / "net.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"checkpoint")
        openings = root / "openings.tsv"
        openings.write_text("".join(f"fen{i}|fen{i}\tw\t0\n" for i in range(20)))
        if prior is not None:
            (root / "balance").mkdir()
            (root / "balance" / "prior.json").write_text(json.dumps(prior))
        self.commands, self.balance_calls, self.claims = [], [], {}
        self.stages = {"probe": self.probe, "minitrain": self.minitrain, "score": self.score, **(stages or {})}
        args = dict(games=4, segment_games=2, val_games=2, window=3, new_passes=1.5, replay_passes=0.5,
                    parallel_games=2, resign_threshold=0.0, tree_reuse=False, checkpoint_games=0, match_games=4,
                    match_movetime=100, time_model="clock", fixed_time_model="clock", anchor_after=None,
                    generator=self.generator, generator_checkpoint=self.checkpoint, batch_size=8, openings=openings,
                    runs=self.runs, repo="user/net", no_upload=True, archive="user/archive", keep_selfplay=True,
                    archive_only=False, pause_for="", distributed=False, local_selfplay=True,
                    selfplay_engine=root / "engine-sp1", match_engine=root / "engine", promote_above=None)
        args.update(options)
        self.args = SimpleNamespace(**args)
        monkeypatch.setattr(subprocess, "run", self.run)
        monkeypatch.setattr(subprocess, "Popen", self.popen())
        monkeypatch.setattr(train, "positions", lambda dirs: 100 * len(list(dirs)))
        monkeypatch.setattr(train, "policy_positions", lambda dirs: 90 * len(list(dirs)))
        monkeypatch.setattr(rl_loop, "RETRY_SECONDS", 0)
        monkeypatch.setattr(module, "RETRY_SECONDS", 0)

    def loop(self):
        return self.module.Loop(self.args, generator=self.generator, checkpoint=self.checkpoint,
                                teacher_val=self.root / "no-teacher-val")

    def norm(self, value):
        return json.loads(json.dumps(value).replace(str(self.root), "<RUN>").replace(sys.executable, "<PYTHON>"))

    # ---------------------------------------------------------------- the stub processes
    def run(self, command, *args, **kwargs):
        command = [str(c) for c in command] if isinstance(command, (list, tuple)) else command
        if isinstance(command, list) and command[-1] == "--help":
            return SimpleNamespace(returncode=0, stdout="--deep-fraction F\n--fast-nodes B\n", stderr="")
        if isinstance(command, list) and command[:2] == [sys.executable, "-c"]:
            return SimpleNamespace(returncode=0, stdout="False\n", stderr="")
        if isinstance(command, list) and command[:3] == [sys.executable, "-m", "hivemind.cli.balance"]:
            self.balance_calls.append(command[3])
            self.commands.append(self.norm(command))
            if command[3] in self.stages:
                return SimpleNamespace(returncode=self.stages[command[3]](command, kwargs))
            return REAL_RUN(command, *args, **kwargs)
        if isinstance(command, list) and command[:2] == [sys.executable, str(CONTROLLER)]:
            self.balance_calls.append(command[2] + (" --dry" if "--dry" in command else ""))
            self.commands.append(self.norm(command))
            return REAL_RUN(command, *args, **kwargs)
        self.commands.append(self.norm(command))
        if command[1] == "selfplay":
            return self.selfplay(command, kwargs["stdout"])
        if "distill-train" in command:
            out = Path(arg(command, "--out"))
            out.mkdir(parents=True)
            (out / "last.pt").write_text(f"weights {arg(command, '--steps')} {arg(command, '--seed')}")
            (out / "distill-test-v3.0.onnx").write_text(f"net {arg(command, '--steps')} {arg(command, '--seed')}")
            return SimpleNamespace(returncode=0)
        return REAL_RUN(command, *args, **kwargs)

    def selfplay(self, command, stream):
        games, seed = int(arg(command, "--games")), int(arg(command, "--seed"))
        data, records = Path(arg(command, "--distill-output")), Path(arg(command, "--output"))
        claim = data.parents[2] / "stages" / f"{data.name}.claim"
        if claim.is_file():
            self.claims[data.name] = json.loads(claim.read_text())
        data.mkdir(parents=True, exist_ok=True)
        records.mkdir(parents=True, exist_ok=True)
        (data / f"search_{seed}_00000.dst").write_bytes(b"chunk %d" % seed)
        (records / "seat_clocks.tsv").write_text("game\tnodes\n" + "".join(f"{g + 1}\t800\n" for g in range(games)))
        if "--deep-fraction" in command:
            (data / "cap_counts.json").write_text(json.dumps({"records": 10 * games, "deep": games, "masked": 0,
                                                              "deep_visits": 3200 * games,
                                                              "fast_visits": 7200 * games}))
        for g in range(games):
            stream.write(f"selfplay game {g + 1}/{games} winner 0\n")
        stream.write(f"selfplay complete {games} games\n")
        stream.flush()
        return SimpleNamespace(returncode=0)

    def popen(self):
        world = self

        class Process:
            def __init__(self, command, **kwargs):
                world.commands.append(world.norm(command))
                out = Path(command[command.index("--output") + 1])
                games = int(command[command.index("--games") + 1])
                out.mkdir(parents=True, exist_ok=True)
                results = [1 if i % 4 else -1 for i in range(games)]
                rows = [f"{i}\t{r}\t{1 - i % 2}\t0\tcheckmate\t10\t1000\t1000000000\t10\t500\t1000000000\t{i + 1}"
                        for i, r in enumerate(results)]
                (out / "games.tsv").write_text("\n".join(["config\tstub", *rows]) + "\n")
                (out / "games.pgn").write_bytes(b"x" * games)
                wins, losses = results.count(1), results.count(-1)
                score = wins / games
                (out / "summary.json").write_text(json.dumps({
                    "games": games, "contender_wins": wins, "baseline_wins": losses, "draws": 0,
                    "contender_elo": 400 * math.log10(score / (1 - score)), "elo_confidence_95": None,
                    "performance": {"contender": {"nps": 1000.0}, "baseline": {"nps": 500.0}},
                    "contender_breakdown": {}, "terminations": {"checkmate": games}}))

            def wait(self, timeout=None):
                return 0

        def popen(command, *args, **kwargs):
            # subprocess.run itself opens processes through Popen: only tournaments are stubbed.
            if isinstance(command, list) and len(command) > 1 and str(command[1]) == "tournament":
                return Process(command, **kwargs)
            return REAL_POPEN(command, *args, **kwargs)

        return popen

    # ---------------------------------------------------------------- stub torch/engine balance stages
    def paths(self, command):
        run, n = Path(command[4]), int(command[5])
        return run, n, run / "runs" / f"rl-it{n}" / "stages", run / "balance" / f"it{n}"

    def probe(self, command, kwargs):
        run, n, stages, _ = self.paths(command)
        (stages / "balance_probe.done").write_text(json.dumps({"version": 1, "gpu_s": 30.0}))
        return 0

    def minitrain(self, command, kwargs):
        run, n, stages, bit = self.paths(command)
        (bit / "mini").mkdir(parents=True, exist_ok=True)
        for name in ("inc_s0", "inc_s1", "chal_s0", "chal_s1"):
            (bit / "mini" / name).mkdir(exist_ok=True)
            (bit / "mini" / name / "distill-mini-v3.0.onnx").write_text(name)
            (bit / "mini" / f"{name}.done").write_text(json.dumps(
                {"onnx": str(bit / "mini" / name / "distill-mini-v3.0.onnx"), "gpu_s": 10.0}))
        (stages / "balance_minitrain.done").write_text(json.dumps({"gpu_s": 40.0}))
        return 0

    def score(self, command, kwargs):
        run, n, stages, bit = self.paths(command)
        scores = {"inc_s0": 2.0, "chal_s0": 1.99, "inc_s1": 2.01, "chal_s1": 1.995}
        (bit / "score.json").write_text(json.dumps({"schema": 1, "iteration": n, "probe_version": 1,
                                                    "scores": scores}, sort_keys=True))
        (stages / "balance_score.done").write_text(json.dumps({"gpu_s": 5.0}))
        return 0


def engine_commands(world):
    """The self-play, training and match commands (not the balance controller or stages)."""
    return [c for c in world.commands if c[:3] != ["<PYTHON>", "-m", "hivemind.cli.balance"]
            and not (c[0] == "<PYTHON>" and c[1].endswith("search_balance.py"))]


def snapshot(world):
    """Every file under RUN (paths and bytes, the run root replaced), the log without timestamps."""
    files = {}
    for path in sorted(world.root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            data = path.read_bytes().replace(str(world.root).encode(), b"<RUN>")
            if path.name == "rl-loop.log":
                data = re.sub(rb"(?m)^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", b"", data)
            files[str(path.relative_to(world.root))] = data
    return files


VARIANTS = {"local": {}, "shared": {"shared_selfplay": True, "selfplay_hook": None},
            "sprt": {"gate_sprt": (0.0, 15.0), "gate_sprt_alpha": 0.05, "gate_sprt_beta": 0.05,
                     "gate_chunk": 2, "gate_max_games": None}}


def run_world(tmp_path, name, module, monkeypatch, variant, iterations=2, **options):
    world = World(tmp_path / name, module, monkeypatch, **{**VARIANTS[variant], **options})
    world.loop().run(1, iterations)
    return world


# ---------------------------------------------------------------- off: byte-identical to HEAD

@pytest.mark.parametrize("variant", VARIANTS)
def test_without_the_flag_commands_markers_and_log_are_heads(tmp_path, monkeypatch, head, variant):
    old = run_world(tmp_path, "head", head, monkeypatch, variant)
    new = run_world(tmp_path, "new", rl_loop, monkeypatch, variant)
    assert new.commands == old.commands
    files = snapshot(new)
    if variant == "shared":
        # The one addition: a shared iteration records that it began shared (rl-loop's "switching" rule).
        for n in (1, 2):
            assert json.loads(files.pop(f"runs/rl-it{n}/stages/selfplay_mode.done")) == {
                "shared": True, "segment_games": 2}
    assert files == snapshot(old)
    assert not (new.root / "balance").exists()
    assert len([c for c in new.commands if c[1] == "selfplay"]) == 2 * 3  # 2 segments + val, 2 iterations


def test_the_defaults_are_off(head):
    assert rl_loop.WAVE2_DEFAULTS["search_balance"] is False
    assert {k: v for k, v in rl_loop.WAVE2_DEFAULTS.items() if not k.startswith("search_balance")} \
        == head.WAVE2_DEFAULTS
    # A claim without a balance entry is the same JSON as before.
    assert rl_loop.Claims.take.__defaults__ == (None,)


# ---------------------------------------------------------------- on, safe state: today's recipe

@pytest.mark.parametrize("variant", VARIANTS)
def test_on_without_a_prior_plays_and_trains_todays_recipe(tmp_path, monkeypatch, head, variant):
    old = run_world(tmp_path, "head", head, monkeypatch, variant)
    new = run_world(tmp_path, "new", rl_loop, monkeypatch, variant, search_balance=True)
    engine = engine_commands(new)
    # Every self-play, training and match command is the one the loop runs without the flag.
    assert engine == old.commands
    assert new.balance_calls == ["plan", "accounting", "realise", "decide"] * 2
    for n in (1, 2):
        plan = json.loads((new.root / "balance" / f"it{n}.plan.json").read_text())
        assert plan["mode"] == "frozen" and plan["challenger"] is None
        result = json.loads((new.root / "balance" / f"it{n}.json").read_text())
        assert result["decision"]["action"] == "none"
        stages = new.runs / f"rl-it{n}" / "stages"
        marker = json.loads((stages / "balance_plan.done").read_text())
        assert {s["theta"] for s in marker["segments"].values()} == {"inc"}
        assert all(not s["engine_args"] for s in marker["segments"].values())
        for seg in ("seg00", "seg01"):
            done = json.loads((stages / f"{seg}.done").read_text())
            assert done["balance"]["theta"] == "inc" and done["balance"]["engine_args"] == []
            assert done["gpu_s"] >= 0
        assert json.loads((stages / "balance_decide.done").read_text())["action"] == "none"
        assert json.loads((stages / "train.done").read_text())["search_balance"]["steps_rule"] == "positions"
        realised = json.loads((new.root / "balance" / f"it{n}" / "realised.json").read_text())
        assert set(realised) == {"seg00", "seg01"} and realised["seg00"]["visits"] == 1600
        accounting = json.loads((new.root / "balance" / f"it{n}" / "accounting.json").read_text())
        assert accounting["sources"]["seg00"] == "marker gpu_s"
    log = (new.runs / "rl-loop.log").read_text()
    assert "it1 search balance plan: mode frozen, challenger none" in log
    assert "it2 search balance decision: none" in log


# ---------------------------------------------------------------- on, with a challenger

def test_a_challenger_plan_runs_end_to_end(tmp_path, monkeypatch):
    world = run_world(tmp_path, "run", rl_loop, monkeypatch, "local", search_balance=True, prior=CHALLENGER_PRIOR)
    assert world.balance_calls == ["plan", "probe", "minitrain", "score", "decide --dry", "accounting", "realise",
                                   "decide", "plan", "minitrain", "score", "decide --dry", "accounting", "realise",
                                   "decide"]
    for n in (1, 2):
        selfplay = {arg(c, "--distill-output").rsplit("/", 1)[1]: c for c in world.commands if c[1] == "selfplay"
                    and f"rl-it{n}/" in arg(c, "--distill-output")}
        assert "--deep-fraction" not in selfplay["seg00"] and "--deep-fraction" not in selfplay["val"]
        chal = selfplay["seg01"]
        assert chal[-4:] == ["--deep-fraction", "0.125", "--deep-multiplier", "4"]
        assert arg(chal, "--seed") == arg(selfplay["seg00"], "--seed") == str(1000 * n)  # the twin's seed
        assert arg(chal, "--games") == "1"  # round(2 / c), c = 1.3713
        stages = world.runs / f"rl-it{n}" / "stages"
        assert json.loads((stages / "seg01.done").read_text())["balance"]["theta"] == "chal"
        done = json.loads((stages / "train.done").read_text())
        assert done["search_balance"]["steps_rule"] == "policy_positions"
        assert done["new_positions"] == 180  # policy_positions of the two segments
        assert set(done["search_balance"]["main_mix"]) == {"inc", "chal"}
        result = json.loads((world.root / "balance" / f"it{n}.json").read_text())
        assert result["decision"]["action"] in ("continue", "none", "reject")
        assert result["results"]["d"] == pytest.approx(0.0125)
    trains = [c for c in world.commands if "distill-train" in c]
    assert all("--search-q-blend" not in c for c in trains)  # the incumbent's lambda is 0
    stage = [c for c in world.commands if c[:3] == ["<PYTHON>", "-m", "hivemind.cli.balance"]]
    minitrain = next(c for c in stage if c[3] == "minitrain")
    assert minitrain[4:] == ["<RUN>", "1", "--batch-size", "8", "--lr", "1e-4", "--gpus", "1.0"]
    log = (world.runs / "rl-loop.log").read_text()
    assert "it1 self-play seg01: 1 games, seed 1000, search balance arm chal --deep-fraction 0.125" in log


def test_the_incumbents_trainer_args_reach_training(tmp_path, monkeypatch):
    world = World(tmp_path / "run", rl_loop, monkeypatch, search_balance=True, **VARIANTS["local"])
    loop = world.loop()
    loop.n, loop.it, loop.arms = 1, world.runs / "rl-it1", world.runs / "rl-it1" / "arms"
    for directory in ("stages", "search", "arms", "models"):
        (loop.it / directory).mkdir(parents=True)
    (loop.it / "stages" / "seg00.done").write_text(json.dumps({"games": 2, "positions": 100}))
    (loop.it / "search" / "train" / "seg00").mkdir(parents=True)
    loop.mark("balance_plan", {"plan": "balance/it1.plan.json", "segments": {"seg00": {
        "theta": "inc", "seed": 1000, "games": 2, "engine_args": []}}, "trainer_args": ["--search-q-blend", "0.25"],
        "steps_rule": "positions"})
    loop.train()
    command = next(c for c in world.commands if "distill-train" in c)
    assert command[-2:] == ["--search-q-blend", "0.25"]
    assert json.loads((loop.it / "stages" / "train.done").read_text())["search_balance"]["trainer_args"] \
        == ["--search-q-blend", "0.25"]


def test_the_mini_gate_runs_when_decide_says_it_can_promote(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_loop, "last_json_line", lambda text: {"eligible": True, "run_minigate": True})
    world = run_world(tmp_path, "run", rl_loop, monkeypatch, "local", iterations=1, search_balance=True,
                      prior=CHALLENGER_PRIOR)
    gate = next(c for c in world.commands if c[1] == "tournament" and "balance_minigate" in arg(c, "--output"))
    assert arg(gate, "--contender").endswith("mini/chal_s0/distill-mini-v3.0.onnx")
    assert arg(gate, "--baseline").endswith("mini/inc_s0/distill-mini-v3.0.onnx")
    assert arg(gate, "--games") == "100" and arg(gate, "--seed") == "1997"
    minigate = json.loads((world.root / "balance" / "it1" / "minigate.json").read_text())
    assert minigate["games"] == 100 and minigate["wins"] == 75
    result = json.loads((world.root / "balance" / "it1.json").read_text())
    assert result["results"]["minigate"]["score"] == 0.75
    assert "minigate" in json.loads((world.root / "balance" / "it1" / "overhead.json").read_text())["gpu_s"]


# ---------------------------------------------------------------- resume, failures

def test_a_rerun_resumes_at_the_first_unfinished_balance_step(tmp_path, monkeypatch):
    world = run_world(tmp_path, "run", rl_loop, monkeypatch, "local", iterations=1, search_balance=True,
                      prior=CHALLENGER_PRIOR)
    result = world.root / "balance" / "it1.json"
    before = result.read_bytes()
    stages = world.runs / "rl-it1" / "stages"
    for name in ("iteration", "balance_decide"):
        (stages / f"{name}.done").unlink()
    result.unlink()
    world.balance_calls.clear()
    engines = len(engine_commands(world))
    world.loop().run(1, 1)
    # Only the steps after the last marker rerun (accounting and the dry decide are rerunnable), and decide
    # writes the same bytes.
    assert world.balance_calls == ["decide --dry", "accounting", "decide"]
    assert len(engine_commands(world)) == engines  # no self-play, training or match again
    assert result.read_bytes() == before


def test_a_failing_balance_stage_is_skipped_and_decide_still_keeps_the_ledger(tmp_path, monkeypatch):
    world = run_world(tmp_path, "run", rl_loop, monkeypatch, "local", search_balance=True, prior=CHALLENGER_PRIOR,
                      stages={"minitrain": lambda command, kwargs: 2})
    assert world.balance_calls[:8] == ["plan", "probe", "minitrain", "minitrain", "minitrain", "accounting",
                                       "realise", "decide"]
    assert world.balance_calls[8] == "plan"  # decide ran once (--no-score), no retries
    decide = next(c for c in world.commands if c[1].endswith("search_balance.py") and c[2] == "decide")
    assert "--no-score" in decide
    assert (world.runs / "rl-it1" / "stages" / "iteration.done").is_file()
    result = json.loads((world.root / "balance" / "it1.json").read_text())
    assert result["decision"] == {"action": "none", "reason": "no mini-train scores: no evidence, no move"}
    assert result["state"]["overhead"]["cum_gpu_h"] > 0  # the probe's GPU time reached the ledger
    assert result["state"]["probe"]["iter"] == 1  # the probe that was built is kept
    plan = json.loads((world.root / "balance" / "it2.plan.json").read_text())
    assert plan["mode"] != "failsafe"
    log = (world.runs / "rl-loop.log").read_text()
    assert "it1: search balance minitrain skipped after 3 attempts" in log
    assert "it1: search balance has no mini-train scores this iteration: decide records no evidence" in log
    assert "decide skipped" not in log


def test_without_a_controller_the_iteration_plays_stock(tmp_path, monkeypatch, head):
    old = run_world(tmp_path, "head", head, monkeypatch, "local", iterations=1)
    world = run_world(tmp_path, "run", rl_loop, monkeypatch, "local", iterations=1, search_balance=True,
                      search_balance_controller=tmp_path / "missing.py")
    assert world.commands == old.commands
    marker = json.loads((world.runs / "rl-it1" / "stages" / "balance_plan.done").read_text())
    assert marker["failsafe"].startswith("no controller")
    assert [s["games"] for s in marker["segments"].values()] == [2, 2]
    assert "failsafe, stock self-play" in (world.runs / "rl-loop.log").read_text()


def test_an_iteration_started_without_the_flag_finishes_as_it_began(tmp_path, monkeypatch):
    world = World(tmp_path / "run", rl_loop, monkeypatch, search_balance=True, **VARIANTS["local"])
    stages = world.runs / "rl-it1" / "stages"
    stages.mkdir(parents=True)
    (stages / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    world.loop().run(1, 1)
    assert json.loads((stages / "balance_plan.done").read_text()) == {
        "skipped": "the iteration started without --search-balance"}
    assert world.balance_calls == []
    assert "balance" not in json.loads((stages / "seg00.done").read_text())


# ---------------------------------------------------------------- shared self-play: workers read the plan

def test_workers_play_the_planned_segments_with_their_flags(tmp_path, monkeypatch):
    world = World(tmp_path / "run", rl_loop, monkeypatch, prior=CHALLENGER_PRIOR, search_balance=True,
                  **VARIANTS["shared"])
    main = world.loop()
    main.n, main.it, main.arms = 1, world.runs / "rl-it1", world.runs / "rl-it1" / "arms"
    for directory in ("stages", "search", "arms", "models"):
        (main.it / directory).mkdir(parents=True)
    main.balance_plan()
    main.mark("selfplay_mode", {"shared": True, "segment_games": 2})
    (main.it / "stages" / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    # A worker started with the loop's settings but without --search-balance.
    world.args = SimpleNamespace(**{**vars(world.args), "search_balance": False})
    worker = world.loop()
    assert worker.work() == 2
    played = {arg(c, "--distill-output").rsplit("/", 1)[1]: c for c in world.commands if c[1] == "selfplay"}
    assert sorted(played) == ["seg00", "seg01"]
    assert played["seg01"][-4:] == ["--deep-fraction", "0.125", "--deep-multiplier", "4"]
    assert arg(played["seg01"], "--seed") == "1000" and arg(played["seg01"], "--games") == "1"
    assert world.claims["seg01"]["balance"] == {"theta": "chal", "engine_args": [
        "--deep-fraction", "0.125", "--deep-multiplier", "4"]}
    assert world.claims["seg01"]["games"] == 1
    assert world.claims["seg00"]["balance"]["engine_args"] == []
    assert worker.work() == 0  # every planned segment is done, though the games are 3 of --games 4
    assert not main.selfplay_left()


# ---------------------------------------------------------------- guards

def test_validation_with_cap_flags_is_refused(tmp_path, monkeypatch):
    world = World(tmp_path / "run", rl_loop, monkeypatch, search_balance=True, **VARIANTS["local"])
    loop = world.loop()
    loop.n, loop.it, loop.arms = 1, world.runs / "rl-it1", world.runs / "rl-it1" / "arms"
    for directory in ("stages", "search", "arms", "models"):
        (loop.it / directory).mkdir(parents=True)
    (loop.it / "stages" / "generator").write_text(rl_loop.sha256(world.generator) + "\n")
    loop.mark("balance_plan", {"segments": {"seg00": {"theta": "inc", "seed": 1000, "games": 2,
                                                      "engine_args": []}}})
    monkeypatch.setattr(rl_loop.selfplay, "time_model_options", lambda model: ("--deep-fraction", "0.5"))
    with pytest.raises(rl_loop.StageFailed, match="must play stock"):
        loop.selfplay("val", 2, 1999, loop.it / "search" / "val", loop.it / "search" / "val_run")


@pytest.mark.parametrize("argv, message", [
    (["--search-balance-mini-steps", "20"], "need --search-balance"),
    (["--search-balance", "--games", "5", "--segment-games", "2"], "multiple of --segment-games"),
    (["--search-balance", "--games", "4", "--segment-games", "2", "--search-balance-controller", "/no/such.py"],
     "no controller"),
    (["--search-balance", "--games", "4", "--segment-games", "2", "--distributed"], "without --distributed"),
])
def test_the_flags_are_checked(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, "argv", ["rl-loop", *argv])
    with pytest.raises(SystemExit) as raised:
        rl_loop.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_the_controller_is_found_above_the_fork(monkeypatch, tmp_path):
    monkeypatch.delenv("HIVEMIND_SEARCH_BALANCE", raising=False)
    assert rl_loop.search_balance_controller() == PROJECT_ROOT.parents[1] / "cluster" / "search_balance.py"
    script = tmp_path / "controller.py"
    script.write_text("")
    monkeypatch.setenv("HIVEMIND_SEARCH_BALANCE", str(script))
    assert rl_loop.search_balance_controller() == script.resolve()
    assert rl_loop.search_balance_controller(tmp_path / "missing.py") is None
