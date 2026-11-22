"""rl-loop --gate-sprt: the sequential gate against the generator."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from hivemind.cli import rl_loop


def fishtest_llr_logistic(elo0, elo1, results):
    """fishtest's LLRcalc.LLR_logistic on [losses, draws, wins], transcribed
    from its pdf form (regularize, results_to_pdf, stats, LLR_alt2) as an
    independent check of rl_loop.sprt_llr."""
    results = [r if r > 0 else 1e-3 for r in results]
    n = sum(results)
    pdf = [(i / 2, r / n) for i, r in enumerate(results)]
    mu = sum(value * p for value, p in pdf)
    var = sum(p * (value - mu) ** 2 for value, p in pdf)
    s0, s1 = (1 / (1 + 10 ** (-elo / 400)) for elo in (elo0, elo1))
    return n * (s1 - s0) * (2 * mu - s0 - s1) / var / 2


def test_bounds_are_fishtests():
    lower, upper = rl_loop.sprt_bounds(0.05, 0.05)
    assert lower == pytest.approx(-2.944439, abs=1e-6) and upper == pytest.approx(2.944439, abs=1e-6)
    assert rl_loop.sprt_bounds(0.05, 0.1) == pytest.approx((math.log(0.1 / 0.95), math.log(0.9 / 0.05)))


@pytest.mark.parametrize("wins, losses, draws", [(60, 40, 0), (55, 45, 0), (40, 60, 0), (520, 480, 30),
                                                 (300, 250, 450), (100, 0, 0), (0, 100, 0)])
@pytest.mark.parametrize("elo0, elo1", [(0, 15), (0, 5), (-5, 5)])
def test_llr_matches_fishtest(wins, losses, draws, elo0, elo1):
    expected = fishtest_llr_logistic(elo0, elo1, [losses, draws, wins])
    assert rl_loop.sprt_llr(wins, losses, draws, elo0, elo1) == pytest.approx(expected, rel=1e-9)


def test_llr_by_hand():
    # 60-40 with no draws: mean 0.6, variance 0.24 per game (the 1e-3 draw
    # fishtest puts in for the zero count moves it in the sixth digit).
    s1 = 1 / (1 + 10 ** (-15 / 400))
    assert rl_loop.sprt_llr(60, 40, 0, 0, 15) == pytest.approx(100 * (s1 - 0.5) * (1.2 - 0.5 - s1) / 0.48, rel=1e-4)
    assert rl_loop.sprt_llr(60, 40, 0, 0, 15) == pytest.approx(0.80193, abs=1e-5)
    assert rl_loop.sprt_llr(0, 0, 0, 0, 15) == 0.0
    assert rl_loop.sprt_decision(3.0, (-2.94, 2.94)) == "H1"
    assert rl_loop.sprt_decision(-3.0, (-2.94, 2.94)) == "H0"
    assert rl_loop.sprt_decision(0.0, (-2.94, 2.94)) is None


def test_openings_rotate_by_the_pairs_played(tmp_path):
    path = tmp_path / "openings.tsv"
    path.write_text("# header\na\tw\t0\n\nb\tb\t0\nc\tw\t1\n")
    assert rl_loop.rotated_openings(path, 0) == "a\tw\t0\nb\tb\t0\nc\tw\t1\n"
    assert rl_loop.rotated_openings(path, 4) == "b\tb\t0\nc\tw\t1\na\tw\t0\n"


def sprt_loop(tmp_path, outcome, **args):
    """A loop at it3 with a stub engine whose game i of a tournament ends
    outcome(seed, i) for the contender (1 win, 0 draw, -1 loss)."""
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs = tmp_path
    loop.n, loop.it = 3, tmp_path / "rl-it3"
    loop.arms = loop.it / "arms"
    for directory in ("stages", "arms", "models", "search"):
        (loop.it / directory).mkdir(parents=True)
    loop.generator = tmp_path / "net.onnx"
    loop.generator.write_bytes(b"net")
    openings = tmp_path / "openings.tsv"
    openings.write_text("".join(f"fen{i}|fen{i}\tw\t0\n" for i in range(100)))
    loop.args = SimpleNamespace(**{
        "match_engine": tmp_path / "engine", "time_model": "clock", "match_games": 400, "match_movetime": 100,
        "openings": openings, "pause_for": "", "gate_sprt": (0.0, 15.0), "gate_sprt_alpha": 0.05,
        "gate_sprt_beta": 0.05, "gate_chunk": 100, "gate_max_games": None, **args})
    loop.resume_matches = True
    loop.logged = []
    loop.log = loop.logged.append
    loop.commands = []

    class Process:
        def __init__(self, command, **kwargs):
            loop.commands.append(command)
            out = Path(command[command.index("--output") + 1])
            games, seed = int(command[command.index("--games") + 1]), int(command[command.index("--seed") + 1])
            out.mkdir(parents=True, exist_ok=True)
            results = [outcome(seed, i) for i in range(games)]
            rows = [f"{i}\t{r}\t{1 - i % 2}\t0\tcheckmate\t10\t1000\t1000000000\t10\t500\t1000000000\t{10 * (i + 1)}"
                    for i, r in enumerate(results)]
            (out / "games.tsv").write_text("\n".join(["config\tstub", *rows]) + "\n")
            (out / "games.pgn").write_bytes(b"x" * 10 * games)
            wins, losses = results.count(1), results.count(-1)
            score = (wins + results.count(0) / 2) / games
            (out / "summary.json").write_text(json.dumps({
                "games": games, "seed": seed, "time_model": "clock", "move_time_ms": 100,
                "contender_wins": wins, "baseline_wins": losses, "draws": results.count(0),
                "contender_elo": None if score in (0, 1) else 400 * math.log10(score / (1 - score)),
                "elo_confidence_95": None,
                "performance": {"contender": {"nps": 1000.0}, "baseline": {"nps": 500.0}},
                "contender_breakdown": {"white": {"wins": wins, "losses": losses, "draws": 0}},
                "terminations": {"checkmate": games, "draw": 0}}))

        def wait(self, timeout=None):
            return 0

    rl_loop.subprocess.Popen = Process
    return loop


@pytest.fixture(autouse=True)
def restore_popen(monkeypatch):
    monkeypatch.setattr(rl_loop.subprocess, "Popen", rl_loop.subprocess.Popen)


def promote(loop, promote_above=0.5):
    loop.args.promote_above = promote_above
    loop.generator_sp1, loop.checkpoint = loop.runs / "net-sp1.onnx", loop.runs / "net.pt"
    installs = []
    rl_loop.install = lambda source, target: installs.append(target)
    loop.promote()
    return json.loads((loop.it / "stages" / "promote.done").read_text()), installs


@pytest.fixture(autouse=True)
def restore_install(monkeypatch):
    monkeypatch.setattr(rl_loop, "install", rl_loop.install)


def test_a_lopsided_gate_stops_after_one_chunk_and_promotes(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 10 else 0)  # 90 wins, 10 draws a chunk
    loop.match("generator", loop.generator)
    command, = loop.commands
    assert command[command.index("--games") + 1] == "100" and command[command.index("--seed") + 1] == "3000"
    out = loop.arms / "match_generator"
    assert Path(command[command.index("--output") + 1]) == out / "chunk00"
    marker = json.loads((loop.it / "stages" / "match_generator.done").read_text())
    assert marker["games"] == 100 and (marker["wins"], marker["losses"], marker["draws"]) == (90, 0, 10)
    assert marker["sprt"]["decision"] == "H1" and marker["sprt"]["llr"] >= marker["sprt"]["bounds"][1]
    assert marker["sprt"]["elo0"] == 0 and marker["sprt"]["elo1"] == 15 and marker["sprt"]["chunks"] == 1
    assert marker["movetime"] == 100 and marker["time_model"] == "clock"
    # The merged match sits where the readers of a match look.
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games"] == 100 and summary["contender_wins"] == 90 and summary["seeds"] == [3000]
    assert summary["terminations"] == {"checkmate": 100, "draw": 0}
    assert summary["performance"]["contender"]["nps"] == pytest.approx(1000.0)
    assert len((out / "games.tsv").read_text().splitlines()) == 101
    decided, installs = promote(loop)
    assert decided == {"promoted": True, "score": 0.95, "sprt": "H1"} and len(installs) == 3


def test_a_losing_gate_accepts_h0_and_keeps_the_generator(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 10 == 0 else -1)  # 10-90 a chunk
    loop.match("generator", loop.generator)
    marker = json.loads((loop.it / "stages" / "match_generator.done").read_text())
    assert marker["games"] == 100 and marker["sprt"]["decision"] == "H0"
    decided, installs = promote(loop, promote_above=0.0)  # H0 overrides even a lenient score rule
    assert decided["promoted"] is False and decided["sprt"] == "H0" and not installs


def test_a_close_gate_runs_to_its_cap_and_falls_back_to_the_score_rule(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 2 else -1, match_games=200, gate_max_games=400)
    loop.match("generator", loop.generator)
    seeds = [int(c[c.index("--seed") + 1]) for c in loop.commands]
    assert seeds == [3000, 3001, 3002, 3003]
    # Chunk k opens with the openings one 400-game match would play from pair 50 k.
    out = loop.arms / "match_generator"
    assert (out / "chunk01.openings.tsv").read_text().splitlines()[0].startswith("fen50|")
    assert (out / "chunk02.openings.tsv").read_text().splitlines()[0].startswith("fen0|")
    marker = json.loads((loop.it / "stages" / "match_generator.done").read_text())
    assert marker["games"] == 400 and marker["sprt"]["decision"] == "inconclusive"
    assert marker["sprt"]["max_games"] == 400 and marker["score"] == 0.5
    rows = (out / "games.tsv").read_text().splitlines()[1:]
    assert [int(row.split("\t")[0]) for row in rows] == list(range(400))
    assert int(rows[-1].split("\t")[11]) == (out / "games.pgn").stat().st_size == 4000
    decided, installs = promote(loop)
    assert decided == {"promoted": False, "score": 0.5, "sprt": "inconclusive"} and not installs


def test_the_cap_cuts_the_last_chunk(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 2 else -1, match_games=250, gate_chunk=100)
    loop.match("generator", loop.generator)
    assert [c[c.index("--games") + 1] for c in loop.commands] == ["100", "100", "50"]
    assert json.loads((loop.it / "stages" / "match_generator.done").read_text())["games"] == 250


def test_a_rerun_keeps_its_finished_chunks(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 2 else -1)
    out = loop.arms / "match_generator"
    original = loop.tournament
    calls = []

    def cut_off(*args):
        calls.append(args)
        original(*args)
        if len(calls) == 2:
            raise rl_loop.StageFailed("job hit its time limit")

    loop.tournament = cut_off
    with pytest.raises(rl_loop.StageFailed):
        loop.match("generator", loop.generator)
    assert (out / "chunk00.done").is_file() and not (out / "chunk01.done").exists()
    loop.commands.clear()
    loop.tournament = original
    loop.match("generator", loop.generator)
    seeds = [int(c[c.index("--seed") + 1]) for c in loop.commands]
    assert seeds == [3001, 3002, 3003]
    assert json.loads((loop.it / "stages" / "match_generator.done").read_text())["games"] == 400


def test_the_fixed_baseline_match_keeps_its_size(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 10 else 0)
    loop.match("it0", tmp_path / "it0.onnx", "flag")
    command, = loop.commands
    assert command[command.index("--games") + 1] == "400" and command[command.index("--seed") + 1] == "3"
    assert Path(command[command.index("--output") + 1]) == loop.arms / "match_it0"
    assert "sprt" not in json.loads((loop.it / "stages" / "match_it0.done").read_text())


def test_without_the_flag_the_gate_is_unchanged(tmp_path):
    loop = sprt_loop(tmp_path, lambda seed, i: 1 if i % 2 else -1, match_games=2)
    del loop.args.gate_sprt  # an args namespace from before the option: WAVE2_DEFAULTS
    loop.match("generator", loop.generator)
    out = loop.arms / "match_generator"
    command, = loop.commands
    assert command == [str(tmp_path / "engine"), "tournament", "--contender", str(loop.it / "models" / "C.onnx"),
                       "--baseline", str(loop.generator), "--movetime", "100", "--games", "2", "--seed", "3",
                       "--positions", str(loop.args.openings), "--output", str(out), "--time-model", "clock",
                       "--resume", "true"]
    assert loop.logged[0] == f"it3 match vs generator ({loop.generator}, 2 games at 100 ms, on a simulated clock)"
    marker = (loop.it / "stages" / "match_generator.done").read_text()
    assert marker == json.dumps({"wins": 1, "losses": 1, "draws": 0, "score": 0.5, "elo": 0.0, "ci": None,
                                 "nps": [1000.0, 500.0], "movetime": 100, "time_model": "clock"}) + "\n"
    assert not list(out.glob("chunk*"))
    decided, _ = promote(loop)
    assert decided == {"promoted": False, "score": 0.5}


def test_a_sequential_gate_is_not_reused_as_the_fixed_match(tmp_path):
    generator = tmp_path / "twin.onnx"
    generator.write_bytes(b"it0")
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.args = SimpleNamespace(anchor_after=None, games=0, segment_games=1000, val_games=0, keep_selfplay=True,
                                time_model="clock", gate_sprt=(0.0, 15.0))
    loop.runs, loop.exchange, loop.generator = tmp_path, None, generator
    loop.fixed, loop.fixed_name, loop.fixed_time_model = generator, "it0", "clock"
    loop.it = tmp_path / "rl-it2"
    (loop.it / "stages").mkdir(parents=True)
    (loop.it / "stages" / "generator").write_text(rl_loop.sha256(generator) + "\n")
    loop.log = lambda message: None
    loop.selfplay = lambda *a: None
    loop.attempt = lambda stage, *a: stage(*a)
    loop.train = lambda: None
    loop.promote = loop.upload = lambda: None
    matches = []

    def match(name, baseline, time_model=None):
        matches.append(name)
        (loop.it / "stages" / f"match_{name}.done").write_text(json.dumps({"score": 0.9, "sprt": {}}) + "\n")

    loop.match = match
    loop.iteration(2)
    assert matches == ["generator", "it0"]


def test_cli_parses_the_bounds():
    assert rl_loop.sprt_elos("0,15") == (0.0, 15.0)
    assert rl_loop.sprt_elos("-2.5,2.5") == (-2.5, 2.5)
    for bad in ("15,0", "5", "a,b"):
        with pytest.raises(rl_loop.argparse.ArgumentTypeError):
            rl_loop.sprt_elos(bad)


def test_no_sprt_decision_before_the_minimum_sample():
    bounds = rl_loop.sprt_bounds(0.05, 0.05)
    llr = rl_loop.sprt_llr(2, 0, 0, 0, 15)
    assert llr > bounds[1]
    assert rl_loop.sprt_decision(llr, bounds, 2, 400) is None
    assert rl_loop.sprt_decision(llr, bounds, 4, 4) == "H1"   # a cap below the minimum still decides at the cap
    assert rl_loop.sprt_decision(llr, bounds, 100, 400) == "H1"
