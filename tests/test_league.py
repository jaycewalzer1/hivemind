"""hivemind.league (spec v7 section 5.3): the league and pace segment plan rl-loop freezes
per iteration, its validation, PFSP weights and the segment summaries."""

import argparse
import json

import pytest

from hivemind import league


def parse(*argv, games=10_000):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=games)
    league.add_arguments(parser)
    args = parser.parse_args(list(argv))
    league.validate(args, parser)
    return args


def refused(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    league.add_arguments(parser)
    with pytest.raises(SystemExit):
        league.validate(parser.parse_args(list(argv)), parser)


@pytest.fixture
def lineage(tmp_path):
    """runs/ with a generator, the fixed baseline and helpers to add iterations."""
    runs = tmp_path / "runs"
    runs.mkdir()
    generator = tmp_path / "twin-s-clock-w2.onnx"
    generator.write_bytes(b"generator")
    fixed = tmp_path / "twin-s-clock-w2-it0.onnx"
    fixed.write_bytes(b"it13")

    def iteration(m, promoted, content=None):
        it = runs / f"rl-it{m}"
        (it / "stages").mkdir(parents=True)
        (it / "models").mkdir()
        (it / "models" / "C.onnx").write_bytes(content or f"C{m}".encode())
        (it / "stages" / "promote.done").write_text(json.dumps({"promoted": promoted, "score": 0.5}))
        return it

    return runs, generator, fixed, iteration


def kinds(plan):
    return [spec["theta"].split(":")[0] for spec in plan["segments"].values()]


def test_defaults_are_off_and_active_follows_the_switches():
    args = parse()
    assert not league.active(args)
    assert league.pace_args(args) == []
    assert league.active(parse("--pace"))
    assert league.active(parse("--pace", "--league"))
    # Old namespaces (workers, markers) lack the options.
    assert not league.active(argparse.Namespace())


def test_validation_errors():
    refused("--league")                                   # needs --pace
    refused("--pace-k-range", "100:400")                  # knob without --pace
    refused("--league-past", "2", "--pace")               # knob without --league
    refused("--league-pfsp", "off", "--pace")
    refused("--pace", "--pace-k-range", "400:80")
    refused("--pace", "--pace-k-range", "0:80")
    refused("--pace", "--pace-nodes", "300:200")
    refused("--pace", "--pace-ref-k", "0")
    refused("--pace", "--league", "--league-shares", "0,0.5,0.5")
    refused("--pace", "--league", "--league-shares", "0.5,0.5")
    refused("--pace", "--league", "--league-fixed-k", "0")
    refused("--pace", "--league", "--league-fixed-model", "/no/such.onnx")
    with pytest.raises(SystemExit):
        parse("--pace-k-range", "80-400")


def test_pace_only_plan_is_self_segments_with_pace_args(lineage):
    runs, generator, fixed, _ = lineage
    args = parse("--pace", "--pace-k-range", "90:300", "--pace-nodes", "100:2000", games=2500)
    plan = league.plan(args, runs, 4, generator, fixed, 1000)
    assert sorted(plan["segments"]) == ["seg00", "seg01", "seg02"]
    pace = ["--pace-k-min", "90", "--pace-k-max", "300", "--pace-ref-k", "110",
            "--pace-min-nodes", "100", "--pace-max-nodes", "2000"]
    for k, (name, spec) in enumerate(sorted(plan["segments"].items())):
        assert spec["theta"] == "self"
        assert spec["seed"] == 4000 + k
        assert spec["engine_args"] == pace
        assert spec["opponent_model"] is None
    assert [s["games"] for s in plan["segments"].values()] == [1000, 1000, 500]
    assert "pace k 90:300" in plan["summary"]


def test_iteration_one_has_no_past_so_its_share_goes_to_fixed(lineage):
    runs, generator, fixed, _ = lineage
    args = parse("--pace", "--league")
    plan = league.plan(args, runs, 1, generator, fixed, 500)  # 20 segments
    assert len(plan["segments"]) == 20
    assert kinds(plan).count("self") == 10
    assert kinds(plan).count("fixed") == 10
    assert kinds(plan).count("past") == 0
    # The fixed paces are cycled evenly: 5 paces, 2 segments each.
    thetas = [s["theta"] for s in plan["segments"].values() if s["theta"].startswith("fixed")]
    assert sorted(set(thetas)) == sorted(f"fixed:k{k}" for k in (80, 110, 160, 250, 400))
    assert all(thetas.count(theta) == 2 for theta in set(thetas))
    # Self and league segments are interleaved, not in two blocks.
    assert kinds(plan)[:2] == ["self", "fixed"]


def test_allocation_with_past_generators(lineage):
    runs, generator, fixed, iteration = lineage
    iteration(1, True)
    iteration(2, False)
    iteration(3, True)
    iteration(4, True, content=b"generator")  # the current generator: never its own opponent
    iteration(5, True, content=b"C3")         # same network as it3: once
    iteration(6, True)
    args = parse("--pace", "--league")
    plan = league.plan(args, runs, 7, generator, fixed, 500)
    counts = {kind: kinds(plan).count(kind) for kind in ("self", "past", "fixed")}
    assert counts == {"self": 10, "past": 5, "fixed": 5}
    past = [s["theta"] for s in plan["segments"].values() if s["theta"].startswith("past")]
    # Newest first, up to 3: it6, it5 (= it3's network, newest copy), it1; dealt 2, 2, 1.
    assert sorted(set(past)) == ["past:it1", "past:it5", "past:it6"]
    assert past.count("past:it6") == 2 and past.count("past:it5") == 2 and past.count("past:it1") == 1


def test_engine_args_per_segment(lineage):
    runs, generator, fixed, iteration = lineage
    iteration(1, True)
    args = parse("--pace", "--league")
    plan = league.plan(args, runs, 2, generator, fixed, 1000)
    pace = league.pace_args(args)
    for spec in plan["segments"].values():
        theta = spec["theta"]
        if theta == "self":
            assert spec["engine_args"] == pace and spec["opponent_model"] is None
            assert spec["league"] == {"kind": "self"}
        elif theta.startswith("past:"):
            assert spec["engine_args"] == [*pace, "--opponent-model", "{opponent_model}"]
            assert spec["opponent_model"] == str(runs / "rl-it1" / "models" / "C.onnx")
            assert spec["league"]["kind"] == "past" and spec["league"]["iteration"] == 1
        else:
            k = int(theta.split(":k")[1])
            assert spec["engine_args"] == [*pace, "--opponent-model", "{opponent_model}", "--opponent-k", str(k)]
            assert spec["opponent_model"] == str(fixed)
            assert spec["league"]["k"] == k
    other = runs.parent / "other.onnx"
    other.write_bytes(b"other")
    explicit = league.plan(parse("--pace", "--league", "--league-fixed-model", str(other)),
                           runs, 2, generator, fixed, 1000)
    fixed_models = {s["opponent_model"] for s in explicit["segments"].values() if s["theta"].startswith("fixed")}
    assert fixed_models == {str(other)}


def test_small_iterations_keep_one_self_segment(lineage):
    runs, generator, fixed, _ = lineage
    args = parse("--pace", "--league", games=500)
    plan = league.plan(args, runs, 1, generator, fixed, 500)
    assert kinds(plan) == ["self"]
    plan = league.plan(parse("--pace", "--league", "--league-shares", "0.1,0.45,0.45", games=1000),
                       runs, 1, generator, fixed, 500)
    assert kinds(plan) == ["self", "fixed"]


def test_pfsp_weights_from_the_previous_iteration(lineage):
    runs, generator, fixed, iteration = lineage
    iteration(1, True)
    previous = iteration(2, False)
    # The learner beat fixed:k80 (0.9 over 300 games), lost to fixed:k400 (0.2 over 400) and
    # met fixed:k110 too few times to count.
    for name, theta, wins, draws, losses in (("seg00", "fixed:k80", 270, 0, 30), ("seg01", "fixed:k400", 80, 0, 200),
                                             ("seg02", "fixed:k400", 0, 0, 120), ("seg03", "fixed:k110", 10, 0, 50),
                                             ("seg04", "self", 0, 0, 0)):
        summary = {"theta": theta, "learner_games": wins + draws + losses, "wins": wins, "draws": draws,
                   "losses": losses}
        (previous / "stages" / f"{name}.done").write_text(json.dumps({"games": 1, "league": summary}))
    assert league.pfsp_weight(None) == 1.0
    assert league.pfsp_weight({"games": 199, "score": 1.0}) == 1.0
    assert league.pfsp_weight({"games": 200, "score": 1.0}) == pytest.approx(0.0025)
    assert league.pfsp_weight({"games": 300, "score": 0.9}) == pytest.approx(0.01)
    plan = league.plan(parse("--pace", "--league", "--league-fixed-k", "80,110,400"),
                       runs, 3, generator, fixed, 500)  # 20 segments: past 5 (it1), fixed 5
    opponents = plan["opponents"]
    assert opponents["fixed:k80"]["weight"] == pytest.approx(0.01)
    assert opponents["fixed:k400"]["weight"] == pytest.approx(0.64)
    assert opponents["fixed:k110"]["weight"] == 1.0
    assert opponents["fixed:k400"]["previous"] == {"games": 400, "score": 0.2}
    # 5 fixed segments by weights 0.01 : 1 : 0.64 -> 0, 3, 2.
    fixed_thetas = [s["theta"] for s in plan["segments"].values() if s["theta"].startswith("fixed")]
    assert fixed_thetas.count("fixed:k80") == 0
    assert fixed_thetas.count("fixed:k110") == 3
    assert fixed_thetas.count("fixed:k400") == 2
    # PFSP off: equal weights.
    flat = league.plan(parse("--pace", "--league", "--league-fixed-k", "80,110,400", "--league-pfsp", "off"),
                       runs, 3, generator, fixed, 500)
    assert {o["weight"] for o in flat["opponents"].values()} == {1.0}


def test_apportion_and_spread():
    assert league.apportion(5, [1, 1, 1]) == [2, 2, 1]
    assert league.apportion(0, [1, 2]) == [0, 0]
    assert league.apportion(4, [0.01, 1, 0.64]) == [0, 2, 2]
    assert league.spread([["s"] * 2, ["p"], ["f"]]) == ["s", "p", "f", "s"]


GAME = ("selfplay game {n}/4 raw 3 samples 50 events 70 termination checkmate lead 0->726 ms clocks 1 2 3 4 "
        "hybrid 0/13/87 recent w0 b0 k_white_team {kw} k_black_team {kb}{league}{think}")


def game_line(n, kw, kb, team=None, result=None, scale=None):
    return GAME.format(n=n, kw=kw, kb=kb,
                       league="" if team is None else f" learner_team {team} learner_result {result}",
                       think="" if scale is None else f" mean_s {scale:.3f} think_missing 0")


def test_segment_summary_parses_learner_results_and_paces(tmp_path):
    log = tmp_path / "seg01.log"
    log.write_text("\n".join([
        "Self-play using 1 inference engine(s)",
        game_line(1, 100, 250, "white", "win", 1.2),
        game_line(2, 80, 250, "black", "loss", 0.8),
        game_line(2, 90, 250, "black", "draw", 1.0),  # a resumed run's replay: the last line counts
        game_line(3, 250, 120, "black", "win", 1.0),
        "selfplay complete 4 games",
    ]))
    summary = league.segment_summary(log, {"theta": "fixed:k250"})
    assert summary == {"theta": "fixed:k250", "games": 3, "learner_games": 3, "wins": 2, "draws": 1, "losses": 0,
                       "score": pytest.approx(0.8333, abs=1e-4), "mean_k": pytest.approx(176.6667, abs=1e-4),
                       "mean_k_learner": pytest.approx(156.6667, abs=1e-4),
                       "mean_k_opponent": pytest.approx(196.6667, abs=1e-4),
                       "mean_s": pytest.approx(1.0667, abs=1e-4)}
    # A self segment (pace only) and a missing log.
    log.write_text(game_line(1, 100, 300) + "\n")
    summary = league.segment_summary(log, {"theta": "self"})
    assert summary["learner_games"] == 0 and summary["score"] is None
    assert summary["mean_k"] == pytest.approx(200.0) and summary["mean_s"] is None
    assert league.segment_summary(tmp_path / "missing.log", {"theta": "self"})["games"] == 0


def test_segment_summary_reads_the_engine_line_format(tmp_path):
    """The exact line the engine printed in the Mac smoke run (it13 vs it13, pace 80:400)."""
    log = tmp_path / "seg.log"
    log.write_text("selfplay game 3/4 raw 3 samples 95 events 122 termination checkmate lead 0->349 ms clocks "
                   "137641 145640 137292 145989 hybrid 0/23/157 recent w0 b0 k_white_team 135 k_black_team 225 "
                   "learner_team black learner_result loss\n")
    summary = league.segment_summary(log, {"theta": "past:it2"})
    assert (summary["losses"], summary["mean_k_learner"], summary["mean_k_opponent"]) == (1, 225.0, 135.0)
