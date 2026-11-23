"""--shared-selfplay: segment claims on the shared filesystem, --selfplay-worker, the hook."""

import json
import os
import time
from types import SimpleNamespace

from hivemind.cli import rl_loop


def stale(path, seconds=rl_loop.STALE_CLAIM_SECONDS + 5):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_a_claim_is_exclusive_and_counts_while_fresh(tmp_path):
    a, b = rl_loop.Claims(tmp_path, "a"), rl_loop.Claims(tmp_path, "b")
    assert a.take(0, 500)
    assert not b.take(0, 500)
    assert a.live() == {0: 500}
    (tmp_path / "seg00.done").write_text('{"games": 500}\n')
    assert a.live() == {}  # finished: its games count through the .done marker
    a.release(0)
    assert not (tmp_path / "seg00.claim").exists()


def test_a_dead_claim_is_taken_over_once(tmp_path):
    a, b, c = (rl_loop.Claims(tmp_path, name) for name in "abc")
    assert a.take(3, 500)
    stale(a.path(3))
    assert b.live() == {}
    assert b.take(3, 500)
    assert not c.take(3, 500)  # b's claim is fresh again
    assert json.loads(b.path(3).read_text())["owner"] == "b"
    assert len(list(tmp_path.glob("seg03.claim.stale-*"))) == 1


def shared_loop(tmp_path, target, segment_games, n=5, began_shared=True):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs, loop.exchange = tmp_path, None
    loop.n, loop.it = n, tmp_path / f"rl-it{n}"
    loop.arms = loop.it / "arms"
    (loop.it / "stages").mkdir(parents=True)
    loop.args = SimpleNamespace(games=target, segment_games=segment_games, val_games=100, keep_selfplay=True,
                                anchor_after=None, shared_selfplay=True, selfplay_hook=None)
    if began_shared:  # as a loop with --shared-selfplay leaves a fresh iteration
        loop.mark("selfplay_mode", {"shared": True, "segment_games": segment_games})
    (loop.it / "stages" / "generator").write_text("generator\n")
    loop.checkpoint_selfplay = True
    loop.generator = tmp_path / "generator.onnx"
    loop.fixed, loop.fixed_name = tmp_path / "missing-it0.onnx", "it0"
    loop.log = lambda message: None
    return loop


def test_the_main_loop_skips_segments_workers_hold_and_plays_validation_while_they_finish(tmp_path, monkeypatch):
    loop = shared_loop(tmp_path, target=3000, segment_games=1000)
    stages = loop.it / "stages"
    other = rl_loop.Claims(stages, "worker")
    assert other.take(0, 1000)  # a worker is playing seg00
    calls = []

    def selfplay(name, games, seed, data, records):
        if loop.done(name):  # as the real stage
            return
        calls.append((name, games, seed))
        if name != "val":
            assert (stages / f"{name}.claim").is_file()
        loop.mark(name, {"games": games})

    def sleep(seconds):  # the worker finishes while the main loop waits
        calls.append(("wait",))
        (stages / "seg00.done").write_text('{"games": 1000}\n')
        other.release(0)

    monkeypatch.setattr(rl_loop.time, "sleep", sleep)
    loop.selfplay = selfplay
    loop.train = lambda: calls.append(("train",))
    loop.match = lambda *args: None
    loop.promote = loop.upload = lambda: None
    loop.iteration(5)
    assert calls == [("seg01", 1000, 5001), ("seg02", 1000, 5002), ("val", 100, 5999), ("wait",), ("train",)]
    assert not list(stages.glob("*.claim"))


def test_a_worker_plays_what_is_left_resumes_a_dead_claim_and_exits(tmp_path):
    loop = shared_loop(tmp_path, target=2500, segment_games=1000, n=1)  # the worker finds it1 by itself
    loop.check_stop = lambda: None
    stages = loop.it / "stages"
    (stages / "seg00.done").write_text('{"games": 1000}\n')
    dead = rl_loop.Claims(stages, "killed job")
    assert dead.take(1, 1000)
    stale(dead.path(1))
    progress = loop.it / "search" / "train_run" / "seg01" / "checkpoint" / "progress"
    progress.parent.mkdir(parents=True)
    progress.write_text("seed 1001\ngames 1000\ndone 400\n")
    calls = []

    def selfplay(name, games, seed, data, records):
        calls.append((name, games, seed))
        loop.mark(name, {"games": games})

    loop.selfplay = selfplay
    assert loop.work() == 2
    assert calls == [("seg01", 1000, 1001), ("seg02", 500, 1002)]
    assert loop.games_played() == 2500
    assert loop.work() == 0  # nothing left


def test_a_worker_before_the_iteration_starts_exits(tmp_path):
    loop = shared_loop(tmp_path, target=1000, segment_games=500, n=1)
    (loop.it / "stages" / "generator").unlink()
    assert loop.work() == 0


def test_the_hook_runs_once_per_iteration_with_its_number(tmp_path, monkeypatch):
    loop = shared_loop(tmp_path, target=500, segment_games=500)
    loop.args.selfplay_hook = "submit workers"
    hooks = []

    def run(command, **kwargs):
        hooks.append((command, kwargs["env"]["RL_ITERATION"]))
        return SimpleNamespace(returncode=0, stdout="submitted 4\n", stderr="")

    monkeypatch.setattr(rl_loop.subprocess, "run", run)
    loop.selfplay = lambda name, games, *a: loop.done(name) or loop.mark(name, {"games": games})
    loop.train = loop.promote = loop.upload = lambda: None
    loop.match = lambda *args: None
    loop.iteration(5)
    assert hooks == [("submit workers", "5")]


def test_the_heartbeat_keeps_a_claim_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_loop, "HEARTBEAT_SECONDS", 0.05)
    loop = shared_loop(tmp_path, target=500, segment_games=500)
    claims = rl_loop.Claims(loop.it / "stages", loop.owner)
    assert claims.take(0, 500)
    stale(claims.path(0))
    loop.claimed = 0
    with loop.heartbeat():
        time.sleep(0.3)
    assert time.time() - claims.path(0).stat().st_mtime < 5


# ---------------------------------------------------------------- switching a run to or from shared self-play

def stub_stages(loop, calls):
    def selfplay(name, games, seed, data, records):
        if loop.done(name):
            return
        calls.append((name, games, seed))
        loop.mark(name, {"games": games})
    loop.selfplay = selfplay
    loop.check_stop = lambda: None
    loop.train = lambda: loop.done("train") or (calls.append(("train",)), loop.mark("train", {}))
    loop.match = lambda name, *args: calls.append(("match", name))
    loop.promote = loop.upload = lambda: None


def hook_recorder(loop, monkeypatch):
    hooks = []
    loop.args.selfplay_hook = "submit workers"
    monkeypatch.setattr(rl_loop.subprocess, "run", lambda command, **kw: hooks.append(kw["env"]["RL_ITERATION"])
                        or SimpleNamespace(returncode=0, stdout="", stderr=""))
    return hooks


def test_an_iteration_begun_unshared_finishes_alone_with_its_own_segment_size(tmp_path, monkeypatch):
    # it2 began with --segment-games 2000 and no --shared-selfplay; the next job passes --shared-selfplay
    # --segment-games 500 and a hook (launch_full_run.sh REPLACE_PENDING=1 WORKERS=5).
    loop = shared_loop(tmp_path, target=10000, segment_games=500, n=2, began_shared=False)
    stages = loop.it / "stages"
    loop.mark("seg00", {"games": 2000})
    loop.mark("seg01", {"games": 2000})
    calls = []
    stub_stages(loop, calls)
    hooks = hook_recorder(loop, monkeypatch)
    loop.iteration(2)
    assert hooks == []  # no workers for it
    assert calls == [("seg02", 2000, 2002), ("seg03", 2000, 2003), ("seg04", 2000, 2004), ("val", 100, 2999),
                     ("train",), ("match", "generator")]
    assert json.loads((stages / "selfplay_mode.done").read_text()) == {
        "shared": False, "segment_games": 2000, "skipped": "the iteration started without --shared-selfplay"}
    assert not list(stages.glob("*.claim"))


def test_the_segment_size_of_an_unshared_iteration_comes_from_its_checkpoint(tmp_path):
    loop = shared_loop(tmp_path, target=10000, segment_games=500, n=2, began_shared=False)
    progress = loop.it / "search" / "train_run" / "seg00" / "checkpoint" / "progress"
    progress.parent.mkdir(parents=True)
    progress.write_text("seed 2000\ngames 2000\ndone 700\n")
    assert loop.legacy_segment_games() == 2000
    progress.unlink()
    assert loop.legacy_segment_games() == 500  # nothing played yet: this loop's size


def test_an_unshared_iteration_resumed_in_its_gate_then_the_next_begins_shared(tmp_path, monkeypatch):
    loop = shared_loop(tmp_path, target=1000, segment_games=500, n=1, began_shared=False)
    stages = loop.it / "stages"
    for name, games in (("seg00", 1000), ("val", 100), ("train", 0)):
        loop.mark(name, {"games": games})
    calls = []
    stub_stages(loop, calls)
    loop.generator.write_bytes(b"net")
    hooks = hook_recorder(loop, monkeypatch)
    loop.iteration(1)
    assert calls == [("match", "generator")] and hooks == []
    assert json.loads((stages / "selfplay_mode.done").read_text())["shared"] is False
    loop.iteration(2)
    stages2 = tmp_path / "rl-it2" / "stages"
    assert json.loads((stages2 / "selfplay_mode.done").read_text()) == {"shared": True, "segment_games": 500}
    assert hooks == ["2"]
    assert calls[1:] == [("seg00", 500, 2000), ("seg01", 500, 2001), ("val", 100, 2999), ("train",),
                         ("match", "generator")]


def test_the_mode_marker_is_written_before_the_generator_marker(tmp_path, monkeypatch):
    loop = shared_loop(tmp_path, target=500, segment_games=500, n=1)
    loop.args.search_balance = True
    order = []
    loop.balance_plan = lambda: order.append(("plan", (loop.it / "stages" / "generator").exists()))
    real_mark = loop.mark

    def mark(stage, value):
        if stage == "selfplay_mode":
            order.append(("mode", (loop.it / "stages" / "generator").exists()))
        real_mark(stage, value)
    loop.mark = mark
    calls = []
    stub_stages(loop, calls)
    loop.balance_after_train = loop.balance_finish = lambda: None
    loop.generator.write_bytes(b"net")
    loop.iteration(3)
    assert order == [("plan", False), ("mode", False)]


def test_workers_leave_an_unshared_iteration_alone(tmp_path):
    loop = shared_loop(tmp_path, target=1000, segment_games=500, n=1, began_shared=False)
    calls = []
    stub_stages(loop, calls)
    assert loop.work() == 0  # begun by a loop that never ran with --shared-selfplay
    loop.mark("selfplay_mode", {"shared": False, "segment_games": 1000})
    assert loop.work() == 0  # resumed by one that does
    assert calls == []


def test_an_iteration_begun_shared_finishes_shared_without_the_flag(tmp_path, monkeypatch):
    # Switching back to WORKERS=0: the next job passes neither --shared-selfplay nor a hook, and
    # --segment-games 2000; a worker still holds seg01.
    loop = shared_loop(tmp_path, target=2000, segment_games=500, n=4)
    loop.args.shared_selfplay, loop.args.segment_games = False, 2000
    stages = loop.it / "stages"
    loop.mark("seg00", {"games": 500})
    worker = rl_loop.Claims(stages, "worker")
    assert worker.take(1, 500)
    calls = []
    stub_stages(loop, calls)

    def sleep(seconds):
        calls.append(("wait",))
        loop.mark("seg01", {"games": 500})
        worker.release(1)
    monkeypatch.setattr(rl_loop.time, "sleep", sleep)
    loop.iteration(4)
    assert calls == [("seg02", 500, 4002), ("seg03", 500, 4003), ("val", 100, 4999), ("wait",), ("train",),
                     ("match", "generator")]
    assert loop.games_played() == 2000


def test_without_the_flag_no_mode_marker_is_written(tmp_path):
    loop = shared_loop(tmp_path, target=1000, segment_games=500, n=1, began_shared=False)
    loop.args.shared_selfplay = False
    (loop.it / "stages" / "generator").unlink()
    calls = []
    stub_stages(loop, calls)
    loop.generator.write_bytes(b"net")
    loop.iteration(1)
    assert not (loop.it / "stages" / "selfplay_mode.done").exists()
    assert calls[:2] == [("seg00", 500, 1000), ("seg01", 500, 1001)]
