"""Wave 2 training: engine options, value-label hygiene, the trunk learning
rate and the rl-loop settings that use them."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hivemind import wave2
from hivemind.cli import rl_loop, train
from hivemind.distill import labels
from hivemind.distill.data import HEADER_BYTES, MAGIC, PLANES, VERSION, read_chunk
from hivemind.distill.train import parameter_groups
from hivemind.architectures.twin_board import get_twin_board_model


# ---- options

def test_stock_options_add_nothing_to_the_engine_command():
    assert wave2.STOCK.is_stock
    assert wave2.STOCK.engine_args() == []


def test_wave2_preset_sends_every_changed_option():
    args = wave2.WAVE2.engine_args()
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs == {"--durative-tree": "true", "--sit-gate": "false", "--flag-loss": "true",
                     "--instant-transfer": "false",
                     "--base-mix": "60000:0.25,120000:0.25,180000:0.35,300000:0.15",
                     "--seat-move-times": "true"}


def test_overrides_apply_on_top_of_the_preset():
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", "--sit-gate", "true", "--pass-cap-k", "4",
                                                 "--base-mix", "120000:1"]))
    assert options.sit_gate and options.durative_tree and options.pass_cap_k == 4
    assert options.base_mix == ((120_000, 1.0),)
    assert wave2.from_args(parser.parse_args([])) == wave2.STOCK
    # One ablation without the preset: stock plus the durative tree.
    ablation = wave2.from_args(parser.parse_args(["--durative-tree", "true"]))
    assert ablation.engine_args() == ["--durative-tree", "true"]


def test_flag_loss_needs_seat_clocks():
    with pytest.raises(ValueError):
        wave2.Options(flag_loss=True)
    with pytest.raises(ValueError):
        wave2.Options(pass_cap_k=0)


@pytest.mark.parametrize("text, mix", [
    ("off", ()), ("", ()), ("60000:0.5, 180000:0.5", ((60_000, 0.5), (180_000, 0.5))),
])
def test_base_mix_parses(text, mix):
    assert wave2.parse_base_mix(text) == mix
    if mix:
        assert wave2.parse_base_mix(wave2.format_base_mix(mix)) == mix


@pytest.mark.parametrize("text", ["60000", "60000:0", "0:1", "x:1"])
def test_bad_base_mix_is_refused(text):
    with pytest.raises(ValueError):
        wave2.parse_base_mix(text)


def test_team_tree_reuse_options():
    """Wave 2c fix 1: off by default (stock and the WAVE2 preset send nothing
    new), on top of the preset it sends its flags, and it needs the durative
    tree on seat clocks."""
    assert not wave2.STOCK.team_tree_reuse and not wave2.WAVE2.team_tree_reuse
    assert "--team-tree-reuse" not in wave2.WAVE2.engine_args()
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", "--team-tree-reuse", "true",
                                                 "--team-tree-reuse-tolerance-ms", "1000"]))
    pairs = dict(zip(options.engine_args()[::2], options.engine_args()[1::2]))
    assert pairs["--team-tree-reuse"] == "true"
    assert pairs["--team-tree-reuse-tolerance-ms"] == "1000"
    only_reuse = wave2.from_args(parser.parse_args(["--wave2", "--team-tree-reuse", "true"]))
    assert "--team-tree-reuse-tolerance-ms" not in only_reuse.engine_args()
    with pytest.raises(ValueError):
        wave2.Options(team_tree_reuse=True)
    with pytest.raises(ValueError):
        wave2.Options(durative_tree=True, team_tree_reuse=True)
    with pytest.raises(ValueError):
        wave2.replace(wave2.WAVE2, team_tree_reuse_tolerance_ms=5001)
    assert wave2.missing_engine_options(options, "--durative-tree --sit-gate --flag-loss "
                                        "--instant-transfer --base-mix --seat-move-times") == [
        "--team-tree-reuse", "--team-tree-reuse-tolerance-ms"]


def test_joint_plans_options():
    """Wave 2c fix 2: off by default (stock and WAVE2 send nothing new), on
    top of the preset it sends its flags, and it needs the durative tree only
    (not seat clocks: it acts inside each search)."""
    assert not wave2.STOCK.joint_plans and not wave2.WAVE2.joint_plans
    assert "--joint-plans" not in wave2.WAVE2.engine_args()
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", "--joint-plans", "true",
                                                 "--joint-plan-free-permille", "300"]))
    pairs = dict(zip(options.engine_args()[::2], options.engine_args()[1::2]))
    assert pairs["--joint-plans"] == "true"
    assert pairs["--joint-plan-free-permille"] == "300"
    only_plans = wave2.from_args(parser.parse_args(["--wave2", "--joint-plans", "true"]))
    assert "--joint-plan-free-permille" not in only_plans.engine_args()
    assert wave2.Options(durative_tree=True, joint_plans=True).joint_plans
    with pytest.raises(ValueError):
        wave2.Options(joint_plans=True)
    with pytest.raises(ValueError):
        wave2.replace(wave2.WAVE2, joint_plan_free_permille=1001)
    assert wave2.missing_engine_options(options, "--durative-tree --sit-gate --flag-loss "
                                        "--instant-transfer --base-mix --seat-move-times") == [
        "--joint-plans", "--joint-plan-free-permille"]


def test_plan_carry_options():
    """Wave 2c fix 3: off by default (stock and WAVE2 send nothing new), on
    top of the preset it sends its flags, and like fix 1 it needs the
    durative tree on seat clocks."""
    assert not wave2.STOCK.plan_carry and not wave2.WAVE2.plan_carry
    assert "--plan-carry" not in wave2.WAVE2.engine_args()
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", "--plan-carry", "true",
                                                 "--plan-carry-permille", "450"]))
    pairs = dict(zip(options.engine_args()[::2], options.engine_args()[1::2]))
    assert pairs["--plan-carry"] == "true"
    assert pairs["--plan-carry-permille"] == "450"
    only_carry = wave2.from_args(parser.parse_args(["--wave2", "--plan-carry", "true"]))
    assert "--plan-carry-permille" not in only_carry.engine_args()
    with pytest.raises(ValueError):
        wave2.Options(plan_carry=True)
    with pytest.raises(ValueError):
        wave2.Options(durative_tree=True, plan_carry=True)
    with pytest.raises(ValueError):
        wave2.replace(wave2.WAVE2, plan_carry_permille=1001)
    assert wave2.missing_engine_options(options, "--durative-tree --sit-gate --flag-loss "
                                        "--instant-transfer --base-mix --seat-move-times") == [
        "--plan-carry", "--plan-carry-permille"]


def test_continuity_sets_the_three_switches():
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", "--continuity"]))
    assert options.team_tree_reuse and options.joint_plans and options.plan_carry
    pairs = dict(zip(options.engine_args()[::2], options.engine_args()[1::2]))
    assert pairs["--team-tree-reuse"] == pairs["--joint-plans"] == pairs["--plan-carry"] == "true"
    # The knobs keep their defaults, so they are not sent.
    assert not {"--team-tree-reuse-tolerance-ms", "--joint-plan-free-permille",
                "--plan-carry-permille"} & set(options.engine_args())
    # An explicit switch overrides the preset.
    ablation = wave2.from_args(parser.parse_args(["--wave2", "--continuity", "--joint-plans", "false"]))
    assert ablation.team_tree_reuse and ablation.plan_carry and not ablation.joint_plans
    assert wave2.from_args(parser.parse_args(["--wave2"])) == wave2.WAVE2


def test_continuity_knob_needs_its_switch():
    """A wave 2c knob without its switch does nothing, and the engine would
    still take it as a search change and switch a stock run to the clock time
    model: it is refused (the engine's validate() refuses it too)."""
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    for flag, value in (("--team-tree-reuse-tolerance-ms", "400"),
                        ("--joint-plan-free-permille", "400"),
                        ("--plan-carry-permille", "200")):
        with pytest.raises(ValueError):
            wave2.from_args(parser.parse_args([flag, value]))
        with pytest.raises(ValueError):
            wave2.from_args(parser.parse_args(["--wave2", flag, value]))
    with pytest.raises(ValueError):
        wave2.Options(plan_carry_permille=200)
    # With the switch (or the preset) the knob is sent; at its default it is fine alone.
    options = wave2.from_args(parser.parse_args(["--wave2", "--continuity", "--plan-carry-permille", "200"]))
    assert "--plan-carry-permille" in options.engine_args()
    assert wave2.Options(plan_carry_permille=300).is_stock


def test_continuity_flags_are_in_the_engine_help():
    """missing_engine_options is empty against the built engine's --help (spec v5 5.2)."""
    import shutil
    import subprocess

    engine = Path(__file__).resolve().parents[1] / "engine" / "build-ninja" / "hivemind"
    if not engine.is_file() and shutil.which("hivemind") is None:
        pytest.skip("no built engine")
    text = subprocess.run([str(engine), "--help"], capture_output=True, text=True, timeout=60).stdout
    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    every = wave2.from_args(parser.parse_args([
        "--wave2", "--continuity", "--team-tree-reuse-tolerance-ms", "1000",
        "--joint-plan-free-permille", "300", "--plan-carry-permille", "450"]))
    assert wave2.missing_engine_options(every, text) == []


def test_missing_options_are_read_off_the_engine_help():
    assert wave2.missing_engine_options(wave2.WAVE2, "--durative-tree --sit-gate --flag-loss") == [
        "--instant-transfer", "--base-mix", "--seat-move-times"]
    assert wave2.missing_engine_options(wave2.STOCK, "") == []


def test_init_lineage_copies_it13_once(tmp_path):
    def fetch(variant, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{variant}.bin"
        path.write_bytes(variant.encode())
        return path

    paths = wave2.init_lineage("w2", tmp_path / "models", tmp_path / "artifacts", fetch=fetch)
    assert paths["generator"].read_bytes() == paths["fixed"].read_bytes() == b"clock-it13"
    assert paths["checkpoint"].read_bytes() == b"clock-it13-checkpoint"
    paths["generator"].write_bytes(b"promoted")
    wave2.init_lineage("w2", tmp_path / "models", tmp_path / "artifacts", fetch=fetch)
    assert paths["generator"].read_bytes() == b"promoted"


# ---- value labels

def write_chunk(path, outcomes, wdl, planes=None):
    """An HDST v2 chunk: one policy entry (the pass) per board, per record."""
    n = len(outcomes)
    bits = np.zeros((n, PLANES), np.int64) if planes is None else planes
    raw = [MAGIC, np.array([VERSION, PLANES], np.uint32).tobytes(), np.array([n, 2 * n], np.uint64).tobytes(),
           bits.tobytes(), np.zeros((n, PLANES), np.float16).tobytes(), np.zeros(n, np.float32).tobytes(),
           np.asarray(wdl, np.float32).reshape(n, 3).tobytes(), np.full(n, 0.5, np.float16).tobytes(),
           np.arange(n, dtype=np.uint16).tobytes(), np.asarray(outcomes, np.int8).tobytes(),
           np.ones((n, 2), np.uint16).tobytes(), np.zeros(2 * n, np.uint16).tobytes(),
           np.ones(2 * n, np.float16).tobytes()]
    path.write_bytes(b"".join(raw))
    assert len(raw[0]) + 4 + 4 + 8 + 8 == HEADER_BYTES


def one_hot(outcomes):
    wdl = np.zeros((len(outcomes), 3), np.float32)
    wdl[np.arange(len(outcomes)), np.asarray(outcomes) + 1] = 1
    return wdl


def game_line(number, samples, termination, total=9):
    return f"selfplay game {number}/{total} raw 0 samples {samples} events 9 termination {termination} lead 0->5 ms"


def segment(tmp_path, games, lines=None):
    """games: (termination, outcomes, labelled) per game."""
    data = tmp_path / "data"
    data.mkdir()
    outcomes = [o for _, game, _ in games for o in game]
    wdl = np.concatenate([one_hot(game) if labelled else np.zeros((len(game), 3))
                          for _, game, labelled in games])
    write_chunk(data / "search_00000.dst", outcomes[:3], wdl[:3])
    write_chunk(data / "search_00001.dst", outcomes[3:], wdl[3:])
    log = tmp_path / "seg.log"
    log.write_text("\n".join(lines or [game_line(k, len(game), t) for k, (t, game, _) in enumerate(games, 1)])
                   + "\n")
    assert read_chunk(data / "search_00000.dst")["outcome"].tolist() == outcomes[:3]
    return log, data


def test_clean_labels_pass(tmp_path):
    log, data = segment(tmp_path, [("checkmate", [1, -1, 1], True), ("flag", [-1, 1], True),
                                   ("macro-ply limit", [0, 0], False), ("double_flag", [0], True)])
    report = labels.label_report(log, data)
    assert report["problems"] == []
    assert report["games"] == 4 and report["records"] == 8
    assert report["flag_games"] == 1 and report["flag_records"] == 2
    assert report["terminations"] == {"checkmate": 1, "flag": 1, "macro-ply limit": 1, "double_flag": 1}


def test_a_flag_labelled_as_a_draw_is_a_problem(tmp_path):
    log, data = segment(tmp_path, [("checkmate", [1, -1, 1], True), ("flag", [0, 0], True)])
    assert any("flag labelled as a draw" in p for p in labels.label_report(log, data)["problems"])


def test_an_unlabelled_flag_is_a_problem(tmp_path):
    log, data = segment(tmp_path, [("checkmate", [1, -1, 1], True), ("flag", [0, 0], False)])
    assert any("without a WDL" in p for p in labels.label_report(log, data)["problems"])


def test_an_unfinished_game_must_not_carry_a_result(tmp_path):
    log, data = segment(tmp_path, [("macro-ply limit", [1, -1, 1], True), ("checkmate", [1], True)])
    assert any("unfinished" in p for p in labels.label_report(log, data)["problems"])


def test_dropped_records_are_caught(tmp_path):
    # The log says 3 + 3 samples; the chunks hold 5: one was deduplicated away.
    games = [("checkmate", [1, -1, 1], True), ("checkmate", [1, -1], True)]
    log, data = segment(tmp_path, games, lines=[game_line(1, 3, "checkmate"), game_line(2, 3, "checkmate")])
    assert any("dropped" in p for p in labels.label_report(log, data)["problems"])


def test_a_replayed_game_line_counts_once(tmp_path):
    # A resumed run plays game 2 again and prints its line again.
    games = [("checkmate", [1, -1, 1], True), ("flag", [1, -1], True)]
    lines = [game_line(1, 3, "checkmate"), game_line(2, 4, "checkmate"), game_line(2, 2, "flag")]
    log, data = segment(tmp_path, games, lines=lines)
    assert labels.label_report(log, data)["problems"] == []


def test_termination_with_spaces_and_trailing_keys():
    lines = ["selfplay game 1/2 raw 3 samples 7 events 12 termination macro-ply limit",
             "selfplay game 2/2 raw 0 samples 5 events 9 termination no legal action resign-check right",
             "noise", "selfplay complete 2 games"]
    assert labels.game_lines(lines) == [(7, "macro-ply limit"), (5, "no legal action")]


def test_a_pass_then_a_move_in_the_same_position_is_kept_and_counted(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    planes = np.zeros((3, PLANES), np.int64)
    planes[2, 0] = 1  # records 0 and 1: the same position (a pass, then the move)
    write_chunk(data / "search_00000.dst", [1, -1, 1], one_hot([1, -1, 1]), planes=planes)
    log = tmp_path / "seg.log"
    log.write_text(game_line(1, 3, "checkmate") + "\n")
    report = labels.label_report(log, data)
    assert report["problems"] == [] and report["records"] == 3 and report["repeat_records"] == 1


# ---- training

def test_trunk_and_heads_get_their_own_learning_rates():
    model = get_twin_board_model("twin-s", attention_dim=0, value_diff=True, se_every=0)
    single, = parameter_groups(model, 1e-4)
    assert len(single["params"]) == len(list(model.parameters()))
    trunk, heads = parameter_groups(model, 1e-4, 5e-5)
    assert trunk["lr"] == 5e-5 and heads["lr"] == 1e-4
    assert len(trunk["params"]) + len(heads["params"]) == len(list(model.parameters()))
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(names[id(p)].startswith(("policy_head.", "value_head.")) for p in heads["params"])
    assert any(names[id(p)].startswith("stem.") for p in trunk["params"])
    optimizer = torch.optim.AdamW([trunk, heads], lr=1e-4)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 1e-4]


def test_the_stock_training_command_is_unchanged(tmp_path):
    command = train.distill_command(tmp_path / "init.pt", [tmp_path / "d"], [tmp_path / "v"], [], 10, 0, 1024, 1,
                                    tmp_path / "out")
    assert command[command.index("--lr") + 1] == "1e-4"
    assert "--trunk-lr" not in command
    command = train.distill_command(tmp_path / "init.pt", [tmp_path / "d"], [tmp_path / "v"], [], 10, 0, 1024, 1,
                                    tmp_path / "out", lr="1e-4", trunk_lr="5e-5")
    assert command[command.index("--trunk-lr") + 1] == "5e-5"


# ---- rl-loop

def loop_at(tmp_path, n=3, **args):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs = tmp_path
    loop.n, loop.it = n, tmp_path / f"rl-it{n}"
    loop.arms = loop.it / "arms"
    for directory in ("stages", "arms", "models", "search"):
        (loop.it / directory).mkdir(parents=True)
    loop.generator = tmp_path / "w2.onnx"
    loop.generator.write_bytes(b"generator")
    loop.generator_sp1 = tmp_path / "w2-sp1.onnx"
    loop.checkpoint = tmp_path / "w2.pt"
    (loop.it / "stages" / "generator").write_text(rl_loop.sha256(loop.generator) + "\n")
    loop.args = SimpleNamespace(**{
        "selfplay_engine": tmp_path / "sp1", "match_engine": tmp_path / "engine", "parallel_games": 6,
        "time_model": "clock", "match_games": 2, "match_movetime": 100, "openings": tmp_path / "o.tsv",
        "pause_for": "", "keep_selfplay": True, "resign_threshold": 0.0, "tree_reuse": False,
        "checkpoint_games": 0, **args})
    loop.checkpoint_selfplay = loop.resume_matches = False
    loop.teacher_val = []
    loop.log = lambda message: None
    return loop


def fake_selfplay(monkeypatch, termination, outcomes, labelled):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        data = Path(command[command.index("--distill-output") + 1])
        data.mkdir(parents=True, exist_ok=True)
        wdl = one_hot(outcomes) if labelled else np.zeros((len(outcomes), 3))
        write_chunk(data / "search_00000.dst", outcomes, wdl)
        kwargs["stdout"].write(game_line(1, len(outcomes), termination, total=1) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rl_loop.subprocess, "run", run)
    return commands


def test_wave2_selfplay_sends_the_options_and_records_the_labels(tmp_path, monkeypatch):
    loop = loop_at(tmp_path, wave2_options=wave2.WAVE2, label_check=True)
    commands = fake_selfplay(monkeypatch, "flag", [1, -1, 1], True)
    loop.selfplay("seg00", 1, 3000, loop.it / "search" / "train" / "seg00", loop.it / "search" / "train_run" / "seg00")
    command, = commands
    assert command[command.index("--durative-tree") + 1] == "true"
    assert command[command.index("--sit-gate") + 1] == "false"
    assert command[command.index("--time-model") + 1] == "clock"
    marker = json.loads((loop.it / "stages" / "seg00.done").read_text())
    assert marker["labels"]["flag_games"] == 1 and marker["wave2"]["sit_gate"] is False


def test_bad_labels_fail_the_segment(tmp_path, monkeypatch):
    loop = loop_at(tmp_path, wave2_options=wave2.WAVE2, label_check=True)
    fake_selfplay(monkeypatch, "flag", [0, 0], False)
    with pytest.raises(rl_loop.StageFailed, match="value labels"):
        loop.selfplay("seg00", 1, 3000, loop.it / "search" / "train" / "seg00",
                      loop.it / "search" / "train_run" / "seg00")
    assert not (loop.it / "stages" / "seg00.done").exists()


def test_upstream_selfplay_is_unchanged_without_wave2_settings(tmp_path, monkeypatch):
    loop = loop_at(tmp_path)
    commands = fake_selfplay(monkeypatch, "flag", [0, 0], False)  # unchecked without --label-check
    loop.selfplay("seg00", 1, 3000, loop.it / "search" / "train" / "seg00", loop.it / "search" / "train_run" / "seg00")
    assert "--durative-tree" not in commands[0]
    assert "labels" not in json.loads((loop.it / "stages" / "seg00.done").read_text())


@pytest.mark.parametrize("above, score, promoted", [(None, 0.5, True), (0.5, 0.5, False), (0.5, 0.51, True)])
def test_promotion_keeps_the_better_network(tmp_path, monkeypatch, above, score, promoted):
    loop = loop_at(tmp_path, promote_above=above)
    loop.mark("match_generator", {"score": score})
    (loop.it / "models" / "C.onnx").write_bytes(b"candidate")
    (loop.arms / "C").mkdir()
    (loop.arms / "C" / "last.pt").write_bytes(b"candidate weights")
    monkeypatch.setattr(rl_loop, "install", lambda source, target: target.write_bytes(source.read_bytes()))
    loop.promote()
    assert (loop.generator.read_bytes() == b"candidate") == promoted
    assert json.loads((loop.it / "stages" / "promote.done").read_text())["promoted"] == promoted


@pytest.mark.parametrize("train_from, expected", [("previous", "previous"), ("generator", "generator")])
def test_training_starts_from_the_promoted_network(tmp_path, monkeypatch, train_from, expected):
    loop = loop_at(tmp_path, train_from=train_from, lr="1e-4", trunk_lr="5e-5", window=1, new_passes=1.5,
                   replay_passes=0.5, batch_size=8)
    previous = tmp_path / "rl-it2" / "arms" / "C"
    previous.mkdir(parents=True)
    (previous / "last.pt").write_bytes(b"unpromoted")
    loop.checkpoint.write_bytes(b"generator")
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 100 if dirs else 0)
    monkeypatch.setattr(rl_loop, "train_dirs", lambda runs, n: [runs / f"d{n}"])
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        raise SystemExit  # stop before the outputs are checked

    monkeypatch.setattr(rl_loop.subprocess, "run", run)
    with pytest.raises(SystemExit):
        loop.train()
    command, = commands
    init = Path(command[command.index("--init") + 1])
    assert init == (previous / "last.pt" if expected == "previous" else loop.checkpoint).resolve()
    assert command[command.index("--trunk-lr") + 1] == "5e-5"


def test_rl_loop_wave2_flags_parse(monkeypatch, tmp_path):
    """--wave2 implies the clock model and the conservative fine-tune."""
    for name in ("generator", "selfplay", "match", "openings"):
        (tmp_path / name).write_bytes(b"x")
    captured = {}

    class Stop(Exception):
        pass

    def fake_loop(args, generator, checkpoint):
        captured["args"] = args
        raise Stop

    monkeypatch.setattr(rl_loop, "Loop", fake_loop)
    monkeypatch.setattr(rl_loop, "engine_help", lambda engine: " ".join(wave2.WAVE2.engine_args()[::2]))
    monkeypatch.setattr("sys.argv", ["rl-loop", "--wave2", "--no-upload", "--keep-selfplay",
                                     "--generator", str(tmp_path / "generator"),
                                     "--selfplay-engine", str(tmp_path / "selfplay"),
                                     "--match-engine", str(tmp_path / "match"),
                                     "--openings", str(tmp_path / "openings"), "--runs", str(tmp_path / "runs")])
    with pytest.raises(Stop):
        rl_loop.main()
    args = captured["args"]
    assert args.time_model == "clock" and args.train_from == "generator" and args.promote_above == 0.5
    assert args.lr == "1e-4" and args.trunk_lr == "5e-5" and args.wave2_options == wave2.WAVE2


def test_rl_loop_refuses_an_engine_without_the_wave2_options(monkeypatch, tmp_path):
    for name in ("generator", "selfplay", "match", "openings"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(rl_loop, "engine_help", lambda engine: "")
    argv = ["rl-loop", "--wave2", "--no-upload", "--generator", str(tmp_path / "generator"),
            "--selfplay-engine", str(tmp_path / "selfplay"), "--match-engine", str(tmp_path / "match"),
            "--openings", str(tmp_path / "openings"), "--runs", str(tmp_path / "runs")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="lack the wave 2 options"):
        rl_loop.main()


def test_training_refuses_less_than_one_batch(tmp_path, monkeypatch):
    """distill-train would loop forever looking for a whole batch."""
    loop = loop_at(tmp_path, train_from="generator", window=1, new_passes=1.5, replay_passes=0.5, batch_size=1024)
    loop.checkpoint.write_bytes(b"generator")
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 291 if dirs else 0)
    monkeypatch.setattr(rl_loop, "train_dirs", lambda runs, n: [runs / f"d{n}"])
    with pytest.raises(rl_loop.StageFailed, match="fewer than one batch"):
        loop.train()
