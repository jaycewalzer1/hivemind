"""Optional clock planes (custom engine spec, section 2): 41 channels per board."""

import importlib.util
import math
import struct
import sys
from pathlib import Path

import chess
import numpy as np
import polars as pl
import pytest
import torch

from hivemind.architectures.model_config import (
    ModelArgs,
    create_model,
    get_model_args,
    model_args_from_state_dict,
)
from hivemind.constants import (
    NUM_BUGHOUSE_CHANNELS,
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCK_BUGHOUSE_CHANNELS,
    board_channels_from_total,
    channels_per_board,
)
from hivemind.data.convert_selfplay_data import (
    HEADER,
    JOINT_POLICY_ENTRY,
    POLICY_ENTRY,
    SAMPLE_METADATA,
    convert_to_split_parquet,
    is_validation_game,
    read_binary_shard,
)
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes
from hivemind.training.data_loaders import flip_bughouse_sample, load_rl_parquet_shard
from hivemind.training.train_loop import checkpoint_input_channels, rl_parquet_input_channels

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_clock_net", TOOLS / "make_clock_net.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_clock_net"] = module
    spec.loader.exec_module(module)
    return module


def _some_position():
    board = BughouseBoard()
    for board_num, uci in [(0, "e2e4"), (1, "d2d4"), (0, "d7d5"), (0, "e4d5"),
                           (1, "g8f6"), (0, "d8d5")]:
        board.push_uci(board_num, uci)
    return board


# ---------------------------------------------------------------- constants --

def test_default_channel_counts_are_unchanged():
    assert NUM_BUGHOUSE_CHANNELS_PER_BOARD == 37
    assert NUM_BUGHOUSE_CHANNELS == 74
    assert NUM_CLOCK_BUGHOUSE_CHANNELS == 82
    assert channels_per_board() == 37
    assert channels_per_board(True) == 41
    assert ModelArgs().input_shape == (74, 8, 8)
    assert get_model_args().input_shape == (74, 8, 8)
    assert get_model_args(clock_planes=True).input_shape == (82, 8, 8)


@pytest.mark.parametrize("total", [73, 76, 80, 84])
def test_unsupported_channel_counts_are_rejected(total):
    with pytest.raises(ValueError):
        board_channels_from_total(total)


# ------------------------------------------------------------- board2planes --

def test_board2planes_default_is_74_channels():
    planes = board2planes(_some_position(), chess.WHITE)
    assert planes.shape == (74, 8, 8)


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("flip", [False, True])
def test_clock_layout_keeps_the_stock_planes(team, flip):
    board = _some_position()
    stock = board2planes(board, team, flip=flip)
    clock = board2planes(board, team, flip=flip, clocks=(61000, 2000, 300000, 450000))
    assert clock.shape == (82, 8, 8)
    np.testing.assert_array_equal(clock[:37], stock[:37])
    np.testing.assert_array_equal(clock[41:78], stock[37:])


def test_clock_planes_are_zero_without_clocks():
    planes = board2planes(_some_position(), chess.WHITE, clock_planes=True)
    assert planes.shape == (82, 8, 8)
    assert not planes[37:41].any() and not planes[78:82].any()


def _expected(ms):
    t = ms / 1000.0
    return [min(t / 300.0, 1.0), math.exp(-t / 10.0)]


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
def test_clock_planes_follow_the_spec(team):
    clocks = {"aw": 61000, "ab": 2500, "bw": 450000, "bb": 9000}
    planes = board2planes(_some_position(), team,
                          clocks=(clocks["aw"], clocks["ab"], clocks["bw"], clocks["bb"]))
    # Board A: our seat has team's colour; board B: our seat is the partner (other colour).
    a_ours, a_theirs = (clocks["aw"], clocks["ab"]) if team == chess.WHITE else (clocks["ab"], clocks["aw"])
    b_ours, b_theirs = (clocks["bb"], clocks["bw"]) if team == chess.WHITE else (clocks["bw"], clocks["bb"])
    for base, ours, theirs in ((0, a_ours, a_theirs), (41, b_ours, b_theirs)):
        expected = _expected(ours) + _expected(theirs)
        for offset, value in enumerate(expected):
            plane = planes[base + 37 + offset]
            assert np.all(plane == plane.flat[0]), "clock planes are constant"
            assert plane.flat[0] == pytest.approx(value, abs=1e-6)


def test_flip_swaps_clock_planes_with_their_board():
    board = _some_position()
    clocks = (61000, 2500, 450000, 9000)
    planes = board2planes(board, chess.WHITE, clocks=clocks)
    flipped = board2planes(board, chess.WHITE, flip=True, clocks=clocks)
    np.testing.assert_array_equal(flipped[:41], planes[41:])
    np.testing.assert_array_equal(flipped[41:], planes[:41])


def test_flagged_clock_counts_as_zero():
    planes = board2planes(_some_position(), chess.WHITE, clocks=(-500, 1000, 1000, 1000))
    assert planes[37].flat[0] == 0.0 and planes[38].flat[0] == 1.0


def test_clocks_without_clock_layout_are_an_error():
    with pytest.raises(ValueError):
        board2planes(_some_position(), chess.WHITE, clocks=(1, 2, 3, 4), clock_planes=False)


# ------------------------------------------------------------- architecture --

def _cross_board_model(channels, seed=0):
    torch.manual_seed(seed)
    args = get_model_args(architecture="crossboard-risev33", clock_planes=channels == 82)
    args.use_wdl = args.use_plys_to_end = True
    args.shared_policy_trunk = False
    model = create_model(args)
    # Non-trivial BatchNorm statistics so eval mode is a real test.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def test_cross_board_model_accepts_82_channels_and_rejects_others():
    model = _cross_board_model(82)
    assert model.board_channels == 41
    out = model(torch.rand(2, 82, 8, 8))
    assert out[0].shape == (2, 1) and out[1][0].shape == (2, 4672)
    args = get_model_args(architecture="crossboard-risev33")
    args.input_shape = (76, 8, 8)
    with pytest.raises(ValueError):
        create_model(args)


def test_dual_stream_model_accepts_82_channels():
    args = get_model_args(architecture="dualstream-memory-risev33", clock_planes=True)
    model = create_model(args).eval()
    assert model.shared_stem.body[0].weight.shape[1] == 41
    out = model(torch.rand(1, 82, 8, 8))
    assert out[0].shape == (1, 1)


def test_checkpoint_shape_detection_reports_input_channels():
    for channels in (74, 82):
        state = _cross_board_model(channels).state_dict()
        assert model_args_from_state_dict(state).input_shape == (channels, 8, 8)
        assert checkpoint_input_channels(state) == channels


# ---------------------------------------------------------------- widening --

def test_widened_network_matches_original_whatever_the_clocks_hold():
    tool = _load_tool()
    old = _cross_board_model(74)
    old.use_wdl = True
    old_state = old.state_dict()
    new = tool.load_model(tool.widen_state_dict(old_state, 41))
    assert new.board_channels == 41

    rng = np.random.default_rng(0)
    planes74 = rng.uniform(0, 1, size=(6, 74, 8, 8)).astype(np.float32)
    for clocks in (None, rng.uniform(-10, 10, size=(6, 2, 4))):
        planes82 = tool.widen_planes(planes74, clocks, target=41)
        np.testing.assert_array_equal(tool.strip_clock_planes(planes82), planes74)
        diffs = tool.max_diffs(tool._torch_outputs(old, planes74), tool._torch_outputs(new, planes82))
        assert max(diffs.values()) < 1e-5, diffs

    # The check is not vacuous: a non-zero clock weight changes the outputs.
    with torch.no_grad():
        new.body_spatial[0].body[0].weight[:, 37:41] = 0.5
    planes82 = tool.widen_planes(planes74, np.ones((6, 2, 4)), target=41)
    diffs = tool.max_diffs(tool._torch_outputs(old, planes74), tool._torch_outputs(new, planes82))
    assert max(diffs.values()) > 1e-3


def test_widened_optimizer_state_loads_and_steps():
    tool = _load_tool()
    old = _cross_board_model(74)
    optimizer = torch.optim.SGD(old.parameters(), lr=0.01, momentum=0.9)
    old.train()
    loss = old(torch.rand(2, 74, 8, 8))[0].sum()
    loss.backward()
    optimizer.step()
    checkpoint = {"model_state_dict": old.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "training_iteration": 1, "evaluation_step": 0, "batch_steps": 1}
    widened = tool.widen_checkpoint(checkpoint, 41)
    new = tool.load_model(widened["model_state_dict"]).train()
    new_optimizer = torch.optim.SGD(new.parameters(), lr=0.01, momentum=0.9)
    new_optimizer.load_state_dict(widened["optimizer_state_dict"])
    stem = new.body_spatial[0].body[0].weight
    buffer = new_optimizer.state[stem]["momentum_buffer"]
    assert buffer.shape == stem.shape
    assert not buffer[:, 37:41].any() and not buffer[:, 78:82].any()
    new(torch.rand(2, 82, 8, 8))[0].sum().backward()
    new_optimizer.step()


# ------------------------------------------------------------- data format --

def write_chunk(path, game_ids, channels=82, version=5, seed=0):
    """An HVM chunk in selfplay.cc's binary format with random planes."""
    rng = np.random.default_rng(seed)
    magic = {4: b"HVM4", 5: b"HVM5"}[version]
    with path.open("wb") as output:
        output.write(HEADER.pack(magic, version, channels, 4672, len(game_ids)))
        for game_id in game_ids:
            output.write(SAMPLE_METADATA.pack(game_id, 100, 1, 20, 0, 0, 1, 2, 0.5))
            output.write(rng.integers(0, 256, size=channels * 64, dtype=np.uint8).tobytes())
            for _ in range(2):
                output.write(struct.pack("<H", 1))
                output.write(POLICY_ENTRY.pack(int(rng.integers(0, 4672)), np.float32(1.0)))
            if version >= 5:
                output.write(struct.pack("<H", 1))
                output.write(JOINT_POLICY_ENTRY.pack(4672, 12, np.float32(1.0)))


@pytest.mark.parametrize("channels", [74, 82])
def test_binary_shards_of_either_width_are_read(tmp_path, channels):
    path = tmp_path / "chunk_run_000000.hvm"
    write_chunk(path, [1, 2], channels=channels)
    samples = read_binary_shard(path)
    assert [len(s["x"]) for s in samples] == [channels * 64] * 2
    other = 74 if channels == 82 else 82
    with pytest.raises(ValueError, match=f"Expected {other} channels"):
        read_binary_shard(path, input_channels=other)


def test_unsupported_shard_width_is_rejected(tmp_path):
    path = tmp_path / "chunk_run_000000.hvm"
    write_chunk(path, [1], channels=78)
    with pytest.raises(ValueError, match="channels"):
        read_binary_shard(path)


def _split_games():
    val = next(g for g in range(1000) if is_validation_game("run", g, 0.5, 7))
    train = next(g for g in range(1000) if not is_validation_game("run", g, 0.5, 7))
    return train, val


def test_conversion_does_not_mix_widths(tmp_path):
    source = tmp_path / "training_data"
    source.mkdir()
    train, val = _split_games()
    write_chunk(source / "chunk_run_000000.hvm", [train, val], channels=82)
    write_chunk(source / "chunk_run_000001.hvm", [train, val], channels=74)
    with pytest.raises(ValueError, match="Expected 82 channels"):
        convert_to_split_parquet(source, tmp_path / "rl", validation_fraction=0.5, split_seed=7)


def test_82_channel_shards_reach_the_rl_loader(tmp_path):
    source = tmp_path / "training_data"
    source.mkdir()
    train, val = _split_games()
    write_chunk(source / "chunk_run_000000.hvm", [train, val, train], channels=82)
    train_dir, val_dir, n_train, n_val = convert_to_split_parquet(
        source, tmp_path / "rl", validation_fraction=0.5, split_seed=7)
    assert (n_train, n_val) == (2, 1)
    shard = sorted(train_dir.glob("*.parquet"))[0]
    assert rl_parquet_input_channels(shard) == 82
    x, *_ = load_rl_parquet_shard(shard, include_joint_policy=True)
    assert x.shape == (2, 82, 8, 8)
    raw = np.frombuffer(pl.read_parquet(shard)["x"][0], dtype=np.uint8).reshape(82, 8, 8)
    np.testing.assert_allclose(x[0].numpy(), raw / 255.0, rtol=0, atol=1e-7)
    flipped, pol_a, pol_b = flip_bughouse_sample(x[0], torch.zeros(4672), torch.ones(4672))
    assert torch.equal(flipped[:41], x[0][41:]) and torch.equal(flipped[41:], x[0][:41])


def test_stock_flip_is_unchanged():
    x = torch.arange(74 * 64, dtype=torch.float32).view(74, 8, 8)
    flipped, _, _ = flip_bughouse_sample(x, torch.zeros(1), torch.zeros(1))
    assert torch.equal(flipped[:37], x[37:]) and torch.equal(flipped[37:], x[:37])
