"""Custom engine spec v3 (network side, audit WP-F inputs 1): the diagonal clock
plane, channel 43 of each board block (44 per board, 88 total),
tanh((t_ourA - t_oppB) / s), identical in both blocks."""

import importlib.util
import math
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from hivemind.architectures.model_config import create_model, get_model_args
from hivemind.constants import (
    DIAGONAL_CHANNEL_OFFSET,
    DIAGONAL_SCALE_S,
    NUM_DIAGONAL_BUGHOUSE_CHANNELS,
    SUPPORTED_CHANNELS_PER_BOARD,
    board_channels_from_total,
    channels_per_board,
)
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes, diagonal_plane_value

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_clock_net_v3", TOOLS / "make_clock_net.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_clock_net_v3"] = module
    spec.loader.exec_module(module)
    return module


def _position():
    board = BughouseBoard()
    for board_num, uci in [(0, "e2e4"), (1, "d2d4"), (0, "d7d5"), (0, "e4d5"),
                           (1, "g8f6"), (0, "d8d5")]:
        board.push_uci(board_num, uci)
    return board


def _model(channels, aux, seed=0):
    torch.manual_seed(seed)
    args = get_model_args(architecture="crossboard-risev33", input_channels=channels,
                          aux_heads=aux)
    args.use_wdl = args.use_plys_to_end = True
    model = create_model(args)
    torch.manual_seed(seed + 1)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


# ------------------------------------------------------------------- planes --

def test_v3_channel_counts():
    assert DIAGONAL_SCALE_S == 15.0
    assert DIAGONAL_CHANNEL_OFFSET == 43
    assert channels_per_board(diagonal_plane=True) == 44
    assert channels_per_board(True, True) == 43  # v2 unchanged
    assert channels_per_board() == 37  # stock unchanged
    assert board_channels_from_total(88) == 44
    assert NUM_DIAGONAL_BUGHOUSE_CHANNELS == 88
    assert SUPPORTED_CHANNELS_PER_BOARD == (37, 41, 43, 44)
    assert get_model_args(input_channels=88).input_shape == (88, 8, 8)


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("flip", [False, True])
def test_diagonal_layout_keeps_every_v2_plane(team, flip):
    board = _position()
    clocks = (61000, 2000, 300000, 450000)
    elapsed = (1500, 25000)
    v2 = board2planes(board, team, flip=flip, clocks=clocks, elapsed=elapsed)
    v3 = board2planes(board, team, flip=flip, clocks=clocks, elapsed=elapsed, diagonal_plane=True)
    assert v3.shape == (88, 8, 8)
    np.testing.assert_array_equal(v3[:43], v2[:43])
    np.testing.assert_array_equal(v3[44:87], v2[43:86])


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("scale", [10.0, 15.0, 20.0])
def test_diagonal_follows_the_spec(team, scale):
    # A: white 40 s, black 100 s; B: white 25 s, black 70 s.
    clocks = (40000, 100000, 25000, 70000)
    ours, diagonal = (40.0, 25.0) if team == chess.WHITE else (100.0, 70.0)
    planes = board2planes(_position(), team, clocks=clocks, elapsed=(0, 0),
                          diagonal_plane=True, diagonal_scale_s=scale)
    expected = math.tanh((ours - diagonal) / scale)
    for channel in (43, 87):
        assert np.all(planes[channel] == planes[channel].flat[0])
        assert planes[channel].flat[0] == pytest.approx(expected, abs=1e-6)
    assert diagonal_plane_value(team, clocks, scale) == pytest.approx(expected)


def test_diagonal_sign_scale_and_edges():
    pos = _position()
    # A 15 s lead at the default scale is tanh(1); a 15 s deficit is -tanh(1).
    lead = board2planes(pos, chess.WHITE, clocks=(30000, 1, 15000, 1), diagonal_plane=True)
    behind = board2planes(pos, chess.WHITE, clocks=(15000, 1, 30000, 1), diagonal_plane=True)
    assert lead[43].flat[0] == pytest.approx(math.tanh(1.0), abs=1e-6)
    assert behind[43].flat[0] == pytest.approx(-math.tanh(1.0), abs=1e-6)
    # Negative clocks count as 0, as in planes.cc.
    flagged = board2planes(pos, chess.WHITE, clocks=(-3000, 5000, 2000, 5000), diagonal_plane=True)
    assert flagged[43].flat[0] == pytest.approx(math.tanh(-2.0 / 15.0), abs=1e-6)
    # No clocks: 0, like every other clock plane.
    none = board2planes(pos, chess.WHITE, diagonal_plane=True)
    assert none.shape == (88, 8, 8) and not none[37:44].any() and not none[81:88].any()
    # Flip swaps the blocks; the plane is the same in both.
    clocks = (61000, 2000, 300000, 450000)
    flipped = board2planes(pos, chess.WHITE, flip=True, clocks=clocks, diagonal_plane=True)
    straight = board2planes(pos, chess.WHITE, clocks=clocks, diagonal_plane=True)
    np.testing.assert_array_equal(flipped[:44], straight[44:])
    assert flipped[43].flat[0] == straight[43].flat[0]
    with pytest.raises(ValueError):
        board2planes(pos, chess.WHITE, clocks=clocks, diagonal_plane=True, elapsed_planes=False)
    with pytest.raises(ValueError):
        diagonal_plane_value(chess.WHITE, (1, 2, 3))


def test_default_board2planes_is_unchanged():
    assert board2planes(_position(), chess.WHITE).shape == (74, 8, 8)
    assert board2planes(_position(), chess.WHITE, clocks=(1, 2, 3, 4)).shape == (82, 8, 8)
    assert board2planes(_position(), chess.WHITE, clocks=(1, 2, 3, 4),
                        elapsed=(1, 2)).shape == (86, 8, 8)


# ---------------------------------------------------------------- widening --

@pytest.mark.parametrize("source", [74, 86])
def test_widening_to_44_planes_matches_the_source_network(source):
    tool = _load_tool()
    old = _model(source, aux=source == 86)
    widened = tool.widen_checkpoint({"model_state_dict": old.state_dict()}, 44)
    assert widened["diagonal_plane"] and widened["elapsed_planes"] and widened["aux_heads"]
    new = tool.load_model(widened["model_state_dict"])
    assert new.board_channels == 44 and new.aux_heads
    rng = np.random.default_rng(1)
    planes = rng.uniform(0, 1, size=(4, source, 8, 8)).astype(np.float32)
    per = source // 2
    for fill in (0.0, None):
        wide = np.zeros((4, 88, 8, 8), dtype=np.float32)
        wide[:, :per] = planes[:, :per]
        wide[:, 44:44 + per] = planes[:, per:]
        new_cols = [c for c in range(88) if not (c < per or 44 <= c < 44 + per)]
        wide[:, new_cols] = fill if fill is not None else rng.uniform(-5, 5, size=wide[:, new_cols].shape)
        ref = tool._torch_outputs(old, planes)
        diffs = tool.max_diffs(ref, tool._torch_outputs(new, wide))
        assert max(diffs.values()) < 1e-5, diffs
        if source == 74:
            think, arrivals = tool._torch_aux_outputs(new, wide)
            assert not think.any() and not arrivals.any()
    # Not vacuous: a non-zero diagonal weight changes the outputs.
    with torch.no_grad():
        new.body_spatial[0].body[0].weight[:, [43, 87]] = 0.5
    wide[:, [43, 87]] = 1.0
    diffs = tool.max_diffs(tool._torch_outputs(old, planes), tool._torch_outputs(new, wide))
    assert max(diffs.values()) > 1e-3


def test_widened_v3_optimizer_state_loads_and_steps():
    tool = _load_tool()
    old = _model(74, aux=False).train()
    optimizer = torch.optim.SGD(old.parameters(), lr=0.01, momentum=0.9)
    old(torch.rand(2, 74, 8, 8))[0].sum().backward()
    optimizer.step()
    widened = tool.widen_checkpoint({"model_state_dict": old.state_dict(),
                                     "optimizer_state_dict": optimizer.state_dict()}, 44)
    new = tool.load_model(widened["model_state_dict"]).train()
    new_optimizer = torch.optim.SGD(new.parameters(), lr=0.01, momentum=0.9)
    new_optimizer.load_state_dict(widened["optimizer_state_dict"])
    stem = new.body_spatial[0].body[0].weight
    buffer = new_optimizer.state[stem]["momentum_buffer"]
    assert buffer.shape == stem.shape and not buffer[:, 37:44].any() and not buffer[:, 81:88].any()
    out = new(torch.rand(2, 88, 8, 8))
    (out[0].sum() + out[5].sum() + out[6].sum()).backward()
    new_optimizer.step()


def test_v3_planes_for_uses_the_diagonal_layout():
    tool = _load_tool()
    positions = tool.random_positions(3, seed=2, max_moves=10)
    p74, wide = tool.planes_for(positions, 44, diagonal_scale_s=10.0)
    assert p74.shape == (3, 74, 8, 8) and wide.shape == (3, 88, 8, 8)
    np.testing.assert_array_equal(tool.strip_new_planes(wide), p74)
    for (_, team, clocks, _), planes in zip(positions, wide):
        assert planes[43].flat[0] == pytest.approx(diagonal_plane_value(team, clocks, 10.0), abs=1e-6)
