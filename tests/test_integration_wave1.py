"""Audit wave 1 integration: 88-channel (spec v3 diagonal plane) data through the converter and loader."""
import chess
import torch

from hivemind.constants import DIAGONAL_CHANNEL_OFFSET
from hivemind.data.convert_selfplay_data import SUPPORTED_INPUT_CHANNELS
from hivemind.domain.board2planes import diagonal_plane_value
from hivemind.training.data_loaders import (
    CLOCKS_MASK_BIT, ELAPSED_MASK_BIT, rebuild_clock_planes,
)


def test_converter_accepts_the_88_channel_layout():
    assert 88 in SUPPORTED_INPUT_CHANNELS


def test_loader_rebuilds_the_signed_diagonal_plane_from_exact_clocks():
    clocks = torch.tensor([[40000.0, 70000.0, 55000.0, 90000.0],
                           [40000.0, 70000.0, 55000.0, 90000.0],
                           [-5.0, 10000.0, 3000.0, 0.0]])
    elapsed = torch.zeros(3, 2)
    team = torch.tensor([0, 1, 0])
    mask = torch.full((3,), CLOCKS_MASK_BIT | ELAPSED_MASK_BIT)
    x = torch.zeros(3, 88, 8, 8)
    rebuild_clock_planes(x, clocks, elapsed, mask, team)
    for row in range(3):
        side = chess.WHITE if int(team[row]) == 0 else chess.BLACK
        want = diagonal_plane_value(side, clocks[row].tolist())
        for base in (0, 44):
            got = x[row, base + DIAGONAL_CHANNEL_OFFSET]
            assert torch.allclose(got, torch.full((8, 8), want), atol=1e-6)
    assert float(x[0, DIAGONAL_CHANNEL_OFFSET, 0, 0]) < 0  # behind the diagonal: the negative half survives


def test_rows_without_clocks_and_86_channel_data_keep_their_planes():
    x = torch.full((1, 88, 8, 8), 0.5)
    rebuild_clock_planes(x, torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long),
                         torch.zeros(1, dtype=torch.long))
    assert bool((x == 0.5).all())
    y = torch.full((1, 86, 8, 8), 0.5)
    rebuild_clock_planes(y, torch.tensor([[1000.0, 2000.0, 3000.0, 4000.0]]), torch.zeros(1, 2),
                         torch.tensor([CLOCKS_MASK_BIT]), torch.zeros(1, dtype=torch.long))
    assert float(y[0, 42, 0, 0]) == 0.5 and float(y[0, 85, 0, 0]) == 0.5  # elapsed planes untouched
