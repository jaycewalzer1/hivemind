"""Custom engine spec v2 (network side): elapsed-time planes (43 per board),
think_gain and arrivals heads, the HVM6 shard trailing block."""

import importlib.util
import math
import struct
import sys
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch

from hivemind.architectures.model_config import (
    create_model,
    get_model_args,
    model_args_from_state_dict,
)
from hivemind.constants import (
    NUM_ELAPSED_BUGHOUSE_CHANNELS,
    board_channels_from_total,
    channels_per_board,
)
from hivemind.data.convert_selfplay_data import (
    AUX_LABELS,
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
from hivemind.training.data_loaders import (
    StreamingRLDataset,
    flip_aux_labels,
    load_rl_parquet_shard,
)
from hivemind.training.train_loop import checkpoint_has_aux_heads, rl_parquet_input_channels
from hivemind.training.trainer_agent import (
    aux_head_losses,
    export_to_onnx,
    split_model_extras,
    split_rl_batch_extras,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_clock_net_v2", TOOLS / "make_clock_net.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_clock_net_v2"] = module
    spec.loader.exec_module(module)
    return module


def _position():
    board = BughouseBoard()
    for board_num, uci in [(0, "e2e4"), (1, "d2d4"), (0, "d7d5"), (0, "e4d5"),
                           (1, "g8f6"), (0, "d8d5")]:
        board.push_uci(board_num, uci)
    return board


# ------------------------------------------------------------------- planes --

def test_v2_channel_counts():
    assert channels_per_board(True, True) == 43
    assert channels_per_board(False, True) == 43
    assert board_channels_from_total(86) == 43
    assert NUM_ELAPSED_BUGHOUSE_CHANNELS == 86
    assert get_model_args(elapsed_planes=True).input_shape == (86, 8, 8)
    assert get_model_args(input_channels=86).input_shape == (86, 8, 8)
    assert get_model_args().input_shape == (74, 8, 8)


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("flip", [False, True])
def test_elapsed_layout_keeps_stock_and_clock_planes(team, flip):
    board = _position()
    clocks = (61000, 2000, 300000, 450000)
    stock = board2planes(board, team, flip=flip)
    v1 = board2planes(board, team, flip=flip, clocks=clocks)
    v2 = board2planes(board, team, flip=flip, clocks=clocks, elapsed=(1500, 25000))
    assert v2.shape == (86, 8, 8)
    np.testing.assert_array_equal(v2[:41], v1[:41])
    np.testing.assert_array_equal(v2[43:84], v1[41:82])
    np.testing.assert_array_equal(v2[:37], stock[:37])
    np.testing.assert_array_equal(v2[43:80], stock[37:])


def _expected(ms):
    e = max(ms, 0) / 1000.0
    return min(e / 10.0, 1.0), 1.0 - math.exp(-e / 2.0)


@pytest.mark.parametrize("team", [chess.WHITE, chess.BLACK])
def test_elapsed_planes_follow_the_spec(team):
    elapsed = (3500, 42000)
    planes = board2planes(_position(), team, clocks=(1, 2, 3, 4), elapsed=elapsed)
    for base, ms in ((0, elapsed[0]), (43, elapsed[1])):
        for offset, value in enumerate(_expected(ms)):
            plane = planes[base + 41 + offset]
            assert np.all(plane == plane.flat[0])
            assert plane.flat[0] == pytest.approx(value, abs=1e-6)
    # Board planes, not seat planes: the team does not change them.
    other = board2planes(_position(), not team, clocks=(1, 2, 3, 4), elapsed=elapsed)
    np.testing.assert_array_equal(planes[[41, 42, 84, 85]], other[[41, 42, 84, 85]])


def test_elapsed_edge_cases():
    no_timing = board2planes(_position(), chess.WHITE, elapsed_planes=True)
    assert no_timing.shape == (86, 8, 8)
    assert not no_timing[37:43].any() and not no_timing[80:86].any()
    clocks = (60000, 60000, 60000, 60000)
    negative = board2planes(_position(), chess.WHITE, clocks=clocks, elapsed=(-300, 0))
    assert not negative[41:43].any()
    long_wait = board2planes(_position(), chess.WHITE, clocks=clocks, elapsed=(10 ** 7, 0))
    assert long_wait[41].flat[0] == 1.0 and long_wait[42].flat[0] == pytest.approx(1.0)
    flipped = board2planes(_position(), chess.WHITE, flip=True, clocks=clocks, elapsed=(1000, 9000))
    straight = board2planes(_position(), chess.WHITE, clocks=clocks, elapsed=(1000, 9000))
    np.testing.assert_array_equal(flipped[:43], straight[43:])
    with pytest.raises(ValueError):
        board2planes(_position(), chess.WHITE, elapsed=(1, 2), elapsed_planes=False)
    with pytest.raises(ValueError):
        board2planes(_position(), chess.WHITE, elapsed=(1, 2), clock_planes=False)
    with pytest.raises(ValueError):
        board2planes(_position(), chess.WHITE, elapsed=(1, 2, 3))


def test_elapsed_without_clocks_matches_the_engine():
    # planes.cc fills 41..42 only when the board has clocks and elapsed times, and UCI ignores
    # 'elapsed' without 'clocks': Python gives the same 86 planes with elapsed planes 0.
    alone = board2planes(_position(), chess.WHITE, elapsed=(5000, 30000))
    none = board2planes(_position(), chess.WHITE, elapsed_planes=True)
    assert alone.shape == (86, 8, 8)
    np.testing.assert_array_equal(alone, none)
    with_clocks = board2planes(_position(), chess.WHITE, clocks=(1, 2, 3, 4), elapsed=(5000, 30000))
    assert with_clocks[41].any() and with_clocks[84].any()


# ------------------------------------------------------------------- heads --

def _model(channels, aux, seed=0, joint_rank=0):
    torch.manual_seed(seed)
    args = get_model_args(architecture="crossboard-risev33", input_channels=channels,
                          aux_heads=aux)
    args.use_wdl = args.use_plys_to_end = True
    args.joint_policy_rank = joint_rank
    model = create_model(args)
    torch.manual_seed(seed + 1)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def test_aux_heads_append_two_zero_outputs_and_change_nothing_else():
    plain = _model(86, aux=False)
    with_aux = _model(86, aux=True)
    x = torch.rand(3, 86, 8, 8)
    with torch.no_grad():
        a, b = plain(x), with_aux(x)
    assert len(a) == 5 and len(b) == 7
    assert b[5].shape == (3, 2) and b[6].shape == (3, 20)
    assert not b[5].any() and not b[6].any()
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1][0], b[1][0], rtol=0, atol=0)
    # Heads are registered last, so old parameters keep their positions.
    names = [n for n, _ in with_aux.named_parameters()]
    assert names[:len(list(plain.parameters()))] == [n for n, _ in plain.named_parameters()]
    args = model_args_from_state_dict(with_aux.state_dict())
    assert args.aux_heads and args.aux_head_hidden == 64 and args.input_shape == (86, 8, 8)
    assert checkpoint_has_aux_heads(with_aux.state_dict())
    assert not checkpoint_has_aux_heads(plain.state_dict())


def test_output_order_with_joint_and_aux_heads():
    model = _model(86, aux=True, joint_rank=2)
    out = model(torch.rand(2, 86, 8, 8))
    joint, aux = split_model_extras(model, out)
    assert joint[0].shape == (2, 2 * 4673)
    assert aux[0].shape == (2, 2) and aux[1].shape == (2, 20)
    assert split_model_extras(_model(74, aux=False), out[:5]) == (None, None)


@pytest.mark.parametrize("architecture", ["risev33", "dualstream-memory-risev33"])
def test_other_architectures_take_aux_heads(architecture):
    args = get_model_args(architecture=architecture, input_channels=86, aux_heads=True)
    args.use_wdl = args.use_plys_to_end = True
    model = create_model(args).eval()
    out = model(torch.rand(1, 86, 8, 8))
    assert out[-2].shape == (1, 2) and out[-1].shape == (1, 20)


def test_onnx_export_has_named_aux_outputs(tmp_path):
    model = _model(86, aux=True)
    with torch.no_grad():
        model.think_gain_head.body[2].bias.copy_(torch.tensor([0.25, -0.5]))
    path = export_to_onnx(model, 1, torch.zeros(1, 86, 8, 8), tmp_path, "v2",
                          has_auxiliary_output=True, dynamic_batch_size=True,
                          fp16=False, onnx_name="v2.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    assert names == ["value", "pi_a", "pi_b", "wdl_out", "moves_left", "think_gain", "arrivals"]
    x = np.random.default_rng(0).uniform(0, 1, size=(3, 86, 8, 8)).astype(np.float32)
    think, arrivals, value = session.run(["think_gain", "arrivals", "value"], {"data": x})
    assert think.shape == (3, 2) and arrivals.shape == (3, 20)
    np.testing.assert_allclose(think, [[0.25, -0.5]] * 3, atol=1e-6)
    with torch.no_grad():
        np.testing.assert_allclose(value, model(torch.from_numpy(x))[0].numpy(), atol=1e-4)


# ---------------------------------------------------------------- widening --

@pytest.mark.parametrize("source", [74, 82])
def test_widening_to_43_planes_matches_the_source_network(source):
    tool = _load_tool()
    old = _model(source, aux=False)
    checkpoint = {"model_state_dict": old.state_dict()}
    widened = tool.widen_checkpoint(checkpoint, 43)
    new = tool.load_model(widened["model_state_dict"])
    assert new.board_channels == 43 and new.aux_heads
    rng = np.random.default_rng(1)
    planes = rng.uniform(0, 1, size=(4, source, 8, 8)).astype(np.float32)
    per = source // 2
    for fill in (0.0, None):
        wide = np.zeros((4, 86, 8, 8), dtype=np.float32)
        wide[:, :per] = planes[:, :per]
        wide[:, 43:43 + per] = planes[:, per:]
        new_cols = [c for c in range(86) if not (c < per or 43 <= c < 43 + per)]
        wide[:, new_cols] = fill if fill is not None else rng.uniform(-5, 5, size=wide[:, new_cols].shape)
        diffs = tool.max_diffs(tool._torch_outputs(old, planes), tool._torch_outputs(new, wide))
        assert max(diffs.values()) < 1e-5, diffs
        think, arrivals = tool._torch_aux_outputs(new, wide)
        assert not think.any() and not arrivals.any()
    # Not vacuous: a non-zero elapsed-plane weight changes the outputs.
    with torch.no_grad():
        new.body_spatial[0].body[0].weight[:, 41:43] = 0.5
    wide[:, 41:43] = 1.0
    diffs = tool.max_diffs(tool._torch_outputs(old, planes), tool._torch_outputs(new, wide))
    assert max(diffs.values()) > 1e-3


def test_widened_v2_optimizer_state_covers_the_new_heads():
    tool = _load_tool()
    old = _model(74, aux=False).train()
    optimizer = torch.optim.SGD(old.parameters(), lr=0.01, momentum=0.9)
    old(torch.rand(2, 74, 8, 8))[0].sum().backward()
    optimizer.step()
    checkpoint = {"model_state_dict": old.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict()}
    widened = tool.widen_checkpoint(checkpoint, 43)
    new = tool.load_model(widened["model_state_dict"]).train()
    new_optimizer = torch.optim.SGD(new.parameters(), lr=0.01, momentum=0.9)
    new_optimizer.load_state_dict(widened["optimizer_state_dict"])
    stem = new.body_spatial[0].body[0].weight
    buffer = new_optimizer.state[stem]["momentum_buffer"]
    assert buffer.shape == stem.shape and not buffer[:, 37:43].any()
    out = new(torch.rand(2, 86, 8, 8))
    (out[0].sum() + out[5].sum() + out[6].sum()).backward()
    new_optimizer.step()
    assert new.think_gain_head.body[2].weight.abs().sum() > 0


def test_v1_widening_is_unchanged():
    tool = _load_tool()
    old = _model(74, aux=False)
    widened = tool.widen_checkpoint({"model_state_dict": old.state_dict()}, 41)
    new = tool.load_model(widened["model_state_dict"])
    assert new.board_channels == 41 and not new.aux_heads
    assert widened["aux_heads"] is False


# ------------------------------------------------------------- shard format --

def write_chunk(path, samples, channels=86, version=6, seed=0):
    """HVM5/HVM6 chunk; samples = [(game_id, think(2) or None, masks(2), arrivals(20))]."""
    rng = np.random.default_rng(seed)
    magic = {5: b"HVM5", 6: b"HVM6"}[version]
    with path.open("wb") as output:
        output.write(HEADER.pack(magic, version, channels, 4672, len(samples)))
        for game_id, think, masks, arrivals in samples:
            output.write(SAMPLE_METADATA.pack(game_id, 100, 1, 20, 0, 0, 1, 2, 0.5))
            output.write(rng.integers(0, 256, size=channels * 64, dtype=np.uint8).tobytes())
            for _ in range(2):
                output.write(struct.pack("<H", 1))
                output.write(POLICY_ENTRY.pack(int(rng.integers(0, 4672)), np.float32(1.0)))
            output.write(struct.pack("<H", 1))
            output.write(JOINT_POLICY_ENTRY.pack(4672, 12, np.float32(1.0)))
            if version >= 6:
                output.write(AUX_LABELS.pack(*think, *masks, *arrivals))


ARRIVALS = [float(i % 3) for i in range(20)]


def test_trailing_block_is_90_bytes():
    assert AUX_LABELS.size == 90
    assert AUX_LABELS.format == "<2f2B20f"


def test_hvm6_round_trip(tmp_path):
    path = tmp_path / "chunk_run_000000.hvm"
    nan = [float("nan")] * 20
    write_chunk(path, [
        (1, (1.0, 0.0), (1, 0), ARRIVALS),
        (1, (0.0, 0.0), (0, 0), nan),
        (2, (0.0, 1.0), (1, 1), [0.0] * 20),
    ])
    samples = read_binary_shard(path)
    assert len(samples) == 3
    first, second, third = samples
    assert (first["think_gain_a"], first["think_gain_mask_a"]) == (1.0, 1)
    assert (first["think_gain_b"], first["think_gain_mask_b"]) == (0.0, 0)
    np.testing.assert_array_equal(np.frombuffer(first["arrivals"], dtype=np.float32), ARRIVALS)
    assert first["arrivals_mask"] == 1
    assert second["arrivals_mask"] == 0 and not np.frombuffer(second["arrivals"], np.float32).any()
    assert (third["think_gain_b"], third["think_gain_mask_b"], third["arrivals_mask"]) == (1.0, 1, 1)
    assert len(first["x"]) == 86 * 64 and first["joint_policy_count"] == 1


def test_old_shards_read_with_masked_labels(tmp_path):
    path = tmp_path / "chunk_run_000000.hvm"
    write_chunk(path, [(1, None, None, None)], channels=82, version=5)
    (sample,) = read_binary_shard(path)
    assert sample["think_gain_mask_a"] == sample["think_gain_mask_b"] == 0
    assert sample["arrivals_mask"] == 0 and len(sample["arrivals"]) == 80


def test_bad_trailing_blocks_are_rejected(tmp_path):
    path = tmp_path / "chunk_run_000000.hvm"
    write_chunk(path, [(1, (0.0, 0.0), (2, 0), ARRIVALS)])
    with pytest.raises(ValueError, match="mask byte"):
        read_binary_shard(path)
    write_chunk(path, [(1, (3.0, 0.0), (1, 0), ARRIVALS)])
    with pytest.raises(ValueError, match="outside"):
        read_binary_shard(path)
    # Version marker and block must agree: an HVM6 header without blocks is truncated.
    write_chunk(path, [(1, None, None, None)], version=5)
    data = bytearray(path.read_bytes())
    data[0:8] = HEADER.pack(b"HVM6", 6, 86, 4672, 1)[0:8]
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="Truncated"):
        read_binary_shard(path)
    data[0:8] = HEADER.pack(b"HVM5", 6, 86, 4672, 1)[0:8]
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="Unsupported"):
        read_binary_shard(path)


def _split_games():
    val = next(g for g in range(1000) if is_validation_game("run", g, 0.5, 7))
    train = next(g for g in range(1000) if not is_validation_game("run", g, 0.5, 7))
    return train, val


def test_conversion_writes_aux_columns_from_old_and_new_chunks(tmp_path):
    source = tmp_path / "training_data"
    source.mkdir()
    train, val = _split_games()
    write_chunk(source / "chunk_run_000000.hvm", [
        (train, (1.0, 0.0), (1, 0), ARRIVALS),
        (val, (0.0, 1.0), (1, 1), ARRIVALS),
    ])
    write_chunk(source / "chunk_run_000001.hvm", [(train, None, None, None)], version=5)
    train_dir, val_dir, n_train, n_val = convert_to_split_parquet(
        source, tmp_path / "rl", validation_fraction=0.5, split_seed=7)
    assert (n_train, n_val) == (2, 1)
    shard = sorted(train_dir.glob("*.parquet"))[0]
    assert rl_parquet_input_channels(shard) == 86
    df = pl.read_parquet(shard)
    assert df["think_gain_mask_a"].to_list() == [1, 0]
    assert df["arrivals_mask"].to_list() == [1, 0]

    loaded = load_rl_parquet_shard(shard, include_joint_policy=True, include_aux_labels=True)
    assert len(loaded) == 14
    think, think_mask, arrivals, arrivals_mask = loaded[10:]
    assert loaded[0].shape == (2, 86, 8, 8)
    assert think.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert think_mask.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert arrivals[0].tolist() == ARRIVALS and arrivals_mask.tolist() == [1.0, 0.0]
    assert len(load_rl_parquet_shard(shard)) == 6
    assert len(load_rl_parquet_shard(shard, include_aux_labels=True)) == 10


def test_parquet_without_aux_columns_gives_masked_labels(tmp_path):
    shard = tmp_path / "old.parquet"
    pl.DataFrame({
        "x": [bytes(74 * 64)],
        "policy_a": [np.eye(1, 4672, dtype=np.float32)[0].tobytes()],
        "policy_b": [np.eye(1, 4672, dtype=np.float32)[0].tobytes()],
        "y_value": [1.0], "y_wdl": [2], "y_moves_left": [10],
    }).write_parquet(shard)
    *_, think, think_mask, arrivals, arrivals_mask = load_rl_parquet_shard(
        shard, include_aux_labels=True)
    assert not think_mask.any() and not arrivals_mask.any()
    assert arrivals.shape == (1, 20)


def test_flip_swaps_board_labels():
    think = torch.tensor([0.0, 1.0])
    mask = torch.tensor([1.0, 0.0])
    arrivals = torch.arange(20, dtype=torch.float32)
    f_think, f_mask, f_arrivals, f_amask = flip_aux_labels(think, mask, arrivals, torch.tensor(1.0))
    assert f_think.tolist() == [1.0, 0.0] and f_mask.tolist() == [0.0, 1.0]
    assert f_arrivals.tolist() == list(range(10, 20)) + list(range(10))
    assert float(f_amask) == 1.0
    # Twice is the identity.
    again = flip_aux_labels(f_think, f_mask, f_arrivals, f_amask)
    assert torch.equal(again[2], arrivals)


def test_streaming_dataset_yields_flipped_aux_labels(tmp_path):
    source = tmp_path / "training_data"
    source.mkdir()
    train, _ = _split_games()
    write_chunk(source / "chunk_run_000000.hvm", [(train, (1.0, 0.0), (1, 0), ARRIVALS)])
    out = tmp_path / "parquet"
    from hivemind.data.convert_selfplay_data import convert_to_parquet
    convert_to_parquet(str(source), str(out))
    files = sorted(str(p) for p in out.glob("*.parquet"))
    dataset = StreamingRLDataset(files, shuffle_files=False, augment_flip=True,
                                 shuffle_samples=False, include_joint_policy=True,
                                 include_aux_labels=True)
    original, flipped = list(dataset)
    assert len(original) == len(flipped) == 14
    assert original[10].tolist() == [1.0, 0.0] and flipped[10].tolist() == [0.0, 1.0]
    assert flipped[12].tolist() == ARRIVALS[10:] + ARRIVALS[:10]
    joint, aux = split_rl_batch_extras(original, aux_head_labels=True)
    assert len(joint) == 4 and len(aux) == 4
    assert split_rl_batch_extras(original[:10], aux_head_labels=False)[1] is None


# ------------------------------------------------------------------ losses --

def test_aux_losses_use_only_labelled_samples():
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]], requires_grad=True)
    arrivals = torch.zeros(2, 20, requires_grad=True)
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    arrival_label = torch.ones(2, 20)
    arrival_mask = torch.tensor([0.0, 1.0])
    think, think_n, arr, arr_n = aux_head_losses(
        logits, arrivals, label, mask, arrival_label, arrival_mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(2.0), torch.tensor(1.0))
    assert float(think) == pytest.approx(float(expected))
    assert float(think_n) == 1.0 and float(arr) == pytest.approx(1.0) and float(arr_n) == 1.0
    (think + arr).backward()
    assert logits.grad[0, 0] != 0 and not logits.grad[0, 1] and not logits.grad[1].any()
    assert not arrivals.grad[0].any() and arrivals.grad[1].abs().sum() > 0

    none = aux_head_losses(logits, arrivals, label, torch.zeros(2, 2),
                           arrival_label, torch.zeros(2))
    assert float(none[0]) == 0.0 and float(none[2]) == 0.0
