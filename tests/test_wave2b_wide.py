"""Wave 2b (talisman fork): the wide planes, HDST v3, the widened network and
its arrivals head."""
import math
import struct

import chess
import numpy as np
import pytest
import torch

from hivemind import wave2, wide
from hivemind.architectures.twin_board import (
    get_twin_board_model,
    load_twin_board_checkpoint,
    widen_to_wide_planes,
)
from hivemind.constants import (
    NUM_CLOCKED_CHANNELS,
    NUM_CLOCKED_CHANNELS_PER_BOARD,
    NUM_WIDE_CHANNELS,
    NUM_WIDE_CHANNELS_PER_BOARD,
)
from hivemind.distill.data import (
    HEADER_BYTES,
    MAGIC,
    PLANES,
    TIMED_VERSION,
    VERSION,
    WIDE_PLANES,
    DistillDataset,
    chunk_problem,
    head_chunk,
    read_chunk,
)
from hivemind.distill.train import ARRIVALS_TYPE_WEIGHTS, arrivals_loss, layout, losses
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes
from hivemind.domain.wide_planes import (
    WideTiming,
    board2planes_wide,
    extend_planes,
    narrow_planes,
    parse_seat_arrivals,
    recent_counts,
    relative_order,
    relative_timing,
)

N = NUM_CLOCKED_CHANNELS_PER_BOARD   # 39: the extra planes start here in each block
W = NUM_WIDE_CHANNELS_PER_BOARD      # 65


def _board_with_pockets():
    """Captures on B: Bw takes a pawn and a knight (to its partner Ab), Bb
    takes a pawn (to Aw)."""
    board = BughouseBoard()
    for number, move in ((1, "e2e4"), (1, "d7d5"), (1, "e4d5"), (1, "g8f6"), (1, "d1f3"),
                         (1, "f6d5"), (1, "f3d5")):
        board.push_uci(number, move)
    return board


def _extra(planes, block):
    return planes[block * W + N:(block + 1) * W, 0, 0]


# ---- planes ---------------------------------------------------------------

def test_wide_planes_keep_the_clocked_block_and_add_26_per_board():
    board = _board_with_pockets()
    for team in (chess.WHITE, chess.BLACK):
        planes = board2planes_wide(board, team, lead_plane=0.25, speed_plane=-0.5)
        assert planes.shape == (NUM_WIDE_CHANNELS, 8, 8)
        base = board2planes(board, team)
        for block in range(2):
            np.testing.assert_array_equal(planes[block * W:block * W + 37], base[block * 37:(block + 1) * 37])
            assert (planes[block * W + 37] == 0.25).all() and (planes[block * W + 38] == -0.5).all()
            for offset in range(26):
                plane = planes[block * W + N + offset]
                assert (plane == plane[0, 0]).all(), "extra planes are constant"


def test_without_timing_only_the_presence_planes_are_set():
    board = _board_with_pockets()
    planes = board2planes_wide(board, chess.WHITE)
    a, b = _extra(planes, 0), _extra(planes, 1)
    assert a[:6].tolist() == [0] * 6 and a[16:].tolist() == [0] * 10
    assert a[6:11].tolist() == [1, 0, 0, 0, 0]       # ours (Aw): P
    assert a[11:16].tolist() == [1, 1, 0, 0, 0]      # theirs (Ab): P, N
    assert b.tolist() == [0] * 26                    # nothing in the B pockets
    black = _extra(board2planes_wide(board, chess.BLACK), 0)
    assert black[6:11].tolist() == [1, 1, 0, 0, 0] and black[11:16].tolist() == [1, 0, 0, 0, 0]


def test_timing_planes_follow_the_contract_from_each_teams_view():
    board = _board_with_pockets()
    timing = WideTiming({"Aw": 150_000, "Ab": 30_000, "Bw": 600_000, "Bb": 0}, (4_000, 25_000),
                        [("Aw", "N", 4_999), ("Aw", "N", 100), ("Aw", "N", 200), ("Bw", "q", 5_000),
                         ("Bb", "R", 0), ("Ab", "P", 2_500)])
    a = _extra(board2planes_wide(board, chess.WHITE, timing=timing), 0)
    b = _extra(board2planes_wide(board, chess.WHITE, timing=timing), 1)
    assert a[0] == pytest.approx(0.5) and a[1] == pytest.approx(math.exp(-15), rel=1e-3)
    assert a[2] == pytest.approx(0.1) and a[3] == pytest.approx(math.exp(-3), rel=1e-3)
    assert a[4] == pytest.approx(0.4) and a[5] == pytest.approx(1 - math.exp(-2), rel=1e-3)
    assert a[16:21].tolist() == [0, 1, 0, 0, 0]       # three knights -> min(3/2, 1)
    assert a[21:26].tolist() == [0.5, 0, 0, 0, 0]     # one pawn -> 0.5
    # Block B for the white team: ours = Bb (clock 0), theirs = Bw (600 s).
    assert b[0] == 0 and b[1] == 1 and b[2] == 1 and b[3] == pytest.approx(math.exp(-60), abs=1e-6)
    assert b[4] == 1 and b[5] == pytest.approx(1 - math.exp(-12.5), rel=1e-3)
    assert b[16:21].tolist() == [0, 0, 0, 0.5, 0]     # Bb's rook
    assert b[21:26].tolist() == [0] * 5               # Bw's queen landed 5 s ago: outside the window
    # The black team sees the same seats the other way round.
    a_black = _extra(board2planes_wide(board, chess.BLACK, timing=timing), 0)
    assert a_black[0] == pytest.approx(0.1) and a_black[2] == pytest.approx(0.5)
    assert a_black[16:21].tolist() == [0.5, 0, 0, 0, 0] and a_black[21:26].tolist() == [0, 1, 0, 0, 0]


def test_flip_swaps_the_wide_blocks():
    board = _board_with_pockets()
    timing = WideTiming({"Aw": 1, "Ab": 2, "Bw": 3, "Bb": 4}, (5, 6), [])
    planes = board2planes_wide(board, chess.WHITE, timing=timing)
    flipped = board2planes_wide(board, chess.WHITE, timing=timing, flip=True)
    np.testing.assert_array_equal(flipped[:W], planes[W:])
    np.testing.assert_array_equal(flipped[W:], planes[:W])


def test_seat_arrivals_option_parses_and_counts_the_last_5_seconds():
    assert parse_seat_arrivals("") == []
    landings = parse_seat_arrivals("Aw:n:120, Bb:Q:4999,Ab:P:5000")
    assert landings == [("Aw", "N", 120), ("Bb", "Q", 4999), ("Ab", "P", 5000)]
    counts = recent_counts(landings)
    assert counts["Aw"] == [0, 1, 0, 0, 0] and counts["Bb"] == [0, 0, 0, 0, 1] and counts["Ab"] == [0] * 5
    for bad in ("Cw:P:1", "Aw:K:1", "Aw:P", "Aw:P:-1"):
        with pytest.raises(ValueError):
            parse_seat_arrivals(bad)
    assert relative_order(chess.WHITE) == ("Aw", "Ab", "Bb", "Bw")
    assert relative_order(chess.BLACK) == ("Ab", "Aw", "Bw", "Bb")


def test_extend_planes_matches_the_board_encoder():
    """The chunk reader's vectorised widening equals board2planes_wide."""
    rng = np.random.default_rng(3)
    for index, (board, team) in enumerate(wide.random_positions(12, seed=3)):
        timing = wide.random_timing(rng) if index % 2 else None
        expected = board2planes_wide(board, team, 0.1, 0.2, timing)
        narrow = narrow_planes(expected[None, :, 0, 0])
        clocks, elapsed, recent = relative_timing(timing, team)
        got = extend_planes(narrow, None if clocks is None else clocks[None],
                            None if elapsed is None else elapsed[None], recent[None])[0]
        np.testing.assert_allclose(got, expected[:, 0, 0], atol=1e-6)


# ---- HDST v3 --------------------------------------------------------------

def _write_v3(path, rng, n=8, planes_count=PLANES, version=TIMED_VERSION, labelled=True):
    """A chunk with constant planes (values only) and the v3 timing block."""
    values = np.zeros((n, planes_count), np.float16)
    values[:, 12:22] = rng.integers(0, 3, (n, 10)) / 16.0
    half = planes_count // 2
    values[:, half + 12:half + 22] = rng.integers(0, 3, (n, 10)) / 16.0
    if planes_count == WIDE_PLANES:
        values[:, N:W] = 0.75
        values[:, W + N:] = 0.25
    clocks = rng.uniform(0, 300_000, (n, 4)).astype(np.float32)
    clocks[0] = -1                                          # no timing on the first record
    elapsed = rng.uniform(0, 20_000, (n, 2)).astype(np.float32)
    elapsed[0] = -1
    recent = rng.integers(0, 3, (n, 20)).astype(np.uint8)
    arrivals = rng.integers(0, 3, (n, 20)).astype(np.float32)
    if not labelled:
        arrivals[:, 0] = -1
    counts = np.ones((n, 2), np.uint16)
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", version, planes_count))
        stream.write(struct.pack("<QQ", n, 2 * n))
        stream.write(np.zeros((n, planes_count), np.int64).tobytes())
        stream.write(values.tobytes())
        stream.write(np.zeros(n, np.float32).tobytes())
        stream.write(np.tile(np.array([0, 0, 1], np.float32), (n, 1)).tobytes())
        stream.write(np.full(n, 0.5, np.float16).tobytes())
        stream.write(np.zeros(n, np.uint16).tobytes())
        stream.write(np.ones(n, np.int8).tobytes())
        stream.write(counts.tobytes())
        stream.write(np.arange(2 * n, dtype=np.uint16).tobytes())
        stream.write(np.ones(2 * n, np.float16).tobytes())
        if version == TIMED_VERSION:
            for array in (clocks, elapsed, recent, arrivals):
                stream.write(array.tobytes())
    return values, clocks, elapsed, recent, arrivals


def test_v3_chunk_is_widened_from_its_timing_block(tmp_path):
    rng = np.random.default_rng(0)
    values, clocks, elapsed, recent, arrivals = _write_v3(tmp_path / "c.dst", rng)
    chunk = read_chunk(tmp_path / "c.dst", layout="wide")
    assert chunk["plane_values"].shape == (8, WIDE_PLANES)
    expected = extend_planes(values.astype(np.float32), clocks, elapsed, recent)
    np.testing.assert_allclose(chunk["plane_values"].astype(np.float32), expected, atol=1e-3)
    # No timing on record 0: clock/elapsed/landing planes 0, presence set.
    extra = chunk["plane_values"][0, N:W].astype(np.float32)
    assert (extra[:6] == 0).all() and (extra[16:] == 0).all()
    np.testing.assert_array_equal(extra[6:16], values[0, 12:22] > 0)
    np.testing.assert_array_equal(chunk["arrivals"], arrivals)
    assert chunk["has_arrivals"].all()
    narrow = read_chunk(tmp_path / "c.dst")
    assert narrow["plane_values"].shape == (8, PLANES)
    np.testing.assert_array_equal(narrow["plane_values"], values)
    assert chunk_problem(tmp_path / "c.dst") is None
    assert head_chunk(chunk, 3)["arrivals"].shape == (3, 20)


def test_wide_chunks_keep_their_own_planes_and_narrow_cuts_them(tmp_path):
    rng = np.random.default_rng(1)
    values, *_ = _write_v3(tmp_path / "c.dst", rng, planes_count=WIDE_PLANES, labelled=False)
    chunk = read_chunk(tmp_path / "c.dst", layout="wide")
    np.testing.assert_array_equal(chunk["plane_values"], values)
    assert not chunk["has_arrivals"].any() and (chunk["arrivals"] == 0).all()
    narrow = read_chunk(tmp_path / "c.dst", layout="narrow")
    np.testing.assert_array_equal(narrow["plane_values"], narrow_planes(values))


def test_v2_chunks_read_wide_get_presence_only(tmp_path):
    rng = np.random.default_rng(2)
    values, *_ = _write_v3(tmp_path / "c.dst", rng, version=VERSION)
    chunk = read_chunk(tmp_path / "c.dst", layout="wide")
    np.testing.assert_allclose(chunk["plane_values"].astype(np.float32),
                               extend_planes(values.astype(np.float32)), atol=1e-3)
    assert not chunk["has_arrivals"].any()
    dataset = DistillDataset([chunk], torch.device("cpu"))
    assert dataset.planes(torch.arange(3)).shape == (3, WIDE_PLANES, 8, 8)


def test_v2_header_with_wide_planes_is_refused(tmp_path):
    rng = np.random.default_rng(2)
    _write_v3(tmp_path / "c.dst", rng, version=VERSION, planes_count=WIDE_PLANES)
    with pytest.raises(ValueError):
        read_chunk(tmp_path / "c.dst")


# ---- network --------------------------------------------------------------

def _perturbed(model, scale=0.02):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("arrivals_head"):
                parameter.add_(scale * torch.randn_like(parameter))
    return model


def _narrow_of(planes):
    return torch.cat((planes[:, :N], planes[:, W:W + N]), dim=1)


def test_widened_network_equals_the_narrow_one_whatever_the_new_planes_hold():
    torch.manual_seed(0)
    narrow = _perturbed(get_twin_board_model("twin-s", value_diff=True, attention_dim=0, se_every=0)).eval()
    checkpoint = {"model": narrow.state_dict(), "size": "twin-s",
                  "variant": {"value_diff": True, "attention_dim": 0, "se_every": 0}}
    widened, _, variant = load_twin_board_checkpoint(checkpoint, wide=True)
    widened.eval()
    assert variant["planes_per_board"] == W and variant["arrivals"]
    assert layout(widened) == "wide" and layout(narrow) == "narrow"
    planes = torch.rand(16, NUM_WIDE_CHANNELS, 8, 8) * 4 - 2
    with torch.no_grad():
        expected = narrow(_narrow_of(planes))
        got = widened(planes)
    for e, g in zip((expected[0], *expected[1], *expected[2:]), (got[0], *got[1], *got[2:5])):
        assert float((e - g).abs().max()) < 1e-5
    assert got[5].shape == (16, 20) and float(got[5].abs().max()) == 0


def test_a_wide_checkpoint_stays_wide_and_a_narrow_one_narrow():
    model = get_twin_board_model("twin-s", planes_per_board=W, arrivals=True)
    variant = {"planes_per_board": W, "arrivals": True}
    loaded, _, loaded_variant = load_twin_board_checkpoint(
        {"model": model.state_dict(), "size": "twin-s", "variant": variant})
    assert loaded.planes_per_board == W and loaded.arrivals_head is not None
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    narrow, _, _ = load_twin_board_checkpoint({"model": get_twin_board_model("twin-s").state_dict(),
                                               "size": "twin-s", "variant": {}})
    assert narrow.planes_per_board == N and narrow.arrivals_head is None
    widened = widen_to_wide_planes(get_twin_board_model("twin-s").state_dict())
    assert widened["stem.body.0.weight"].shape[1] == W
    assert widened["stem.body.0.weight"][:, N:].abs().sum() == 0


def test_arrivals_head_is_swap_equivariant():
    torch.manual_seed(1)
    model = get_twin_board_model("twin-s", planes_per_board=W, arrivals=True).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    x = torch.rand(4, NUM_WIDE_CHANNELS, 8, 8)
    swapped = torch.cat((x[:, W:], x[:, :W]), dim=1)
    with torch.no_grad():
        arrivals, arrivals_s = model(x)[5], model(swapped)[5]
    torch.testing.assert_close(arrivals_s[:, :10], arrivals[:, 10:], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(arrivals_s[:, 10:], arrivals[:, :10], atol=1e-5, rtol=1e-4)


def test_arrivals_loss_weights_types_and_skips_unlabelled_rows(tmp_path):
    predicted = torch.zeros(1, 20)
    for piece, weight in enumerate(ARRIVALS_TYPE_WEIGHTS):
        target = torch.zeros(1, 20)
        target[0, 5 * 2 + piece] = 1.0
        assert float(arrivals_loss(predicted, target)) == pytest.approx(weight / 44)

    rng = np.random.default_rng(4)
    _write_v3(tmp_path / "a.dst", rng, n=8)
    _write_v3(tmp_path / "b.dst", rng, n=8, labelled=False)
    dataset = DistillDataset([read_chunk(tmp_path / f"{name}.dst", layout="wide") for name in "ab"],
                             torch.device("cpu"))
    model = get_twin_board_model("twin-s", planes_per_board=W, arrivals=True)
    batch = dataset.batch(torch.arange(16))
    weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 1.0, "moves_left": 0.1, "arrivals": 0.05}
    _, terms, _ = losses(model, batch, weights)
    # The head starts at zero: the loss is the labelled rows' weighted target energy.
    labelled = dataset.arrivals[:8]
    expected = arrivals_loss(torch.zeros_like(labelled), labelled).mean()
    assert float(terms["arrivals"].detach()) == pytest.approx(float(expected), rel=1e-5)
    # A flat weight dict without "arrivals" trains without it.
    total, _, _ = losses(model, batch, {k: v for k, v in weights.items() if k != "arrivals"})
    assert torch.isfinite(total)


def test_a_wide_training_step_moves_the_arrivals_head(tmp_path):
    rng = np.random.default_rng(5)
    _write_v3(tmp_path / "a.dst", rng, n=16)
    dataset = DistillDataset([read_chunk(tmp_path / "a.dst", layout="wide")], torch.device("cpu"))
    model = get_twin_board_model("twin-s", planes_per_board=W, arrivals=True)
    weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 1.0, "moves_left": 0.1, "arrivals": 1.0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = dataset.batch(torch.arange(16))
    first = None
    for _ in range(15):
        total, terms, _ = losses(model, batch, weights)
        first = float(terms["arrivals"].detach()) if first is None else first
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    assert float(losses(model, batch, weights)[1]["arrivals"]) < first


def test_wide_export_has_130_inputs_and_an_arrivals_output(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    from hivemind.distill.train import export

    torch.manual_seed(0)
    model = get_twin_board_model("twin-s", planes_per_board=W, arrivals=True, attention_dim=0, se_every=0)
    torch.save({"model": model.state_dict(), "size": "twin-s",
                "variant": {"planes_per_board": W, "arrivals": True, "attention_dim": 0, "se_every": 0}},
               tmp_path / "best.pt")
    export(tmp_path, "twin-s", path=tmp_path / "w.onnx", fp32_path=tmp_path / "w-fp32.onnx")
    for name in ("w.onnx", "w-fp32.onnx"):
        session = onnxruntime.InferenceSession(str(tmp_path / name), providers=["CPUExecutionProvider"])
        assert session.get_inputs()[0].shape[1] == NUM_WIDE_CHANNELS
        names = [output.name for output in session.get_outputs()]
        assert names == ["value", "pi_a", "pi_b", "wdl_out", "moves_left", "arrivals"]
        out = session.run(None, {"data": np.random.rand(2, NUM_WIDE_CHANNELS, 8, 8).astype(np.float32)})
        assert out[5].shape == (2, 20) and np.abs(out[5]).max() == 0


# ---- it13 itself ----------------------------------------------------------

def test_it13_widened_equals_it13_on_300_positions():
    """The contract's parity test on the real it13 checkpoint (skipped when it
    is not downloaded): random legal positions, random lead/speed, and the
    new planes empty, timed or uniform noise in [-2, 2]."""
    from hivemind.network import DESTINATIONS

    path = DESTINATIONS["clock-it13-checkpoint"]
    if not path.is_file():
        pytest.skip(f"no it13 checkpoint at {path}")
    source = torch.load(path, map_location="cpu")
    narrow, _, _ = load_twin_board_checkpoint(source, wide=False)
    widened, _, _ = load_twin_board_checkpoint(source, wide=True)
    report = wide.parity(narrow.eval(), widened.eval(), wide.parity_inputs(300, seed=11))
    assert max(report[name] for name in wide.OUTPUTS) < wide.PARITY_TOLERANCE
    assert report["arrivals_max_abs"] == 0


# ---- lineage --------------------------------------------------------------

def test_init_lineage_wide_starts_from_the_widened_net_and_gates_against_it13(tmp_path):
    def fetch(variant, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{variant}.bin"
        path.write_text(variant)
        return path

    def fake_widen(checkpoint, out_dir, models):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{wide.WIDE_NAME}.pt").write_text("wide pt")
        (models / f"{wide.WIDE_NAME}.onnx").write_text("wide onnx")

    paths = wave2.init_lineage("w2w", tmp_path / "models", tmp_path / "artifacts", fetch=fetch, wide=True,
                               widen=fake_widen)
    assert paths["generator"].read_text() == "wide onnx"
    assert paths["checkpoint"].read_text() == "wide pt"
    assert paths["fixed"].read_text() == "clock-it13"
