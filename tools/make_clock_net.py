#!/usr/bin/env python3
"""Widen a 37-plane-per-board Hivemind network for the custom engine.

Spec v3 (default, `--planes 44`): the v2 layout below plus channel 43 of each
board block, the diagonal clock plane tanh((t_ourA - t_oppB) / s) (s = 15 s by
default; its weights start at 0 like every new plane, so s does not change the
network, only the planes it is checked on). Same v2 heads. Output
`hivemind-clock-v3*`.

Spec v2 (`--planes 43`): every board block grows from 37 to 43
channels: 37..40 the clock planes (our / opponent seat, min(t/300, 1) and
exp(-t/10)) and 41..42 the elapsed-time planes (min(e/10, 1), 1 - exp(-e/2)).
It also adds the two v2 auxiliary heads, exported as extra ONNX outputs:
  - think_gain [batch, 2]: one logit per board (A, B), the chance a 4x longer
    search plays a different move there;
  - arrivals [batch, 20]: predicted pieces arriving within 5 s, pocket-major
    (ours A, theirs A, ours B, theirs B) x (P, N, B, R, Q).
Each head is global pooling of the post-attention board maps -> one hidden
layer (64, ReLU) -> linear output whose weight and bias are exactly 0.

Spec v1 (`--planes 41`): 41 channels per board (clock planes only), no heads;
this reproduces the v1 network `hivemind-clock*`.

All new input weights are exactly 0, so the widened network computes the same
value, policy, wdl and moves_left as the source network whatever the new planes
hold, and the new heads output exactly 0 at the start.

Which weights read the raw input (crossboard-risev33, the it04 architecture),
with S = source and T = target channels per board:
  - body_spatial.0.body.0.weight  [384, 2S, 3, 3] -> [384, 2T, 3, 3]
    (the stem sees both boards: board A's S channels stay at 0..S-1, board B's
     move to T..T+S-1; the rest are new zeros)
  - board_token_projections.{0,1}.0.weight [192, 384 + S, 1, 1]
    -> [192, 384 + T, 1, 1] (each board's raw planes follow the 384 trunk
     channels; the new columns are zeros)
For dualstream-memory-risev33 the shared per-board stem
(shared_stem.body.0.weight [C, S, 3, 3]) gets T - S zero columns.
The source may be a 37-plane (it04), a 41-plane (v1) or a 43-plane (v2) network.

The optimizer state (SGD momentum buffers) is widened the same way; the new
head parameters are appended to the last parameter group without state, so
`train_loop.py --mode rl --checkpoint <out> --resume-training` stays possible.

    python tools/make_clock_net.py                # it04 -> engine/models/hivemind-clock-v3* (44 planes)
    python tools/make_clock_net.py --planes 43    # it04 -> engine/models/hivemind-clock-v2*
    python tools/make_clock_net.py --planes 41    # it04 -> engine/models/hivemind-clock* (v1)
    python tools/make_clock_net.py --check        # also run the parity check
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "src"
if SOURCE_DIR.is_dir() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from hivemind.architectures.model_config import (  # noqa: E402
    create_model,
    model_args_from_state_dict,
)
from hivemind.constants import (  # noqa: E402
    NUM_ARRIVAL_OUTPUTS,
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    DIAGONAL_SCALE_S,
    NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_DIAGONAL_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_ELAPSED_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_THINK_GAIN_OUTPUTS,
    board_channels_from_total,
)

MODELS = REPO / "engine" / "models"
DEFAULT_SOURCE = MODELS / "hivemind-it04-crossboard-risev33-loss1.556-p82.0.tar"
DEFAULT_PREFIXES = {
    NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD: "hivemind-clock",
    NUM_ELAPSED_BUGHOUSE_CHANNELS_PER_BOARD: "hivemind-clock-v2",
    NUM_DIAGONAL_BUGHOUSE_CHANNELS_PER_BOARD: "hivemind-clock-v3",
}
FP16_CONVERTER = REPO / "engine" / "scripts" / "convert_onnx_fp16.py"

OLD = NUM_BUGHOUSE_CHANNELS_PER_BOARD  # 37
V1 = NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD  # 41
V2 = NUM_ELAPSED_BUGHOUSE_CHANNELS_PER_BOARD  # 43
V3 = NUM_DIAGONAL_BUGHOUSE_CHANNELS_PER_BOARD  # 44
NEW = V3  # default target

OUTPUT_NAMES = ("value", "pi_a", "pi_b", "wdl_out", "moves_left")
AUX_OUTPUT_NAMES = ("think_gain", "arrivals")
AUX_KEY_PREFIXES = ("think_gain_head.", "arrivals_head.")
STEM_KEYS = ("body_spatial.0.body.0.weight", "shared_stem.body.0.weight")


# ----------------------------------------------------------------- widening --

def _widen_two_board_stem(tensor: torch.Tensor, source: int, target: int) -> torch.Tensor:
    """[O, 2S, k, k] -> [O, 2T, k, k]; old A block at 0, old B block at T."""
    if tensor.shape[1] != 2 * source:
        raise ValueError(f"Expected a {2 * source}-channel stem, found {tuple(tensor.shape)}")
    shape = list(tensor.shape)
    shape[1] = 2 * target
    widened = tensor.new_zeros(shape)
    widened[:, :source] = tensor[:, :source]
    widened[:, target:target + source] = tensor[:, source:]
    return widened


def _append_zero_columns(tensor: torch.Tensor, expected_in: int, extra: int) -> torch.Tensor:
    """[O, expected_in, ...] -> [O, expected_in + extra, ...] with zero new columns."""
    if tensor.shape[1] != expected_in:
        raise ValueError(f"Expected {expected_in} input channels, found {tuple(tensor.shape)}")
    shape = list(tensor.shape)
    shape[1] += extra
    widened = tensor.new_zeros(shape)
    widened[:, :expected_in] = tensor
    return widened


def widen_tensor(key: str, tensor: torch.Tensor, trunk_channels: int,
                 source: int = OLD, target: int = NEW) -> torch.Tensor:
    """Widen one parameter (or its momentum buffer) if it reads the raw planes."""
    if source == target:
        return tensor
    if key == "body_spatial.0.body.0.weight":
        return _widen_two_board_stem(tensor, source, target)
    if key == "shared_stem.body.0.weight":
        return _append_zero_columns(tensor, source, target - source)
    if key.startswith("board_token_projections.") and key.endswith(".0.weight"):
        return _append_zero_columns(tensor, trunk_channels + source, target - source)
    return tensor


def widened_keys(state_dict) -> list[str]:
    return [
        key for key in state_dict
        if key in STEM_KEYS
        or (key.startswith("board_token_projections.") and key.endswith(".0.weight"))
    ]


def _trunk_channels(state_dict) -> int:
    for key in STEM_KEYS:
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Checkpoint has no recognised input stem")


def source_board_channels(state_dict) -> int:
    return board_channels_from_total(model_args_from_state_dict(state_dict).input_shape[0])


def widen_state_dict(state_dict, target: int = NEW) -> dict:
    """Copy of a 37- (41- or 43-) channel-per-board state dict widened to target."""
    source = source_board_channels(state_dict)
    if source not in (OLD, V1, V2) or target not in (V1, V2, V3) or target <= source:
        raise ValueError(
            f"Cannot widen a {source}-plane-per-board network to {target} planes"
        )
    trunk = _trunk_channels(state_dict)
    return {key: widen_tensor(key, value, trunk, source, target)
            for key, value in state_dict.items()}


def add_aux_heads(state_dict, seed: int = 0) -> dict:
    """Add the v2 think_gain / arrivals heads: hidden layer seeded, output layer 0."""
    if any(key.startswith(AUX_KEY_PREFIXES) for key in state_dict):
        return dict(state_dict)
    args = model_args_from_state_dict(state_dict)
    args.aux_heads = True
    torch.manual_seed(seed)
    model = create_model(args)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith(AUX_KEY_PREFIXES)]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected keys adding aux heads: missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    full = model.state_dict()
    for name in ("think_gain_head", "arrivals_head"):
        if full[f"{name}.body.2.weight"].abs().max() != 0 or full[f"{name}.body.2.bias"].abs().max() != 0:
            raise AssertionError(f"{name} output layer is not zero")
    # Original tensors first (same order), new head tensors after.
    out = dict(state_dict)
    for key, value in full.items():
        if key.startswith(AUX_KEY_PREFIXES):
            out[key] = value.clone()
    return out


def widen_optimizer_state(optimizer_state: dict, model: torch.nn.Module, trunk: int,
                          source: int = OLD, target: int = NEW,
                          new_parameter_count: int = 0) -> dict:
    """Widen per-parameter optimizer buffers to match the widened parameters.

    torch optimizers key state by parameter position in model.parameters(),
    which equals named_parameters() order; widening changes shapes, not order.
    new_parameter_count parameters registered after all old ones (the aux
    heads) are appended to the last parameter group, without state.
    """
    names = [name for name, _ in model.named_parameters()]
    flat_ids = [pid for group in optimizer_state["param_groups"] for pid in group["params"]]
    if len(flat_ids) != len(names):
        raise ValueError(
            f"Optimizer covers {len(flat_ids)} parameters, model has {len(names)}"
        )
    id_to_name = dict(zip(flat_ids, names))
    widened = copy.deepcopy(optimizer_state)
    for pid, buffers in widened["state"].items():
        name = id_to_name[pid]
        for buffer_name, value in list(buffers.items()):
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                buffers[buffer_name] = widen_tensor(name, value, trunk, source, target)
    if new_parameter_count:
        next_id = max(flat_ids, default=-1) + 1
        widened["param_groups"][-1]["params"].extend(
            range(next_id, next_id + new_parameter_count)
        )
    return widened


def widen_checkpoint(checkpoint: dict, target: int = NEW,
                     aux_heads: bool | None = None) -> dict:
    """Widen model weights and, when present, the optimizer state.

    aux_heads (default: on for the v2 and v3 targets) also adds the think_gain
    and arrivals heads with zero output layers (kept as they are when the
    source already has them).
    """
    if aux_heads is None:
        aux_heads = target in (V2, V3)
    state_dict = checkpoint["model_state_dict"]
    source = source_board_channels(state_dict)
    old_model = create_model(model_args_from_state_dict(state_dict))
    old_model.load_state_dict(state_dict)
    new_state = widen_state_dict(state_dict, target)
    added = 0
    if aux_heads:
        before = len(new_state)
        new_state = add_aux_heads(new_state)
        added = len(new_state) - before
    widened = dict(checkpoint)
    widened["model_state_dict"] = new_state
    if checkpoint.get("optimizer_state_dict") is not None:
        widened["optimizer_state_dict"] = widen_optimizer_state(
            checkpoint["optimizer_state_dict"], old_model, _trunk_channels(state_dict),
            source, target, new_parameter_count=added,
        )
    widened["clock_planes"] = True
    widened["elapsed_planes"] = target in (V2, V3)
    widened["diagonal_plane"] = target == V3
    widened["aux_heads"] = bool(aux_heads)
    return widened


def load_model(state_dict) -> torch.nn.Module:
    model = create_model(model_args_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ------------------------------------------------------------------- planes --

def widen_planes(planes74: np.ndarray, extra_values: np.ndarray | None = None,
                 target: int = NEW) -> np.ndarray:
    """[N, 74, 8, 8] -> [N, 2T, 8, 8]; new planes from extra_values [N, 2, T-37] or 0."""
    n = planes74.shape[0]
    out = np.zeros((n, 2 * target, 8, 8), dtype=planes74.dtype)
    out[:, :OLD] = planes74[:, :OLD]
    out[:, target:target + OLD] = planes74[:, OLD:]
    if extra_values is not None:
        out[:, OLD:target] = extra_values[:, 0, :, None, None]
        out[:, target + OLD:] = extra_values[:, 1, :, None, None]
    return out


def strip_new_planes(planes: np.ndarray) -> np.ndarray:
    """[N, 2T, 8, 8] -> [N, 74, 8, 8] (drop channels 37..T-1 of each board)."""
    target = board_channels_from_total(planes.shape[1])
    return np.concatenate((planes[:, :OLD], planes[:, target:target + OLD]), axis=1)


strip_clock_planes = strip_new_planes  # v1 name


def export_onnx(model: torch.nn.Module, path: Path, input_channels: int) -> Path:
    """FP32 ONNX with the engine's I/O names (data -> value, pi_a, pi_b, wdl_out,
    moves_left, and think_gain, arrivals when the model has the v2 heads)."""
    from hivemind.training.trainer_agent import export_to_onnx

    dummy = torch.zeros(1, input_channels, 8, 8)
    return export_to_onnx(
        copy.deepcopy(model).cpu(), 1, dummy, path.parent, path.stem,
        has_auxiliary_output=True, dynamic_batch_size=True,
        fp16=False, onnx_name=path.name,
    )


def convert_fp16(src: Path, dst: Path) -> None:
    subprocess.run([sys.executable, str(FP16_CONVERTER), str(src), str(dst)], check=True)


# ---------------------------------------------------------------- positions --

def random_positions(count: int, seed: int = 0, max_moves: int = 120):
    """Legal positions from random bughouse play, with random clocks and elapsed (ms)."""
    import chess
    from hivemind.domain.board import BughouseBoard

    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        board = BughouseBoard()
        for _ in range(int(rng.integers(0, max_moves))):
            board_num = int(rng.integers(0, 2))
            moves = list(board.boards[board_num].legal_moves)
            if not moves or board.is_checkmate():
                break
            board.push(board_num, moves[int(rng.integers(0, len(moves)))])
        clocks = tuple(float(v) for v in rng.uniform(-2000, 400000, size=4))
        elapsed = tuple(float(v) for v in rng.uniform(-500, 40000, size=2))
        team = bool(rng.integers(0, 2))
        positions.append((copy.deepcopy(board), chess.WHITE if team else chess.BLACK,
                          clocks, elapsed))
    return positions


def logged_positions(paths, limit: int):
    """Positions from arena event logs (real engine games), with clocks and the
    time since each board's last move (from the event timestamps)."""
    import json
    import chess
    from hivemind.domain.board import BughouseBoard

    positions = []
    for path in paths:
        board = None
        try:
            with open(path) as handle:
                for line in handle:
                    event = json.loads(line)
                    if event.get("type") == "game_start":
                        board = BughouseBoard()
                        board.set_fen(f"{event['fens']['A']} | {event['fens']['B']}")
                        last = {"A": event.get("ts", 0), "B": event.get("ts", 0)}
                    elif event.get("type") == "move" and board is not None:
                        board.push_uci(0 if event["board"] == "A" else 1, event["uci"])
                        c = event["clocks"]
                        clocks = (c["A_white"], c["A_black"], c["B_white"], c["B_black"])
                        board.times = [[c["A_black"], c["A_white"]], [c["B_black"], c["B_white"]]]
                        ts = event.get("ts", 0)
                        last[event["board"]] = ts
                        elapsed = (ts - last["A"], ts - last["B"])
                        for team in (chess.WHITE, chess.BLACK):
                            positions.append((copy.deepcopy(board), team, clocks, elapsed))
                        if len(positions) >= limit:
                            return positions
        except (ValueError, KeyError, AssertionError) as error:
            print(f"  stopped replaying {path} at an unsupported move: {error}")
    return positions


def planes_for(positions, target: int = NEW, diagonal_scale_s: float = DIAGONAL_SCALE_S):
    """(74-channel planes, target-width planes from board2planes) for positions."""
    from hivemind.domain.board2planes import board2planes

    p74 = np.stack([board2planes(p[0], p[1]) for p in positions])
    if target == V3:
        wide = np.stack([board2planes(b, team, clocks=clocks, elapsed=elapsed,
                                      diagonal_plane=True, diagonal_scale_s=diagonal_scale_s)
                         for b, team, clocks, elapsed in positions])
    elif target == V2:
        wide = np.stack([board2planes(b, team, clocks=clocks, elapsed=elapsed)
                         for b, team, clocks, elapsed in positions])
    else:
        wide = np.stack([board2planes(p[0], p[1], clocks=p[2]) for p in positions])
    return p74, wide


# ------------------------------------------------------------------- parity --

def _torch_outputs(model, planes: np.ndarray) -> list[np.ndarray]:
    with torch.no_grad():
        outputs = model(torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)))
    value, (pi_a, pi_b), _aux, wdl, moves_left = outputs[:5]
    return [t.numpy() for t in (value, pi_a, pi_b, wdl, moves_left)]


def _torch_aux_outputs(model, planes: np.ndarray) -> list[np.ndarray]:
    with torch.no_grad():
        outputs = model(torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)))
    return [t.numpy() for t in outputs[-2:]]


def _ort_outputs(path: Path, planes: np.ndarray, names=OUTPUT_NAMES) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: np.ascontiguousarray(planes, dtype=np.float32)}
    return session.run(list(names), feed)


def max_diffs(a: list[np.ndarray], b: list[np.ndarray], names=OUTPUT_NAMES) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for name, x, y in zip(names, a, b)
    }


def plane_variants(wide: np.ndarray, rng) -> list[tuple[str, np.ndarray]]:
    """The same positions with different contents in every new plane."""
    target = board_channels_from_total(wide.shape[1])
    new = list(range(OLD, target)) + list(range(target + OLD, 2 * target))
    zero = wide.copy()
    zero[:, new] = 0
    noise = wide.copy()
    noise[:, new] = rng.uniform(-5, 5, size=noise[:, new].shape)
    return [("board2planes", wide), ("zero new planes", zero), ("noise new planes", noise)]


clock_variants = plane_variants  # v1 name


def run_parity(old_model, new_model, positions_sets, onnx_pair=None, batch=64, seed=0):
    """Compare the source net (74 planes) with the widened net. Returns the worst
    diff over every source output, and of the new heads from 0."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    has_aux = getattr(new_model, "aux_heads", False)
    for label, p74, wide in positions_sets:
        if not np.array_equal(strip_new_planes(wide), p74):
            raise AssertionError(f"{label}: widened encoding differs outside the new planes")
        for variant, planes in plane_variants(wide, rng):
            for start in range(0, len(p74), batch):
                ref = _torch_outputs(old_model, p74[start:start + batch])
                got = _torch_outputs(new_model, planes[start:start + batch])
                diffs = max_diffs(ref, got)
                if has_aux:
                    aux = _torch_aux_outputs(new_model, planes[start:start + batch])
                    diffs.update({n: float(np.max(np.abs(a))) for n, a in zip(AUX_OUTPUT_NAMES, aux)})
                worst = max(worst, *diffs.values())
            print(f"  torch  {label:>14} / {variant:<17} max |diff| "
                  + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
            if onnx_pair is not None:
                ref = _ort_outputs(onnx_pair[0], p74)
                names = OUTPUT_NAMES + (AUX_OUTPUT_NAMES if has_aux else ())
                got = _ort_outputs(onnx_pair[1], planes, names)
                diffs = max_diffs(ref, got[:len(OUTPUT_NAMES)])
                for name, values in zip(names[len(OUTPUT_NAMES):], got[len(OUTPUT_NAMES):]):
                    diffs[name] = float(np.max(np.abs(values)))
                worst = max(worst, *diffs.values())
                print(f"  ort    {label:>14} / {variant:<17} max |diff| "
                      + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    return worst


def check_onnx_outputs(path: Path, input_channels: int, aux_heads: bool) -> list[tuple[str, list]]:
    """Load with onnxruntime and check the input and output names/shapes."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape[1:] != [input_channels, 8, 8]:
        raise AssertionError(f"{path}: input {[(i.name, i.shape) for i in inputs]}")
    outputs = [(o.name, o.shape) for o in session.get_outputs()]
    names = [name for name, _ in outputs]
    expected = list(OUTPUT_NAMES) + (list(AUX_OUTPUT_NAMES) if aux_heads else [])
    if names != expected:
        raise AssertionError(f"{path}: outputs {names}, expected {expected}")
    if aux_heads:
        shapes = dict(outputs)
        if shapes["think_gain"][1:] != [NUM_THINK_GAIN_OUTPUTS] or \
                shapes["arrivals"][1:] != [NUM_ARRIVAL_OUTPUTS]:
            raise AssertionError(f"{path}: aux output shapes {outputs}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=MODELS)
    parser.add_argument("--planes", type=int, choices=(V1, V2, V3), default=V3,
                        help="channels per board: 44 (spec v3, default), 43 (v2) or 41 (v1)")
    parser.add_argument("--prefix", default=None,
                        help="output name prefix (default hivemind-clock-v3; hivemind-clock-v2 "
                             "with --planes 43, hivemind-clock with --planes 41)")
    parser.add_argument("--diagonal-scale", type=float, default=DIAGONAL_SCALE_S,
                        help="--check with --planes 44: s of the diagonal plane in the "
                             "parity positions (default 15; the network does not depend on it)")
    aux = parser.add_mutually_exclusive_group()
    aux.add_argument("--aux-heads", dest="aux_heads", action="store_true", default=None,
                     help="add think_gain/arrivals heads (default with --planes 43 and 44)")
    aux.add_argument("--no-aux-heads", dest="aux_heads", action="store_false")
    parser.add_argument("--no-fp16", action="store_true", help="skip the FP16 ONNX")
    parser.add_argument("--check", action="store_true",
                        help="parity check: widened net vs source, torch and ONNX Runtime")
    parser.add_argument("--random-positions", type=int, default=256)
    parser.add_argument("--logged-positions", type=int, default=512)
    parser.add_argument("--game-logs", type=Path, nargs="*", default=None,
                        help="arena event logs (jsonl) to take real positions from")
    args = parser.parse_args()
    target = args.planes
    aux_heads = (target in (V2, V3)) if args.aux_heads is None else args.aux_heads
    prefix = args.prefix or DEFAULT_PREFIXES[target]

    torch.set_num_threads(4)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if source_board_channels(checkpoint["model_state_dict"]) != OLD:
        raise SystemExit("The parity check compares against a 37-plane source; use it04")
    widened = widen_checkpoint(checkpoint, target, aux_heads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / f"{prefix}.tar"
    torch.save(widened, tar_path)
    print(f"checkpoint {tar_path} ({target} planes per board, aux heads "
          f"{'on' if aux_heads else 'off'}; widened: "
          f"{', '.join(widened_keys(checkpoint['model_state_dict']))}; "
          f"optimizer state {'widened' if 'optimizer_state_dict' in widened else 'absent'})")

    new_model = load_model(widened["model_state_dict"])
    fp32_path = export_onnx(new_model, args.out_dir / f"{prefix}-fp32.onnx", 2 * target)
    print(f"onnx fp32  {fp32_path}: {check_onnx_outputs(fp32_path, 2 * target, aux_heads)}")
    if not args.no_fp16:
        fp16_path = args.out_dir / f"{prefix}-fp16.onnx"
        convert_fp16(fp32_path, fp16_path)
        print(f"onnx fp16  {fp16_path}")

    if not args.check:
        return

    import tempfile

    old_model = load_model(checkpoint["model_state_dict"])
    with tempfile.TemporaryDirectory() as tmp:
        # Reference: it04 exported from its checkpoint by the same FP32 path.
        ref_path = export_onnx(old_model, Path(tmp) / "it04-fp32.onnx", 2 * OLD)
        sets = []
        rnd = random_positions(args.random_positions, seed=1)
        sets.append(("random", *planes_for(rnd, target, args.diagonal_scale)))
        logs = args.game_logs
        if logs is None:
            runs = REPO.parents[1] / "runs"
            logs = sorted(runs.glob("*/game_*.jsonl")) if runs.is_dir() else []
        if logs and args.logged_positions:
            real = logged_positions(logs, args.logged_positions)
            if real:
                sets.append(("logged games", *planes_for(real, target, args.diagonal_scale)))
        for label, p74, _ in sets:
            print(f"positions: {label} {len(p74)}")
        worst = run_parity(old_model, new_model, sets, onnx_pair=(ref_path, fp32_path))
    print(f"worst max |diff| = {worst:.3e} ({'PASS' if worst < 1e-5 else 'FAIL'} at 1e-5)")
    if worst >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
