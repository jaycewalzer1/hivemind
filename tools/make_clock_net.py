#!/usr/bin/env python3
"""Widen a 37-plane-per-board Hivemind network to 41 planes (clock planes).

The custom engine feeds four extra planes per board (channels 37..40 of each
board block: our / opponent seat, min(t/300, 1) and exp(-t/10)). This tool
takes the it04 checkpoint and adds those input channels with weights set to
exactly 0, so the widened network computes the same function as it04 whatever
the clock planes hold. Training then learns the new weights.

Which weights read the raw input (crossboard-risev33, the it04 architecture):
  - body_spatial.0.body.0.weight  [384, 74, 3, 3] -> [384, 82, 3, 3]
    (the stem sees both boards at once: board A 0..36 stays at 0..36,
     board B 37..73 moves to 41..77; 37..40 and 78..81 are new zeros)
  - board_token_projections.{0,1}.0.weight [192, 384 + 37, 1, 1]
    -> [192, 384 + 41, 1, 1] (each board's raw planes are concatenated after
     the 384 trunk channels; the four new columns are zeros)
For dualstream-memory-risev33 the shared per-board stem
(shared_stem.body.0.weight [C, 37, 3, 3]) gets four zero columns.

The optimizer state (SGD momentum buffers) is widened the same way, so
`train_loop.py --mode rl --checkpoint <out> --resume-training` stays possible;
the momentum of the new weights starts at 0.

    python tools/make_clock_net.py            # it04 -> engine/models/hivemind-clock*
    python tools/make_clock_net.py --check    # also run the parity check
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
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCK_CHANNELS_PER_BOARD,
)

MODELS = REPO / "engine" / "models"
DEFAULT_SOURCE = MODELS / "hivemind-it04-crossboard-risev33-loss1.556-p82.0.tar"
DEFAULT_PREFIX = "hivemind-clock"
FP16_CONVERTER = REPO / "engine" / "scripts" / "convert_onnx_fp16.py"

OLD = NUM_BUGHOUSE_CHANNELS_PER_BOARD  # 37
NEW = NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD  # 41
EXTRA = NUM_CLOCK_CHANNELS_PER_BOARD  # 4

OUTPUT_NAMES = ("value", "pi_a", "pi_b", "wdl_out", "moves_left")


def _widen_two_board_stem(tensor: torch.Tensor) -> torch.Tensor:
    """[O, 74, k, k] -> [O, 82, k, k]; old A block at 0, old B block at 41."""
    if tensor.shape[1] != 2 * OLD:
        raise ValueError(f"Expected a {2 * OLD}-channel stem, found {tuple(tensor.shape)}")
    shape = list(tensor.shape)
    shape[1] = 2 * NEW
    widened = tensor.new_zeros(shape)
    widened[:, :OLD] = tensor[:, :OLD]
    widened[:, NEW:NEW + OLD] = tensor[:, OLD:]
    return widened


def _append_zero_columns(tensor: torch.Tensor, expected_in: int) -> torch.Tensor:
    """[O, expected_in, ...] -> [O, expected_in + 4, ...] with zero new columns."""
    if tensor.shape[1] != expected_in:
        raise ValueError(f"Expected {expected_in} input channels, found {tuple(tensor.shape)}")
    shape = list(tensor.shape)
    shape[1] += EXTRA
    widened = tensor.new_zeros(shape)
    widened[:, :expected_in] = tensor
    return widened


def widen_tensor(key: str, tensor: torch.Tensor, trunk_channels: int) -> torch.Tensor:
    """Widen one parameter (or its momentum buffer) if it reads the raw planes."""
    if key == "body_spatial.0.body.0.weight":
        return _widen_two_board_stem(tensor)
    if key == "shared_stem.body.0.weight":
        return _append_zero_columns(tensor, OLD)
    if key.startswith("board_token_projections.") and key.endswith(".0.weight"):
        return _append_zero_columns(tensor, trunk_channels + OLD)
    return tensor


def widened_keys(state_dict) -> list[str]:
    return [
        key for key in state_dict
        if key in ("body_spatial.0.body.0.weight", "shared_stem.body.0.weight")
        or (key.startswith("board_token_projections.") and key.endswith(".0.weight"))
    ]


def _trunk_channels(state_dict) -> int:
    for key in ("body_spatial.0.body.0.weight", "shared_stem.body.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Checkpoint has no recognised input stem")


def widen_state_dict(state_dict) -> dict:
    """Return a copy of a 74-channel state dict widened to 82 channels."""
    args = model_args_from_state_dict(state_dict)
    if args.input_shape[0] != 2 * OLD:
        raise ValueError(
            f"Source network has {args.input_shape[0]} input channels, expected {2 * OLD}"
        )
    trunk = _trunk_channels(state_dict)
    return {key: widen_tensor(key, value, trunk) for key, value in state_dict.items()}


def widen_optimizer_state(optimizer_state: dict, model: torch.nn.Module, trunk: int) -> dict:
    """Widen per-parameter optimizer buffers to match the widened parameters.

    torch optimizers key state by parameter position in model.parameters(),
    which equals named_parameters() order; widening changes shapes, not order.
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
                buffers[buffer_name] = widen_tensor(name, value, trunk)
    return widened


def widen_checkpoint(checkpoint: dict) -> dict:
    """Widen model weights and, when present, the optimizer state."""
    state_dict = checkpoint["model_state_dict"]
    old_model = create_model(model_args_from_state_dict(state_dict))
    old_model.load_state_dict(state_dict)
    new_state = widen_state_dict(state_dict)
    widened = dict(checkpoint)
    widened["model_state_dict"] = new_state
    if checkpoint.get("optimizer_state_dict") is not None:
        widened["optimizer_state_dict"] = widen_optimizer_state(
            checkpoint["optimizer_state_dict"], old_model, _trunk_channels(state_dict)
        )
    widened["clock_planes"] = True
    return widened


def load_model(state_dict) -> torch.nn.Module:
    model = create_model(model_args_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def widen_planes(planes74: np.ndarray, clock_values: np.ndarray | None = None) -> np.ndarray:
    """[N, 74, 8, 8] -> [N, 82, 8, 8]; clock planes from clock_values [N, 2, 4] or 0."""
    n = planes74.shape[0]
    out = np.zeros((n, 2 * NEW, 8, 8), dtype=planes74.dtype)
    out[:, :OLD] = planes74[:, :OLD]
    out[:, NEW:NEW + OLD] = planes74[:, OLD:]
    if clock_values is not None:
        out[:, OLD:NEW] = clock_values[:, 0, :, None, None]
        out[:, NEW + OLD:] = clock_values[:, 1, :, None, None]
    return out


def strip_clock_planes(planes82: np.ndarray) -> np.ndarray:
    """[N, 82, 8, 8] -> [N, 74, 8, 8] (drop channels 37..40 of each board)."""
    return np.concatenate((planes82[:, :OLD], planes82[:, NEW:NEW + OLD]), axis=1)


def export_onnx(model: torch.nn.Module, path: Path, input_channels: int) -> Path:
    """FP32 ONNX with the engine's I/O names (data -> value, pi_a, pi_b, wdl_out, moves_left)."""
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
    """Legal positions from random bughouse play, with random clocks (ms)."""
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
        team = bool(rng.integers(0, 2))
        positions.append((copy.deepcopy(board), chess.WHITE if team else chess.BLACK, clocks))
    return positions


def logged_positions(paths, limit: int):
    """Positions from arena event logs (real engine games), with their clocks."""
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
                    elif event.get("type") == "move" and board is not None:
                        board.push_uci(0 if event["board"] == "A" else 1, event["uci"])
                        c = event["clocks"]
                        clocks = (c["A_white"], c["A_black"], c["B_white"], c["B_black"])
                        board.times = [[c["A_black"], c["A_white"]], [c["B_black"], c["B_white"]]]
                        for team in (chess.WHITE, chess.BLACK):
                            positions.append((copy.deepcopy(board), team, clocks))
                        if len(positions) >= limit:
                            return positions
        except (ValueError, KeyError, AssertionError) as error:
            print(f"  stopped replaying {path} at an unsupported move: {error}")
    return positions


def planes_for(positions):
    from hivemind.domain.board2planes import board2planes

    p74 = np.stack([board2planes(b, team) for b, team, _ in positions])
    p82 = np.stack([board2planes(b, team, clocks=clocks) for b, team, clocks in positions])
    return p74, p82


# ------------------------------------------------------------------- parity --

def _torch_outputs(model, planes: np.ndarray) -> list[np.ndarray]:
    with torch.no_grad():
        outputs = model(torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)))
    value, (pi_a, pi_b), _aux, wdl, moves_left = outputs[:5]
    return [t.numpy() for t in (value, pi_a, pi_b, wdl, moves_left)]


def _ort_outputs(path: Path, planes: np.ndarray) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: np.ascontiguousarray(planes, dtype=np.float32)}
    return session.run(list(OUTPUT_NAMES), feed)


def max_diffs(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for name, x, y in zip(OUTPUT_NAMES, a, b)
    }


def clock_variants(p82: np.ndarray, rng) -> list[tuple[str, np.ndarray]]:
    """The same positions with different clock-plane contents."""
    zero = p82.copy()
    zero[:, OLD:NEW] = 0
    zero[:, NEW + OLD:] = 0
    noise = p82.copy()
    noise[:, OLD:NEW] = rng.uniform(-5, 5, size=noise[:, OLD:NEW].shape)
    noise[:, NEW + OLD:] = rng.uniform(-5, 5, size=noise[:, NEW + OLD:].shape)
    return [("board2planes clocks", p82), ("zero clocks", zero), ("noise clocks", noise)]


def run_parity(old_model, new_model, positions_sets, onnx_pair=None, batch=64, seed=0):
    """Compare it04 (74 planes) with the widened net (82 planes). Returns the worst diff."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for label, p74, p82 in positions_sets:
        if not np.array_equal(strip_clock_planes(p82), p74):
            raise AssertionError(f"{label}: 82-plane encoding differs outside the clock planes")
        for variant, planes in clock_variants(p82, rng):
            for start in range(0, len(p74), batch):
                ref = _torch_outputs(old_model, p74[start:start + batch])
                got = _torch_outputs(new_model, planes[start:start + batch])
                diffs = max_diffs(ref, got)
                worst = max(worst, *diffs.values())
            print(f"  torch  {label:>14} / {variant:<19} max |diff| "
                  + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
            if onnx_pair is not None:
                ref = _ort_outputs(onnx_pair[0], p74)
                got = _ort_outputs(onnx_pair[1], planes)
                diffs = max_diffs(ref, got)
                worst = max(worst, *diffs.values())
                print(f"  ort    {label:>14} / {variant:<19} max |diff| "
                      + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=MODELS)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--no-fp16", action="store_true", help="skip the FP16 ONNX")
    parser.add_argument("--check", action="store_true",
                        help="parity check: widened net vs source, torch and ONNX Runtime")
    parser.add_argument("--random-positions", type=int, default=256)
    parser.add_argument("--logged-positions", type=int, default=512)
    parser.add_argument("--game-logs", type=Path, nargs="*", default=None,
                        help="arena event logs (jsonl) to take real positions from")
    args = parser.parse_args()

    torch.set_num_threads(4)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    widened = widen_checkpoint(checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / f"{args.prefix}.tar"
    torch.save(widened, tar_path)
    print(f"checkpoint {tar_path} (widened: {', '.join(widened_keys(checkpoint['model_state_dict']))}; "
          f"optimizer state {'widened' if 'optimizer_state_dict' in widened else 'absent'})")

    new_model = load_model(widened["model_state_dict"])
    fp32_path = export_onnx(new_model, args.out_dir / f"{args.prefix}-fp32.onnx", 2 * NEW)
    print(f"onnx fp32  {fp32_path}")
    if not args.no_fp16:
        fp16_path = args.out_dir / f"{args.prefix}-fp16.onnx"
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
        sets.append(("random", *planes_for(rnd)))
        logs = args.game_logs
        if logs is None:
            runs = REPO.parents[1] / "runs"
            logs = sorted(runs.glob("*/game_*.jsonl")) if runs.is_dir() else []
        if logs and args.logged_positions:
            real = logged_positions(logs, args.logged_positions)
            if real:
                sets.append(("logged games", *planes_for(real)))
        for label, p74, _ in sets:
            print(f"positions: {label} {len(p74)}")
        worst = run_parity(old_model, new_model, sets, onnx_pair=(ref_path, fp32_path))
    print(f"worst max |diff| = {worst:.3e} ({'PASS' if worst < 1e-5 else 'FAIL'} at 1e-5)")
    if worst >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
