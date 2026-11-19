"""Wave 2b (talisman fork): twin-s-clock it13 widened to the wide planes, with
an arrivals head.

`python -m hivemind.wave2 widen` loads the it13 training checkpoint, appends
26 zero-weight input planes to each board block (hivemind.domain.wide_planes
has the order) and adds the arrivals head with a zero last layer, so every
output it13 has is unchanged whatever the new planes hold. It saves
twin-s-clock-wide.pt and exports twin-s-clock-wide.onnx (FP16 weights, as
it13 is published) and twin-s-clock-wide-fp32.onnx (that file converted by
engine/scripts/convert_onnx_fp32.py, as the published it13 FP32 copy is),
checks parity and writes the SHA-256 of each file to twin-s-clock-wide.json
next to the ONNX.

Parity, three ways, on random legal positions with random lead/speed and
random values in the new planes: the PyTorch networks; it13 and the wide net
exported to FP32 ONNX by the same exporter (ONNX Runtime CPU); and, when the
published it13 files are given, the wide ONNX files against them.
"""
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from hivemind.constants import (
    NUM_CLOCKED_CHANNELS_PER_BOARD,
    NUM_WIDE_CHANNELS,
    NUM_WIDE_CHANNELS_PER_BOARD,
)

WIDE_NAME = "twin-s-clock-wide"
PARITY_TOLERANCE = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def random_positions(count, seed=0, max_plies=80):
    """`count` positions (BughouseBoard, team) from random legal walks of
    0..max_plies actions on either board, drops included."""
    from hivemind.domain.board import BughouseBoard

    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        board = BughouseBoard()
        for _ in range(int(rng.integers(0, max_plies + 1))):
            number = int(rng.integers(0, 2))
            moves = list(board.boards[number].legal_moves)
            if not moves:
                number = 1 - number
                moves = list(board.boards[number].legal_moves)
            if not moves:
                break
            board.push(number, moves[int(rng.integers(0, len(moves)))])
        positions.append((board, chess.WHITE if rng.random() < 0.5 else chess.BLACK))
    return positions


def random_timing(rng):
    from hivemind.domain.wide_planes import PIECE_LETTERS, SEATS, WideTiming

    clocks = {seat: float(rng.uniform(0, 300_000)) for seat in SEATS}
    landings = [(SEATS[int(rng.integers(0, 4))], PIECE_LETTERS[int(rng.integers(0, 5))],
                 int(rng.integers(0, 8000))) for _ in range(int(rng.integers(0, 6)))]
    return WideTiming(clocks, (float(rng.uniform(0, 20_000)), float(rng.uniform(0, 20_000))), landings)


def parity_inputs(count, seed=0):
    """Wide inputs [count, 130, 8, 8] for the parity check: random legal
    positions with random lead and speed planes; a third with no timing, a
    third with random timing, a third with uniform noise in [-2, 2] on every
    new plane (parity must hold whatever they hold)."""
    from hivemind.domain.wide_planes import board2planes_wide

    rng = np.random.default_rng(seed)
    planes = []
    for index, (board, team) in enumerate(random_positions(count, seed)):
        lead, speed = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        timing = random_timing(rng) if index % 3 == 1 else None
        wide = board2planes_wide(board, team, lead, speed, timing)
        if index % 3 == 2:
            for block in range(2):
                start = block * NUM_WIDE_CHANNELS_PER_BOARD + NUM_CLOCKED_CHANNELS_PER_BOARD
                stop = (block + 1) * NUM_WIDE_CHANNELS_PER_BOARD
                wide[start:stop] = rng.uniform(-2, 2, wide[start:stop].shape)
        planes.append(wide)
    return torch.from_numpy(np.stack(planes).astype(np.float32))


def narrow_inputs(wide):
    """The clocked 78 planes inside wide inputs [N, 130, 8, 8]."""
    narrow, block = NUM_CLOCKED_CHANNELS_PER_BOARD, NUM_WIDE_CHANNELS_PER_BOARD
    return torch.cat((wide[:, :narrow], wide[:, block:block + narrow]), dim=1)


OUTPUTS = ("value", "pi_a", "pi_b", "wdl_out", "moves_left")


def torch_outputs(model, planes):
    with torch.no_grad():
        value, (pi_a, pi_b), _, wdl, plys = model(planes)[:5]
    return dict(zip(OUTPUTS, (value, pi_a, pi_b, wdl, plys)))


def parity(narrow_model, wide_model, planes):
    """Max |difference| per it13 output between the narrow network on the
    clocked planes and the wide one on all planes, and the arrivals output's
    max |value| (0 while its last layer is zero)."""
    narrow = torch_outputs(narrow_model, narrow_inputs(planes))
    wide = torch_outputs(wide_model, planes)
    report = {name: float((narrow[name] - wide[name]).abs().max()) for name in OUTPUTS}
    with torch.no_grad():
        report["arrivals_max_abs"] = float(wide_model(planes)[5].abs().max())
    return report


def _convert_fp32(source, destination):
    import importlib.util

    from hivemind.network import FP32_CONVERTER

    spec = importlib.util.spec_from_file_location("convert_onnx_fp32", FP32_CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.convert(str(source), str(destination))


def onnx_parity(narrow_onnx, wide_onnx, planes):
    """The same comparison between two FP32 ONNX files, on ONNX Runtime CPU."""
    import onnxruntime as ort

    def run(path, data):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [output.name for output in session.get_outputs()]
        return dict(zip(names, session.run(None, {"data": data.numpy()})))

    narrow = run(narrow_onnx, narrow_inputs(planes))
    wide = run(wide_onnx, planes)
    report = {name: float(np.abs(narrow[name] - wide[name]).max()) for name in OUTPUTS}
    report["arrivals_max_abs"] = float(np.abs(wide["arrivals"]).max())
    return report


def widen(checkpoint, out_dir, models_dir, positions=300, seed=0, scratch=None, published=None):
    """Widen the it13 checkpoint, export, check parity, record SHA-256s.
    Returns the record (also written to MODELS/twin-s-clock-wide.json).
    Raises if any it13 output differs by PARITY_TOLERANCE or more."""
    import tempfile

    from hivemind.architectures.twin_board import load_twin_board_checkpoint
    from hivemind.distill.train import export

    out_dir, models_dir = Path(out_dir), Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    source = torch.load(checkpoint, map_location="cpu")
    narrow, size, narrow_variant = load_twin_board_checkpoint(source, wide=False)
    torch.manual_seed(seed)  # the arrivals head's hidden layer
    wide, _, wide_variant = load_twin_board_checkpoint(source, wide=True)
    narrow.eval()
    wide.eval()
    if wide.planes_per_board * 2 != NUM_WIDE_CHANNELS:
        raise RuntimeError("the widened network does not read the wide layout")

    pt = out_dir / f"{WIDE_NAME}.pt"
    torch.save({"model": wide.state_dict(), "size": size, "variant": wide_variant,
                "widened_from": {"path": str(checkpoint), "sha256": sha256(checkpoint)}}, pt)
    onnx_path, fp32_path = models_dir / f"{WIDE_NAME}.onnx", models_dir / f"{WIDE_NAME}-fp32.onnx"

    planes = parity_inputs(positions, seed)
    report = {"torch": parity(narrow, wide, planes)}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp = Path(tmp)
        export(out_dir, size, checkpoint_name=pt.name, path=onnx_path, fp32_path=tmp / "wide-exact.onnx")
        _convert_fp32(onnx_path, fp32_path)
        # it13 exported by the same exporter, FP32, as the ONNX reference.
        torch.save({"model": narrow.state_dict(), "size": size, "variant": narrow_variant}, tmp / "it13.pt")
        export(tmp, size, checkpoint_name="it13.pt", path=tmp / "it13.onnx", fp32_path=tmp / "it13-fp32.onnx")
        report["onnx_fp32"] = onnx_parity(tmp / "it13-fp32.onnx", tmp / "wide-exact.onnx", planes)
    for name, (reference, ours) in (published or {}).items():
        report[f"published_{name}"] = onnx_parity(reference, {"fp16": onnx_path, "fp32": fp32_path}[ours], planes)
    record = {
        "network": WIDE_NAME, "widened_from": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "size": size, "variant": wide_variant, "positions": positions, "seed": seed,
        "tolerance": PARITY_TOLERANCE, "parity": report,
        "files": {path.name: sha256(path) for path in (pt, onnx_path, fp32_path)},
    }
    (models_dir / f"{WIDE_NAME}.json").write_text(json.dumps(record, indent=1) + "\n")
    worst = max(value for part in report.values() for name, value in part.items() if name in OUTPUTS)
    if worst >= PARITY_TOLERANCE or any(part["arrivals_max_abs"] != 0 for part in report.values()):
        raise RuntimeError(f"widened network differs from it13: {report}")
    return record
