"""Search balance trainer support: --search-q-blend and policy_positions.

Uses the int-shared fixture (runs/wave2/int-shared of the talisman repo, or
$HIVEMIND_INT_SHARED) read-only; anything rewritten goes to tmp_path.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.architectures.twin_board import get_twin_board_model
from hivemind.cli import train as cli_train
from hivemind.distill import train
from hivemind.distill.data import HEADER_BYTES, MAGIC, DistillDataset, chunk_problem, read_chunk

FORK = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get("HIVEMIND_INT_SHARED", FORK.parents[1] / "runs" / "wave2" / "int-shared"))
SEARCH = FIXTURE / "runs" / "rl-it1" / "search"


def fixture_chunks():
    paths = sorted(SEARCH.glob("train/seg*/*.dst"))
    if not paths:
        pytest.skip(f"no int-shared fixture at {FIXTURE}")
    return paths


@pytest.fixture(scope="module")
def head_train(tmp_path_factory):
    """distill/train.py as committed (before --search-q-blend), as a module."""
    try:
        source = subprocess.run(["git", "show", "HEAD:src/hivemind/distill/train.py"], cwd=FORK,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the fork's git HEAD is not available")
    path = tmp_path_factory.mktemp("head") / "head_train.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("head_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(seed=0):
    torch.manual_seed(seed)
    return get_twin_board_model("twin-s").eval()


def search_batch(paths, rows=None):
    dataset = DistillDataset([read_chunk(path) for path in paths], torch.device("cpu"), source=1)
    index = torch.arange(len(dataset) if rows is None else min(rows, len(dataset)))
    return dataset.batch(index)


WEIGHTS = {"raw": {k: 1.0 for k in train.TERMS},
           "search": {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 0.0, "moves_left": 0.1,
                      "arrivals": 0.05}}


# --- lambda = 0 is today's trainer, bit for bit ------------------------------------------------

def test_lambda_zero_losses_and_gradients_are_bit_identical(head_train):
    batch = search_batch(fixture_chunks()[:2], rows=256)
    results = []
    for losses, kwargs in ((head_train.losses, {}), (train.losses, {}), (train.losses, {"q_blend": 0.0})):
        net = model()
        total, terms, _ = losses(net, batch, WEIGHTS, **kwargs)
        total.backward()
        results.append((total.detach(), {k: v.detach() for k, v in terms.items()},
                        [p.grad.clone() for p in net.parameters() if p.grad is not None]))
    reference = results[0]
    for total, terms, grads in results[1:]:
        assert torch.equal(total, reference[0])
        assert terms.keys() == reference[1].keys()
        assert all(torch.equal(terms[k], reference[1][k]) for k in terms)
        assert len(grads) == len(reference[2]) and all(torch.equal(a, b) for a, b in zip(grads, reference[2]))
    assert train.wdl_target(batch, 0.0) is batch["wdl"]


def test_lambda_zero_training_run_is_bit_identical(head_train, tmp_path, monkeypatch):
    """A whole distill-train run (3 steps, CPU) with --search-q-blend 0 and
    without the flag gives the same weights and history as the committed code."""
    chunks = fixture_chunks()
    val = sorted(SEARCH.glob("val/*.dst"))[:1] or chunks[-1:]
    common = ["--search-data", *map(str, chunks[:2]), "--search-val", *map(str, val),
              "--search-fraction", "1", "--size", "twin-s", "--batch-size", "32", "--steps", "3",
              "--eval-every", "3", "--eval-batch-size", "64", "--train-probe", "0", "--no-compile",
              "--search-value-weight", "0", "--search-wdl-weight", "1", "--search-moves-left-weight", "0.1",
              "--seed", "5", "--export", "last"]
    runs = {"head": (head_train, []), "new": (train, []), "new0": (train, ["--search-q-blend", "0"])}
    for name, (module, flags) in runs.items():
        monkeypatch.setattr(module, "export", lambda *args, **kwargs: None)
        monkeypatch.setattr(sys, "argv", ["distill-train", *common, *flags, "--out", str(tmp_path / name)])
        module.main()
    reference = torch.load(tmp_path / "head" / "last.pt", map_location="cpu", weights_only=True)["model"]
    history = [{k: v for k, v in h.items() if k != "seconds"}
               for h in json.loads((tmp_path / "head" / "history.json").read_text())]
    for name in ("new", "new0"):
        state = torch.load(tmp_path / name / "last.pt", map_location="cpu", weights_only=True)["model"]
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[k], reference[k]) for k in state)
        assert [{k: v for k, v in h.items() if k != "seconds"}
                for h in json.loads((tmp_path / name / "history.json").read_text())] == history


# --- lambda > 0 --------------------------------------------------------------------------------

def test_q_wdl_orientation_matches_the_engine():
    """Outcome +1 (a win) is WDL index 2; q = +1 is the same certain win."""
    torch.testing.assert_close(train.q_wdl(torch.tensor([1.0, -1.0, 0.0, 0.5])),
                               torch.tensor([[0.0, 0, 1], [1, 0, 0], [0.5, 0, 0.5], [0.25, 0, 0.75]]))
    chunk = read_chunk(fixture_chunks()[0])
    has = chunk["wdl"].sum(1) > 0
    assert (chunk["wdl"][has].argmax(1) == chunk["outcome"][has] + 1).all()


def test_lambda_one_target_is_the_root_q_wdl():
    batch = search_batch(fixture_chunks()[:1])
    rows = train.q_blend_rows(batch)
    assert rows.all()                     # stock fixture: finished games, every record has a policy
    expected = train.q_wdl(batch["value"])
    torch.testing.assert_close(train.wdl_target(batch, 1.0), expected, rtol=0, atol=0)
    half = train.wdl_target(batch, 0.25)
    torch.testing.assert_close(half, 0.75 * batch["wdl"] + 0.25 * expected)
    torch.testing.assert_close(half.sum(1), torch.ones(len(half)))

    net = model()
    with torch.no_grad():
        per_row, _ = train._loss_rows(net, batch, q_blend=1.0)
        logits = net(batch["planes"])[3].float()
    torch.testing.assert_close(per_row["wdl"][0], -(expected * torch.log_softmax(logits, 1)).sum(1))


def test_q_blend_leaves_other_rows_on_the_outcome():
    batch = search_batch(fixture_chunks()[:1], rows=8)
    batch["wdl"][0] = 0                                   # macro-ply limit: no result
    batch["value"][1] = float("nan")                      # no valid root Q
    for board in (0, 1):                                  # a fast (masked) record
        target, legal, mask = batch["policy"][board]
        mask[2] = False
    batch["source"][3] = 0                                # a teacher row
    rows = train.q_blend_rows(batch)
    assert rows.tolist() == [False, False, False, False, True, True, True, True]
    target = train.wdl_target(batch, 0.5)
    assert torch.equal(target[:4], batch["wdl"][:4])
    assert torch.isfinite(target).all()
    assert not torch.equal(target[4:], batch["wdl"][4:])


def test_cli_flag_is_validated_and_trains(tmp_path, monkeypatch):
    chunks = fixture_chunks()
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    base = ["distill-train", "--search-data", str(chunks[0]), "--search-val", str(chunks[1]),
            "--search-fraction", "1", "--size", "twin-s", "--batch-size", "32", "--steps", "2",
            "--eval-every", "2", "--train-probe", "0", "--no-compile", "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*base, "--search-q-blend", "1.5"])
    with pytest.raises(SystemExit):
        train.main()
    monkeypatch.setattr(sys, "argv", [*base, "--search-q-blend", "0.5"])
    train.main()
    assert (tmp_path / "out" / "last.pt").is_file()


# --- masked (playout-cap fast) records and policy_positions -------------------------------------

def write_chunk(path, chunk, version, planes):
    """Re-serialise read_chunk's arrays (on-disk plane layout) as an HDST chunk."""
    n = len(chunk["value"])
    entries = int(chunk["policy_count"].astype(np.int64).sum())
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", version, planes))
        stream.write(struct.pack("<QQ", n, entries))
        for key in ("plane_bits", "plane_values", "value", "wdl", "moves_left", "ply", "outcome",
                    "policy_count", "policy_index", "policy_probability"):
            stream.write(np.ascontiguousarray(chunk[key]).tobytes())
        if version == 3:
            for key in ("seat_clocks_ms", "elapsed_ms", "recent_landed"):
                stream.write(np.ascontiguousarray(chunk[key]).tobytes())
            arrivals = np.where(chunk["has_arrivals"][:, None], chunk["arrivals"], -1).astype(np.float32)
            stream.write(arrivals.tobytes())


def masked_copy(source, target, masked):
    """A copy of a fixture chunk whose `masked` records lost both policies,
    as the engine writes a playout-cap fast record."""
    raw = source.read_bytes()[:HEADER_BYTES]
    version, planes = struct.unpack("<II", raw[4:12])
    chunk = read_chunk(source)
    if chunk["plane_bits"].shape[1] != planes:
        pytest.skip("fixture chunk needs plane conversion")
    counts = chunk["policy_count"].astype(np.int64)
    starts = (np.cumsum(counts.ravel()) - counts.ravel()).reshape(-1, 2)
    keep = np.ones(len(chunk["policy_index"]), bool)
    for record in masked:
        for board in (0, 1):
            keep[starts[record, board]:starts[record, board] + counts[record, board]] = False
    chunk = dict(chunk)
    chunk["policy_count"] = chunk["policy_count"].copy()
    chunk["policy_count"][masked] = 0
    chunk["policy_index"] = chunk["policy_index"][keep]
    chunk["policy_probability"] = chunk["policy_probability"][keep]
    write_chunk(target, chunk, version, planes)
    return read_chunk(target)


@pytest.fixture
def masked_segment(tmp_path):
    source = fixture_chunks()[0]
    segment = tmp_path / "seg00"
    segment.mkdir()
    n = len(read_chunk(source)["value"])
    masked = np.arange(0, n, 3)
    # A round trip without masking reproduces the engine's bytes exactly.
    masked_copy(source, segment / "unchanged.dst", [])
    assert (segment / "unchanged.dst").read_bytes() == source.read_bytes()
    (segment / "unchanged.dst").unlink()
    chunk = masked_copy(source, segment / source.name, masked)
    assert chunk_problem(segment / source.name) is None
    return segment, n, masked


def test_masked_records_train_wdl_and_moves_left_but_not_policy(masked_segment):
    segment, n, masked = masked_segment
    batch = search_batch(sorted(segment.glob("*.dst")))
    is_masked = torch.zeros(n, dtype=torch.bool)
    is_masked[torch.from_numpy(masked)] = True
    per_row, _ = train._loss_rows(model(), batch)
    for board in ("policy_a", "policy_b"):
        loss, valid = per_row[board]
        assert (valid[is_masked] == 0).all()
        assert (loss[is_masked] == 0).all()               # empty target: nothing to fit
    assert (per_row["wdl"][1][is_masked] == 1).all()      # finished games keep their result
    assert (per_row["moves_left"][1][is_masked] == (batch["moves_left"][is_masked] >= 0)).all()
    assert not train.q_blend_rows(batch)[is_masked].any()
    assert train.q_blend_rows(batch)[~is_masked].all()
    # Masked rows count in the WDL mean but not in the policy means.
    _, terms, _ = train.losses(model(), batch, WEIGHTS)
    loss, valid = per_row["policy_a"]
    torch.testing.assert_close(terms["policy_a"], (loss * valid).sum() / valid.sum())
    assert valid.sum() < n and per_row["wdl"][1].sum() == n


def test_policy_positions_counts_policy_bearing_records(masked_segment):
    segment, n, masked = masked_segment
    chunk = next(segment.glob("*.dst"))
    expected = n - len(masked)
    assert cli_train.chunk_policy_positions(chunk) == expected
    # No sidecar (a run without the cap): every record, like positions().
    assert cli_train.policy_positions([segment]) == cli_train.positions([segment]) == n
    (segment / "cap_counts.json").write_text(json.dumps(
        {"records": n, "deep": 5, "masked": len(masked), "deep_visits": 1, "fast_visits": 2}) + "\n")
    assert cli_train.policy_positions([segment]) == expected
    assert cli_train.positions([segment]) == n
    # A sidecar that disagrees with the chunks: count the chunks themselves.
    (segment / "cap_counts.json").write_text(json.dumps({"records": n + 7, "masked": 0}))
    assert cli_train.policy_positions([segment]) == expected
    # A chunk file given directly next to a sidecar.
    assert cli_train.policy_positions([chunk]) == expected
    assert cli_train.STEPS_RULES["positions"] is cli_train.positions
    assert cli_train.STEPS_RULES["policy_positions"] is cli_train.policy_positions
    steps, _ = cli_train.plan(cli_train.policy_positions([segment]), 0, 1.5, 0.5, 32)
    assert steps == -(-int(1.5 * expected) // 32)


def test_policy_positions_equals_positions_on_the_stock_fixture():
    segments = sorted({path.parent for path in fixture_chunks()})
    assert cli_train.policy_positions(segments) == cli_train.positions(segments)
    assert sum(cli_train.chunk_policy_positions(path) for path in fixture_chunks()) == \
        cli_train.positions(segments)


def test_distill_command_adds_q_blend_only_when_non_zero(tmp_path):
    args = (tmp_path / "init.pt", [tmp_path / "d"], [tmp_path / "v"], [], 10, 0.0, 32, 1, tmp_path / "o")
    stock = cli_train.distill_command(*args)
    assert cli_train.distill_command(*args, q_blend=None) == stock
    assert cli_train.distill_command(*args, q_blend=0) == stock
    assert "--search-q-blend" not in stock
    blended = cli_train.distill_command(*args, q_blend=0.25, extra=["--x"])
    assert blended == [*stock, "--search-q-blend", "0.25", "--x"]
