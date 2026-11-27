"""Spec v7 section 6 (WP4): the think_gain head, its trainer term, labels and rl-loop hooks.

Chunk tests use the int-shared fixture (runs/wave2/int-shared of the talisman repo, or $HIVEMIND_INT_SHARED)
read-only and skip without it; anything written goes to tmp_path.
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind import think
from hivemind.architectures.twin_board import get_twin_board_model, load_twin_board_checkpoint
from hivemind.distill import train
from hivemind.distill.data import THINK_LABELS_NAME, DistillDataset, chunk_positions, read_chunk

FORK = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get("HIVEMIND_INT_SHARED", FORK.parents[1] / "runs" / "wave2" / "int-shared"))
SEARCH = FIXTURE / "runs" / "rl-it1" / "search"
TINY = {"channels": 16, "blocks": 2, "attention_dim": 0, "se_every": 0}


def fixture_chunks():
    paths = sorted(SEARCH.glob("train/seg*/*.dst"))
    if not paths:
        pytest.skip(f"no int-shared fixture at {FIXTURE}")
    return paths


def tiny(seed=0, **variant):
    torch.manual_seed(seed)
    return get_twin_board_model("twin-s", **TINY, **variant).eval()


def planes(model, count=4, seed=0):
    return torch.rand(count, 2 * model.planes_per_board, 8, 8, generator=torch.Generator().manual_seed(seed))


def save_checkpoint(path, model, variant):
    torch.save({"model": model.state_dict(), "size": "twin-s", "variant": variant}, path)
    return path


# --- the head ----------------------------------------------------------------------------------

def test_head_shapes_and_order():
    plain = tiny()
    assert plain.think_gain_head is None and len(plain(planes(plain))) == 5
    headed = tiny(think_gain=True)
    outputs = headed(planes(headed))
    assert len(outputs) == 6 and outputs[-1].shape == (4, 2)
    both = tiny(think_gain=True, arrivals=True)
    outputs = both(planes(both))
    assert len(outputs) == 7 and outputs[5].shape == (4, 20) and outputs[6].shape == (4, 2)


def test_a_fresh_head_says_one_third_and_swapping_boards_swaps_its_logits():
    headed = tiny(think_gain=True)
    x = planes(headed)
    with torch.no_grad():
        logits = headed(x)[-1]
    assert torch.allclose(logits, torch.full_like(logits, -math.log(2.0)))
    assert torch.allclose(torch.sigmoid(logits), torch.full_like(logits, 1 / 3))
    with torch.no_grad():
        torch.nn.init.normal_(headed.think_gain_head.body[-1].weight)
        logits = headed(x)[-1]
        swapped = headed(torch.cat(x.split(headed.planes_per_board, dim=1)[::-1], dim=1))[-1]
    assert torch.allclose(logits, swapped.flip(1), atol=1e-5)


def test_zero_start_parity_every_other_output_identical():
    plain = tiny()
    headed = tiny(think_gain=True)
    missing, unexpected = headed.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(key.startswith("think_gain_head.") for key in missing)
    x = planes(plain)
    with torch.no_grad():
        a, b = plain(x), headed(x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1][0], b[1][0]) and torch.equal(a[1][1], b[1][1])
    assert all(torch.equal(a[i], b[i]) for i in (2, 3, 4))


def test_loader_adds_keeps_and_drops_the_head(tmp_path):
    plain = tiny()
    variant = {**TINY}
    source = save_checkpoint(tmp_path / "plain.pt", plain, variant)
    model, size, loaded = load_twin_board_checkpoint(source)
    assert model.think_gain_head is None and "think_gain" not in loaded  # defaults unchanged
    model, _, loaded = load_twin_board_checkpoint(source, think_gain=True)
    assert model.think_gain_head is not None and loaded["think_gain"] is True
    x = planes(plain)
    with torch.no_grad():
        assert torch.equal(model.eval()(x)[0], plain(x)[0])
    headed = save_checkpoint(tmp_path / "headed.pt", model, loaded)
    kept, _, kept_variant = load_twin_board_checkpoint(headed)
    assert kept.think_gain_head is not None and kept_variant["think_gain"] is True
    assert torch.equal(kept.think_gain_head.body[0].weight, model.think_gain_head.body[0].weight)
    dropped, _, dropped_variant = load_twin_board_checkpoint(headed, think_gain=False)
    assert dropped.think_gain_head is None and "think_gain" not in dropped_variant
    # Anything but the think_gain (or arrivals) head missing is still an error.
    state = {k: v for k, v in plain.state_dict().items() if not k.startswith("value_head.")}
    torch.save({"model": state, "size": "twin-s", "variant": variant}, tmp_path / "broken.pt")
    with pytest.raises(RuntimeError):
        load_twin_board_checkpoint(tmp_path / "broken.pt", think_gain=True)


# --- the trainer term --------------------------------------------------------------------------

def test_masked_bce_no_labels_no_loss_and_gradient_only_on_labelled_boards():
    logits = torch.zeros(3, 2, requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mask = torch.tensor([[True, False], [False, False], [True, True]])
    loss, count = train.think_gain_loss(logits, target, mask)
    assert count.tolist() == [1, 0, 2]
    assert loss[1] == 0
    assert torch.allclose(loss[0], torch.tensor(math.log(2.0)))
    (loss * count).sum().backward()
    assert logits.grad[0, 1] == 0 and (logits.grad[1] == 0).all()
    assert logits.grad[0, 0] != 0 and (logits.grad[2] != 0).all()


def search_batch(paths, rows=64):
    dataset = DistillDataset([read_chunk(path) for path in paths], torch.device("cpu"), source=1)
    return dataset.batch(torch.arange(min(rows, len(dataset))))


WEIGHTS = {"raw": {k: 1.0 for k in train.TERMS},
           "search": {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 0.0, "moves_left": 0.1,
                      "arrivals": 0.05, "think_gain": 0.1}}


def test_unlabelled_batches_leave_the_loss_unchanged():
    batch = search_batch(fixture_chunks()[:1])
    assert batch["think_gain"].shape == batch["has_think_gain"].shape == (len(batch["value"]), 2)
    assert not batch["has_think_gain"].any()
    plain = tiny()
    headed = tiny(think_gain=True)
    headed.load_state_dict(plain.state_dict(), strict=False)
    a, terms_a, _ = train.losses(plain, batch, WEIGHTS)
    b, terms_b, _ = train.losses(headed, batch, WEIGHTS)
    assert torch.equal(a, b) and terms_b["think_gain"] == 0 and "think_gain" not in terms_a
    # With labels the term trains the head alone among the heads' new parameters.
    batch["has_think_gain"][:8] = True
    batch["think_gain"][:8] = 1.0
    total, terms, _ = train.losses(headed, batch, WEIGHTS)
    assert terms["think_gain"] > 0 and total > a
    names = [name for name, _ in headed.named_parameters()]
    assert any(name.startswith("think_gain_head.") for name in names)
    groups = train.parameter_groups(headed, 1e-3, 1e-4)
    heads = {id(p) for p in groups[1]["params"]}
    assert all(id(p) in heads for p in headed.think_gain_head.parameters())


# --- labels next to the chunks -----------------------------------------------------------------

@pytest.fixture
def segment(tmp_path):
    chunks = fixture_chunks()
    source = chunks[0].parent
    paths = sorted(source.glob("*.dst"))
    if len(paths) < 2:
        paths = chunks[:2]
    target = tmp_path / "seg00"
    target.mkdir()
    for k, path in enumerate(paths[:2]):
        shutil.copyfile(path, target / f"search_{k:05d}.dst")
    return target


def test_npz_attaches_by_record_index_in_chunk_order(segment):
    paths = sorted(segment.glob("*.dst"))
    first = chunk_positions(paths[0])
    plain = [read_chunk(path) for path in paths]
    assert all("think_gain" not in chunk for chunk in plain)
    index = np.array([0, first - 1, first, first + 2])
    label = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], np.float32)
    mask = np.array([[True, False], [True, True], [False, True], [True, True]])
    np.savez(segment / THINK_LABELS_NAME, record_index=index, label=label, mask=mask)
    a, b = (read_chunk(path) for path in paths)
    assert a["has_think_gain"][0].tolist() == [True, False] and a["think_gain"][0].tolist() == [1, 0]
    assert a["has_think_gain"][first - 1].tolist() == [True, True]
    assert b["has_think_gain"][0].tolist() == [False, True] and b["think_gain"][0].tolist() == [0, 1]
    assert b["has_think_gain"][2].tolist() == [True, True] and b["think_gain"][2].tolist() == [0, 0]
    assert a["has_think_gain"].sum() + b["has_think_gain"].sum() == mask.sum()
    # Everything else is what it was.
    for old, new in zip(plain, (a, b)):
        assert all(np.array_equal(old[key], new[key]) for key in old)
    data = DistillDataset([a, plain[1]], torch.device("cpu"))
    batch = data.batch(torch.arange(len(data)))
    assert batch["has_think_gain"].sum() == 3  # the second chunk read without labels


def test_bad_record_index_is_an_error(segment):
    total = sum(chunk_positions(p) for p in segment.glob("*.dst"))
    np.savez(segment / THINK_LABELS_NAME, record_index=np.array([total]), label=np.zeros((1, 2), np.float32),
             mask=np.ones((1, 2), bool))
    with pytest.raises(ValueError):
        read_chunk(sorted(segment.glob("*.dst"))[0])


def test_a_headless_checkpoint_trains_with_think_weight_on_fake_labels(segment, tmp_path, monkeypatch):
    paths = sorted(segment.glob("*.dst"))
    total = sum(chunk_positions(p) for p in paths)
    rng = np.random.default_rng(0)
    index = np.arange(0, total, 2)
    np.savez(segment / THINK_LABELS_NAME, record_index=index,
             label=rng.integers(0, 2, (len(index), 2)).astype(np.float32), mask=rng.random((len(index), 2)) < 0.7)
    init = save_checkpoint(tmp_path / "init.pt", tiny(), {**TINY})
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "distill-train", "--init", str(init), "--search-data", str(segment), "--search-val", str(paths[0]),
        "--search-fraction", "1", "--batch-size", "32", "--steps", "3", "--eval-every", "3",
        "--train-probe", "0", "--no-compile", "--think-weight", "0.1", "--lr", "0.01",
        "--out", str(tmp_path / "out")])
    train.main()
    saved = torch.load(tmp_path / "out" / "last.pt", map_location="cpu", weights_only=True)
    assert saved["variant"]["think_gain"] is True
    bias = saved["model"]["think_gain_head.body.2.bias"]
    assert not torch.allclose(bias, torch.tensor([-math.log(2.0)]))
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert "search_think_gain" in history[-1]


def test_think_weight_zero_trains_no_head(segment, tmp_path, monkeypatch):
    init = save_checkpoint(tmp_path / "init.pt", tiny(), {**TINY})
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "distill-train", "--init", str(init), "--search-data", str(segment),
        "--search-val", str(sorted(segment.glob("*.dst"))[0]), "--search-fraction", "1", "--batch-size", "32",
        "--steps", "1", "--eval-every", "1", "--train-probe", "0", "--no-compile", "--out", str(tmp_path / "out")])
    train.main()
    saved = torch.load(tmp_path / "out" / "last.pt", map_location="cpu", weights_only=True)
    assert "think_gain" not in saved["variant"]
    assert not any(key.startswith("think_gain_head.") for key in saved["model"])


# --- labels from (fake) reanalyse --------------------------------------------------------------

def test_make_labels_differs_same_pass_masked():
    base = {3: (10, 20), 7: (11, None), 9: (0, 5), 12: (None, None)}
    deep = {3: (10, 21), 7: (12, None), 9: (0, 5), 12: (None, None)}
    index, label, mask = think.make_labels(base, deep)
    assert index.tolist() == [3, 7, 9, 12]
    assert label.tolist() == [[0, 1], [1, 0], [0, 0], [0, 0]]
    assert mask.tolist() == [[True, True], [True, False], [True, True], [False, False]]
    # A pass is an entry like a move: pass (index 0 here) against a move differs.
    _, label, _ = think.make_labels({1: (0, 4)}, {1: (6, 4)})
    assert label.tolist() == [[1, 0]]
    assert think.disagreement(base, {3: (10, 22), 7: (11, 3)}) == (3, 1)


def test_top_entries_reads_the_most_visited_entry():
    chunk = {"policy_count": np.array([[2, 0], [1, 3]], np.uint16),
             "policy_index": np.array([5, 9, 0, 7, 8, 4], np.uint16),
             "policy_probability": np.array([0.3, 0.7, 1.0, 0.2, 0.5, 0.3], np.float16)}
    assert think.top_entries(chunk) == [(9, None), (0, 8)]


def write_states(path, total, games=3):
    """A state dump in the engine's shape (header, game lines, record lines) for `total` records."""
    lines = [json.dumps({"type": "header", "nodes": 800})]
    per_game = -(-total // games)
    index = 0
    for game in range(games):
        lines.append(json.dumps({"type": "game", "game": game}))
        for record in range(min(per_game, total - index)):
            nodes = 0 if index % 17 == 5 else 800 + index
            lines.append(f'{{"type": "record", "game": {game}, "record": {record}, "target_nodes": {nodes}}}')
            index += 1
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def iteration(tmp_path, segment):
    it = tmp_path / "runs" / "rl-it2"
    data = it / "search" / "train" / "seg00"
    data.parent.mkdir(parents=True)
    shutil.copytree(segment, data)
    total = sum(chunk_positions(p) for p in data.glob("*.dst"))
    records = it / "search" / "train_run" / "seg00"
    records.mkdir(parents=True)
    write_states(records / "states.jsonl", total)
    (it / "stages").mkdir()
    (it / "stages" / "seg00.done").write_text(json.dumps({"games": 3}))
    # A second segment without a dump is skipped.
    (it / "stages" / "seg01.done").write_text("{}")
    (it / "search" / "train" / "seg01").mkdir()
    model = tmp_path / "gen.onnx"
    model.write_bytes(b"onnx")
    return it, model, total


def test_labels_on_a_fake_reanalyse(iteration, monkeypatch, capsys):
    it, model, total = iteration
    calls = []

    def fake_search(job, dump, indices, offsets, wave2_args, root, prefix):
        calls.append((prefix, sorted(indices), [dump.records[i] for i in sorted(indices)], job))
        tops = {}
        for i in indices:
            on_b = i % 3 != 0
            if prefix == "deep":
                tops[i] = (i % 5 if i % 2 else 100, (i % 7) if on_b else None)
            else:
                tops[i] = (i % 5, (i % 7 if prefix == "base" or i % 4 else 99) if on_b else None)
        return tops, {"gpu_s": 1.5}

    monkeypatch.setattr(think, "search", fake_search)
    code = think.main(["labels", "--it", str(it), "--iteration", "2", "--model", str(model), "--engine", "hm",
                       "--fraction", "0.5", "--factor", "4"])
    assert code == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(result) == {"records", "labelled", "rate_a", "rate_b", "noise_rate", "gpu_s", "segments"}
    assert result["records"] == total and result["segments"] == 1
    prefixes = [c[0] for c in calls]
    assert prefixes[:2] == ["base", "deep"]
    chosen = calls[0][1]
    assert chosen == calls[1][1] and 0 < len(chosen) < total
    # Raw-policy records (target_nodes 0) are never chosen; the deep search asks for 4x the budget.
    for base_line, deep_line in zip(calls[0][2], calls[1][2]):
        nodes = json.loads(base_line)["target_nodes"]
        assert nodes > 0 and json.loads(deep_line)["target_nodes"] == 4 * nodes
    assert calls[0][3].nodes == 0 and calls[0][3].dirichlet is False and calls[0][3].replay_noise is False
    with np.load(it / "search" / "train" / "seg00" / THINK_LABELS_NAME) as saved:
        index, label, mask = saved["record_index"], saved["label"], saved["mask"]
    assert index.tolist() == chosen
    assert mask[:, 0].all()
    assert mask[:, 1].tolist() == [i % 3 != 0 for i in chosen]
    assert label[:, 0].tolist() == [float(i % 2 == 0 and i % 5 != 100) for i in chosen]
    assert result["rate_a"] == pytest.approx(label[:, 0].mean(), abs=1e-6)
    assert result["rate_b"] == 0
    assert len(calls) == 3  # about 5% of ~300 chosen records are repeated
    if len(calls) == 3:
        assert calls[2][0] == "repeat" and set(calls[2][1]) <= set(chosen)
        repeat = calls[2][1]
        compared = sum(1 + (i % 3 != 0) for i in repeat)
        differ = sum(i % 3 != 0 and i % 4 == 0 for i in repeat)
        assert result["noise_rate"] == pytest.approx(differ / compared, abs=1e-6)
        assert result["gpu_s"] == 4.5
    # The chunks now carry the labels.
    chunk = read_chunk(sorted((it / "search" / "train" / "seg00").glob("*.dst"))[0])
    assert chunk["has_think_gain"].any()
    # A rerun keeps them and searches nothing.
    calls.clear()
    assert think.main(["labels", "--it", str(it), "--iteration", "2", "--model", str(model), "--engine", "hm",
                       "--fraction", "0.5"]) == 0
    assert calls == []


# --- add-head ----------------------------------------------------------------------------------

def test_add_head_round_trip_and_onnx_output(tmp_path):
    pytest.importorskip("onnxsim")
    ort = pytest.importorskip("onnxruntime")
    source = save_checkpoint(tmp_path / "tiny.pt", tiny(), {**TINY})
    record = think.add_head(source, tmp_path / "out", positions=6)
    for part in ("torch", "onnx"):
        assert all(record["parity"][part][name] == 0 for name in think.PARITY_OUTPUTS)
    assert record["parity"]["onnx"]["think_gain_shape"] == [6, 2]
    onnx_path = tmp_path / "out" / "tiny-think.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()][-1] == "think_gain"
    model, _, variant = load_twin_board_checkpoint(tmp_path / "out" / "tiny-think.pt")
    assert variant["think_gain"] is True and model.think_gain_head is not None
    assert json.loads((tmp_path / "out" / "tiny-think.json").read_text())["files"]


# --- rl-loop hooks -----------------------------------------------------------------------------

def hook_args(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-games", type=int, default=6)
    think.add_arguments(parser)
    args = parser.parse_args(list(argv))
    think.validate(args, parser)
    return args


def test_hooks_are_off_by_default():
    args = hook_args()
    assert think.selfplay_args(args) == []
    assert think.label_command(args, 3, Path("it"), Path("g.onnx"), Path("hm")) is None
    assert think.trainer_args(args) == []
    # An old namespace without the flags (a worker) is off too.
    empty = argparse.Namespace()
    assert think.selfplay_args(empty) == [] and think.trainer_args(empty) == []
    assert think.label_command(empty, 1, Path("it"), Path("g"), Path("e")) is None


def test_hooks_when_on():
    args = hook_args("--think-labels", "--think-policy", "--parallel-games", "4")
    assert think.selfplay_args(args) == ["--record-state-output", "{segment_dir}/states.jsonl",
                                         "--think-policy", "true"]
    assert think.trainer_args(args) == ["--think-weight", "0.1"]
    command = think.label_command(args, 3, Path("/r/rl-it3"), Path("/g.onnx"), Path("/hm"))
    assert command[:4] == [sys.executable, "-m", "hivemind.think", "labels"]
    assert command[command.index("--iteration") + 1] == "3"
    assert command[command.index("--fraction") + 1] == "0.02"
    assert command[command.index("--factor") + 1] == "4"
    assert command[command.index("--parallel-engines") + 1] == "4"
    assert think.selfplay_args(hook_args("--think-policy")) == ["--think-policy", "true"]
    assert think.trainer_args(hook_args("--think-labels", "--think-weight", "0.3")) == ["--think-weight", "0.3"]


@pytest.mark.parametrize("argv", [["--think-weight", "0.1"], ["--think-labels", "--think-label-fraction", "0"],
                                  ["--think-labels", "--think-factor", "1"],
                                  ["--think-labels", "--think-weight", "-1"]])
def test_hook_validation(argv):
    with pytest.raises(SystemExit):
        hook_args(*argv)
