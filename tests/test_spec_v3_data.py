"""Spec v3 WP-D data pipeline: exact clock scalars from the recorder sidecar, sample weights, q-value blend,
opening-pair split, train/val plane dedupe, end-reason report, arrivals end-of-game mask, multi-iteration replay."""

import json
import math
import struct

import chess
import numpy as np
import polars as pl
import pytest
import torch

from hivemind.data.convert_selfplay_data import (
    AUX_LABELS,
    HEADER,
    JOINT_POLICY_ENTRY,
    POLICY_ENTRY,
    SAMPLE_METADATA,
    convert_to_split_parquet,
    is_validation_key,
    read_binary_shard,
    sample_weight,
)
from hivemind.domain.board2planes import _fill_clock_planes, _fill_elapsed_planes
from hivemind.training.data_loaders import StreamingRLDataset, load_rl_parquet_shard

CH = 86
ARR = [1.0] + [0.0] * 19


def _sample(game_id, *, team=0, outcome=1, root_q=0.5, nodes=100, planes=None, seed=0):
    rng = np.random.default_rng(seed)
    out = [SAMPLE_METADATA.pack(game_id, nodes, 1, 20, team, 0, outcome, outcome + 1, root_q)]
    out.append(planes if planes is not None else rng.integers(0, 256, size=CH * 64, dtype=np.uint8).tobytes())
    for _ in range(2):
        out.append(struct.pack("<H", 1) + POLICY_ENTRY.pack(int(rng.integers(0, 4672)), np.float32(1.0)))
    out.append(struct.pack("<H", 1) + JOINT_POLICY_ENTRY.pack(4672, 12, np.float32(1.0)))
    out.append(AUX_LABELS.pack(0.0, 0.0, 0, 0, *ARR))
    return b"".join(out)


def _chunk(path, samples, metas=None):
    with path.open("wb") as f:
        f.write(HEADER.pack(b"HVM6", 6, CH, 4672, len(samples)))
        for s in samples:
            f.write(s)
    if metas is not None:
        with open(str(path) + ".pos.jsonl", "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")


def _meta(game_id, opening, reason="checkmate", t=0, end=60000, **extra):
    return {"gameId": game_id, "t": t, "opening": opening, "reason": reason, "gameEndMs": end,
            "arrivalWindowMs": 5000, "clocks": {"A_white": 60000, "A_black": 60000, "B_white": 60000,
                                                "B_black": 60000}, "elapsed": {"A": 0, "B": 0}, **extra}


def test_loader_rebuilds_clock_planes_from_sidecar_to_1e6(tmp_path):
    src = tmp_path / "training_data"
    src.mkdir()
    clocks = [(59999.5, 31234.25, 1234.0, 70.5), (12.0, 299999.0, 45678.9, 61999.0)]
    elapsed = [(499.5, 12345.0), (0.0, 1999.0)]
    samples, metas = [], []
    for i in range(40):  # many games: whichever split each lands in, both splits get samples
        c, e = clocks[i % 2], elapsed[i % 2]
        samples.append(_sample(i, team=i % 2, seed=i))
        metas.append(_meta(i, f"op{i}", clocks_ms=list(c), elapsed_ms=list(e)))
    _chunk(src / "chunk_r_000000.hvm", samples, metas)
    train, val, _, _ = convert_to_split_parquet(src, tmp_path / "rl", validation_fraction=0.3, split_seed=1)
    for d in (train, val):
        for shard in sorted(d.glob("*.parquet")):
            df = pl.read_parquet(shard)
            x = load_rl_parquet_shard(shard)[0]
            raw = load_rl_parquet_shard(shard, exact_clocks=False)[0]
            for row, gid in enumerate(df["game_id"].to_list()):
                expect = np.zeros((CH, 8, 8))
                team_side = chess.WHITE if gid % 2 == 0 else chess.BLACK
                _fill_clock_planes(expect, CH // 2, team_side, clocks[gid % 2])
                _fill_elapsed_planes(expect, CH // 2, elapsed[gid % 2])
                for c in [*range(37, 43), *range(80, 86)]:
                    np.testing.assert_allclose(x[row, c].numpy(), expect[c], rtol=0, atol=1e-6)
                other = [c for c in range(CH) if c not in (*range(37, 43), *range(80, 86))]
                assert torch.equal(x[row, other], raw[row, other])


def test_v2_sidecar_clocks_are_used_when_exact_fields_are_missing(tmp_path):
    path = tmp_path / "chunk_r_000000.hvm"
    m = _meta(1, "o")
    m["clocks"]["B_black"] = 1500
    _chunk(path, [_sample(1)], [m])
    (s,) = read_binary_shard(path)
    assert s["clock_scalars_mask"] == 3
    assert np.frombuffer(s["clocks_ms"], np.float32).tolist() == [60000, 60000, 60000, 1500]


def test_weight_column_and_value_blend(tmp_path):
    path = tmp_path / "chunk_r_000000.hvm"
    _chunk(path, [_sample(1, nodes=100, root_q=0.5, outcome=1), _sample(1, nodes=900, root_q=-0.5, outcome=-1)])
    a, b = read_binary_shard(path)  # 86 channels: clock-build defaults
    assert a["sample_weight"] == pytest.approx(0.25) and b["sample_weight"] == 1.0  # N_ref 400
    assert a["y_value"] == pytest.approx(0.8 * 1 + 0.2 * 0.5) and b["y_value"] == pytest.approx(-0.9)
    a, _ = read_binary_shard(path, q_value_ratio=0.0, weight_nodes_ref=0)
    assert a["y_value"] == 1.0 and a["sample_weight"] == 1.0
    assert sample_weight(50, 200) == 0.25 and sample_weight(5000, 200) == 1.0 and sample_weight(7, 0) == 1.0


def test_nodes_searched_from_sidecar_sets_the_weight(tmp_path):
    path = tmp_path / "chunk_r_000000.hvm"
    _chunk(path, [_sample(1, nodes=100)], [_meta(1, "o", nodes_searched=200)])
    assert read_binary_shard(path)[0]["sample_weight"] == pytest.approx(0.5)


def test_split_by_opening_and_no_plane_in_both_splits(tmp_path):
    src = tmp_path / "training_data"
    src.mkdir()
    shared = np.random.default_rng(9).integers(0, 256, size=CH * 64, dtype=np.uint8).tobytes()
    samples, metas = [], []
    for g in range(60):
        opening = f"pair{g % 12}"
        samples.append(_sample(g, seed=g))
        metas.append(_meta(g, opening))
        samples.append(_sample(g, planes=shared))  # the same position in every game
        metas.append(_meta(g, opening))
    _chunk(src / "chunk_r_000000.hvm", samples, metas)
    train, val, n_train, n_val = convert_to_split_parquet(src, tmp_path / "rl", validation_fraction=0.3,
                                                          split_seed=3)
    tr = pl.concat([pl.read_parquet(p) for p in train.glob("*.parquet")])
    va = pl.concat([pl.read_parquet(p) for p in val.glob("*.parquet")])
    assert set(tr["opening"]) & set(va["opening"]) == set()
    assert all(is_validation_key(o, 0.3, 3) for o in va["opening"])
    assert set(tr["plane_hash"]) & set(va["plane_hash"]) == set()
    report = json.loads((tmp_path / "rl" / "report.json").read_text())
    assert report["split_by"] == "opening" and report["val_samples_dropped_as_duplicates"] > 0
    assert report["train_cross_game_duplicate_rate"] > 0.4  # half the train samples repeat across games
    assert (n_train, n_val) == (tr.height, va.height)


def test_report_by_end_reason_and_non_training_games_dropped(tmp_path):
    src = tmp_path / "training_data"
    src.mkdir()
    rows = [  # (game, reason, outcome, root_q)
        (1, "flag", 1, 0.2), (1, "flag", -1, 0.0), (2, "checkmate", 1, 0.9), (3, "max_plies", 0, 0.1),
        (4, "aborted", 0, 0.0), (5, "checkmate", -1, -0.7),
    ]
    samples = [_sample(g, outcome=o, root_q=q, seed=i) for i, (g, _, o, q) in enumerate(rows)]
    metas = [_meta(g, f"o{g}", reason=r) for g, r, _, _ in rows]
    _chunk(src / "chunk_r_000000.hvm", samples, metas)
    convert_to_split_parquet(src, tmp_path / "rl", validation_fraction=0.5, split_seed=2, split_by="game")
    rep = json.loads((tmp_path / "rl" / "report.json").read_text())["samples_by_reason"]
    assert set(rep) == {"flag", "checkmate"}  # max_plies / aborted kept out
    assert rep["flag"]["samples"] == 2 and rep["flag"]["games"] == 1
    assert rep["flag"]["mean_abs_root_q_minus_outcome"] == pytest.approx((0.8 + 1.0) / 2)
    assert rep["checkmate"]["mean_abs_root_q_minus_outcome"] == pytest.approx((0.1 + 0.3) / 2)


def test_arrivals_masked_when_window_passes_game_end(tmp_path):
    path = tmp_path / "chunk_r_000000.hvm"
    _chunk(path, [_sample(1), _sample(1)], [_meta(1, "o", t=1000, end=9000), _meta(1, "o", t=5000, end=9000)])
    a, b = read_binary_shard(path, mask_arrivals_past_end=True)
    assert a["arrivals_mask"] == 1 and b["arrivals_mask"] == 0
    assert [s["arrivals_mask"] for s in read_binary_shard(path)] == [1, 1]  # the reader alone keeps labels


def test_replay_from_several_iterations(tmp_path):
    src = tmp_path / "it03"
    src.mkdir()
    _chunk(src / "chunk_r3_000000.hvm", [_sample(g, seed=g) for g in range(20)],
           [_meta(g, f"o{g}") for g in range(20)])
    replay = []
    for it in (0, 1, 2):
        d = tmp_path / f"it0{it}"
        d.mkdir()
        _chunk(d / f"chunk_r{it}_000000.hvm", [_sample(100 + it, seed=50 + it)], [_meta(100 + it, "x")])
        replay.append(d)
    _, _, n_train, n_val = convert_to_split_parquet(src, tmp_path / "rl", validation_fraction=0.3, split_seed=1,
                                                    replay_input_dir=replay)
    assert n_train + n_val == 23
    report = json.loads((tmp_path / "rl" / "report.json").read_text())
    assert report["replay_chunks"] == 3 and len(report["replay_dirs"]) == 3


def test_stock_data_keeps_the_old_defaults(tmp_path):
    path = tmp_path / "chunk_r_000000.hvm"
    with path.open("wb") as f:
        f.write(HEADER.pack(b"HVM5", 5, 74, 4672, 1))
        f.write(SAMPLE_METADATA.pack(1, 10, 1, 20, 0, 0, 1, 2, -0.5) + bytes(74 * 64))
        for _ in range(2):
            f.write(struct.pack("<H", 1) + POLICY_ENTRY.pack(3, np.float32(1.0)))
        f.write(struct.pack("<H", 0))
    (s,) = read_binary_shard(path)
    assert s["y_value"] == 1.0 and s["sample_weight"] == 1.0 and s["clock_scalars_mask"] == 0


def test_sample_weights_reach_the_streaming_loader(tmp_path):
    src = tmp_path / "training_data"
    src.mkdir()
    _chunk(src / "chunk_r_000000.hvm", [_sample(g, nodes=40 * (g + 1), seed=g) for g in range(30)],
           [_meta(g, f"o{g}") for g in range(30)])
    train, _, n_train, _ = convert_to_split_parquet(src, tmp_path / "rl", validation_fraction=0.3, split_seed=1)
    shard = sorted(train.glob("*.parquet"))[0]
    loaded = load_rl_parquet_shard(shard, include_aux_labels=True, include_sample_weights=True)
    assert len(loaded) == 11
    gids = pl.read_parquet(shard)["game_id"].to_list()
    assert loaded[-1].tolist() == pytest.approx([min(1.0, 40 * (g + 1) / 400) for g in gids])
    ds = StreamingRLDataset([str(shard)], shuffle_files=False, augment_flip=True, shuffle_samples=False,
                            include_aux_labels=True, include_sample_weights=True)
    items = list(ds)
    assert len(items) == 2 * n_train and all(len(item) == 11 for item in items)
    assert float(items[0][-1]) == float(items[1][-1]) == pytest.approx(min(1.0, 40 * (gids[0] + 1) / 400))
    assert not math.isnan(float(items[0][-1]))
