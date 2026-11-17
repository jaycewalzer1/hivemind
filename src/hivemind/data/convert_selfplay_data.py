#!/usr/bin/env python3
"""Convert native HVM3-HVM6 self-play chunks to Parquet training shards.

HVM6 (custom engine spec v2) is HVM5 plus one trailing block per sample.
Chunk layout, little endian, no padding:

    header   '<4sIHHQ'  magic b"HVM6", version 6, input channels, policy values, sample count
    sample   the HVM5 sample unchanged: '<QIHHBBbBf' metadata, planes, policyA,
             policyB, jointPolicy
    trailing '<2f2B20f' (90 bytes):
             think_gain_a, think_gain_b      float32 labels (1 = a 4x longer search
                                             plays a different move on that board)
             think_gain_mask_a, _mask_b      uint8, 1 = label present, 0 = absent
                                             (the float is then ignored, written 0)
             arrivals[20]                    float32 counts of pieces arriving within
                                             5 s, pocket-major: ours A, theirs A,
                                             ours B, theirs B; each P, N, B, R, Q

The trailing block is present exactly when the header version is >= 6 (the
version marker); HVM3-HVM5 chunks are read as before. Every sample becomes one
Parquet row with the columns think_gain_a, think_gain_b, think_gain_mask_a,
think_gain_mask_b, arrivals (20 float32 as bytes) and arrivals_mask; samples
from older chunks carry zero labels and zero masks. An arrivals block that
holds a NaN is treated as absent (arrivals_mask 0).

Spec v3 (WP-D) columns, written for every sample (defaults when a chunk has no
`<chunk>.hvm.pos.jsonl` sidecar, see docs/events.md):
    clocks_ms, elapsed_ms        4 and 2 float32 as bytes: the exact clocks
                                 (A_white A_black B_white B_black) and elapsed
                                 times (A, B) the engine was sent
    clock_scalars_mask           bit 0: clocks present, bit 1: elapsed present;
                                 the loader then rebuilds planes 37-42 from them
    sample_weight                min(1, nodes / weight_nodes_ref), 1 when off
    end_reason, opening          the game's end reason and opening-pair key
    plane_hash                   64-bit hash of the uint8 planes (signed int)
The clock build (more than 74 channels) also gets, by default: value target
(1 - r) * outcome + r * root_q with r = 0.2, sample weights with
weight_nodes_ref 400, the validation split by opening pair, validation samples
whose planes also occur in training dropped, the arrivals label masked when its
window passes the game end, games ending aborted / max_plies left out, and a
report (<output>/report.json: samples by end reason, mean |root_q - outcome|
by reason, duplicate rates). Stock (74-channel) data keeps the old behaviour.
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from hivemind.constants import (
    NUM_BUGHOUSE_CHANNELS,
    NUM_CLOCK_BUGHOUSE_CHANNELS,
    NUM_ELAPSED_BUGHOUSE_CHANNELS,
    NUM_DIAGONAL_BUGHOUSE_CHANNELS,
    NUM_MOVE_CHANNELS,
)

# Constants matching C++ code
NB_INPUT_CHANNELS = NUM_BUGHOUSE_CHANNELS
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE
# Plane widths a chunk header may declare: 74 (stock), 82 (clock planes),
# 86 (clock and elapsed-time planes, spec v2) or 88 (plus the spec v3 diagonal
# clock plane).
SUPPORTED_INPUT_CHANNELS = (
    NB_INPUT_CHANNELS, NUM_CLOCK_BUGHOUSE_CHANNELS, NUM_ELAPSED_BUGHOUSE_CHANNELS,
    NUM_DIAGONAL_BUGHOUSE_CHANNELS,
)
NB_POLICY_VALUES = NUM_MOVE_CHANNELS * BOARD_SIZE * BOARD_SIZE
HEADER = struct.Struct('<4sIHHQ')
SAMPLE_METADATA_V3 = struct.Struct('<QIHHBBbB')
SAMPLE_METADATA_V4 = struct.Struct('<QIHHBBbBf')
SAMPLE_METADATA = SAMPLE_METADATA_V4
POLICY_ENTRY = struct.Struct('<Hf')
JOINT_POLICY_ENTRY = struct.Struct('<HHf')
# Spec v2 trailing block per sample (HVM6): see the module docstring.
AUX_LABEL_VERSION = 6
AUX_LABEL_MAGIC = b'HVM6'
NB_ARRIVAL_VALUES = 20
AUX_LABELS = struct.Struct('<2f2B20f')
SUPPORTED_CHUNKS = {
    b'HVM3': (3,),
    b'HVM4': (4,),
    b'HVM5': (5,),
    b'HVM6': (6,),
}
PASS_POLICY_INDEX = NB_POLICY_VALUES
MAX_JOINT_POLICY_ENTRIES = 256
DEFAULT_RL_VALIDATION_FRACTION = 0.05
DEFAULT_REPLAY_FILES = 25
DEFAULT_REPLAY_SELECTION_FRACTION = 0.05
DEFAULT_Q_VALUE_RATIO = 0.0
DEFAULT_RL_SAMPLES_PER_SHARD = 4096
# Spec v3 defaults for clock-build data (more than NB_INPUT_CHANNELS channels).
DEFAULT_CLOCK_Q_VALUE_RATIO = 0.2
DEFAULT_CLOCK_WEIGHT_NODES_REF = 400
NON_TRAINING_REASONS = ("aborted", "max_plies")
CLOCK_SEAT_ORDER = ("A_white", "A_black", "B_white", "B_black")
CLOCKS_MASK_BIT = 1
ELAPSED_MASK_BIT = 2


def is_clock_data(input_channels: int) -> bool:
    return int(input_channels) > NB_INPUT_CHANNELS


def resolve_q_value_ratio(q_value_ratio, input_channels: int) -> float:
    """None -> 0.2 for clock-build data, 0.0 for stock data."""
    if q_value_ratio is not None:
        return float(q_value_ratio)
    return DEFAULT_CLOCK_Q_VALUE_RATIO if is_clock_data(input_channels) else DEFAULT_Q_VALUE_RATIO


def resolve_weight_nodes_ref(weight_nodes_ref, input_channels: int) -> float:
    """None -> 400 for clock-build data, 0 (off: every weight 1) for stock data."""
    if weight_nodes_ref is not None:
        return float(weight_nodes_ref)
    return float(DEFAULT_CLOCK_WEIGHT_NODES_REF) if is_clock_data(input_channels) else 0.0


def sample_weight(nodes: float, weight_nodes_ref: float) -> float:
    """min(1, nodes / N_ref); 1 when N_ref <= 0."""
    if weight_nodes_ref <= 0:
        return 1.0
    return float(min(1.0, max(0.0, float(nodes)) / float(weight_nodes_ref)))


def plane_hash(planes: bytes) -> int:
    """Signed 64-bit hash of a sample's uint8 planes (fits a Parquet Int64)."""
    return int.from_bytes(hashlib.blake2b(planes, digest_size=8).digest(), "little", signed=True)


def sidecar_path(chunk_path) -> Path:
    return Path(str(chunk_path) + ".pos.jsonl")


def read_sidecar(chunk_path) -> list[dict] | None:
    """The arena recorder's per-sample JSON lines next to a chunk, or None."""
    path = sidecar_path(chunk_path)
    if not path.is_file():
        return None
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def exact_clock_scalars(meta: dict | None) -> tuple[np.ndarray, np.ndarray, int]:
    """(clocks_ms[4], elapsed_ms[2], mask) from one sidecar line.

    Prefers the exact float fields (clocks_ms, elapsed_ms, spec v3); falls back
    to the v2 dicts, rounded as the arena sent them (max(0, int(ms))). Elapsed
    counts only together with clocks, as in the engine.
    """
    clocks = np.zeros(4, dtype=np.float32)
    elapsed = np.zeros(2, dtype=np.float32)
    mask = 0
    if not meta:
        return clocks, elapsed, mask
    if meta.get("clocks_ms") is not None:
        clocks[:] = [max(0.0, float(v)) for v in meta["clocks_ms"]]
        mask |= CLOCKS_MASK_BIT
    elif isinstance(meta.get("clocks"), dict):
        clocks[:] = [max(0, int(meta["clocks"][seat])) for seat in CLOCK_SEAT_ORDER]
        mask |= CLOCKS_MASK_BIT
    if mask & CLOCKS_MASK_BIT:
        if meta.get("elapsed_ms") is not None:
            elapsed[:] = [max(0.0, float(v)) for v in meta["elapsed_ms"]]
            mask |= ELAPSED_MASK_BIT
        elif isinstance(meta.get("elapsed"), dict):
            elapsed[:] = [max(0, int(meta["elapsed"][b])) for b in ("A", "B")]
            mask |= ELAPSED_MASK_BIT
    return clocks, elapsed, mask


def arrivals_past_end(meta: dict | None) -> bool:
    """True when the sample's arrivals window runs past the game end."""
    if not meta or meta.get("gameEndMs") is None or meta.get("t") is None:
        return False
    window = meta.get("arrivalWindowMs") or 5000
    return float(meta["t"]) + float(window) > float(meta["gameEndMs"])


def read_exact(stream, size: int) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(
            f"Truncated HVM chunk: expected {size} bytes, found {len(payload)}"
        )
    return payload


def read_sparse_policy(f) -> np.ndarray:
    """Read sparse policy entries and convert to dense array."""
    num_entries = struct.unpack('<H', read_exact(f, 2))[0]
    
    # Create dense policy array
    policy = np.zeros(NB_POLICY_VALUES, dtype=np.float32)
    
    for _ in range(num_entries):
        index, probability = POLICY_ENTRY.unpack(
            read_exact(f, POLICY_ENTRY.size)
        )
        if index >= NB_POLICY_VALUES:
            raise ValueError(
                f"Policy index {index} exceeds {NB_POLICY_VALUES}"
            )
        policy[index] = probability

    if not np.isclose(policy.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"Sparse policy sums to {policy.sum()}, expected 1"
        )
    
    return policy


def read_sparse_joint_policy(f) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read HVM5 joint visits into fixed-width, probability-sorted arrays."""
    num_entries = struct.unpack('<H', read_exact(f, 2))[0]
    entries = [
        JOINT_POLICY_ENTRY.unpack(read_exact(f, JOINT_POLICY_ENTRY.size))
        for _ in range(num_entries)
    ]
    entries.sort(key=lambda entry: entry[2], reverse=True)
    entries = entries[:MAX_JOINT_POLICY_ENTRIES]

    indices_a = np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.uint16)
    indices_b = np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.uint16)
    probabilities = np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.float32)
    for position, (index_a, index_b, probability) in enumerate(entries):
        if index_a > PASS_POLICY_INDEX or index_b > PASS_POLICY_INDEX:
            raise ValueError(
                f"Joint policy index ({index_a}, {index_b}) exceeds "
                f"{PASS_POLICY_INDEX}"
            )
        indices_a[position] = index_a
        indices_b[position] = index_b
        probabilities[position] = probability

    retained = len(entries)
    total = float(probabilities.sum())
    if retained and total > 0.0:
        probabilities[:retained] /= total
    return indices_a, indices_b, probabilities, retained


def empty_joint_policy() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return (
        np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.uint16),
        np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.uint16),
        np.zeros(MAX_JOINT_POLICY_ENTRIES, dtype=np.float32),
        0,
    )


def empty_aux_labels() -> dict:
    """Aux-head columns of a sample without labels (all masked)."""
    return {
        'think_gain_a': 0.0,
        'think_gain_b': 0.0,
        'think_gain_mask_a': 0,
        'think_gain_mask_b': 0,
        'arrivals': np.zeros(NB_ARRIVAL_VALUES, dtype=np.float32).tobytes(),
        'arrivals_mask': 0,
    }


def read_aux_labels(f) -> dict:
    """Read one HVM6 trailing block into the aux-head columns."""
    values = AUX_LABELS.unpack(read_exact(f, AUX_LABELS.size))
    think = values[0:2]
    masks = values[2:4]
    arrivals = np.asarray(values[4:], dtype=np.float32)
    for mask in masks:
        if mask not in (0, 1):
            raise ValueError(f"think_gain mask byte {mask}, expected 0 or 1")
    labels = {}
    for board, label, mask in zip('ab', think, masks):
        if mask and not (0.0 <= label <= 1.0):
            raise ValueError(f"think_gain_{board} label {label} outside [0, 1]")
        labels[f'think_gain_{board}'] = float(label) if mask else 0.0
        labels[f'think_gain_mask_{board}'] = int(mask)
    if np.isnan(arrivals).any():
        labels['arrivals'] = np.zeros(NB_ARRIVAL_VALUES, dtype=np.float32).tobytes()
        labels['arrivals_mask'] = 0
    else:
        labels['arrivals'] = arrivals.tobytes()
        labels['arrivals_mask'] = 1
    return labels


def read_shard_channels(filepath: str | Path) -> int:
    """Input channels declared in a chunk header (74 stock, 82 clocks, 86 v2, 88 v3)."""
    with open(filepath, 'rb') as f:
        _, _, channels, _, _ = HEADER.unpack(read_exact(f, HEADER.size))
    return int(channels)


def read_binary_shard(
    filepath: str | Path,
    q_value_ratio: float | None = None,
    input_channels: int | None = None,
    sidecar="auto",
    weight_nodes_ref: float | None = None,
    drop_non_training: bool = False,
    mask_arrivals_past_end: bool = False,
) -> list[dict]:
    """Read one HVM3, HVM4, HVM5 or HVM6 chunk and return its samples.

    The plane width comes from the chunk header and may be 74, 82, 86 or 88 channels.
    input_channels, when given, makes any other width an error.
    q_value_ratio / weight_nodes_ref: None picks the default for the chunk's
    width (resolve_q_value_ratio, resolve_weight_nodes_ref).
    sidecar: "auto" reads <chunk>.pos.jsonl when present; a list of dicts (one
    per sample) or None. drop_non_training / mask_arrivals_past_end use the
    sidecar's reason / gameEndMs (convert_to_split_parquet turns both on for
    clock-build data; the reader alone returns every sample as written).
    """
    samples = []
    metas = read_sidecar(filepath) if isinstance(sidecar, str) and sidecar == "auto" else sidecar
    
    with open(filepath, 'rb') as f:
        magic, version, channels, policy_values, num_samples = HEADER.unpack(
            read_exact(f, HEADER.size)
        )
        if version not in SUPPORTED_CHUNKS.get(magic, ()):
            raise ValueError(
                f"Unsupported self-play chunk {filepath}: {magic!r} v{version}"
            )
        if channels not in SUPPORTED_INPUT_CHANNELS or (
            input_channels is not None and channels != input_channels
        ):
            expected = (
                input_channels
                if input_channels is not None
                else " or ".join(map(str, SUPPORTED_INPUT_CHANNELS))
            )
            raise ValueError(
                f"Expected {expected} channels, found {channels}"
            )
        input_values = channels * BOARD_SIZE * BOARD_SIZE
        if policy_values != NB_POLICY_VALUES:
            raise ValueError(
                f"Expected {NB_POLICY_VALUES} policy values, found {policy_values}"
            )
        
        q_value_ratio = resolve_q_value_ratio(q_value_ratio, channels)
        weight_nodes_ref = resolve_weight_nodes_ref(weight_nodes_ref, channels)
        if metas is not None and len(metas) != num_samples:
            raise ValueError(
                f"Sidecar of {filepath} has {len(metas)} lines for {num_samples} samples"
            )
        sample_struct = SAMPLE_METADATA_V4 if version >= 4 else SAMPLE_METADATA_V3
        # Read samples
        for sample_index in range(num_samples):
            if version >= 4:
                (
                    game_id,
                    nodes,
                    macro_ply,
                    moves_left,
                    team,
                    has_time_advantage,
                    outcome,
                    wdl,
                    root_q,
                ) = sample_struct.unpack(read_exact(f, sample_struct.size))
            else:
                (
                    game_id,
                    nodes,
                    macro_ply,
                    moves_left,
                    team,
                    has_time_advantage,
                    outcome,
                    wdl,
                ) = sample_struct.unpack(read_exact(f, sample_struct.size))
                root_q = float(outcome)

            planes = read_exact(f, input_values)
            policy_a = read_sparse_policy(f)
            policy_b = read_sparse_policy(f)
            joint_a, joint_b, joint_probability, joint_count = (
                read_sparse_joint_policy(f)
                if version >= 5
                else empty_joint_policy()
            )
            aux_labels = (
                read_aux_labels(f)
                if version >= AUX_LABEL_VERSION
                else empty_aux_labels()
            )

            meta = metas[sample_index] if metas is not None else None
            reason = str(meta.get("reason") or "unknown") if meta else "unknown"
            if drop_non_training and reason in NON_TRAINING_REASONS:
                continue
            if mask_arrivals_past_end and aux_labels['arrivals_mask'] and arrivals_past_end(meta):
                aux_labels['arrivals'] = np.zeros(NB_ARRIVAL_VALUES, dtype=np.float32).tobytes()
                aux_labels['arrivals_mask'] = 0
            clocks_ms, elapsed_ms, clock_mask = exact_clock_scalars(meta)
            nodes_searched = int(meta.get("nodes_searched") or nodes) if meta else int(nodes)

            blended_value = (1.0 - q_value_ratio) * float(outcome) + q_value_ratio * float(root_q)

            samples.append({
                'x': planes,
                'policy_a': policy_a.tobytes(),
                'policy_b': policy_b.tobytes(),
                'joint_policy_a': joint_a.tobytes(),
                'joint_policy_b': joint_b.tobytes(),
                'joint_policy_probability': joint_probability.tobytes(),
                'joint_policy_count': joint_count,
                'y_value': float(blended_value),
                'root_q': float(root_q),
                'outcome': float(outcome),
                'y_wdl': int(wdl),
                'y_moves_left': int(moves_left),
                'game_id': int(game_id),
                'macro_ply': int(macro_ply),
                'team': int(team),
                'has_time_advantage': bool(has_time_advantage),
                'search_nodes': int(nodes),
                **aux_labels,
                'clocks_ms': clocks_ms.tobytes(),
                'elapsed_ms': elapsed_ms.tobytes(),
                'clock_scalars_mask': int(clock_mask),
                'sample_weight': sample_weight(nodes_searched, weight_nodes_ref),
                'end_reason': reason,
                'opening': str(meta.get("opening") or "") if meta else "",
                'plane_hash': plane_hash(planes),
            })
        if f.read(1):
            raise ValueError(f"Trailing bytes in chunk {filepath}")
    
    return samples


def _selfplay_run_id(path: Path) -> str:
    match = re.fullmatch(r"chunk_(.+)_\d+", path.stem)
    return match.group(1) if match else path.stem


def is_validation_game(
    run_id: str,
    game_id: int,
    validation_fraction: float,
    split_seed: int,
) -> bool:
    """Assign an entire self-play game to a deterministic split."""
    digest = hashlib.blake2b(
        f"{split_seed}:{run_id}:{game_id}".encode("ascii"),
        digest_size=8,
    ).digest()
    unit_value = int.from_bytes(digest, "little") / 2**64
    return unit_value < validation_fraction


def select_replay_files(
    replay_input_dir: str | Path,
    replay_files: int | None,
    replay_selection_fraction: float,
    seed: int,
) -> list[Path]:
    """Select archived chunks from the newest replay-memory window."""
    if not 0.0 < replay_selection_fraction <= 1.0:
        raise ValueError("replay_selection_fraction must be in (0, 1]")

    input_p = Path(replay_input_dir)
    available = sorted(input_p.glob("*.hvm"), reverse=True)
    if not available:
        available = sorted(input_p.rglob("*.hvm"), reverse=True)
    if not available:
        raise FileNotFoundError(f"No HVM chunks found in replay directory: {input_p}")

    if replay_files is None or len(available) <= replay_files:
        return available

    if replay_files <= 0:
        raise ValueError("replay_files must be positive")

    window_size = max(
        int(len(available) * replay_selection_fraction + 0.5),
        replay_files,
    )
    window_size = min(window_size, len(available))
    candidates = available[:window_size]
    return sorted(random.Random(seed).sample(candidates, replay_files))


def is_validation_key(key: str, validation_fraction: float, split_seed: int) -> bool:
    """Assign a whole opening pair (or any string key) to a deterministic split."""
    digest = hashlib.blake2b(
        f"{split_seed}:opening:{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / 2**64 < validation_fraction


def _replay_dirs(replay_input_dir) -> list[Path]:
    if replay_input_dir is None:
        return []
    if isinstance(replay_input_dir, (str, Path)):
        replay_input_dir = [replay_input_dir]
    return [Path(d) for d in replay_input_dir if d]


def _sidecar_openings(binary_files) -> set[str] | None:
    """Opening keys from the input chunks' sidecars; None when any chunk lacks them."""
    openings = set()
    for path in binary_files:
        metas = read_sidecar(path)
        if metas is None or any(not m.get("opening") for m in metas):
            return None
        openings.update(str(m["opening"]) for m in metas)
    return openings


class _ConversionStats:
    """Per-iteration data report (spec v3 WP-D): samples and games by end
    reason, mean |root_q - outcome| by reason, weights, duplicate rates."""

    def __init__(self):
        self.by_reason = {}
        self.weight_sum = 0.0
        self.samples = 0
        self.first_game_of_hash = {}
        self.cross_game_duplicates = 0
        self.train_samples = 0

    def add(self, sample: dict, game_key, split: str, fresh: bool) -> None:
        if fresh:
            entry = self.by_reason.setdefault(
                sample["end_reason"], {"samples": 0, "games": set(), "abs_q_err_sum": 0.0}
            )
            entry["samples"] += 1
            entry["games"].add(game_key)
            entry["abs_q_err_sum"] += abs(float(sample["root_q"]) - float(sample["outcome"]))
            self.weight_sum += float(sample["sample_weight"])
            self.samples += 1
        if split == "train":
            self.train_samples += 1
            first = self.first_game_of_hash.setdefault(sample["plane_hash"], game_key)
            if first != game_key:
                self.cross_game_duplicates += 1

    def report(self) -> dict:
        reasons = {}
        for reason, entry in sorted(self.by_reason.items()):
            reasons[reason] = {
                "samples": entry["samples"],
                "games": len(entry["games"]),
                "mean_abs_root_q_minus_outcome": entry["abs_q_err_sum"] / max(1, entry["samples"]),
            }
        return {
            "samples_by_reason": reasons,
            "mean_sample_weight": self.weight_sum / max(1, self.samples),
            "train_cross_game_duplicate_rate": self.cross_game_duplicates / max(1, self.train_samples),
        }


def convert_to_split_parquet(
    input_dir: str | Path,
    output_dir: str | Path,
    validation_fraction: float = DEFAULT_RL_VALIDATION_FRACTION,
    split_seed: int = 42,
    samples_per_shard: int = DEFAULT_RL_SAMPLES_PER_SHARD,
    replay_input_dir=None,
    replay_files: int = DEFAULT_REPLAY_FILES,
    replay_selection_fraction: float = DEFAULT_REPLAY_SELECTION_FRACTION,
    q_value_ratio: float | None = None,
    input_channels: int | None = None,
    split_by: str = "auto",
    dedupe_validation: bool | None = None,
    weight_nodes_ref: float | None = None,
) -> tuple[Path, Path, int, int]:
    """Convert HVM chunks directly into game-disjoint train/validation shards.

    All chunks (replay included) must share one plane width; by default it is
    taken from the first chunk, so 74- and 82-channel data are never mixed.
    replay_input_dir: one directory or a list (spec v3: the last 3
    iterations); replay_files chunks are drawn from each, into training only.
    split_by: "game", "opening" (whole opening pairs, from the sidecars) or
    "auto" (opening for clock-build data whose chunks all have sidecar
    openings, else game; when the opening split would leave one side empty it
    falls back to games). dedupe_validation (None: on for clock-build data):
    drop validation samples whose planes also occur in training. Writes
    <output_dir>/report.json (_ConversionStats) and prints a summary.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    if split_by not in ("auto", "game", "opening"):
        raise ValueError(f"split_by must be auto, game or opening, got {split_by!r}")

    input_path = Path(input_dir)
    binary_files = sorted(input_path.glob("*.hvm"))
    if not binary_files:
        raise FileNotFoundError(f"No HVM chunks found in {input_path}")
    replay_dirs = _replay_dirs(replay_input_dir)
    selected_replay_files = [
        path
        for directory in replay_dirs
        for path in select_replay_files(
            directory,
            replay_files,
            replay_selection_fraction,
            split_seed,
        )
    ]

    if input_channels is None:
        input_channels = read_shard_channels(binary_files[0])
    clock_data = is_clock_data(input_channels)
    q_value_ratio = resolve_q_value_ratio(q_value_ratio, input_channels)
    weight_nodes_ref = resolve_weight_nodes_ref(weight_nodes_ref, input_channels)
    if dedupe_validation is None:
        dedupe_validation = clock_data

    openings = _sidecar_openings(binary_files) if split_by != "game" else None
    if split_by == "opening" and openings is None:
        raise ValueError("split_by='opening' needs sidecars with an opening for every sample")
    effective_split = "game"
    if openings is not None and (split_by == "opening" or clock_data):
        val_openings = {
            o for o in openings
            if is_validation_key(o, validation_fraction, split_seed)
        }
        if val_openings and val_openings != openings:
            effective_split = "opening"
        else:
            print(
                f"Opening split puts {len(val_openings)} of {len(openings)} opening "
                "pairs in validation; splitting by game instead"
            )

    output_path = Path(output_dir)
    if selected_replay_files:
        print(
            f"Selected {len(selected_replay_files)} replay chunks from "
            + ", ".join(map(str, replay_dirs))
        )
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input or resolved_output in resolved_input.parents:
        raise ValueError("output_dir must not contain the HVM input directory")
    train_path = output_path / "train"
    validation_path = output_path / "val"
    shutil.rmtree(output_path, ignore_errors=True)
    train_path.mkdir(parents=True)
    validation_path.mkdir(parents=True)

    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    shard_indices = {"train": 0, "val": 0}
    stats = _ConversionStats()
    train_hashes = set()
    val_hashes = set()

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        directory = validation_path if split == "val" else train_path
        destination = directory / f"shard_{shard_indices[split]:04d}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        pl.DataFrame(buffers[split]).write_parquet(temporary, compression="zstd")
        temporary.replace(destination)
        shard_indices[split] += 1
        buffers[split].clear()

    def add(split: str, sample: dict, game_key, fresh: bool) -> None:
        (train_hashes if split == "train" else val_hashes).add(sample["plane_hash"])
        stats.add(sample, game_key, split, fresh)
        buffers[split].append(sample)
        counts[split] += 1
        if len(buffers[split]) >= samples_per_shard:
            flush(split)

    read_kw = dict(
        q_value_ratio=q_value_ratio,
        input_channels=input_channels,
        weight_nodes_ref=weight_nodes_ref,
        drop_non_training=clock_data,
        mask_arrivals_past_end=clock_data,
    )
    for binary_file in tqdm(binary_files, desc="Converting HVM chunks"):
        run_id = _selfplay_run_id(binary_file)
        for sample in read_binary_shard(binary_file, **read_kw):
            if effective_split == "opening" and sample["opening"]:
                is_val = is_validation_key(sample["opening"], validation_fraction, split_seed)
            else:
                is_val = is_validation_game(
                    run_id,
                    sample["game_id"],
                    validation_fraction,
                    split_seed,
                )
            add("val" if is_val else "train", sample, (run_id, sample["game_id"]), True)

    for binary_file in tqdm(selected_replay_files, desc="Adding replay chunks"):
        run_id = _selfplay_run_id(binary_file)
        for sample in read_binary_shard(binary_file, **read_kw):
            add("train", sample, (run_id, sample["game_id"]), False)

    flush("train")
    flush("val")

    shared = val_hashes & train_hashes
    dropped = 0
    if dedupe_validation and shared:
        shared_list = list(shared)
        for shard in sorted(validation_path.glob("*.parquet")):
            frame = pl.read_parquet(shard)
            kept = frame.filter(~pl.col("plane_hash").is_in(shared_list))
            dropped += frame.height - kept.height
            if kept.height:
                temporary = shard.with_suffix(".parquet.tmp")
                kept.write_parquet(temporary, compression="zstd")
                temporary.replace(shard)
            else:
                shard.unlink()
        counts["val"] -= dropped

    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError(
            "The game-level split produced an empty train or validation set; "
            "adjust validation_fraction or generate more games"
        )

    report = {
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "split_by": effective_split,
        "q_value_ratio": q_value_ratio,
        "weight_nodes_ref": weight_nodes_ref,
        "replay_dirs": [str(d) for d in replay_dirs],
        "replay_chunks": len(selected_replay_files),
        "val_planes_also_in_train": len(shared),
        "val_samples_dropped_as_duplicates": dropped,
        **stats.report(),
    }
    (output_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Prepared RL data: {counts['train']} train samples, "
        f"{counts['val']} validation samples"
    )
    print(
        f"  split by {effective_split}; q_value_ratio {q_value_ratio}; "
        f"mean sample weight {report['mean_sample_weight']:.3f}; train cross-game "
        f"duplicate rate {report['train_cross_game_duplicate_rate']:.4f}; "
        f"{dropped} validation samples dropped as train duplicates"
    )
    for reason, entry in report["samples_by_reason"].items():
        print(
            f"  {reason}: {entry['samples']} samples, {entry['games']} games, "
            f"mean |root_q - outcome| {entry['mean_abs_root_q_minus_outcome']:.3f}"
        )
    return train_path, validation_path, counts["train"], counts["val"]


def convert_to_parquet(
    input_dir: str,
    output_dir: str,
    samples_per_shard: int = 65536,
    q_value_ratio: float | None = None,
    input_channels: int | None = None,
):
    """Convert all binary shards in input_dir to parquet in output_dir.
    
    Uses streaming to avoid loading all data into memory at once.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bin_files = sorted(input_path.glob('*.hvm'))
    
    if not bin_files:
        raise FileNotFoundError(f"No HVM chunks found in {input_dir}")
    
    print(f"Found {len(bin_files)} binary shard files")
    if input_channels is None:
        input_channels = read_shard_channels(bin_files[0])
    
    # Process files in streaming fashion to avoid OOM
    current_shard_samples = []
    total_samples = 0
    num_output_shards = 0
    
    def write_shard(samples):
        """Write a batch of samples to a parquet file."""
        nonlocal num_output_shards
        shard_id = uuid.uuid4().hex[:8]
        output_file = output_path / f"shard_{shard_id}.parquet"
        
        temporary_file = output_file.with_suffix('.parquet.tmp')
        pl.DataFrame(samples).write_parquet(temporary_file, compression='zstd')
        temporary_file.replace(output_file)
        num_output_shards += 1
        
        if num_output_shards % 10 == 0:
            print(f"  Written {num_output_shards} shards, {total_samples} total samples...")
    
    for bin_file in tqdm(bin_files, desc="Converting files"):
        try:
            samples = read_binary_shard(
                str(bin_file),
                q_value_ratio=q_value_ratio,
                input_channels=input_channels,
            )
            
            for sample in samples:
                current_shard_samples.append(sample)
                total_samples += 1
                
                # Write shard when we have enough samples
                if len(current_shard_samples) >= samples_per_shard:
                    write_shard(current_shard_samples)
                    current_shard_samples = []  # Clear for next batch
                    
        except Exception as e:
            print(f"Error reading {bin_file}: {e}")
            continue
    
    # Write remaining samples
    if current_shard_samples:
        write_shard(current_shard_samples)
    
    print("\nConversion complete!")
    print(f"  Input: {len(bin_files)} binary shards")
    print(f"  Output: {num_output_shards} parquet shards")
    print(f"  Total samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary selfplay data to parquet format'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing HVM chunks (*.hvm)'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to write parquet files'
    )
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=16384,
        help='Number of samples per output parquet shard (default: 16384)'
    )
    parser.add_argument(
        '--q-value-ratio',
        type=float,
        default=None,
        help=(f'Ratio of root Q-value to mix with game outcome (default: '
              f'{DEFAULT_CLOCK_Q_VALUE_RATIO} for clock-build data, {DEFAULT_Q_VALUE_RATIO} for stock)')
    )
    parser.add_argument(
        '--input-channels',
        type=int,
        choices=SUPPORTED_INPUT_CHANNELS,
        default=None,
        help='Plane channels every chunk must have (default: taken from the first chunk)'
    )
    parser.add_argument(
        '--delete-binary',
        action='store_true',
        help='Delete binary files after successful conversion'
    )
    
    args = parser.parse_args()
    
    convert_to_parquet(
        args.input_dir,
        args.output_dir,
        args.samples_per_shard,
        q_value_ratio=args.q_value_ratio,
        input_channels=args.input_channels,
    )
    
    if args.delete_binary:
        input_path = Path(args.input_dir)
        bin_files = list(input_path.glob('*.hvm'))
        for bin_file in bin_files:
            bin_file.unlink()
        print(f"Deleted {len(bin_files)} binary files")


if __name__ == '__main__':
    main()
