"""Reader for the engine's `gendistill` chunks (HDST v2 and v3).

Chunks carry 78 planes, the clock planes closing each board's block, or 74
from before them. Those are widened as they are read, so every position comes
out in the 78-plane layout.

Wave 2b (talisman fork): `read_chunk(path, layout="wide")` gives the 130-plane
layout of hivemind.domain.wide_planes instead. HDST v3 chunks (78 or 130
planes) end with a timing block, per position and team-relative in block
order, after policyProbability:

  seatClocksMs   f32[N*4]   ours A, theirs A, ours B, theirs B; < 0 = unknown
  elapsedMs      f32[N*2]   block A, block B: the side on turn has been
                            thinking this long; < 0 = unknown
  recentLanded   u8[N*20]   pieces that landed in the last 5 s, pockets ours A,
                            theirs A, ours B, theirs B x P, N, B, R, Q
  arrivals       f32[N*20]  the label: pieces landing within the next 5 s
                            after the search started, same order; arrivals[0]
                            < 0 = no label (game unfinished)

A 78-plane chunk is widened from its pocket planes and this block (no block:
no timing, presence planes only); a 130-plane chunk's own extra planes are
used as written and cut off for the narrow layout.

Spec v7 (think_gain head): a chunk directory may hold `think_labels.npz`
(hivemind.think labels) next to its chunks: `record_index` [M] int, `label`
[M, 2] float (boards A, B), `mask` [M, 2] bool. record_index is the record's
position among the directory's records in chunk order (chunk_paths order, as
the engine wrote them). read_chunk then adds `think_gain` [N, 2] float32 and
`has_think_gain` [N, 2] bool for the chunk's rows; chunks without the file
get neither, and DistillDataset fills in unlabelled rows."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from hivemind.constants import (
    NUM_ARRIVALS,
    NUM_BUGHOUSE_CHANNELS,
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCKED_CHANNELS,
    NUM_CLOCKED_CHANNELS_PER_BOARD,
    NUM_WIDE_CHANNELS,
    TIME_ADVANTAGE_CHANNEL,
)
from hivemind.domain.wide_planes import extend_planes, narrow_planes

MAGIC = b"HDST"
VERSION = 2
TIMED_VERSION = 3
HEADER_BYTES = 28
PLANES = NUM_CLOCKED_CHANNELS
LEGACY_PLANES = NUM_BUGHOUSE_CHANNELS
WIDE_PLANES = NUM_WIDE_CHANNELS
POLICY_SIZE = 4672
LAYOUT_PLANES = {"narrow": PLANES, "wide": WIDE_PLANES}
THINK_LABELS_NAME = "think_labels.npz"


def _header(raw, path):
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not an HDST chunk")
    version, planes = (int(v) for v in np.frombuffer(raw, np.uint32, 2, 4))
    allowed = (PLANES, LEGACY_PLANES) if version == VERSION else (PLANES, WIDE_PLANES)
    if version not in (VERSION, TIMED_VERSION) or planes not in allowed:
        raise ValueError(f"{path}: version {version}, {planes} planes "
                         f"(expected {VERSION} with {PLANES} or {LEGACY_PLANES}, "
                         f"or {TIMED_VERSION} with {PLANES} or {WIDE_PLANES})")
    n, entries = (int(v) for v in np.frombuffer(raw, np.uint64, 2, 12))
    return n, entries, planes, version


def widen_legacy_planes(bits, values):
    """Positions from before the clock planes, in the clocked layout. Those
    games gave one team the right to sit for the whole game, so the lead plane
    is +1 for a team that may sit and -1 for one that may not; the speed plane,
    which those games never had, is 0."""
    count = len(bits)
    old, new = NUM_BUGHOUSE_CHANNELS_PER_BOARD, NUM_CLOCKED_CHANNELS_PER_BOARD
    wide_bits = np.zeros((count, PLANES), np.int64)
    wide_values = np.zeros((count, PLANES), np.float16)
    for board in range(2):
        wide_bits[:, board * new:board * new + old] = bits[:, board * old:(board + 1) * old]
        wide_values[:, board * new:board * new + old] = values[:, board * old:(board + 1) * old]
        may_sit = values[:, board * old + TIME_ADVANTAGE_CHANNEL] > 0.5
        wide_values[:, board * new + old] = np.where(may_sit, 1.0, -1.0)
    return wide_bits, wide_values


def chunk_positions(path):
    """Positions in a chunk, from its header alone."""
    with open(path, "rb") as stream:
        return _header(stream.read(HEADER_BYTES), path)[0]


def read_chunk(path, layout="narrow"):
    """A chunk's arrays, its planes in `layout` ("narrow": 78, "wide": 130).
    Every chunk also gets `arrivals` [N, 20] float32 and `has_arrivals` [N]
    bool (False without a label), and v3 chunks keep their timing fields."""
    if layout not in LAYOUT_PLANES:
        raise ValueError(f"unknown layout {layout!r}")
    raw = Path(path).read_bytes()
    n, entries, planes, version = _header(raw, path)
    offset = HEADER_BYTES

    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        return array

    chunk = {
        "plane_bits": take(np.int64, n * planes).reshape(n, planes),
        "plane_values": take(np.float16, n * planes).reshape(n, planes),
        "value": take(np.float32, n),
        "wdl": take(np.float32, 3 * n).reshape(n, 3),
        "moves_left": take(np.float16, n),
        "ply": take(np.uint16, n),
        "outcome": take(np.int8, n),
        "policy_count": take(np.uint16, 2 * n).reshape(n, 2),
        "policy_index": take(np.uint16, entries),
        "policy_probability": take(np.float16, entries),
    }
    if version == TIMED_VERSION:
        chunk["seat_clocks_ms"] = take(np.float32, n * 4).reshape(n, 4)
        chunk["elapsed_ms"] = take(np.float32, n * 2).reshape(n, 2)
        chunk["recent_landed"] = take(np.uint8, n * NUM_ARRIVALS).reshape(n, NUM_ARRIVALS)
        arrivals = take(np.float32, n * NUM_ARRIVALS).reshape(n, NUM_ARRIVALS)
        chunk["has_arrivals"] = arrivals[:, 0] >= 0
        chunk["arrivals"] = np.where(chunk["has_arrivals"][:, None], arrivals, 0).astype(np.float32)
    else:
        chunk["has_arrivals"] = np.zeros(n, bool)
        chunk["arrivals"] = np.zeros((n, NUM_ARRIVALS), np.float32)
    if offset != len(raw) or int(chunk["policy_count"].sum()) != entries:
        raise ValueError(f"{path}: truncated or inconsistent chunk")
    if planes == LEGACY_PLANES:
        chunk["plane_bits"], chunk["plane_values"] = widen_legacy_planes(
            chunk["plane_bits"], chunk["plane_values"])
        planes = PLANES
    if layout == "wide" and planes == PLANES:
        chunk["plane_bits"], chunk["plane_values"] = widen_planes(chunk)
    elif layout == "narrow" and planes == WIDE_PLANES:
        chunk["plane_bits"] = narrow_planes(chunk["plane_bits"])
        chunk["plane_values"] = narrow_planes(chunk["plane_values"])
    attach_think_labels(path, chunk)
    return chunk


@lru_cache(maxsize=64)
def _think_labels(directory, stamp):
    """(labels by record_index, first record_index of each chunk path) of a
    directory's think_labels.npz; `stamp` (the file's mtime and size) keys the
    cache so a rewritten file is read again."""
    with np.load(Path(directory) / THINK_LABELS_NAME) as data:
        index = np.asarray(data["record_index"], np.int64)
        label = np.asarray(data["label"], np.float32).reshape(-1, 2)
        mask = np.asarray(data["mask"], bool).reshape(-1, 2)
    if not (len(index) == len(label) == len(mask)):
        raise ValueError(f"{directory}/{THINK_LABELS_NAME}: record_index, label and mask lengths differ")
    firsts, offset = {}, 0
    for chunk in chunk_paths([directory]):
        firsts[chunk.name] = offset
        offset += chunk_positions(chunk)
    if len(index) and (index.min() < 0 or index.max() >= offset):
        raise ValueError(f"{directory}/{THINK_LABELS_NAME}: record_index beyond the directory's {offset} records")
    return index, label, mask, firsts


def attach_think_labels(path, chunk):
    """Adds `think_gain` and `has_think_gain` [N, 2] to a chunk read from a
    directory with think_labels.npz (module docstring); leaves others alone."""
    path = Path(path)
    labels = path.parent / THINK_LABELS_NAME
    if not labels.is_file():
        return
    stat = labels.stat()
    index, label, mask, firsts = _think_labels(str(path.parent), (stat.st_mtime_ns, stat.st_size))
    if path.name not in firsts:
        return
    n = len(chunk["value"])
    rows = index - firsts[path.name]
    inside = (rows >= 0) & (rows < n)
    gain = np.zeros((n, 2), np.float32)
    has = np.zeros((n, 2), bool)
    gain[rows[inside]] = label[inside]
    has[rows[inside]] = mask[inside]
    chunk["think_gain"] = np.where(has, gain, 0).astype(np.float32)
    chunk["has_think_gain"] = has


def widen_planes(chunk):
    """A 78-plane chunk's bits and values in the wide layout: the extra
    planes are constant (bits 0), built from the pocket planes and the v3
    timing block when the chunk has one."""
    values = extend_planes(chunk["plane_values"].astype(np.float32), chunk.get("seat_clocks_ms"),
                           chunk.get("elapsed_ms"), chunk.get("recent_landed"))
    bits = np.zeros(values.shape, np.int64)
    bits[:, :PLANES // 2] = chunk["plane_bits"][:, :PLANES // 2]
    wide = WIDE_PLANES // 2
    bits[:, wide:wide + PLANES // 2] = chunk["plane_bits"][:, PLANES // 2:]
    return bits, values.astype(np.float16)


# Moves-left of a record with a result but no real game length (a resigned
# game): the trainer masks it, as it masks every target of an unfinished game.
MOVES_LEFT_UNKNOWN = -1.0


def chunk_problem(path):
    """Why a chunk is unfit to train on, or None. Catches contributors whose
    engine wrote broken values: non-finite or out-of-range targets and inputs,
    or move probabilities that do not sum to one."""
    try:
        chunk = read_chunk(path)
    except ValueError as error:  # truncated, or not a chunk
        return f"{Path(path).name}: unreadable ({str(error).replace(str(path), 'it')})"
    for field in ("plane_values", "value", "wdl", "moves_left", "policy_probability", "arrivals"):
        if not np.isfinite(chunk[field].astype(np.float32)).all():
            return f"{Path(path).name}: non-finite {field}"
    if np.abs(chunk["plane_values"].astype(np.float32)).max(initial=0) > 16:
        return f"{Path(path).name}: input planes out of range"
    moves_left = chunk["moves_left"].astype(np.float32)  # remaining plies / 100, capped at 1
    unknown = moves_left == MOVES_LEFT_UNKNOWN
    if (np.abs(chunk["value"]).max(initial=0) > 1 or (chunk["wdl"] < 0).any() or (chunk["wdl"] > 1).any()
            or ((moves_left < 0) & ~unknown).any() or (moves_left > 1).any()):
        return f"{Path(path).name}: value targets out of range"
    if (chunk["arrivals"] < 0).any() or (chunk["arrivals"] > 16).any():
        return f"{Path(path).name}: arrivals labels out of range"
    probabilities = chunk["policy_probability"].astype(np.float64)
    if (chunk["policy_index"] >= POLICY_SIZE).any() or (probabilities < 0).any():
        return f"{Path(path).name}: policy out of range"
    counts = chunk["policy_count"].ravel().astype(np.int64)
    sums = np.add.reduceat(probabilities, np.cumsum(counts)[counts > 0] - counts[counts > 0]) \
        if len(probabilities) else np.zeros(0)
    if (np.abs(sums - 1) > 0.05).any():
        return f"{Path(path).name}: move probabilities do not sum to one"
    return None

def head_chunk(chunk, count):
    """The first `count` positions of a chunk (policy entries follow the counts)."""
    count = min(count, len(chunk["value"]))
    entries = int(chunk["policy_count"][:count].astype(np.int64).sum())
    per_position = {"plane_bits", "plane_values", "value", "wdl", "moves_left", "ply", "outcome",
                    "policy_count", "arrivals", "has_arrivals", "seat_clocks_ms", "elapsed_ms",
                    "recent_landed", "think_gain", "has_think_gain"}
    return {key: array[:count] if key in per_position else array[:entries]
            for key, array in chunk.items()}


def chunk_paths(sources):
    paths = []
    for source in sources:
        source = Path(source)
        paths.extend(sorted(source.glob("*.dst")) if source.is_dir() else [source])
    return paths


class DistillDataset:
    """Positions on one device; planes are rebuilt from bitboards per batch."""

    def __init__(self, chunks, device, source=0):
        """`source` tags every row (0 = teacher data, 1 = search data) so the
        loss can weight the two kinds of target separately."""
        cat = lambda key: np.concatenate([c[key] for c in chunks])
        self.device = device
        self.source = source
        self.plane_bits = torch.from_numpy(cat("plane_bits")).to(device)
        self.plane_values = torch.from_numpy(cat("plane_values").copy()).to(device)
        self.value = torch.from_numpy(cat("value").copy()).to(device)
        self.wdl = torch.from_numpy(cat("wdl").copy()).to(device)
        self.moves_left = torch.from_numpy(cat("moves_left").astype(np.float32)).to(device)
        counts = torch.from_numpy(cat("policy_count").astype(np.int64)).to(device)
        self.policy_counts = counts                                  # [N, 2]
        flat = counts.flatten()
        self.policy_starts = (torch.cumsum(flat, 0) - flat).view(-1, 2)
        # Compact on the device (indices fit int16); widened per batch.
        self.policy_index = torch.from_numpy(cat("policy_index").astype(np.int16)).to(device)
        self.policy_probability = torch.from_numpy(cat("policy_probability").copy()).to(device)
        self.plane_count = self.plane_bits.shape[1]
        self.arrivals = torch.from_numpy(cat("arrivals").astype(np.float32)).to(device)
        self.has_arrivals = torch.from_numpy(cat("has_arrivals").astype(bool)).to(device)
        # Spec v7: think_gain labels where a chunk has them, masked elsewhere.
        think = [c.get("think_gain", np.zeros((len(c["value"]), 2), np.float32)) for c in chunks]
        has_think = [c.get("has_think_gain", np.zeros((len(c["value"]), 2), bool)) for c in chunks]
        self.think_gain = torch.from_numpy(np.concatenate(think).astype(np.float32)).to(device)
        self.has_think_gain = torch.from_numpy(np.concatenate(has_think).astype(bool)).to(device)
        self.shifts = torch.arange(64, device=device, dtype=torch.int64)

    def __len__(self):
        return len(self.value)

    def planes(self, index):
        bits = self.plane_bits[index]                                # [B, 78 or 130]
        values = self.plane_values[index].float()
        squares = ((bits.unsqueeze(-1) >> self.shifts) & 1).float()  # [B, 78, 64]
        filled = torch.where(bits.unsqueeze(-1) != 0, squares, torch.ones_like(squares))
        return (filled * values.unsqueeze(-1)).view(-1, self.plane_count, 8, 8)

    def policy_targets(self, index, board):
        """One board's teacher policy as dense [B, 4672] targets.

        Returns (target, legal, mask): the teacher's probabilities (summing to
        one), which policy indices are legal moves or the pass, and which rows
        have a board on turn at all. Rows without one are marked all-legal so a
        legal-only softmax stays finite; their loss is masked out.
        """
        counts = self.policy_counts[index, board]
        starts = self.policy_starts[index, board]
        total = int(counts.sum())
        rows = torch.repeat_interleave(torch.arange(len(index), device=self.device), counts,
                                       output_size=total)
        offsets = torch.cumsum(counts, 0) - counts
        within = torch.arange(total, device=self.device) - torch.repeat_interleave(
            offsets, counts, output_size=total)
        entries = torch.repeat_interleave(starts, counts, output_size=total) + within
        columns = self.policy_index[entries].long()
        target = torch.zeros(len(index), POLICY_SIZE, device=self.device)
        target.index_put_((rows, columns), self.policy_probability[entries].float(), accumulate=True)
        legal = torch.zeros(len(index), POLICY_SIZE, dtype=torch.bool, device=self.device)
        legal[rows, columns] = True
        mask = counts > 0
        legal[~mask] = True
        mass = target.sum(1, keepdim=True)
        return target / mass.clamp_min(1e-8), legal, mask

    def batch(self, index):
        return {
            "planes": self.planes(index),
            "value": self.value[index],
            "wdl": self.wdl[index],
            "moves_left": self.moves_left[index],
            "policy": [self.policy_targets(index, board) for board in (0, 1)],
            "arrivals": self.arrivals[index],
            "has_arrivals": self.has_arrivals[index],
            "think_gain": self.think_gain[index],
            "has_think_gain": self.has_think_gain[index],
            "source": torch.full((len(index),), self.source, device=self.device, dtype=torch.int8),
        }
