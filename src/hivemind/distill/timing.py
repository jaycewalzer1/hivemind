"""Wave 2b (talisman fork): the seat-clock timing of each self-play record,
moved from the engine's seat_clocks.tsv into its HDST chunks (v3).

The engine writes one HDST record per search with the narrow (78-plane)
encoding, and, for seat-clock games, one seat_clocks.tsv row per record in
the same order (games in game order, records in game order; both are written
by the same call at the end of each game). The tsv row carries what the wide
planes and the arrivals head need beyond the board:

  clock_aw..clock_bb          the four seat clocks (ms) when the search ran
  board_elapsed_a, _b         ms since the last action landed on A and on B
  recent_<pocket>_<piece>     pieces landed in the last 5 s, pockets ours_a,
                              theirs_a, ours_b, theirs_b from the record
                              team's view, pieces p n b r q
  arrivals_<pocket>_<piece>   the label: landed in the 5 s after the search
                              started, same order (written at game end)

`attach_timing` appends them to each chunk as the v3 timing block
(hivemind.distill.data has the layout), so the trainer reads the wide planes
and the arrivals labels from the chunks alone. A game cut off at the
macro-ply limit has its labels cut short by the game's end, so its records
carry no arrivals label; everything else is kept as written.
"""
import csv
import os
from pathlib import Path
import struct

import chess
import numpy as np

from hivemind.distill.data import HEADER_BYTES, MAGIC, TIMED_VERSION, VERSION, chunk_paths
from hivemind.distill.labels import UNFINISHED, game_lines
from hivemind.domain.wide_planes import relative_order

POCKETS = ("ours_a", "theirs_a", "ours_b", "theirs_b")
PIECES = "pnbrq"
SEAT_ROWS = "seat_clocks.tsv"


def _columns(kind):
    return [f"{kind}_{pocket}_{piece}" for pocket in POCKETS for piece in PIECES]


def read_seat_rows(path):
    """The rows of a seat_clocks.tsv as arrays in record order: team [N],
    clocks [N, 4] (team-relative: ours A, theirs A, ours B, theirs B),
    elapsed [N, 2] (board A, B), recent [N, 20], arrivals [N, 20], game [N].
    A resumed run can append a game's rows twice; the last copy counts."""
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if rows and "board_elapsed_a" not in rows[0]:
        raise ValueError(f"{path}: no wave 2b columns (an engine from before wave 2b)")
    by_key = {}
    for row in rows:
        by_key[(int(row["game"]), int(row["record"]))] = row
    ordered = [by_key[key] for key in sorted(by_key)]
    n = len(ordered)
    team = np.array([int(row["team"]) for row in ordered], np.int64)
    clocks = np.zeros((n, 4), np.float32)
    for index, row in enumerate(ordered):
        order = relative_order(chess.WHITE if row["team"] == "0" else chess.BLACK)
        clocks[index] = [float(row[f"clock_{seat.lower()}"]) for seat in order]
    elapsed = np.array([[float(row["board_elapsed_a"]), float(row["board_elapsed_b"])] for row in ordered],
                       np.float32).reshape(n, 2)
    recent = np.array([[int(row[c]) for c in _columns("recent")] for row in ordered], np.uint8).reshape(n, 20)
    arrivals = np.array([[float(row[c]) for c in _columns("arrivals")] for row in ordered],
                        np.float32).reshape(n, 20)
    game = np.array([int(row["game"]) for row in ordered], np.int64)
    return {"team": team, "clocks": clocks, "elapsed": elapsed, "recent": recent, "arrivals": arrivals,
            "game": game}


def timing_block(clocks, elapsed, recent, arrivals):
    """The v3 timing block's bytes for N records."""
    return (np.ascontiguousarray(clocks, np.float32).tobytes()
            + np.ascontiguousarray(elapsed, np.float32).tobytes()
            + np.ascontiguousarray(recent, np.uint8).tobytes()
            + np.ascontiguousarray(arrivals, np.float32).tobytes())


def _write_atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def attach_timing(log, seat_rows, data):
    """Rewrite the v2 chunks under `data` as v3 with the timing of
    `seat_rows` (a seat_clocks.tsv), checking the counts against the game
    lines of `log`. Chunks already v3 are left alone. Returns a summary;
    raises ValueError when records and rows do not line up."""
    rows = read_seat_rows(seat_rows)
    games = game_lines(Path(log).read_text(errors="replace").splitlines())
    expected = sum(samples for samples, _ in games)
    paths = chunk_paths([data])
    headers = []
    for path in paths:
        with open(path, "rb") as stream:
            header = stream.read(HEADER_BYTES)
        if header[:4] != MAGIC:
            raise ValueError(f"{path}: not an HDST chunk")
        version = struct.unpack_from("<I", header, 4)[0]
        count = struct.unpack_from("<Q", header, 12)[0]
        headers.append((path, version, count))
    total = sum(count for _, _, count in headers)
    if total != len(rows["team"]) or total != expected:
        raise ValueError(f"{total} records, {len(rows['team'])} seat rows and {expected} samples in "
                         f"{len(games)} game lines do not line up")
    # Per record: does its game's arrivals label hold (finished games only)?
    finished = np.concatenate([np.full(samples, termination != UNFINISHED)
                               for samples, termination in games]) if games else np.zeros(0, bool)
    arrivals = np.where(finished[:, None], rows["arrivals"], -1.0).astype(np.float32)
    start, rewritten, labelled = 0, 0, int(finished.sum())
    for path, version, count in headers:
        stop = start + count
        if version == VERSION:
            raw = bytearray(Path(path).read_bytes())
            struct.pack_into("<I", raw, 4, TIMED_VERSION)
            raw += timing_block(rows["clocks"][start:stop], rows["elapsed"][start:stop],
                                rows["recent"][start:stop], arrivals[start:stop])
            _write_atomic(Path(path), bytes(raw))
            rewritten += 1
        start = stop
    return {"chunks": len(headers), "rewritten": rewritten, "records": total, "labelled": labelled,
            "arrivals_total": float(rows["arrivals"][finished].sum()) if total else 0.0}
