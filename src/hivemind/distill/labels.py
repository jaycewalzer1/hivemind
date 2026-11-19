"""Value-label hygiene for self-play chunks (audit training 4, wave 2).

The engine writes one record per search, games in order, and prints one
`selfplay game N/M ... samples S ... termination T` line per game. Reading
the two together checks, per game, what the trainer will learn from it:

  * every record is there: the records add up to the games' samples, so
    nothing deduplicated a pass followed by a move in the same position
    (that pair is what teaches when to stop sitting);
  * a finished game labels every record with a one-hot WDL that matches its
    outcome, and an unfinished one (macro-ply limit) labels none;
  * a flag is a decisive result: no flag game record is a draw or unlabelled
    (a team that flags lost; `double_flag` is the rules' draw).

Records that repeat the previous record's planes in the same game (a pass,
then a new decision in the same position) are counted, never removed.
"""

from collections import Counter
from pathlib import Path
import re

import numpy as np

from hivemind.distill.data import chunk_paths, read_chunk

UNFINISHED = "macro-ply limit"
FLAG = "flag"
DRAWN_TERMINATIONS = {"draw", "double_flag", "double flag"}
# The termination may hold spaces ("macro-ply limit"); these keys follow it.
GAME_LINE = re.compile(r"^selfplay game (\d+)/\d+ .*?\bsamples (\d+) .*?\btermination (.+?)"
                       r"(?= (?:lead|resign-check|flag-seat|clocks) |$)")


def game_lines(lines):
    """(samples, termination) per game in game order. A resumed run prints
    a game again when it replays it, so the last line of each game counts."""
    games = {}
    for line in lines:
        match = GAME_LINE.match(line.strip())
        if match:
            games[int(match[1])] = (int(match[2]), match[3].strip())
    return [games[n] for n in sorted(games)]


def _records(data):
    """Per record: outcome, WDL and a hash of the planes, chunk by chunk."""
    for path in chunk_paths([data]):
        chunk = read_chunk(path)
        bits = np.ascontiguousarray(chunk["plane_bits"]).view(np.uint8).reshape(len(chunk["value"]), -1)
        values = np.ascontiguousarray(chunk["plane_values"]).view(np.uint8).reshape(len(chunk["value"]), -1)
        keys = [hash((b.tobytes(), v.tobytes())) for b, v in zip(bits, values)]
        yield chunk["outcome"].astype(np.int64), chunk["wdl"].astype(np.float32), keys


def label_report(log, data):
    """Check `data` (a self-play output directory of HDST chunks) against the
    game lines of its log. Returns counts and a list of problems; an empty
    list means the labels are fit to train on."""
    lines = Path(log).read_text(errors="replace").splitlines()
    games = game_lines(lines)
    outcomes, wdls, keys = [np.zeros(0, np.int64)], [np.zeros((0, 3), np.float32)], []
    for o, w, k in _records(data):
        outcomes.append(o)
        wdls.append(w)
        keys += k
    outcome, wdl = np.concatenate(outcomes), np.concatenate(wdls)
    expected = sum(samples for samples, _ in games)
    report = {"games": len(games), "records": len(outcome), "terminations": dict(Counter(t for _, t in games)),
              "flag_games": 0, "flag_records": 0, "repeat_records": 0, "problems": []}
    problems = report["problems"]
    if len(outcome) != expected:
        problems.append(f"{len(outcome)} records for {expected} samples in {len(games)} game lines: "
                        f"records were dropped (deduplicated?) or added")
        return report
    start = 0
    for number, (samples, termination) in enumerate(games, 1):
        o, w = outcome[start:start + samples], wdl[start:start + samples]
        game_keys = keys[start:start + samples]
        report["repeat_records"] += sum(a == b for a, b in zip(game_keys, game_keys[1:]))
        start += samples
        labelled = w.sum(1) > 0
        if termination == UNFINISHED:
            if labelled.any():
                problems.append(f"game {number}: unfinished but {int(labelled.sum())} records carry a WDL")
            continue
        if not labelled.all():
            problems.append(f"game {number} ({termination}): {int((~labelled).sum())} records without a WDL")
            continue
        if (w.argmax(1) != o + 1).any() or (np.abs(w.sum(1) - 1) > 1e-6).any():
            problems.append(f"game {number} ({termination}): WDL does not match the outcome")
        if termination == FLAG:
            report["flag_games"] += 1
            report["flag_records"] += samples
            if (o == 0).any():
                problems.append(f"game {number}: a flag labelled as a draw in {int((o == 0).sum())} records")
        elif termination in DRAWN_TERMINATIONS and (o != 0).any():
            problems.append(f"game {number} ({termination}): a draw labelled as decisive")
    return report
