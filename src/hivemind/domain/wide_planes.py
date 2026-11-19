"""The wide input layout of wave 2b (talisman fork).

Each board's block is twin-s-clock's 39 planes followed by 26 constant planes,
in the same order in both blocks, "our"/"their" from the searching team's view
on that board (block B: our partner's seat and pocket):

  +0  our seat clock min(t/300, 1)          +1  our seat exp(-t/10)
  +2  opponent's clock min(t/300, 1)        +3  opponent's exp(-t/10)
  +4  side on turn here, thinking min(e/10, 1)
  +5  the same, 1 - exp(-e/2)
  +6..+10   our pocket holds at least one P, N, B, R, Q (0/1)
  +11..+15  their pocket, the same
  +16..+20  our pocket: P, N, B, R, Q that landed in the last 5 s, min(n/2, 1)
  +21..+25  their pocket, the same

t is a seat clock and e the time since the last action landed on that board,
both in seconds. Without timing the clock, thinking and landing planes are 0;
the presence planes are always set, from the pockets.

Two encoders share the arithmetic: `board2planes_wide` from a BughouseBoard
(what the C++ engine's planes are checked against) and `extend_planes`, which
widens a batch of narrow plane values from their pocket planes and the timing
fields an HDST v3 chunk carries (hivemind.distill.data).
"""
from dataclasses import dataclass, field

import chess
import numpy as np

from hivemind.constants import (
    ARRIVALS_WINDOW_MS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_NUM_DROPS,
    NUM_ARRIVALS,
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCKED_CHANNELS_PER_BOARD,
    NUM_DROP_TYPE,
    NUM_WIDE_CHANNELS,
    NUM_WIDE_CHANNELS_PER_BOARD,
    NUM_WIDE_EXTRA_CHANNELS_PER_BOARD,
)

SEATS = ("Aw", "Ab", "Bw", "Bb")
PIECE_LETTERS = "PNBRQ"
# Pocket planes within a board block: ours 12..16, theirs 17..21 (count / 16).
OUR_POCKET_CHANNEL, THEIR_POCKET_CHANNEL = 12, 17
LEAD_CHANNEL, SPEED_CHANNEL = NUM_BUGHOUSE_CHANNELS_PER_BOARD, NUM_BUGHOUSE_CHANNELS_PER_BOARD + 1
# Offsets of the extra planes after the clocked block.
OUR_CLOCK, THEIR_CLOCK, ELAPSED, OUR_PRESENCE, THEIR_PRESENCE, OUR_RECENT, THEIR_RECENT = 0, 2, 4, 6, 11, 16, 21


def team_seats(team_side):
    """The team's view of the seats, in block order: (ours, theirs) on board
    A, then on board B. The white team is Aw and Bb."""
    if team_side == chess.WHITE:
        return ("Aw", "Ab"), ("Bb", "Bw")
    return ("Ab", "Aw"), ("Bw", "Bb")


def relative_order(team_side):
    """Seat names in the arrivals head's pocket order: ours A, theirs A,
    ours B, theirs B."""
    (our_a, their_a), (our_b, their_b) = team_seats(team_side)
    return our_a, their_a, our_b, their_b


def parse_seat_arrivals(text):
    """The UCI SeatArrivalsMs value, '<pocket>:<piece>:<ms_ago>,...' with the
    pocket one of Aw, Ab, Bw, Bb and the piece one of P, N, B, R, Q (either
    case), as a list of (pocket, piece letter, ms ago). Empty -> []."""
    landings = []
    for item in (text or "").replace(" ", "").split(","):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3 or parts[0] not in SEATS or parts[1].upper() not in PIECE_LETTERS:
            raise ValueError(f"bad landing {item!r}: expected <Aw|Ab|Bw|Bb>:<P|N|B|R|Q>:<ms_ago>")
        ms_ago = int(parts[2])
        if ms_ago < 0:
            raise ValueError(f"bad landing {item!r}: ms_ago < 0")
        landings.append((parts[0], parts[1].upper(), ms_ago))
    return landings


def format_seat_arrivals(landings):
    return ",".join(f"{pocket}:{piece}:{int(ms_ago)}" for pocket, piece, ms_ago in landings)


def recent_counts(landings, window_ms=ARRIVALS_WINDOW_MS):
    """{seat: [P, N, B, R, Q] counts} of the landings with 0 <= ms_ago <
    window_ms (the last 5 s)."""
    counts = {seat: [0] * NUM_DROP_TYPE for seat in SEATS}
    for pocket, piece, ms_ago in landings:
        if 0 <= ms_ago < window_ms:
            counts[pocket][PIECE_LETTERS.index(piece)] += 1
    return counts


@dataclass
class WideTiming:
    """What the wide planes need beyond the board, in absolute seat terms.

    seat_clocks_ms: {seat: ms} for Aw, Ab, Bw, Bb, or None (no timing).
    elapsed_ms: ms since the last action landed on board A and on board B
        (the side on turn there has been thinking that long), or None.
    landings: (pocket, piece letter, ms ago) of recent landings.
    """
    seat_clocks_ms: dict = None
    elapsed_ms: tuple = None
    landings: list = field(default_factory=list)


def clock_values(ms):
    """(min(t/300, 1), exp(-t/10)) for a clock of `ms` (array), t in s; 0 and
    0 where the clock is unknown (negative or NaN)."""
    ms = np.asarray(ms, np.float64)
    known = np.isfinite(ms) & (ms >= 0)
    seconds = np.where(known, ms, 0.0) / 1000.0
    return (np.where(known, np.minimum(seconds / 300.0, 1.0), 0.0),
            np.where(known, np.exp(-seconds / 10.0), 0.0))


def elapsed_values(ms):
    """(min(e/10, 1), 1 - exp(-e/2)) for `ms` thinking (array); 0 and 0 where
    unknown (negative or NaN)."""
    ms = np.asarray(ms, np.float64)
    known = np.isfinite(ms) & (ms >= 0)
    seconds = np.where(known, ms, 0.0) / 1000.0
    return np.minimum(seconds / 10.0, 1.0), 1.0 - np.exp(-seconds / 2.0)


def extra_values(our_pocket, their_pocket, clocks_ms, elapsed_ms, recent):
    """The 26 extra plane values of one board block, for N positions.

    our_pocket, their_pocket: [N, 5] piece counts (P..Q); clocks_ms: [N, 2]
    (ours, theirs), negative or NaN = unknown; elapsed_ms: [N]; recent: [N, 10]
    landed counts (ours P..Q, theirs P..Q). Returns float32 [N, 26]."""
    our_pocket, their_pocket = np.asarray(our_pocket), np.asarray(their_pocket)
    clocks_ms = np.asarray(clocks_ms, np.float64)
    recent = np.asarray(recent, np.float64)
    out = np.zeros((len(our_pocket), NUM_WIDE_EXTRA_CHANNELS_PER_BOARD), np.float64)
    out[:, OUR_CLOCK], out[:, OUR_CLOCK + 1] = clock_values(clocks_ms[:, 0])
    out[:, THEIR_CLOCK], out[:, THEIR_CLOCK + 1] = clock_values(clocks_ms[:, 1])
    out[:, ELAPSED], out[:, ELAPSED + 1] = elapsed_values(elapsed_ms)
    out[:, OUR_PRESENCE:OUR_PRESENCE + 5] = our_pocket > 0
    out[:, THEIR_PRESENCE:THEIR_PRESENCE + 5] = their_pocket > 0
    out[:, OUR_RECENT:OUR_RECENT + 10] = np.minimum(np.where(recent > 0, recent, 0) / 2.0, 1.0)
    return out.astype(np.float32)


def extend_planes(values, clocks_ms=None, elapsed_ms=None, recent=None):
    """Narrow per-plane values [N, 78] (constant planes: their value; others
    are rebuilt from bits by the caller) to the wide layout [N, 130].

    The timing fields are team-relative, in block order as an HDST v3 chunk
    stores them: clocks_ms [N, 4] (ours A, theirs A, ours B, theirs B),
    elapsed_ms [N, 2] (block A, block B), recent [N, 20] (the arrivals
    order). None or negative = no timing. Presence comes from the pocket
    planes (count / 16)."""
    values = np.asarray(values)
    count = len(values)
    if clocks_ms is None:
        clocks_ms = np.full((count, 4), -1.0)
    if elapsed_ms is None:
        elapsed_ms = np.full((count, 2), -1.0)
    if recent is None:
        recent = np.zeros((count, NUM_ARRIVALS))
    clocks_ms, elapsed_ms, recent = np.asarray(clocks_ms), np.asarray(elapsed_ms), np.asarray(recent)
    narrow, wide = NUM_CLOCKED_CHANNELS_PER_BOARD, NUM_WIDE_CHANNELS_PER_BOARD
    out = np.zeros((count, NUM_WIDE_CHANNELS), values.dtype)
    for block in range(2):
        start = block * narrow
        out[:, block * wide:block * wide + narrow] = values[:, start:start + narrow]
        pockets = values[:, start + OUR_POCKET_CHANNEL:start + OUR_POCKET_CHANNEL + 10].astype(np.float32)
        # Known landings need known timing: no timing, no landing planes.
        timed = (clocks_ms[:, 2 * block] >= 0)[:, None]
        landed = np.where(timed, recent[:, 10 * block:10 * block + 10], 0)
        out[:, block * wide + narrow:(block + 1) * wide] = extra_values(
            pockets[:, :5], pockets[:, 5:], clocks_ms[:, 2 * block:2 * block + 2],
            elapsed_ms[:, block], landed)
    return out


def narrow_planes(values):
    """Wide per-plane values (or bits) [N, 130] cut back to the clocked
    layout [N, 78] a narrow network reads."""
    narrow, wide = NUM_CLOCKED_CHANNELS_PER_BOARD, NUM_WIDE_CHANNELS_PER_BOARD
    return np.concatenate((values[:, :narrow], values[:, wide:wide + narrow]), axis=1)


def relative_timing(timing, team_side):
    """A WideTiming in the chunk's team-relative block order: clocks [4],
    elapsed [2], recent [20], or (None, None, zeros) without timing."""
    recent = np.zeros(NUM_ARRIVALS)
    if timing is None or timing.seat_clocks_ms is None:
        return None, None, recent
    seats = relative_order(team_side)
    clocks = np.array([timing.seat_clocks_ms[seat] for seat in seats], np.float64)
    elapsed = (np.array(timing.elapsed_ms, np.float64) if timing.elapsed_ms is not None
               else np.full(2, -1.0))
    counts = recent_counts(timing.landings)
    for index, seat in enumerate(seats):
        recent[5 * index:5 * index + 5] = counts[seat]
    return clocks, elapsed, recent


def board2planes_wide(board, team_side, lead_plane=0.0, speed_plane=0.0, timing=None, flip=False):
    """The wide input planes [130, 8, 8] of a BughouseBoard for `team_side`.

    lead_plane / speed_plane: twin-s-clock's planes 37/38 (the engine's
    clock_lead_plane_value / clock_speed_plane_value); timing: a WideTiming,
    or None for no timing (presence planes only). `flip` swaps the two
    blocks, as board2planes does."""
    from hivemind.domain.board2planes import board2planes

    narrow = NUM_BUGHOUSE_CHANNELS_PER_BOARD
    base = board2planes(board, team_side)  # [74, 8, 8], unflipped
    values = np.zeros((1, 2 * NUM_CLOCKED_CHANNELS_PER_BOARD), np.float32)
    clocked = np.zeros((2 * NUM_CLOCKED_CHANNELS_PER_BOARD, BOARD_HEIGHT, BOARD_WIDTH), np.float32)
    for block in range(2):
        start = block * NUM_CLOCKED_CHANNELS_PER_BOARD
        clocked[start:start + narrow] = base[block * narrow:(block + 1) * narrow]
        clocked[start + LEAD_CHANNEL] = lead_plane
        clocked[start + SPEED_CHANNEL] = speed_plane
        # Pocket counts as the chunk stores them (count / 16).
        values[0, start + OUR_POCKET_CHANNEL:start + OUR_POCKET_CHANNEL + 10] = \
            clocked[start + OUR_POCKET_CHANNEL:start + OUR_POCKET_CHANNEL + 10, 0, 0]
    clocks, elapsed, recent = relative_timing(timing, team_side)
    extended = extend_planes(values, None if clocks is None else clocks[None],
                             None if elapsed is None else elapsed[None], recent[None])[0]
    wide = NUM_WIDE_CHANNELS_PER_BOARD
    planes = np.zeros((NUM_WIDE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), np.float32)
    for block in range(2):
        start = block * NUM_CLOCKED_CHANNELS_PER_BOARD
        planes[block * wide:block * wide + NUM_CLOCKED_CHANNELS_PER_BOARD] = \
            clocked[start:start + NUM_CLOCKED_CHANNELS_PER_BOARD]
        planes[block * wide + NUM_CLOCKED_CHANNELS_PER_BOARD:(block + 1) * wide] = \
            extended[block * wide + NUM_CLOCKED_CHANNELS_PER_BOARD:(block + 1) * wide, None, None]
    if flip:
        planes = np.concatenate((planes[wide:], planes[:wide]))
    return planes


def pocket_counts_check(planes):
    """Pocket counts (ours, theirs) x P..Q per block from wide planes, for
    tests: the count planes x 16."""
    wide = NUM_WIDE_CHANNELS_PER_BOARD
    return [np.rint(planes[block * wide + OUR_POCKET_CHANNEL:block * wide + OUR_POCKET_CHANNEL + 10, 0, 0]
                    * MAX_NUM_DROPS).astype(int) for block in range(2)]
