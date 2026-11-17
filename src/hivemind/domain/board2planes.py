import math

import chess
import chess.variant
import numpy as np

from hivemind.domain.board import BughouseBoard
from hivemind.constants import (BOARD_A, BOARD_B, BOARD_HEIGHT, BOARD_WIDTH,
                           MAX_NUM_DROPS, MAX_NUM_NO_PROGRESS,
                           NUM_BUGHOUSE_CHANNELS_PER_BOARD, CLOCK_CHANNEL_OFFSET,
                           channels_per_board)


_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_SQUARE_FLIP_VERTICAL = np.arange(64, dtype=np.int16).reshape(8, 8)[::-1, :].reshape(-1)

_LAST_MOVE_FROM_CHANNEL = 32
_LAST_MOVE_TO_CHANNEL = 33
_HALFMOVE_CLOCK_CHANNEL = 34
_REPETITION_2_CHANNEL = 35
_REPETITION_3_CHANNEL = 36


def _set_plane_from_bitboard(plane: np.ndarray, bitboard: int, needs_vertical_flip: bool) -> None:
    while bitboard:
        lsb = bitboard & -bitboard
        sq = lsb.bit_length() - 1
        if needs_vertical_flip:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        plane.flat[sq] = 1.0
        bitboard ^= lsb


def _orient_square(square: int, needs_vertical_flip: bool) -> int:
    if not needs_vertical_flip:
        return square
    return int(_SQUARE_FLIP_VERTICAL[square])


def _ep_square(single_board):
    # python-chess keeps ep_square after every double push; the engine
    # (Fairy-Stockfish) only sets it when an en passant capture is legal.
    return single_board.ep_square if single_board.has_legal_en_passant() else None


def _last_move_to_square(board: BughouseBoard, board_idx: int, move: chess.Move) -> int:
    # The engine records castling as king to rook square (e1h1/e1a1); a king
    # that moved two files can only have castled.
    king_square = move.to_square
    single_board = board.boards[board_idx]
    if (move.drop is None
            and single_board.piece_type_at(king_square) == chess.KING
            and abs(chess.square_file(move.from_square) - chess.square_file(king_square)) == 2):
        rook_file = 7 if chess.square_file(king_square) > chess.square_file(move.from_square) else 0
        return chess.square(rook_file, chess.square_rank(king_square))
    return king_square


def _last_move_for_board(board: BughouseBoard, board_idx: int):
    for idx in range(len(board.board_order) - 1, -1, -1):
        if board.board_order[idx] == board_idx:
            return board.move_history[idx]
    return None


def _safe_repetition_flag(single_board, count: int) -> float:
    try:
        # `is_repetition` can traverse history via push/pop internally.
        # Run it on a copy so live bughouse board state is never mutated.
        board_copy = single_board.copy(stack=True)
        return float(board_copy.is_repetition(count))
    except Exception:
        # Some historical game records can violate pocket consistency during
        # deep repetition probes; do not fail sample generation for this.
        return 0.0


def clock_plane_values(remaining_ms: float) -> tuple[float, float]:
    """The two clock-plane values for one seat: min(t/300, 1) and exp(-t/10), t in seconds.

    A negative clock (a flag) counts as 0, as in the engine's planes.cc, so the
    planes stay in [0, 1].
    """
    t = max(float(remaining_ms), 0.0) / 1000.0
    return min(t / 300.0, 1.0), math.exp(-t / 10.0)


def _fill_clock_planes(planes, offset, team_side, clocks) -> None:
    """Fill channels 37..40 of both board blocks from clocks (ms).

    clocks = (A_white, A_black, B_white, B_black), the order of the engine's
    `position ... clocks` suffix. Our seat on A is team_side, on B the other
    colour (the partner); the opponent seat is the other colour on that board.
    """
    if len(clocks) != 4:
        raise ValueError("clocks must be (A_white, A_black, B_white, B_black) in ms")
    a_white, a_black, b_white, b_black = clocks
    board_a = {chess.WHITE: a_white, chess.BLACK: a_black}
    board_b = {chess.WHITE: b_white, chess.BLACK: b_black}
    seats = (
        (0, board_a[team_side], board_a[not team_side]),
        (offset, board_b[not team_side], board_b[team_side]),
    )
    for base, ours, theirs in seats:
        c = base + CLOCK_CHANNEL_OFFSET
        planes[c][:, :], planes[c + 1][:, :] = clock_plane_values(ours)
        planes[c + 2][:, :], planes[c + 3][:, :] = clock_plane_values(theirs)


def board2planes(board: BughouseBoard, team_side: chess.Color, flip=False,
                 clocks=None, clock_planes=None) -> np.ndarray:
    """Encode a bughouse position from team_side's view.

    Default: 37 channels per board (74 total), unchanged. With clock_planes=True,
    or when clocks are given, 41 channels per board (82 total): channels 37..40 of
    each board block hold the clock planes (all 0 when clocks is None).
    clocks: remaining ms for (A_white, A_black, B_white, B_black).
    """
    if clock_planes is None:
        clock_planes = clocks is not None
    if clocks is not None and not clock_planes:
        raise ValueError("clocks need clock_planes=True")
    offset = channels_per_board(clock_planes)
    planes = np.zeros(
        (2 * offset, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32
    )

    board_a = board.boards[BOARD_A]
    board_b = board.boards[BOARD_B]
    flip_board_a = team_side == chess.BLACK
    flip_board_b = team_side == chess.WHITE

    # we evaluate castling rights BEFORE we apply transform since king position is used to determine castling rights
    planes[27][:, :] = float(board_a.has_kingside_castling_rights(team_side))
    planes[28][:, :] = float(board_a.has_queenside_castling_rights(team_side))
    planes[29][:, :] = float(board_a.has_kingside_castling_rights(not team_side))
    planes[30][:, :] = float(board_a.has_queenside_castling_rights(not team_side))

    planes[offset + 27][:, :] = float(board_b.has_kingside_castling_rights(not team_side))
    planes[offset + 28][:, :] = float(board_b.has_queenside_castling_rights(not team_side))
    planes[offset + 29][:, :] = float(board_b.has_kingside_castling_rights(team_side))
    planes[offset + 30][:, :] = float(board_b.has_queenside_castling_rights(team_side))

    # pieces
    c = 0
    for color in [team_side, not team_side]:
        for pt in _PIECE_TYPES:
            _set_plane_from_bitboard(
                planes[c],
                int(board_a.pieces_mask(pt, color)),
                flip_board_a,
            )
            _set_plane_from_bitboard(
                planes[offset + c],
                int(board_b.pieces_mask(pt, not color)),
                flip_board_b,
            )
            c += 1

    # pocket pieces
    c = 12
    for color in [team_side, not team_side]:
        for pt in range(1, 6):
            count_a = board_a.pockets[color].count(pt)
            planes[c][:, :] = count_a / MAX_NUM_DROPS

            count_b = board_b.pockets[not color].count(pt)
            planes[offset + c][:, :] = count_b / MAX_NUM_DROPS
            c += 1

    # promoted pieces
    c = 22
    for color in [team_side, not team_side]:
        mask_a = int(board_a.occupied_co[color] & board_a.promoted)
        _set_plane_from_bitboard(planes[c], mask_a, flip_board_a)

        mask_b = int(board_b.occupied_co[not color] & board_b.promoted)
        _set_plane_from_bitboard(planes[offset + c], mask_b, flip_board_b)
        c += 1

    # en passant square
    ep_a = _ep_square(board_a)
    if ep_a is not None:
        sq = ep_a
        if flip_board_a:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        planes[24].flat[sq] = 1.0
    ep_b = _ep_square(board_b)
    if ep_b is not None:
        sq = ep_b
        if flip_board_b:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        planes[offset + 24].flat[sq] = 1.0

    # on turn
    planes[25][:, :] = 1.0 if board_a.turn == team_side else 0.0
    planes[offset + 25][:, :] = 1.0 if board_b.turn == (not team_side) else 0.0

    # constant plane
    planes[26][:, :] = 1.0
    planes[offset + 26][:, :] = 1.0

    # has time advantage (can sit)
    has_time_advantage = 1.0 if board.time_advantage(team_side) > 0 else 0.0
    planes[31][:, :] = has_time_advantage
    planes[offset + 31][:, :] = has_time_advantage

    # last move planes
    last_move_a = _last_move_for_board(board, BOARD_A)
    if last_move_a is not None:
        if last_move_a.drop is None:
            sq = _orient_square(last_move_a.from_square, flip_board_a)
            planes[_LAST_MOVE_FROM_CHANNEL].flat[sq] = 1.0
        sq = _orient_square(_last_move_to_square(board, BOARD_A, last_move_a), flip_board_a)
        planes[_LAST_MOVE_TO_CHANNEL].flat[sq] = 1.0

    last_move_b = _last_move_for_board(board, BOARD_B)
    if last_move_b is not None:
        if last_move_b.drop is None:
            sq = _orient_square(last_move_b.from_square, flip_board_b)
            planes[offset + _LAST_MOVE_FROM_CHANNEL].flat[sq] = 1.0
        sq = _orient_square(_last_move_to_square(board, BOARD_B, last_move_b), flip_board_b)
        planes[offset + _LAST_MOVE_TO_CHANNEL].flat[sq] = 1.0

    # halfmove clock and repetition context
    planes[_HALFMOVE_CLOCK_CHANNEL][:, :] = min(board_a.halfmove_clock, MAX_NUM_NO_PROGRESS) / MAX_NUM_NO_PROGRESS
    planes[offset + _HALFMOVE_CLOCK_CHANNEL][:, :] = min(board_b.halfmove_clock, MAX_NUM_NO_PROGRESS) / MAX_NUM_NO_PROGRESS

    planes[_REPETITION_2_CHANNEL][:, :] = _safe_repetition_flag(board_a, 2)
    planes[offset + _REPETITION_2_CHANNEL][:, :] = _safe_repetition_flag(board_b, 2)
    planes[_REPETITION_3_CHANNEL][:, :] = _safe_repetition_flag(board_a, 3)
    planes[offset + _REPETITION_3_CHANNEL][:, :] = _safe_repetition_flag(board_b, 3)

    if clocks is not None:
        _fill_clock_planes(planes, offset, team_side, clocks)

    if flip:
        a_block = planes[:offset].copy()
        planes[:offset] = planes[offset:]
        planes[offset:] = a_block

    return planes
