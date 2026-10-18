"""board2planes must match the engine's board_to_planes (Fairy-Stockfish rules)."""

import chess
import numpy as np

from hivemind.constants import BOARD_A, NUM_BUGHOUSE_CHANNELS_PER_BOARD
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes

EP_CHANNEL = 24
LAST_TO_CHANNEL = 33
HALFMOVE_CHANNEL = 34


def play(moves):
    board = BughouseBoard()
    for board_num, uci in moves:
        board.push_uci(board_num, uci)
    return board


def test_ep_square_only_when_capture_is_legal():
    # No black pawn can take on e3.
    board = play([(BOARD_A, "e2e4")])
    assert board.boards[BOARD_A].ep_square is not None
    planes = board2planes(board, chess.WHITE)
    assert not planes[EP_CHANNEL].any()

    # d7d5 next to the e5 pawn: e5xd6 is legal.
    board = play([(BOARD_A, "e2e4"), (BOARD_A, "a7a6"),
                  (BOARD_A, "e4e5"), (BOARD_A, "d7d5")])
    planes = board2planes(board, chess.WHITE)
    assert np.flatnonzero(planes[EP_CHANNEL]).tolist() == [chess.D6]


def test_castling_last_move_targets_rook_square():
    board = BughouseBoard()
    board.boards[BOARD_A].set_fen("r3k2r/8/8/8/8/8/8/R3K2R[] w KQkq - 0 1")
    board.push_uci(BOARD_A, "e1g1")
    planes = board2planes(board, chess.WHITE)
    assert np.flatnonzero(planes[LAST_TO_CHANNEL]).tolist() == [chess.H1]

    board = BughouseBoard()
    board.boards[BOARD_A].set_fen("r3k2r/8/8/8/8/8/8/R3K2R[] b KQkq - 0 1")
    board.push_uci(BOARD_A, "e8c8")
    # Team White's view of board A is unflipped, so a8 stays a8.
    planes = board2planes(board, chess.WHITE)
    assert np.flatnonzero(planes[LAST_TO_CHANNEL]).tolist() == [chess.A8]


def test_one_square_king_move_is_not_treated_as_castling():
    board = BughouseBoard()
    board.boards[BOARD_A].set_fen("4k3/8/8/8/8/8/8/4K3[] w - - 0 1")
    board.push_uci(BOARD_A, "e1f1")
    planes = board2planes(board, chess.WHITE)
    assert np.flatnonzero(planes[LAST_TO_CHANNEL]).tolist() == [chess.F1]


def test_pawn_drop_resets_halfmove_clock():
    board = BughouseBoard()
    board.boards[BOARD_A].set_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[Pn] w KQkq - 5 3")
    board.push_uci(BOARD_A, "P@e4")
    assert board.boards[BOARD_A].halfmove_clock == 0

    board.push_uci(BOARD_A, "N@e5")  # piece drops still count
    assert board.boards[BOARD_A].halfmove_clock == 1
    planes = board2planes(board, chess.WHITE)
    assert np.allclose(planes[HALFMOVE_CHANNEL], 1 / 50)
    assert not planes[NUM_BUGHOUSE_CHANNELS_PER_BOARD + HALFMOVE_CHANNEL].any()
