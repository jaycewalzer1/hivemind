import chess
from chess.variant import CrazyhouseBoard

from hivemind.domain.move2planes import make_label_index, make_map, policy_label

# Indices the engine's get_fast_policy_index returns for these actions; the
# released network's policy follows them (pass logit sits on slot 0, O-O on
# e1h1).
ENGINE_PASS_INDEX = 0
ENGINE_KINGSIDE_CASTLE_INDEX = 1412  # e1h1
ENGINE_QUEENSIDE_CASTLE_INDEX = 3268  # e1a1

CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R[] {} KQkq - 0 1"


def label_index(board, uci):
    return make_label_index()[policy_label(board, board.parse_uci(uci))]


def test_pass_uses_first_policy_slot():
    labels = make_map()
    assert labels[:64] == ["pass"] * 64
    assert make_label_index()["pass"] == ENGINE_PASS_INDEX


def test_label_index_keeps_first_slot_of_every_label():
    labels = make_map()
    for label, index in make_label_index().items():
        assert labels.index(label) == index


def test_castling_is_encoded_king_to_rook_for_both_colors():
    white = CrazyhouseBoard(CASTLING_FEN.format("w"))
    black = CrazyhouseBoard(CASTLING_FEN.format("b"))

    assert policy_label(white, white.parse_uci("e1g1")) == "e1h1"
    assert policy_label(white, white.parse_uci("e1c1")) == "e1a1"
    assert policy_label(black, black.parse_uci("e8g8")) == "e1h1"
    assert policy_label(black, black.parse_uci("e8c8")) == "e1a1"
    assert label_index(white, "e1g1") == ENGINE_KINGSIDE_CASTLE_INDEX
    assert label_index(white, "e1c1") == ENGINE_QUEENSIDE_CASTLE_INDEX
    assert label_index(black, "e8g8") == ENGINE_KINGSIDE_CASTLE_INDEX
    assert label_index(black, "e8c8") == ENGINE_QUEENSIDE_CASTLE_INDEX


def test_rook_move_with_castling_like_squares_is_not_rewritten():
    board = CrazyhouseBoard("4k3/8/8/8/8/8/8/4R1K1[] w - - 0 1")
    move = board.parse_uci("e1f1")
    assert not board.is_castling(move)
    assert policy_label(board, move) == "e1f1"

    board = CrazyhouseBoard("4k3/8/8/8/8/8/8/K3R3[] w - - 0 1")
    move = board.parse_uci("e1g1")
    assert not board.is_castling(move)
    assert policy_label(board, move) == "e1g1"


def test_promotions_and_drops():
    board = CrazyhouseBoard("4k3/1P6/8/8/8/8/8/4K3[N] w - - 0 1")
    assert policy_label(board, board.parse_uci("b7b8q")) == "b7b8"
    assert policy_label(board, board.parse_uci("b7b8n")) == "b7b8n"
    assert policy_label(board, board.parse_uci("N@d4")) == "N@d4"

    board = CrazyhouseBoard("4k3/8/8/8/8/8/1p6/4K3[n] b - - 0 1")
    assert policy_label(board, board.parse_uci("b2b1q")) == "b7b8"
    assert policy_label(board, board.parse_uci("b2b1n")) == "b7b8n"
    assert policy_label(board, board.parse_uci("N@d4")) == "N@d5"
