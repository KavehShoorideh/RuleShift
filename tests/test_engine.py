import pytest

from ruleshift.engine import DRAW, LOSS, WIN, Engine
from ruleshift.rules import Ruleset


def E(**kw):
    return Engine(Ruleset(**kw))


# ------------------------------------------------------------------ lines
def test_line_counts():
    assert len(E(m=3, n=3, k=3).lines) == 8
    assert len(E(m=3, n=3, k=3, torus=True).lines) == 12  # 3 rows + 3 cols + 3 + 3 wrapped diags
    assert len(E(m=4, n=4, k=3).lines) == 24
    assert len(E(m=7, n=6, k=4).lines) == 69  # classic Connect Four line count
    assert len(E(m=3, n=3, k=4).lines) == 0  # degenerate: no line fits
    # torus k=4 on width 3: horizontal wrapped lines revisit cells -> only vertical fits
    e = E(m=3, n=6, k=4, torus=True)
    assert all(bin(mask).count("1") == 4 for mask in e.lines)


def test_forbidden_lines_discarded():
    e = E(m=3, n=3, k=3, forbidden=frozenset({(1, 1)}))  # center blocked
    # center is on 4 of the 8 lines
    assert len(e.lines) == 4
    assert e.full == 0b111111111 - (1 << 4)


# ---------------------------------------------------------------- gravity
def test_gravity_moves():
    e = E(m=3, n=3, k=3, gravity=True)
    s = e.initial()
    assert e.legal_moves(s) == [0, 1, 2]  # bottom row
    s = e.apply(s, 1)
    assert e.legal_moves(s) == [0, 4, 2]  # column 1 now lands at (1,1)=4


def test_gravity_forbidden_solid_block():
    e = E(m=3, n=3, k=3, gravity=True, forbidden=frozenset({(0, 1)}))
    # column 1's lowest playable cell is (1,1)=4: stone rests on the block
    assert e.legal_moves(e.initial()) == [0, 4, 2]


def test_gravity_full_column():
    e = E(m=3, n=3, k=4, gravity=True)  # k=4: no lines, lets us fill a column
    s = e.initial()
    for mv in (1, 4, 7):
        s = e.apply(s, mv)
    assert 1 not in [c for c in e.legal_moves(s)]
    assert e.legal_moves(s) == [0, 2]


# ------------------------------------------------------------- terminals
def test_win_and_perspective():
    e = E(m=3, n=3, k=3)
    # X: 0, 1, 2 (bottom row) wins; O plays 3, 4 meanwhile
    _, status = e.play([0, 3, 1, 4, 2])
    assert status == LOSS  # player to move (O) has lost


def test_misere_flips_terminal():
    e = E(m=3, n=3, k=3, misere=True)
    _, status = e.play([0, 3, 1, 4, 2])
    assert status == WIN  # completing a line loses in misere: player to move wins


def test_draw_full_board():
    e = E(m=3, n=3, k=3)
    state, status = e.play([4, 0, 2, 6, 3, 5, 1, 7, 8])
    assert status == DRAW
    assert (state[0] | state[1]) == e.full


def test_torus_wrap_win():
    seq_cells = [1, 5, 6]  # (0,1),(1,2),(2,0): wrapped NE diagonal
    e_t = E(m=3, n=3, k=3, torus=True)
    _, status = e_t.play([1, 0, 5, 3, 6])
    assert status == LOSS  # wrapped diagonal completed
    e = E(m=3, n=3, k=3)
    _, status = e.play([1, 0, 5, 3, 6])
    assert status is None  # broken diagonal is not a line off-torus
    del seq_cells


def test_play_raises_after_end_and_on_illegal():
    e = E(m=3, n=3, k=3)
    with pytest.raises(ValueError):
        e.play([0, 3, 1, 4, 2, 5])  # move after X already won
    with pytest.raises(ValueError):
        e.play([0, 0])


def test_full_status_matches_status_after():
    e = E(m=3, n=3, k=3)
    s, status = e.play([0, 3, 1, 4, 2])
    assert e.full_status(s) == status == LOSS
    s2, st2 = e.play([0, 3, 1, 4])
    assert st2 is None and e.full_status(s2) is None


def test_full_status_rejects_unreachable():
    e = E(m=3, n=3, k=3)
    # player to move already holds the bottom row: impossible
    with pytest.raises(ValueError):
        e.full_status((0b111, 0b11000))


def test_render():
    e = E(m=3, n=3, k=3, forbidden=frozenset({(2, 2)}))
    s, _ = e.play([4, 0])
    out = e.render(s)
    assert out.splitlines() == [". . #", ". X .", "O . ."]
