"""Hand-strength and percentile tests.

The percentiles come from a generated equity table, so these tests pin the
properties that matter (ordering, coverage, calibration) rather than exact
numbers that would break on every regeneration.
"""

from __future__ import annotations

import pytest

from pokerarena.cards import Card, evaluate
from pokerarena.strength import (
    EQUITY_ORDER,
    describe_strength,
    hand_label,
    hand_token,
    postflop_strength,
    preflop_strength,
    strength,
)


def h(*codes: str) -> list[Card]:
    return [Card.from_str(c) for c in codes]


# --------------------------------------------------------------------------
# The generated table itself
# --------------------------------------------------------------------------
def test_equity_order_covers_exactly_the_169_starting_hands():
    assert len(EQUITY_ORDER) == 169
    assert len(set(EQUITY_ORDER)) == 169

    pairs = [t for t in EQUITY_ORDER if len(t) == 2]
    suited = [t for t in EQUITY_ORDER if t.endswith("s") and len(t) == 3]
    offsuit = [t for t in EQUITY_ORDER if t.endswith("o")]
    assert len(pairs) == 13
    assert len(suited) == 78
    assert len(offsuit) == 78

    # Tokens must be written high-card first, using *poker* rank order (the
    # string "A" < "K" alphabetically, so a naive comparison is wrong).
    rank_order = "23456789TJQKA"
    for token in suited + offsuit:
        assert rank_order.index(token[0]) > rank_order.index(
            token[1]
        ), f"{token} should be written high-card first"


def test_equity_order_puts_premium_hands_on_top():
    top = set(EQUITY_ORDER[:10])
    assert "AA" in top and "KK" in top and "QQ" in top
    assert "AKs" in top
    # The worst hands are the classic junk.
    assert set(EQUITY_ORDER[-8:]) & {"72o", "32o", "62o", "42o"}


# --------------------------------------------------------------------------
# Percentile behaviour
# --------------------------------------------------------------------------
def test_aces_are_the_top_and_junk_is_the_bottom():
    assert preflop_strength(h("As", "Ah")) == 1.0
    assert preflop_strength(h("3s", "2h")) == 0.0


def test_percentiles_are_within_range_for_every_hand():
    deck = [Card(r, s) for r in range(13) for s in range(4)]
    seen = set()
    for i, a in enumerate(deck):
        for b in deck[i + 1 :]:
            value = preflop_strength([a, b])
            assert 0.0 <= value <= 1.0
            seen.add(value)
    assert len(seen) == 169, "every distinct hand must map to its own percentile"


@pytest.mark.parametrize(
    "better,worse",
    [
        (("As", "Ah"), ("Ks", "Kh")),          # AA > KK
        (("Ks", "Kh"), ("Qs", "Qh")),          # KK > QQ
        (("Qs", "Qh"), ("Js", "Jh")),          # QQ > JJ
        (("As", "Ks"), ("As", "Qs")),          # AKs > AQs
        (("As", "Ks"), ("As", "Kh")),          # AKs > AKo
        (("As", "Ah"), ("As", "Ks")),          # AA > AKs
        (("8s", "8h"), ("7s", "7h")),          # 88 > 77
        (("As", "Ks"), ("7s", "2h")),          # AKs > 72o
        (("Js", "Ts"), ("9s", "8s")),          # JTs > 98s
    ],
)
def test_known_hand_orderings(better, worse):
    assert preflop_strength(h(*better)) > preflop_strength(h(*worse))


def test_pairs_rank_above_their_suited_broadway_neighbours():
    """A common sanity check: 88 should beat JTs head to head."""
    assert preflop_strength(h("8s", "8h")) > preflop_strength(h("Js", "Ts"))


def test_suited_beats_offsuit_for_the_same_ranks():
    for high, low in [("A", "K"), ("Q", "J"), ("T", "9"), ("6", "5")]:
        suited = preflop_strength(h(f"{high}s", f"{low}s"))
        offsuit = preflop_strength(h(f"{high}s", f"{low}h"))
        assert suited > offsuit, (high, low)


def test_opposite_suit_order_does_not_change_the_percentile():
    assert preflop_strength(h("As", "Kh")) == preflop_strength(h("Ah", "Ks"))
    assert preflop_strength(h("As", "Ks")) == preflop_strength(h("Ah", "Kh"))


def test_hand_token_is_canonical():
    assert hand_token(h("Kh", "As")) == "AKo"
    assert hand_token(h("Ah", "Kh")) == "AKs"
    assert hand_token(h("Ah", "Ad")) == "AA"
    assert hand_token(h("2h", "7d")) == "72o"


# --------------------------------------------------------------------------
# Post-flop strength
# --------------------------------------------------------------------------
def test_postflop_monsters_score_higher_than_weak_hands():
    board = h("Kh", "8d", "2c")
    trips = postflop_strength(h("Kd", "Kc"), board)
    top_pair = postflop_strength(h("Kd", "Qc"), board)
    nothing = postflop_strength(h("4d", "3c"), board)
    assert trips > top_pair > nothing


def test_made_hand_categories_are_ordered_postflop():
    # One base board, chosen so that each extra category must come from the hole
    # cards -- no accidental flush (only one heart) and no accidental straight.
    # Verified combination by combination (tools/_find_combos.py): on this board
    # a pocket pair yields TWO PAIR, not trips and not a full house, because the
    # board's 8d and 9d already pair.
    base = ("Kh", "8d", "2c", "5h", "9d")
    paired_board = ("Kh", "Ks", "2c", "5h", "9d")
    cases = {
        "Four of a Kind": (("Kd", "Kc"), paired_board),
        "Full House": (("9c", "9s"), paired_board),
        "Flush": (("8d", "4d"), ("Kd", "9d", "2d", "5h", "7s")),
        "Straight": (("6d", "7c"), base),
        "Three of a Kind": (("9c", "9s"), base),
        "Two Pair": (("8h", "2d"), base),
        "One Pair": (("3c", "3s"), base),
        "High Card": (("3d", "4c"), base),
    }

    values: dict[str, float] = {}
    for expected, (hole, board) in cases.items():
        hole_cards, board_cards = h(*hole), h(*board)
        actual = evaluate(hole_cards + board_cards).name
        assert actual == expected, f"test setup wrong: {hole} + {board} is {actual}"
        values[expected] = postflop_strength(hole_cards, board_cards)

    ordered = [
        values["Four of a Kind"],
        values["Full House"],
        values["Flush"],
        values["Straight"],
        values["Three of a Kind"],
        values["Two Pair"],
        values["One Pair"],
        values["High Card"],
    ]
    assert ordered == sorted(ordered, reverse=True), values
    # The spread must be meaningful, not all clustered together.
    assert ordered[0] - ordered[-1] > 0.7


def test_flush_draw_gets_credit_but_does_not_beat_a_made_hand():
    board = h("Kh", "8h", "2c")
    four_flush = postflop_strength(h("Ah", "3h"), board)
    made_pair = postflop_strength(h("Kd", "7c"), board)
    assert four_flush > postflop_strength(h("9d", "3c"), board)
    assert made_pair > four_flush


def test_strength_dispatches_on_board_size():
    hole = h("As", "Ah")
    assert strength(hole, []) == preflop_strength(hole)
    assert strength(hole, h("Kh", "8d", "2c")) == postflop_strength(
        hole, h("Kh", "8d", "2c")
    )


def test_preflop_strength_handles_bad_input():
    assert preflop_strength([]) == 0.0
    assert preflop_strength(h("As")) == 0.0


# --------------------------------------------------------------------------
# Human-readable output
# --------------------------------------------------------------------------
def test_hand_label_reads_naturally():
    assert hand_label(h("As", "Ah"), []) == "Pocket As"
    assert hand_label(h("As", "Ks"), []) == "AK suited"
    assert hand_label(h("As", "Kh"), []) == "AK offsuit"
    assert hand_label(h("Kd", "Kc"), h("Kh", "8d", "2c")) == "Three of a Kind"


@pytest.mark.parametrize(
    "hole,board,band",
    [
        (("As", "Ah"), (), "monster"),
        (("7s", "2h"), (), "very weak"),
        (("Kd", "Kc"), ("Kh", "8d", "2c"), "very strong"),  # top set, not a lock
        (("Kd", "7c"), ("Kh", "8d", "2c"), "weak"),         # top pair, weak kicker
        (("3d", "4c"), ("Kh", "8d", "2c"), "very weak"),    # air
    ],
)
def test_describe_strength_bands(hole, board, band):
    text = describe_strength(h(*hole), h(*board))
    assert band in text
    assert "percentile" in text
