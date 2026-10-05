"""Hand evaluator tests, including the classic edge cases.

Includes a brute-force cross-check against an independent reference
implementation so a scoring regression cannot slip through.
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from pokerarena.cards import (
    CATEGORY_NAMES,
    Card,
    Deck,
    best_five,
    describe,
    evaluate,
    evaluate5,
    format_cards,
)


def hand(*codes: str) -> list[Card]:
    return [Card.from_str(code) for code in codes]


# --------------------------------------------------------------------------
# Category detection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "codes,expected",
    [
        (("As", "Ks", "Qs", "Js", "Ts"), "Royal Flush"),
        (("9h", "8h", "7h", "6h", "5h"), "Straight Flush"),
        (("As", "2s", "3s", "4s", "5s"), "Straight Flush"),  # steel wheel
        (("5c", "5d", "5h", "5s", "Kd"), "Four of a Kind"),
        (("2c", "2d", "2h", "2s", "Ad"), "Four of a Kind"),
        (("Kc", "Kd", "Kh", "Qd", "Qh"), "Full House"),
        (("Ac", "Ad", "Ah", "2d", "2h"), "Full House"),
        (("Ad", "Jd", "9d", "6d", "3d"), "Flush"),
        (("Ah", "Kd", "Qc", "Js", "Th"), "Straight"),
        (("Ah", "2d", "3c", "4s", "5h"), "Straight"),  # wheel
        (("7c", "7d", "7h", "Ad", "Kc"), "Three of a Kind"),
        (("Ac", "Ad", "Kh", "Kd", "2c"), "Two Pair"),
        (("Ac", "Ad", "Kh", "Qd", "2c"), "One Pair"),
        (("Ac", "Kd", "Qh", "Js", "9c"), "High Card"),
    ],
)
def test_categories(codes, expected):
    assert describe(evaluate5(hand(*codes))) == expected


def test_wheel_is_not_a_straight_when_broken():
    # A-2-3-4-6 is not a straight.
    assert describe(evaluate5(hand("Ah", "2d", "3c", "4s", "6h"))) == "High Card"


def test_royal_flush_is_the_top_straight_flush():
    royal = evaluate5(hand("As", "Ks", "Qs", "Js", "Ts"))
    king_high = evaluate5(hand("Ks", "Qs", "Js", "Ts", "9s"))
    assert royal > king_high
    assert describe(royal) == "Royal Flush"


def test_category_ordering_matches_poker_rules():
    scores = [
        evaluate5(hand("Ac", "Kd", "Qh", "Js", "9c")),  # high card
        evaluate5(hand("Ac", "Ad", "Kh", "Qd", "2c")),  # one pair
        evaluate5(hand("Ac", "Ad", "Kh", "Kd", "2c")),  # two pair
        evaluate5(hand("7c", "7d", "7h", "Ad", "Kc")),  # trips
        evaluate5(hand("Ah", "Kd", "Qc", "Js", "Th")),  # straight
        evaluate5(hand("Ad", "Jd", "9d", "6d", "3d")),  # flush
        evaluate5(hand("Kc", "Kd", "Kh", "Qd", "Qh")),  # full house
        evaluate5(hand("5c", "5d", "5h", "5s", "Kd")),  # quads
        evaluate5(hand("9h", "8h", "7h", "6h", "5h")),  # straight flush
    ]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


# --------------------------------------------------------------------------
# Tie-breakers
# --------------------------------------------------------------------------
def test_kicker_decides_one_pair():
    better = evaluate5(hand("Ac", "Ad", "Kh", "Qd", "2c"))
    worse = evaluate5(hand("Ac", "Ad", "Jh", "Qd", "2c"))
    assert better > worse


def test_higher_pair_wins_two_pair():
    better = evaluate5(hand("Ac", "Ad", "Kh", "Kd", "2c"))
    worse = evaluate5(hand("Ac", "Ad", "Qh", "Qd", "Kc"))
    assert better > worse


def test_two_pair_kicker_is_the_fifth_card():
    better = evaluate5(hand("Kc", "Kd", "Qh", "Qd", "Ac"))
    worse = evaluate5(hand("Kc", "Kd", "Qh", "Qd", "Jc"))
    assert better > worse


def test_full_house_compares_trips_first():
    better = evaluate5(hand("Kc", "Kd", "Kh", "Qd", "Qh"))
    worse = evaluate5(hand("Qc", "Qd", "Qh", "Kd", "Kh"))
    assert better > worse


def test_flush_compares_fifth_card():
    better = evaluate5(hand("Ad", "Kd", "Qd", "Jd", "9d"))
    worse = evaluate5(hand("Ac", "Kc", "Qc", "Tc", "9c"))
    assert better > worse


def test_wheel_loses_to_six_high_straight():
    wheel = evaluate5(hand("Ah", "2d", "3c", "4s", "5h"))
    six_high = evaluate5(hand("2h", "3d", "4c", "5s", "6h"))
    assert six_high > wheel


def test_ace_high_straight_beats_king_high():
    assert evaluate5(hand("Ah", "Kd", "Qc", "Js", "Th")) > evaluate5(
        hand("Kh", "Qd", "Jc", "Ts", "9h")
    )


def test_identical_ranks_tie_regardless_of_suit():
    a = evaluate5(hand("Ah", "Kd", "Qc", "Js", "Th"))
    b = evaluate5(hand("As", "Kc", "Qd", "Jh", "Tc"))
    assert a == b


def test_quad_kicker_matters():
    assert evaluate5(hand("5c", "5d", "5h", "5s", "Kd")) > evaluate5(
        hand("5c", "5d", "5h", "5s", "Qd")
    )


def test_trips_kickers_matter():
    assert evaluate5(hand("7c", "7d", "7h", "Ad", "Kc")) > evaluate5(
        hand("7c", "7d", "7h", "Ad", "Qc")
    )


# --------------------------------------------------------------------------
# Best-of-N
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hole,board,expected",
    [
        (("Ah", "Kh"), ("Qh", "Jh", "Th", "2d", "3c"), "Royal Flush"),
        (("Ah", "Ad"), ("Ac", "Kh", "Kd", "Qh", "Jc"), "Full House"),
        (("7c", "8c"), ("9c", "Tc", "Jc", "Ah", "Ad"), "Straight Flush"),
        (("Qh", "Qd"), ("Ks", "Kh", "Kc", "2d", "3c"), "Full House"),
        (("8h", "9h"), ("Th", "Jh", "Qd", "2h", "3h"), "Flush"),
        (("Jc", "Jd"), ("Qh", "Qs", "Kc", "Kh", "2d"), "Two Pair"),
        (("5c", "9c"), ("6c", "7c", "8c", "Th", "Jd"), "Straight Flush"),
        (("Ah", "Kd"), ("Qc", "Js", "Th", "9d", "2c"), "Straight"),
        (("7h", "7d"), ("8c", "8s", "9h", "9d", "Tc"), "Two Pair"),
        (("Qh", "Kh"), ("Ah", "2h", "3h", "Td", "Jc"), "Flush"),
        (("4h", "4d"), ("5h", "5d", "6c", "6s", "7h"), "Two Pair"),
        (("As", "2h"), ("3d", "4c", "5s", "Th", "Jd"), "Straight"),
    ],
)
def test_best_hand_from_seven(hole, board, expected):
    assert describe(evaluate(hand(*hole) + hand(*board))) == expected


def test_seven_card_evaluation_handles_all_sizes():
    cards = hand("As", "Ks", "Qs", "Js", "Ts", "2d", "3c")
    assert describe(evaluate(cards[:5])) == "Royal Flush"
    assert describe(evaluate(cards[:6])) == "Royal Flush"
    assert describe(evaluate(cards)) == "Royal Flush"
    with pytest.raises(ValueError):
        evaluate(cards[:4])


def test_best_five_returns_the_actual_cards():
    result, combo = best_five(hand("As", "Ks", "Qs", "Js", "Ts", "2d", "3c"))
    assert result.name == "Royal Flush"
    assert {card.code for card in combo} == {"As", "Ks", "Qs", "Js", "Ts"}


# --------------------------------------------------------------------------
# Brute-force cross-check against an independent reference evaluator
# --------------------------------------------------------------------------
_REF_RANKS = "23456789TJQKA"


def _reference_score(cards: list[Card]) -> tuple:
    """A deliberately naive, independent evaluator used only for cross-checks."""
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = {c.suit for c in cards}
    counts: dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    # Sort by (count, rank) descending -> groups like quads/trips/pairs first.
    groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    shape = [count for _, count in groups]
    ordered_ranks = [rank for rank, _ in groups]

    unique = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [12, 3, 2, 1, 0]:
            straight_high = 3

    if len(suits) == 1 and straight_high is not None:
        return (9 if straight_high == 12 else 8, straight_high)
    if shape == [4, 1]:
        return (7,) + tuple(ordered_ranks)
    if shape == [3, 2]:
        return (6,) + tuple(ordered_ranks)
    if len(suits) == 1:
        return (5,) + tuple(ranks)
    if straight_high is not None:
        return (4, straight_high)
    if shape[0] == 3:
        return (3,) + tuple(ordered_ranks)
    if shape[:2] == [2, 2]:
        return (2,) + tuple(ordered_ranks)
    if shape[0] == 2:
        return (1,) + tuple(ordered_ranks)
    return (0,) + tuple(ranks)


@pytest.mark.parametrize("seed", range(40))
def test_five_card_scores_agree_with_reference(seed):
    rng = random.Random(seed)
    deck = [Card(rank, suit) for rank in range(13) for suit in range(4)]
    for _ in range(120):
        cards = rng.sample(deck, 5)
        mine = evaluate5(cards)
        reference = _reference_score(cards)
        assert (mine // 16**5) == reference[0], (format_cards(cards), reference)
        assert describe(mine) == CATEGORY_NAMES[reference[0]]


@pytest.mark.parametrize("seed", range(12))
def test_ordering_matches_reference_over_random_hands(seed):
    """Compare every random pair: my score order must match the reference order."""
    rng = random.Random(1000 + seed)
    deck = [Card(rank, suit) for rank in range(13) for suit in range(4)]
    hands = [rng.sample(deck, 5) for _ in range(60)]
    for a, b in combinations(hands, 2):
        mine_a, mine_b = evaluate5(a), evaluate5(b)
        ref_a, ref_b = _reference_score(a), _reference_score(b)
        if ref_a == ref_b:
            assert mine_a == mine_b, (format_cards(a), format_cards(b))
        elif ref_a > ref_b:
            assert mine_a > mine_b, (format_cards(a), format_cards(b))


# --------------------------------------------------------------------------
# Deck behaviour
# --------------------------------------------------------------------------
def test_deck_is_complete_and_unique():
    deck = Deck(seed=7)
    unique = {(c.rank, c.suit) for c in deck.cards}
    assert len(deck.cards) == 52
    assert len(unique) == 52


def test_deck_seed_is_reproducible():
    assert [c.code for c in Deck(seed=3).cards] == [c.code for c in Deck(seed=3).cards]


def test_deck_deal_and_burn_accounting():
    deck = Deck(seed=1)
    dealt = deck.deal(5)
    assert len(dealt) == 5
    assert len(deck) == 47
    deck.burn(3)
    assert len(deck) == 44


def test_deck_exhaustion_raises():
    deck = Deck(seed=1)
    deck.deal(52)
    with pytest.raises(ValueError):
        deck.deal_one()


def test_card_parsing_accepts_common_notations():
    assert Card.from_str("As").code == "As"
    assert Card.from_str("10h").code == "Th"
    assert Card.from_str("TD").code == "Td"
    assert Card.from_str(" 2C ").code == "2c"
    with pytest.raises(ValueError):
        Card.from_str("Zz")
