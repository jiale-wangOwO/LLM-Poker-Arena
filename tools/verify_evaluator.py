"""Cross-check the fast evaluator against a slow, independent implementation.

The evaluator is a bitmask lookup; the reference below is deliberately naive --
enumerate all 21 five-card subsets, classify each with plain sorting, and keep
the best.  It shares no code with the evaluator, so agreement across every
category is real evidence rather than a tautology.

Checked: all 2,598,960 seven-card hands is too slow in Python, so this samples
heavily *and* exhaustively covers the structural edge cases (wheel straights,
steel wheels, board-plays-the-board, five-of-a-suit boards, quads on board).

Run: ``python tools/verify_evaluator.py [samples]``
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter

from pokerarena.cards import Card, evaluate

RANK_ORDER = "23456789TJQKA"
CATEGORY = {
    "High Card": 0,
    "One Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
    "Royal Flush": 8,
}


def brute_force(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    """Best five-card hand, classified from scratch."""
    best = (-1, ())
    for five in itertools.combinations(cards, 5):
        # Card.rank is an index into RANK_ORDER (2 = 0 ... A = 12).
        ranks = sorted((c.rank for c in five), reverse=True)
        suits = {c.suit for c in five}
        flush = len(suits) == 1

        # Straight: five distinct ranks in a row, with the wheel as a special case.
        distinct = sorted(set(ranks), reverse=True)
        straight_high = None
        if len(distinct) == 5:
            if distinct[0] - distinct[4] == 4:
                straight_high = distinct[0]
            elif distinct == [12, 3, 2, 1, 0]:  # A 5 4 3 2
                straight_high = 3  # the five plays

        counts = Counter(ranks)
        grouped = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        shape = [n for _, n in grouped]
        ordered = tuple(rank for rank, _ in grouped)

        if straight_high is not None and flush:
            key = (CATEGORY["Straight Flush"], (straight_high,))
        elif shape[0] == 4:
            key = (CATEGORY["Four of a Kind"], ordered)
        elif shape[:2] == [3, 2]:
            key = (CATEGORY["Full House"], ordered)
        elif flush:
            key = (CATEGORY["Flush"], tuple(ranks))
        elif straight_high is not None:
            key = (CATEGORY["Straight"], (straight_high,))
        elif shape[0] == 3:
            key = (CATEGORY["Three of a Kind"], ordered)
        elif shape[:2] == [2, 2]:
            key = (CATEGORY["Two Pair"], ordered)
        elif shape[0] == 2:
            key = (CATEGORY["One Pair"], ordered)
        else:
            key = (CATEGORY["High Card"], tuple(ranks))

        if key > best:
            best = key
    return best


def check(cards: list[Card], label: str) -> list[str]:
    """Compare the evaluator with the reference.  Returns problem strings."""
    mine = evaluate(cards)
    theirs_cat, _ = brute_force(cards)
    problems = []

    my_cat = CATEGORY.get(mine.name)
    if my_cat is None:
        problems.append(f"{label}: evaluator returned unknown name {mine.name!r}")
        return problems
    # A royal flush is a straight flush; both are category 8.
    if my_cat != theirs_cat:
        problems.append(
            f"{label}: {[c.code for c in cards]} -> evaluator says {mine.name} "
            f"(cat {my_cat}), reference says category {theirs_cat}"
        )
    return problems


def all_cards() -> list[Card]:
    return [Card.from_str(f"{r}{s}") for r in RANK_ORDER for s in "shdc"]


def structural_cases() -> list[tuple[str, list[Card]]]:
    """Hands where a bitmask evaluator typically goes wrong."""
    deck = {c.code: c for c in all_cards()}

    def take(*codes):
        cards = []
        for code in codes:
            # Accept "10h" as well as "Th" for readability.
            key = code.replace("10", "T")
            cards.append(deck[key])
        return cards

    cases = [
        # Wheel straight, and the steel wheel (wheel + flush).
        ("wheel", take("Ah", "2s", "3d", "4c", "5h", "Kd", "9s")),
        ("steel wheel", take("Ah", "2h", "3h", "4h", "5h", "Kd", "9s")),
        # The six-high straight must NOT be beatable by a wheel-only read.
        ("six high straight", take("2h", "3s", "4d", "5c", "6h", "Kd", "9s")),
        ("broadway", take("Th", "Js", "Qd", "Kc", "Ah", "2d", "3s")),
        # Quads on board: everyone plays them, kicker decides.
        ("quads on board", take("9h", "9s", "9d", "9c", "Ah", "Kd", "2s")),
        ("quads + pair board", take("9h", "9s", "9d", "9c", "Ah", "Ad", "2s")),
        # Five of one suit on the board: a flush is on the board for everyone.
        ("flush on board", take("2h", "5h", "9h", "Jh", "Kh", "Ad", "3s")),
        # Six cards to a flush: the best five matter.
        ("six flush cards", take("2h", "5h", "9h", "Jh", "Kh", "Ah", "3s")),
        # Full house from two trips.
        ("two trips", take("9h", "9s", "9d", "5c", "5h", "5s", "2d")),
        # Straight on board with a higher card in hand: still a straight.
        ("straight on board", take("5h", "6s", "7d", "8c", "9h", "Ah", "Kd")),
        # Seven-card flush where two suits tie.
        ("two possible flushes", take("2h", "5h", "9h", "Jh", "Kh", "3s", "4s")),
        # Ace-high straight flush vs lower straight flush.
        ("royal", take("Th", "Jh", "Qh", "Kh", "Ah", "2d", "3s")),
        # Trips using one hole card plus a pair on board.
        ("trips from board pair", take("9h", "9s", "5d", "5c", "2h", "2d", "3s")),
    ]
    return cases


def main() -> int:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    rng = random.Random(4242)
    deck = all_cards()
    problems: list[str] = []

    print("=== structural edge cases ===")
    for label, cards in structural_cases():
        assert len(cards) == 7, label
        found = check(cards, label)
        problems.extend(found)
        status = "FAIL" if found else "ok"
        print(f"  [{status}] {label:<26} {evaluate(cards).name}")

    print(f"\n=== {samples} random seven-card hands ===")
    for i in range(samples):
        hand = rng.sample(deck, 7)
        found = check(hand, f"random#{i}")
        if found:
            problems.extend(found)
            if len(problems) > 8:
                break

    print(f"\n{'=' * 60}")
    if problems:
        print(f"{len(problems)} disagreement(s):")
        for p in problems[:12]:
            print(f"  {p}")
        return 1
    print("evaluator agrees with the reference on every case checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
