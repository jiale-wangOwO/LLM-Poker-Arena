"""Deck sanity check: 52 unique cards, dealt and counted.

Run directly (``python test/test_deck.py``) or collect it with pytest.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from card import Deck  # noqa: E402


def test_deck_has_52_unique_cards_and_counts_correctly():
    deck = Deck(seed=1234)
    assert len(deck.cards) == 52
    assert len({(card.rank_char, card.suit_char) for card in deck.cards}) == 52

    value_count: Counter[str] = Counter()
    suit_count: Counter[str] = Counter()
    for _ in range(52):
        card = deck.deal_one()
        value_count[card.rank_char] += 1
        suit_count[card.suit_char] += 1

    assert len(deck) == 0
    assert set(value_count.values()) == {4}, "each rank must appear four times"
    assert set(suit_count.values()) == {13}, "each suit must appear 13 times"


if __name__ == "__main__":
    deck = Deck()
    print("Shuffled deck:")
    print(deck.cards)

    value_count: Counter[str] = Counter()
    suit_count: Counter[str] = Counter()
    for _ in range(52):
        card = deck.deal_one()
        print(card)
        value_count[card.rank_char] += 1
        suit_count[card.suit_char] += 1

    print("Value count:", dict(value_count))
    print("Suit count:", dict(suit_count))
    test_deck_has_52_unique_cards_and_counts_correctly()
    print("test_deck passed!")
