"""Compatibility shim for the original ``hand.py`` module.

The evaluator now lives in :mod:`pokerarena.cards` and is score-based rather
than return-string based.  ``Hand`` here preserves the original public API
(``evaluate_hand``, ``tiebreaker``, ``compare_hands``, ``best_hand``) on top of
the new engine so old call sites keep working.

Prefer :func:`pokerarena.cards.evaluate` in new code: it is faster (bitmask
based) and its scores compare directly with ``>``.
"""

from __future__ import annotations

from itertools import combinations

from pokerarena.cards import (
    CATEGORY_NAMES,
    HIGH_CARD,
    Card,
    HandResult,
    best_five,
    evaluate,
    evaluate5,
)

HAND_RANKINGS = list(CATEGORY_NAMES)

#: Categories that the original code treated as "the tiebreak is just ranks".
_RANKLIST_CATEGORIES = {"Flush", "High Card"}


class Hand:
    """A five-card hand with the original string-oriented API.

    Accepts either :class:`pokerarena.cards.Card` objects or ``(rank, suit)``
    tuples like the original ``card.py`` produced.
    """

    def __init__(self, cards: list):
        self.cards = [self._coerce(card) for card in cards]
        if len(self.cards) != 5:
            raise ValueError(
                f"Hand needs exactly 5 cards, got {len(self.cards)}"
            )
        self.values = [card.rank_char for card in self.cards]
        self.suits = [card.suit_char for card in self.cards]

    @staticmethod
    def _coerce(card) -> Card:
        if isinstance(card, Card):
            return card
        if isinstance(card, tuple) and len(card) == 2:
            rank, suit = card
            return Card.from_str(f"{Hand._rank_char(rank)}{Hand._suit_char(suit)}")
        # Legacy card objects exposed ``.value``/``.suit`` as strings.  Detect
        # them structurally rather than by isinstance, because the old module is
        # now a shim over pokerarena.cards.
        value = getattr(card, "value", None)
        suit = getattr(card, "suit", None)
        if isinstance(value, str) and isinstance(suit, str):
            return Card.from_str(f"{Hand._rank_char(value)}{Hand._suit_char(suit)}")
        raise TypeError(f"Cannot interpret {card!r} as a card")

    @staticmethod
    def _rank_char(rank) -> str:
        text = str(rank).strip()
        if text == "10":
            return "T"
        if len(text) != 1 or text.upper() not in "23456789TJQKA":
            raise ValueError(f"Unknown rank {rank!r}")
        return text.upper()

    @staticmethod
    def _suit_char(suit) -> str:
        text = str(suit).strip()
        lower = text.lower()
        if lower in {"h", "d", "c", "s"}:
            return lower
        mapping = {
            "hearts": "h",
            "diamonds": "d",
            "clubs": "c",
            "spades": "s",
        }
        if lower in mapping:
            return mapping[lower]
        raise ValueError(f"Unknown suit {suit!r}")

    # -- evaluation --------------------------------------------------------
    @property
    def result(self) -> HandResult:
        return evaluate(self.cards)

    def score(self) -> int:
        return evaluate5(self.cards)

    def category_index(self) -> int:
        return evaluate5(self.cards) // 16**5

    def evaluate_hand(self) -> str:
        """Name of the hand category, e.g. ``"Two Pair"``."""
        return CATEGORY_NAMES[self.category_index()]

    def tiebreaker(self) -> tuple:
        """Tiebreaker values for two hands of the same category.

        The new evaluator packs category and tiebreakers into one integer, so
        exposing the packed value as a one-tuple is enough to make
        ``tb1 > tb2`` behave exactly as the original tuple comparison did.
        """
        return (evaluate5(self.cards),)

    # -- comparison --------------------------------------------------------
    @staticmethod
    def best_hand(player_cards: list, community_cards: list) -> "Hand":
        """Best five-card ``Hand`` from hole cards plus the board."""
        all_cards = [Hand._coerce(c) for c in list(player_cards) + list(community_cards)]
        if len(all_cards) < 5:
            raise ValueError("Need at least 5 cards to build a hand")
        _, combo = best_five(all_cards)
        return Hand(list(combo))

    @staticmethod
    def compare_hands(hand1: "Hand", hand2: "Hand") -> str:
        """``"Player 1 wins"``, ``"Player 2 wins"`` or ``"It's a tie"``."""
        left, right = hand1.score(), hand2.score()
        if left > right:
            return "Player 1 wins"
        if left < right:
            return "Player 2 wins"
        return "It's a tie"

    # -- individual predicates (kept for API compatibility) ----------------
    def is_royal_flush(self) -> bool:
        return self.category_index() == CATEGORY_NAMES.index("Royal Flush")

    def is_straight_flush(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("Straight Flush")

    def is_four_of_a_kind(self) -> bool:
        return self.category_index() == CATEGORY_NAMES.index("Four of a Kind")

    def is_full_house(self) -> bool:
        return self.category_index() == CATEGORY_NAMES.index("Full House")

    def is_flush(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("Flush")

    def is_straight(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("Straight")

    def is_three_of_a_kind(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("Three of a Kind")

    def is_two_pair(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("Two Pair")

    def is_one_pair(self) -> bool:
        return self.category_index() >= CATEGORY_NAMES.index("One Pair")

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Hand({self.evaluate_hand()}: {[c.code for c in self.cards]})"


__all__ = ["HAND_RANKINGS", "Hand", "combinations"]
