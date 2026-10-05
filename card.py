"""Compatibility shim for the original ``card.py`` module.

The real implementation now lives in :mod:`pokerarena.cards`.  This shim keeps
``from card import Card, Deck`` (and the old constant names) working so existing
scripts and tests do not break.
"""

from __future__ import annotations

from pokerarena.cards import (  # noqa: F401
    CATEGORY_NAMES,
    Deck,
    HandResult,
    RANKS,
    SUITS,
    Card,
    ascii_suits,
    best_five,
    compare,
    describe,
    evaluate,
    evaluate5,
    format_cards,
    make_deck,
)

# The original module exposed human-readable suit names and T-for-ten ranks.
SUITS_LEGACY = ["Hearts", "Diamonds", "Clubs", "Spades"]
VALUES_ORDER_LEGACY = list(RANKS)
HAND_RANKINGS = list(CATEGORY_NAMES)

#: Old names kept as aliases.
SUITS_LIST = SUITS_LEGACY
VALUES_ORDER = VALUES_ORDER_LEGACY


def __getattr__(name: str):
    """Serve ``from card import *``-style access to legacy names lazily.

    The original module and ``hand.py`` were independent, but the old tests do
    ``from card import *`` and then use ``Hand``.  Importing lazily here avoids a
    module-level circular import between the two shims.
    """
    if name == "Hand":
        from hand import Hand  # noqa: PLC0415

        return Hand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Hand"})


__all__ = [
    "CATEGORY_NAMES",
    "Card",
    "Deck",
    "HAND_RANKINGS",
    "Hand",
    "HandResult",
    "RANKS",
    "SUITS",
    "SUITS_LEGACY",
    "SUITS_LIST",
    "VALUES_ORDER",
    "VALUES_ORDER_LEGACY",
    "ascii_suits",
    "best_five",
    "compare",
    "describe",
    "evaluate",
    "evaluate5",
    "format_cards",
    "make_deck",
]
