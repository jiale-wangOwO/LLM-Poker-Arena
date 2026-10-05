"""Card primitives and a fast bitmask-based hand evaluator.

The evaluator exposes two layers:

* :func:`evaluate5` -- scores exactly five cards.
* :func:`evaluate`  -- scores the best five-card hand out of 5, 6 or 7 cards.

Scores are plain integers ordered so that a larger score is always a strictly
better poker hand.  ``score // 16**5`` yields the hand category index, which
makes category comparisons cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Card representation
# --------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

RANK_INDEX = {r: i for i, r in enumerate(RANKS)}  # '2' -> 0 ... 'A' -> 12
SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}

RANK_NAMES = {
    "T": "10",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
}
SUIT_SYMBOLS = {"c": "\u2663", "d": "\u2666", "h": "\u2665", "s": "\u2660"}
SUIT_ASCII = {"c": "c", "d": "d", "h": "h", "s": "s"}
SUIT_NAMES = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


def ascii_suits() -> bool:
    """True when the active stdout cannot encode suit glyphs.

    Many Windows consoles (and any GBK/CP1252 code page) raise
    ``UnicodeEncodeError`` on U+2660..U+2666, so callers must be able to fall
    back to plain letters.
    """
    import os
    import sys

    forced = os.environ.get("POKERARENA_ASCII")
    if forced is not None:
        return forced.strip().lower() not in ("", "0", "false", "no")
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        "\u2660\u2665\u2666\u2663".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return True
    return False

#: Hand category indices, weakest first.  ``score // 16**5`` returns these.
CATEGORY_NAMES = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",
)
HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_KIND, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_KIND, STRAIGHT_FLUSH, ROYAL_FLUSH = range(10)

_BASE = 16**5


@dataclass(frozen=True, slots=True)
class Card:
    """A single playing card stored as a compact integer.

    ``rank`` is 0-12 (``'2'``..``'A'``) and ``suit`` is 0-3
    (``'c'``, ``'d'``, ``'h'``, ``'s'``).
    """

    rank: int
    suit: int

    # -- construction ------------------------------------------------------
    @classmethod
    def from_str(cls, text: str) -> "Card":
        """Parse ``"As"``, ``"10h"``, ``"TD"`` style notation."""
        cleaned = text.strip()
        if len(cleaned) >= 3 and cleaned[:2] == "10":
            cleaned = "T" + cleaned[2:]
        if len(cleaned) != 2:
            raise ValueError(f"Cannot parse card {text!r}; expected like 'As' or '10h'")
        rank_char, suit_char = cleaned[0].upper(), cleaned[1].lower()
        if rank_char not in RANK_INDEX:
            raise ValueError(f"Unknown rank {rank_char!r} in {text!r}")
        if suit_char not in SUIT_INDEX:
            raise ValueError(f"Unknown suit {suit_char!r} in {text!r}")
        return cls(RANK_INDEX[rank_char], SUIT_INDEX[suit_char])

    # -- rendering ---------------------------------------------------------
    @property
    def rank_char(self) -> str:
        return RANKS[self.rank]

    @property
    def suit_char(self) -> str:
        return SUITS[self.suit]

    @property
    def label(self) -> str:
        """Human label, e.g. ``"10\u2665"`` or ``"10h"`` on a limited console."""
        rank = RANK_NAMES.get(self.rank_char, self.rank_char)
        suit = SUIT_ASCII[self.suit_char] if ascii_suits() else SUIT_SYMBOLS[self.suit_char]
        return f"{rank}{suit}"

    @property
    def code(self) -> str:
        """e.g. ``"Th"`` -- machine friendly."""
        return f"{self.rank_char}{self.suit_char}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.label

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Card('{self.code}')"

    # -- bitmask helpers ---------------------------------------------------
    @property
    def bit(self) -> int:
        return 1 << (self.rank * 4 + self.suit)


def make_deck() -> list[Card]:
    """A fresh, ordered 52-card deck."""
    return [Card(rank, suit) for rank in range(13) for suit in range(4)]


class Deck:
    """A shuffled deck with safe dealing and reshuffling."""

    def __init__(self, rng: random.Random | None = None, seed: int | None = None):
        self._rng = rng or random.Random(seed)
        self.cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        self.cards = make_deck()
        self._rng.shuffle(self.cards)

    def deal(self, count: int = 1) -> list[Card]:
        if count > len(self.cards):
            raise ValueError(f"Cannot deal {count} cards; only {len(self.cards)} left")
        dealt = [self.cards.pop() for _ in range(count)]
        return dealt

    def deal_one(self) -> Card:
        if not self.cards:
            raise ValueError("Deck is empty")
        return self.cards.pop()

    def burn(self, count: int = 1) -> None:
        """Discard the top card(s) -- mirrors live poker procedure."""
        if count > len(self.cards):
            raise ValueError("Not enough cards to burn")
        del self.cards[-count:]

    def __len__(self) -> int:
        return len(self.cards)


# --------------------------------------------------------------------------
# Evaluation internals
# --------------------------------------------------------------------------
_STRAIGHT_MASKS: dict[int, int] = {}
for _high in range(4, 13):  # five-high (index 3) up to ace-high (index 12)
    _mask = 0
    for _offset in range(5):
        _mask |= 1 << (_high - _offset)
    _STRAIGHT_MASKS[_mask] = _high
# Wheel: A-2-3-4-5, ace plays low so the straight is five-high.
_STRAIGHT_MASKS[0b1000000001111] = 3


def _keep_top(rank_mask: int, n: int) -> int:
    kept = 0
    for _ in range(n):
        if not rank_mask:
            break
        highest = rank_mask.bit_length() - 1
        kept |= 1 << highest
        rank_mask ^= 1 << highest
    return kept


@dataclass(frozen=True, slots=True, order=True)
class HandResult:
    """The evaluated value of a five-card hand.

    Ordered by ``category`` then ``score``, so ``result_a > result_b`` means
    ``a`` is the better poker hand.  (``score`` alone would suffice, since the
    category is packed into its high bits, but comparing both fields keeps the
    dataclass-generated ordering explicit and correct.)
    """

    category: int
    score: int

    @property
    def name(self) -> str:
        return CATEGORY_NAMES[self.category]

    def __str__(self) -> str:
        return self.name


def evaluate5(cards: Sequence[Card]) -> int:
    """Score exactly five cards.  Larger is better."""
    if len(cards) != 5:
        raise ValueError(f"evaluate5 needs exactly 5 cards, got {len(cards)}")

    rank_mask = 0
    suit_counts = [0, 0, 0, 0]
    rank_counts = [0] * 13
    for card in cards:
        rank_mask |= 1 << card.rank
        suit_counts[card.suit] += 1
        rank_counts[card.rank] += 1

    is_flush = max(suit_counts) == 5

    straight_high = _STRAIGHT_MASKS.get(rank_mask)
    if is_flush and straight_high is not None:
        category = ROYAL_FLUSH if straight_high == 12 else STRAIGHT_FLUSH
        return category * _BASE + straight_high

    # Count-based categories.  buckets[i] holds the rank mask of ranks that
    # appear exactly i times.
    buckets: list[int] = [0] * 5
    for rank, count in enumerate(rank_counts):
        if count:
            buckets[count] |= 1 << rank

    if buckets[4]:
        quad = buckets[4].bit_length() - 1
        kicker = buckets[1].bit_length() - 1
        return FOUR_KIND * _BASE + (quad << 4) + kicker

    if buckets[3] and buckets[2]:
        triple = buckets[3].bit_length() - 1
        pair = buckets[2].bit_length() - 1
        return FULL_HOUSE * _BASE + (triple << 4) + pair

    if is_flush:
        # All five ranked cards, high to low.
        return FLUSH * _BASE + _pack(rank_mask, 5)

    if straight_high is not None:
        return STRAIGHT * _BASE + straight_high

    if buckets[3]:
        triple = buckets[3].bit_length() - 1
        return THREE_KIND * _BASE + (triple << 8) + _pack(buckets[1], 2)

    if buckets[2].bit_count() == 2:
        high_pair = buckets[2].bit_length() - 1
        low_pair = (buckets[2] ^ (1 << high_pair)).bit_length() - 1
        kicker = buckets[1].bit_length() - 1
        return TWO_PAIR * _BASE + (high_pair << 8) + (low_pair << 4) + kicker

    if buckets[2]:
        pair = buckets[2].bit_length() - 1
        return ONE_PAIR * _BASE + (pair << 12) + _pack(buckets[1], 3)

    return HIGH_CARD * _BASE + _pack(rank_mask, 5)


def _pack(rank_mask: int, count: int) -> int:
    """Pack the top ``count`` ranks of a mask, highest rank in the high nibble."""
    kept = _keep_top(rank_mask, count)
    packed = 0
    shift = (count - 1) * 4
    while kept:
        highest = kept.bit_length() - 1
        packed |= highest << shift
        kept ^= 1 << highest
        shift -= 4
    return packed


def evaluate(cards: Iterable[Card]) -> HandResult:
    """Best five-card score from 5, 6 or 7 cards."""
    card_list = list(cards)
    n = len(card_list)
    if n < 5:
        raise ValueError(f"Need at least 5 cards to evaluate, got {n}")
    if n == 5:
        score = evaluate5(card_list)
    elif n == 6:
        score = max(evaluate5(combo) for combo in combinations(card_list, 5))
    elif n == 7:
        score = max(evaluate5(combo) for combo in combinations(card_list, 5))
    else:
        raise ValueError(f"Cannot evaluate {n} cards (max 7)")
    return HandResult(category=score // _BASE, score=score)


def describe(value: int | HandResult) -> str:
    """Human-readable category name for a score or :class:`HandResult`."""
    score = value.score if isinstance(value, HandResult) else value
    return CATEGORY_NAMES[score // _BASE]


def best_five(cards: Sequence[Card]) -> tuple[HandResult, tuple[Card, ...]]:
    """Return the best score together with the exact five cards achieving it."""
    card_list = list(cards)
    if len(card_list) < 5:
        raise ValueError("Need at least 5 cards")
    best_score = -1
    best_combo: tuple[Card, ...] = ()
    for combo in combinations(card_list, 5):
        score = evaluate5(combo)
        if score > best_score:
            best_score = score
            best_combo = combo
    return HandResult(category=best_score // _BASE, score=best_score), best_combo


def compare(scores: Sequence[int]) -> list[int]:
    """Indices of the winners (more than one on a tie)."""
    if not scores:
        return []
    best = max(scores)
    return [i for i, score in enumerate(scores) if score == best]


def format_cards(cards: Iterable[Card]) -> str:
    return " ".join(card.label for card in cards)
