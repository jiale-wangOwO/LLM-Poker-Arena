"""LLM Poker Arena -- a Texas Hold'em engine for LLM-driven AI players."""

from .cards import (
    CATEGORY_NAMES,
    Card,
    Deck,
    HandResult,
    best_five,
    describe,
    evaluate,
    evaluate5,
    format_cards,
)
from .engine import (
    Action,
    ActionType,
    Event,
    IllegalAction,
    LegalActions,
    Player,
    PlayerStatus,
    Street,
    Table,
    play_hand,
)
from .personas import PERSONAS, Persona, get_persona, persona_keys
from .pot import Pot, PotResult, build_pots, settle
from .strength import postflop_strength, preflop_strength, strength

__all__ = [
    "Action",
    "ActionType",
    "CATEGORY_NAMES",
    "Card",
    "Deck",
    "Event",
    "HandResult",
    "IllegalAction",
    "LegalActions",
    "PERSONAS",
    "Persona",
    "Player",
    "PlayerStatus",
    "Pot",
    "PotResult",
    "Street",
    "Table",
    "best_five",
    "build_pots",
    "describe",
    "evaluate",
    "evaluate5",
    "format_cards",
    "get_persona",
    "persona_keys",
    "play_hand",
    "postflop_strength",
    "preflop_strength",
    "settle",
    "strength",
]

__version__ = "2.0.0"
