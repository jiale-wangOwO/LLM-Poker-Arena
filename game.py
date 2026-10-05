"""Compatibility shim for the original ``game.py`` module.

The rules engine was rewritten as :mod:`pokerarena.engine` (correct betting
closure, side pots, split pots, heads-up blinds, all-in run-outs).  This shim
maps the old ``Player``/``PokerGame`` names onto the new classes so existing
scripts keep importing, and points ``PokerGame`` at the new :class:`Table`.

New code should use :class:`pokerarena.engine.Table` directly; it is pure (it
never asks a player what to do) and exposes ``legal_actions`` so callers can
validate any decision before applying it.
"""

from __future__ import annotations

from pokerarena.ai import safe_fallback
from pokerarena.engine import (  # noqa: F401
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

#: The original class name for the table.
PokerGame = Table

__all__ = [
    "Action",
    "ActionType",
    "Event",
    "IllegalAction",
    "LegalActions",
    "Player",
    "PlayerStatus",
    "PokerGame",
    "Street",
    "Table",
    "play_hand",
    "safe_fallback",
]
