"""What a seat is told before it acts.

The point of these tests: a poker decision is not a function of "my cards and
the pot".  A seat must receive the public picture a real player has -- position,
the ordered betting history of the hand, effective stacks, and what happened
over the last few hands -- or it degenerates into a card-strength calculator.

Each test pins one of those, because each was genuinely missing at some point.
"""

from __future__ import annotations

import pytest

from pokerarena.ai import (
    describe_table,
    recent_hand_summaries,
    format_hand_log,
)
from pokerarena.arena import AISpec, Arena, build_players, heuristic_transports
from pokerarena.config import ArenaConfig
from pokerarena.engine import Player, Table


def make_table(names=("A", "B", "C"), chips=1000, seed=1, button=0):
    players = [Player(name=n, seat=i, chips=chips) for i, n in enumerate(names)]
    table = Table(players, 10, 20, seed=seed)
    table.button = button
    return table


def prompt_for(table: Table, seat: int, recent=None) -> str:
    return describe_table(table, seat, recent_hands=recent)


# --------------------------------------------------------------------------
# The betting history
# --------------------------------------------------------------------------
def test_the_betting_history_is_present_and_ordered():
    table = make_table(seed=3)
    table.start_hand()
    # Everyone calls/checks to the big blind.
    for _ in range(3):
        if table.is_hand_over or table.actor is None:
            break
        table.apply_action(table.actor, _passive(table, table.actor))

    seat = table.actor if table.actor is not None else 0
    prompt = prompt_for(table, seat)
    assert "BETTING SO FAR THIS HAND" in prompt

    log = prompt.split("BETTING SO FAR THIS HAND")[1]
    assert "posts the small blind" in log
    assert "posts the big blind" in log
    # Order matters: the blinds come before anything else.
    assert log.index("small blind") < log.index("big blind")


def test_a_raise_is_visible_as_a_raise():
    """The distinction the old prompt could not express at all."""
    table = make_table(seed=3)
    table.start_hand()
    for _ in range(4):
        seat = table.actor
        if seat is None or table.is_hand_over:
            break
        legal = table.legal_actions(seat)
        if legal.can_raise:
            table.apply_action(seat, _raise_to(table, seat, legal.min_raise_to))
        else:
            table.apply_action(seat, _passive(table, seat))

    text = "\n".join(format_hand_log(table, viewer=0))
    assert "raises to" in text, f"no raise recorded in:\n{text}"


def test_the_log_resets_each_hand():
    table = make_table(seed=3)
    table.start_hand()
    table.apply_action(table.actor, _passive(table, table.actor))
    assert table.hand_log, "the log should have content"
    table.start_hand()
    # Blinds are re-posted for the new hand, but nothing carries over.
    assert all(entry["action"] == "post_blind" for entry in table.hand_log)


# --------------------------------------------------------------------------
# Position
# --------------------------------------------------------------------------
def test_position_is_stated():
    """Position is a property of the table, so it is readable for any seat.

    ``describe_table`` needs a seat that can actually act (it embeds the legal
    menu), so the position helper is exercised directly for the blinds.
    """
    from pokerarena.ai import _position_note

    table = make_table(seed=3)
    table.start_hand()
    small, big = table.blind_seats()

    assert "the button" in _position_note(table, table.button)
    assert "the small blind" in _position_note(table, small)
    assert "the big blind" in _position_note(table, big)
    # And it says who is still to act behind, which is the actionable part.
    button_note = _position_note(table, table.button)
    assert "act after you" in button_note or "act LAST" in button_note


def test_position_appears_in_the_prompt():
    table = make_table(seed=3)
    table.start_hand()
    assert "Your position:" in prompt_for(table, table.actor)


# --------------------------------------------------------------------------
# Stacks
# --------------------------------------------------------------------------
def test_effective_stacks_are_stated():
    table = make_table(chips=1000, seed=3)
    table.start_hand()
    # Shrink one opponent *after* the blinds are posted, so the effective stack
    # is genuinely 200 rather than being absorbed by the blind.
    table.players[1].chips = 200
    table.players[1].street_bet = 0
    prompt = prompt_for(table, 0)
    assert "EFFECTIVE STACKS" in prompt
    assert "200 behind" in prompt
    assert "10 BB" in prompt, "the effective stack should be given in big blinds"


def test_spr_is_stated():
    table = make_table(seed=3)
    table.start_hand()
    table.apply_action(table.actor, _passive(table, table.actor))
    prompt = prompt_for(table, table.actor or 0)
    assert "SPR" in prompt


# --------------------------------------------------------------------------
# Session history
# --------------------------------------------------------------------------
def test_recent_hands_are_summarised_with_showdowns():
    specs = [
        AISpec(name="A", persona_key="maniac", model=""),
        AISpec(name="B", persona_key="rock", model=""),
        AISpec(name="C", persona_key="pro", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(
        players,
        config=ArenaConfig(seed=9, max_rounds=12),
        transports=heuristic_transports(players, 9),
    )
    arena.play_game()
    summaries = recent_hand_summaries(arena.hand_history)
    assert summaries, "there should be hands to remember"
    assert any("won" in line for line in summaries)
    # Newest first.
    assert "Hand " in summaries[0]


def test_recent_hands_are_capped():
    history = [
        {"hand_number": i, "winners": ["A"], "pot": 100, "showdown": []}
        for i in range(1, 40)
    ]
    summaries = recent_hand_summaries(history, limit=4)
    assert len(summaries) == 4
    assert "Hand 39" in summaries[0]
    assert "Hand 36" in summaries[-1]


def test_recent_hands_are_in_the_prompt():
    table = make_table(seed=3)
    table.start_hand()
    prompt = prompt_for(
        table, table.actor, recent=["  - Hand 1: A won 300  (A showed As Ks (One Pair))"]
    )
    assert "RECENT HANDS AT THIS TABLE" in prompt
    assert "A won 300" in prompt


def test_no_recent_hands_section_before_the_first_hand():
    table = make_table(seed=3)
    table.start_hand()
    assert "RECENT HANDS" not in prompt_for(table, table.actor, recent=[])


# --------------------------------------------------------------------------
# The memory records
# --------------------------------------------------------------------------
def test_records_include_what_the_others_did():
    """A seat must be able to notice an opponent who keeps raising."""
    from pokerarena.ai import AIDecider
    from pokerarena.personas import get_persona
    from tests.test_context import ScriptedTransport

    table = make_table(seed=3)
    table.start_hand()
    decider = AIDecider(
        get_persona("pro"),
        ScriptedTransport(["<action>call</action><thought>x</thought>"] * 8),
        memory_turns=8,
    )
    # Seat 0 acts, then seat 1 acts, then seat 0 is asked again.
    first = table.actor
    decider.decide(table, first)
    table.apply_action(first, _passive(table, first))
    other = table.actor
    if other is not None and other != first:
        table.apply_action(other, _passive(table, other))
    if table.actor == first:
        decider.decide(table, first)

    records = [m["content"] for m in decider._sent_messages(table.players[first])]
    joined = "\n".join(records)
    assert "while you waited" in joined, (
        "the record must say what happened between this seat's turns:\n" + joined
    )


def test_the_model_is_asked_to_consider_the_table():
    """The prompt should push back on pure hand-strength thinking."""
    table = make_table(seed=3)
    table.start_hand()
    prompt = prompt_for(table, table.actor)
    assert "what the betting so far tells you" in prompt
    assert "your own line looks like to them" in prompt


def _passive(table: Table, seat: int):
    from pokerarena.engine import Action, ActionType

    legal = table.legal_actions(seat)
    if legal.can_check:
        return Action(ActionType.CHECK)
    return Action(ActionType.CALL, amount=legal.call_cost)


def _raise_to(table: Table, seat: int, amount: int):
    from pokerarena.engine import Action, ActionType

    return Action(ActionType.RAISE, amount=amount)
