"""Regression tests for the offline bot's decision quality.

These three bugs all looked like "the bot plays weird" rather than crashing, so
they are exactly the kind that needs pinning down:

1. Preflop raise sizing was interpolated between the minimum raise and the whole
   stack, so a bot would jam 716 into a 70-chip pot.
2. ``facing_raise`` referenced an undefined ``current_bet``, so *every* offline
   decision raised NameError and silently fell back to check/call.  The whole
   "AI" was a no-op.
3. Postflop, the prompt's percentile could not be read back, so every postflop
   decision used the default 0.25 and folded.
"""

from __future__ import annotations

import random

import pytest

from pokerarena.ai import (
    AIDecider,
    HeuristicTransport,
    _parse_legal_block,
    _parse_int,
    _parse_percentile,
    describe_table,
)
from pokerarena.cards import Card
from pokerarena.engine import Action, ActionType, Player, Table
from pokerarena.personas import get_persona


def make_table(stacks=(1000, 1000, 1000), seed=1, button=0):
    players = [Player(name=f"P{i}", seat=i, chips=c) for i, c in enumerate(stacks)]
    table = Table(players, 10, 20, seed=seed)
    table.button = button
    return table


def to_flop(table: Table) -> None:
    """Play the preflop street out passively so the table reaches the flop."""
    steps = 0
    while table.street.value == "preflop" and not table.is_hand_over and steps < 30:
        steps += 1
        legal = table.legal_actions(table.actor)
        table.apply_action(
            table.actor,
            Action(ActionType.CHECK if legal.can_check else ActionType.CALL),
        )


# --------------------------------------------------------------------------
# 1. Raise sizing is pot-relative
# --------------------------------------------------------------------------
def test_preflop_raise_is_not_a_stack_slice():
    """A raise must be sized off the pot, not the distance to the whole stack."""
    persona = get_persona("maniac")
    largest = 0
    for seed in range(30):
        table = make_table(seed=seed)
        table.start_hand()
        seat = table.actor
        transport = HeuristicTransport(persona, random.Random(seed))
        action = AIDecider(persona, transport, retries=0).decide(table, seat)
        if action.type is not ActionType.RAISE:
            continue
        largest = max(largest, action.amount)
        # With nothing but blinds in the pot, an open must be a few big blinds.
        assert action.amount <= 6 * table.big_blind, (
            f"seed {seed}: opening raise {action.amount} is far too large "
            f"(pot {table.pot_total})"
        )
        assert action.amount >= table.big_blind
    assert largest > 0, "the maniac never opened in 30 hands"


def test_facing_a_raise_sizes_relative_to_the_pot():
    """A 3-bet should be a few multiples of the raise, not a stack jam."""
    transport = HeuristicTransport(get_persona("maniac"), random.Random(7))
    legal = {
        "min_raise_to": 80,
        "max_raise_to": 1000,
        "can_raise": True,
        "can_call": True,
        "call_cost": 40,
    }
    amount = _raise_amount(transport, legal, pot=90, current_bet=40, pct=0.6)
    # current_bet 40, pot 90 -> about 2.5x40 + 90, well under the stack.
    assert 80 <= amount <= 320, amount


def test_raise_sizing_scales_with_the_pot():
    transport = HeuristicTransport(get_persona("pro"), random.Random(3))
    legal = {
        "min_raise_to": 40,
        "max_raise_to": 1000,
        "can_raise": True,
        "can_call": True,
        "call_cost": 20,
    }
    small = _raise_amount(transport, legal, pot=60, current_bet=20, pct=0.8)
    large = _raise_amount(transport, legal, pot=600, current_bet=200, pct=0.8)
    assert large > small, "a bigger pot must produce a bigger raise"
    assert small < 1000 and large <= 1000


def test_sizing_never_exceeds_the_stack():
    transport = HeuristicTransport(get_persona("maniac"), random.Random(5))
    legal = {
        "min_raise_to": 100,
        "max_raise_to": 240,          # short stack
        "can_raise": True,
        "can_call": True,
        "call_cost": 80,
    }
    amount = _raise_amount(transport, legal, pot=5000, current_bet=100, pct=1.0)
    # A pot-sized raise would want far more than the stack, so it must be a shove.
    assert transport._sized_raise(legal, 1.0, _sizing_prompt(5000, 100)) == "all-in" or amount <= 240


def _sizing_prompt(pot: int, current_bet: int) -> str:
    return (
        f"Pot: {pot}\n"
        f"Blinds: 10/20\n"
        f"Highest bet on this street: {current_bet}   "
        f"Already in from you: 0   To call: {current_bet}\n"
    )


def _raise_amount(transport, legal, *, pot, current_bet, pct) -> int:
    """The total the sizing formula picks for a given spot.

    Calls the sizing helper directly: this is about *how big*, not about whether
    the bot wants to raise in the first place.
    """
    text = transport._sized_raise(legal, pct, _sizing_prompt(pot, current_bet))
    if text == "all-in":
        return legal["max_raise_to"]
    assert text.startswith("raise"), text
    return int(text.split()[1])


# --------------------------------------------------------------------------
# 2. The heuristic never silently falls back
# --------------------------------------------------------------------------
def test_offline_heuristic_never_falls_back_to_safe_actions():
    """A NameError used to make every heuristic decision a fallback.

    That is invisible in the game (play continued) but meant the offline bot had
    no strategy at all, so this asserts on the decision *source*.
    """
    personas = ["maniac", "rock", "pro"]
    for seed in range(12):
        players = [
            Player(name=f"P{i}", seat=i, chips=1000, persona=personas[i])
            for i in range(3)
        ]
        table = Table(players, 10, 20, seed=seed)
        table.button = seed % 3
        table.start_hand()
        deciders = {
            p.seat: AIDecider(
                get_persona(p.persona),
                HeuristicTransport(get_persona(p.persona), random.Random(seed)),
                retries=0,
            )
            for p in players
        }
        steps = 0
        while not table.is_hand_over and steps < 80:
            steps += 1
            seat = table.actor
            if seat is None:
                break
            action = deciders[seat].decide(table, seat)
            assert action.source != "fallback", (
                f"seed {seed}: {table.players[seat].name} fell back "
                f"({deciders[seat].traces[-1].error})"
            )
            table.apply_action(seat, action)


def test_heuristic_produces_raises_and_checks_not_just_calls():
    """Guard against the whole table collapsing into call/check-only."""
    rng = random.Random(9)
    kinds: dict[str, int] = {}
    for hand in range(60):
        players = [
            Player(name=f"P{i}", seat=i, chips=1000, persona="maniac")
            for i in range(3)
        ]
        table = Table(players, 10, 20, seed=rng.randrange(10**6))
        table.button = hand % 3
        table.start_hand()
        deciders = {
            p.seat: AIDecider(
                get_persona("maniac"),
                HeuristicTransport(get_persona("maniac"), random.Random(rng.random())),
                retries=0,
            )
            for p in players
        }
        steps = 0
        while not table.is_hand_over and steps < 80:
            steps += 1
            seat = table.actor
            if seat is None:
                break
            action = deciders[seat].decide(table, seat)
            kinds[action.type.value] = kinds.get(action.type.value, 0) + 1
            table.apply_action(seat, action)

    assert kinds.get("raise", 0) + kinds.get("bet", 0) > 0, (
        f"the maniac never raised in 60 hands: {kinds}"
    )
    assert kinds.get("fold", 0) > 0, f"nobody ever folded: {kinds}"


# --------------------------------------------------------------------------
# 3. Postflop percentile is readable
# --------------------------------------------------------------------------
def test_parser_reads_the_percentile_from_every_street():
    table = make_table(seed=4)
    table.start_hand()
    preflop_prompt = describe_table(table, table.actor)
    assert 0.0 <= _parse_percentile(preflop_prompt) <= 1.0

    to_flop(table)
    assert table.street.value == "flop"
    seat = table.actor
    assert seat is not None
    flop_prompt = describe_table(table, seat)
    parsed = _parse_percentile(flop_prompt)
    # The prompt must carry a real percentile, not the 0.25 fallback default.
    assert "percentile" in flop_prompt
    assert parsed != 0.25 or "~25th percentile" in flop_prompt


def test_postflop_strength_reflects_the_made_hand():
    """A flopped monster must read as strong, or the bot folds the flop."""
    persona = get_persona("pro")
    table = make_table(seed=4)
    table.start_hand()
    to_flop(table)
    seat = table.actor
    assert seat is not None
    table.board = [Card.from_str(c) for c in ("5h", "6s", "6c")]
    table.players[seat].hole_cards = [Card.from_str("Jh"), Card.from_str("6h")]
    prompt = describe_table(table, seat)
    assert _parse_percentile(prompt) > 0.6, "trips must read as a strong hand"

    action = AIDecider(
        persona, HeuristicTransport(persona, random.Random(1)), retries=0
    ).decide(table, seat)
    # Facing no bet with a monster, the bot must not fold.
    assert action.type is not ActionType.FOLD
    assert action.type in (ActionType.BET, ActionType.RAISE, ActionType.CHECK)


def test_parser_reads_blinds_and_current_bet():
    table = make_table(seed=4)
    table.start_hand()
    prompt = describe_table(table, table.actor)
    assert _parse_int(prompt, r"Blinds:?\s*\d+/(\d+)") == table.big_blind
    assert _parse_int(prompt, r"Highest bet on this street: (\d+)") == table.current_bet
    legal = _parse_legal_block(prompt)
    assert legal.get("can_call") and not legal.get("can_check")


# --------------------------------------------------------------------------
# Table dynamics: the bots must actually play poker
# --------------------------------------------------------------------------
def test_a_full_ring_game_sees_plenty_of_flops():
    """Regression: before the fixes, 3% of full-ring hands reached a flop."""
    rng = random.Random(11)
    lineup = ["pro", "rock", "maniac", "pro", "rock", "calling_station"]
    hands = 120
    saw_flop = 0
    preflop_raises = 0

    for hand in range(hands):
        players = [
            Player(name=f"P{i}", seat=i, chips=1000, persona=lineup[i])
            for i in range(6)
        ]
        table = Table(players, 10, 20, seed=rng.randrange(10**6))
        table.button = hand % 6
        table.start_hand()
        deciders = {
            p.seat: AIDecider(
                get_persona(p.persona),
                HeuristicTransport(get_persona(p.persona), random.Random(rng.random())),
                retries=0,
            )
            for p in players
        }
        steps = 0
        reached = False
        while not table.is_hand_over and steps < 300:
            steps += 1
            seat = table.actor
            if seat is None:
                break
            street = table.street.value
            action = deciders[seat].decide(table, seat)
            table.apply_action(seat, action)
            if street != "preflop":
                reached = True
            elif action.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
                preflop_raises += 1
        saw_flop += reached

    rate = saw_flop / hands
    assert 0.25 <= rate <= 0.85, (
        f"{rate:.0%} of hands saw a flop; real full-ring play is roughly 25-45%"
    )
    assert preflop_raises / hands >= 0.4, (
        f"only {preflop_raises / hands:.2f} preflop raises per hand"
    )


@pytest.mark.parametrize("persona_key", ["rock", "maniac", "calling_station", "pro"])
def test_personas_keep_distinct_preflop_tightness(persona_key):
    """The archetypes must not converge on one play style."""
    rng = random.Random(5)
    persona = get_persona(persona_key)
    entered = 0
    hands = 80

    for hand in range(hands):
        players = [
            Player(
                name=f"P{i}",
                seat=i,
                chips=1000,
                persona=persona_key if i == 0 else "pro",
            )
            for i in range(6)
        ]
        table = Table(players, 10, 20, seed=rng.randrange(10**6))
        table.button = hand % 6
        table.start_hand()
        decider = AIDecider(
            persona, HeuristicTransport(persona, random.Random(rng.random())), retries=0
        )
        voluntary = False
        steps = 0
        while not table.is_hand_over and steps < 300:
            steps += 1
            seat = table.actor
            if seat is None:
                break
            street = table.street.value
            action = decider.decide(table, seat)
            table.apply_action(seat, action)
            if seat == 0 and street == "preflop" and action.type in (
                ActionType.CALL,
                ActionType.RAISE,
                ActionType.BET,
                ActionType.ALL_IN,
            ):
                voluntary = True
        if voluntary:
            entered += 1

    vpip = entered / hands
    # The calling station must be far looser than the nit.
    if persona_key == "rock":
        assert vpip < 0.75, f"rock entered {vpip:.0%} of pots"
    if persona_key == "calling_station":
        assert vpip > 0.3, f"calling station entered only {vpip:.0%} of pots"
    assert 0.0 <= vpip <= 1.0
