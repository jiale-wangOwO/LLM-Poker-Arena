"""Engine invariants that fuzzing found violations of.

Each test here corresponds to a real bug, not a hypothetical one.  They are
written as *rules* rather than as reproductions of the original symptom, because
the symptom is rarely the whole story.
"""

from __future__ import annotations

import random

import pytest

from pokerarena.engine import (
    Action,
    ActionType,
    IllegalAction,
    Player,
    PlayerStatus,
    Street,
    Table,
)


def make_table(seats=3, chips=1000, sb=10, bb=20, seed=1):
    players = [Player(name=f"P{i}", seat=i, chips=chips) for i in range(seats)]
    for p in players:
        p.seated = True
    return Table(players, sb, bb, seed=seed), players


# --------------------------------------------------------------------------
# A winner must not stay marked all-in
# --------------------------------------------------------------------------
def test_a_player_who_wins_the_pot_is_no_longer_all_in():
    """`ALL_IN` means "nothing left to bet", not "shoved at some point".

    A player who shoved and then won a side pot was left marked all-in while
    holding chips.  The end-of-hand pause renders that state, so the table
    claimed a player with 110 chips had no chips, and the next decision prompt
    would have said the same.
    """
    table, players = make_table(seats=3, chips=100, seed=4)
    table.start_hand()
    guard = 0
    while not table.is_hand_over and guard < 200:
        guard += 1
        seat = table.actor
        if seat is None:
            break
        legal = table.legal_actions(seat)
        if legal.all_in_to is not None and legal.call_cost >= players[seat].chips:
            table.apply_action(seat, Action(ActionType.ALL_IN))
        elif legal.can_check:
            table.apply_action(seat, Action(ActionType.CHECK))
        elif legal.can_call:
            table.apply_action(seat, Action(ActionType.CALL, amount=legal.call_cost))
        else:
            table.apply_action(seat, Action(ActionType.FOLD))

    for p in players:
        if p.status is PlayerStatus.ALL_IN:
            assert p.chips == 0, (
                f"{p.name} is marked all_in but holds {p.chips} chips"
            )


@pytest.mark.parametrize("seed", range(40))
def test_the_all_in_status_never_survives_a_payout(seed):
    """Fuzz the same rule over many random hands and stack shapes."""
    rng = random.Random(seed)
    seats = rng.randint(2, 6)
    chips = [rng.choice([20, 50, 120, 400]) for _ in range(seats)]
    table, players = make_table(seats=seats, chips=1, sb=5, bb=10, seed=seed)
    for player, stack in zip(players, chips):
        player.chips = stack
    table.start_hand()

    guard = 0
    while not table.is_hand_over and guard < 300:
        guard += 1
        seat = table.actor
        if seat is None:
            break
        legal = table.legal_actions(seat)
        options = []
        if legal.can_fold:
            options.append("fold")
        if legal.can_check:
            options.append("check")
        if legal.can_call:
            options.append("call")
        if legal.can_raise or legal.can_bet:
            options.append("raise")
        if legal.all_in_to is not None:
            options.append("allin")
        choice = rng.choice(options)
        if choice == "fold":
            action = Action(ActionType.FOLD)
        elif choice == "check":
            action = Action(ActionType.CHECK)
        elif choice == "call":
            action = Action(ActionType.CALL, amount=legal.call_cost)
        elif choice == "allin":
            action = Action(ActionType.ALL_IN)
        else:
            low = legal.min_bet_to if legal.can_bet else legal.min_raise_to
            high = legal.max_bet_to if legal.can_bet else legal.max_raise_to
            action = Action(ActionType.RAISE, amount=min(high, low))
        table.apply_action(seat, action)

    for p in players:
        assert not (p.status is PlayerStatus.ALL_IN and p.chips > 0), (
            f"seed {seed}: {p.name} all_in with {p.chips} chips"
        )


# --------------------------------------------------------------------------
# Out-of-turn actions
# --------------------------------------------------------------------------
def test_a_seat_that_is_not_to_act_is_refused():
    """The engine must not accept a decision from the wrong seat.

    Two players cannot both be "the actor": the turn order the arena and the UI
    both rely on would be meaningless.
    """
    table, players = make_table(seats=3, seed=2)
    table.start_hand()
    actor = table.actor
    other = next(s for s in range(3) if s != actor)
    with pytest.raises(IllegalAction):
        table.apply_action(other, Action(ActionType.FOLD))
    # And the hand is untouched.
    assert players[other].status is PlayerStatus.ACTIVE
    assert table.actor == actor


def test_legal_actions_is_refused_for_a_seat_that_is_not_to_act():
    table, _ = make_table(seats=3, seed=2)
    table.start_hand()
    other = next(s for s in range(3) if s != table.actor)
    with pytest.raises(IllegalAction):
        table.legal_actions(other)


# --------------------------------------------------------------------------
# The legal menu must be self-consistent
# --------------------------------------------------------------------------
def flop_spot(chips: int, *, pot: int = 100, current_bet: int = 0, seed: int = 5):
    """Seat 0 to act on the flop with exactly ``chips`` behind.

    Built directly rather than by playing to the flop: posting blinds from a
    short stack puts the player all-in before the flop, which is not the state
    these tests are about.
    """
    table, players = make_table(seats=2, chips=1000, sb=10, bb=20, seed=seed)
    table.start_hand()
    table.street = Street.FLOP
    for p in players:
        p.new_street()
    players[0].chips = chips
    players[0].status = PlayerStatus.ACTIVE
    table.current_bet = current_bet
    table.pot = pot
    table.actor = 0
    return table, players


def test_a_short_stack_is_never_offered_an_impossible_bet():
    """Regression: a player with 10 chips facing a 20 big blind was offered
    "bet 20-10" -- an inverted range the engine then refused, so the model was
    blamed for an engine bug."""
    for chips in (5, 10, 19, 20, 21, 50, 1000):
        table, players = flop_spot(chips)
        legal = table.legal_actions(0)
        if legal.can_bet:
            assert legal.min_bet_to <= legal.max_bet_to, (
                f"{chips} behind: bet range is inverted "
                f"{legal.min_bet_to}-{legal.max_bet_to}"
            )
            assert legal.min_bet_to <= chips, (
                f"{chips} behind: minimum bet {legal.min_bet_to} is unaffordable"
            )
        # Going all in must always remain available while chips are held.
        assert legal.all_in_to == chips


@pytest.mark.parametrize("chips", [5, 10, 19, 20, 21, 50, 1000])
def test_the_offered_menu_is_always_applicable(chips):
    """Anything the menu offers must be something apply_action accepts."""
    table, players = flop_spot(chips)
    legal = table.legal_actions(0)

    attempts = []
    if legal.can_check:
        attempts.append(Action(ActionType.CHECK))
    if legal.can_bet:
        attempts.append(Action(ActionType.RAISE, amount=legal.min_bet_to))
        attempts.append(Action(ActionType.RAISE, amount=legal.max_bet_to))
    if legal.can_raise:
        attempts.append(Action(ActionType.RAISE, amount=legal.min_raise_to))
        attempts.append(Action(ActionType.RAISE, amount=legal.max_raise_to))
    if legal.all_in_to is not None:
        attempts.append(Action(ActionType.ALL_IN))

    assert attempts, f"{chips} behind: the menu offered nothing at all"
    for action in attempts:
        probe, _ = flop_spot(chips)
        try:
            probe.apply_action(0, action)
        except IllegalAction as exc:  # pragma: no cover - the failure path
            pytest.fail(
                f"{chips} behind: menu offered {action.type.value} "
                f"{action.amount} but it was refused: {exc}"
            )


def test_a_negative_amount_is_rejected_not_clamped():
    table, _ = make_table(seats=2, seed=6)
    table.start_hand()
    with pytest.raises(IllegalAction):
        table.apply_action(table.actor, Action(ActionType.RAISE, amount=-5))
    with pytest.raises(IllegalAction):
        table.apply_action(table.actor, Action(ActionType.CALL, amount=-5))

def test_check_and_call_are_never_offered_together():
    """A menu that offers both cannot be right: one of them is free."""
    rng = random.Random(11)
    for trial in range(60):
        table, _ = make_table(seats=rng.randint(2, 5), chips=rng.choice([40, 300]), seed=trial)
        table.start_hand()
        guard = 0
        while not table.is_hand_over and guard < 200:
            guard += 1
            seat = table.actor
            if seat is None:
                break
            legal = table.legal_actions(seat)
            assert not (legal.can_check and legal.can_call)
            if legal.can_call:
                table.apply_action(seat, Action(ActionType.CALL, amount=legal.call_cost))
            elif legal.can_check:
                table.apply_action(seat, Action(ActionType.CHECK))
            else:
                table.apply_action(seat, Action(ActionType.FOLD))
