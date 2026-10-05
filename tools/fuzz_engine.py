"""Fuzz the engine for rule violations.

The point is to check invariants that a hand-written test tends to miss, and to
do it *during* play rather than only at the end.  Two classes of check:

* **Temporal** -- things that must hold after every single action: chips are
  conserved, nothing was bet that the player did not have, the player to act is
  entitled to act, a street cannot advance while someone still owes chips.
* **Tamper** -- the action chosen is deliberately *not* taken from
  ``legal_actions``.  A model can return anything, so the engine must reject an
  illegal action rather than quietly apply it.  Trusting the legal menu here
  would test nothing.

Run: ``python tools/fuzz_engine.py [hands]``
"""

from __future__ import annotations

import random
import sys

from pokerarena.engine import Action, ActionType, PlayerStatus, Street, Table
from pokerarena.engine import Player


class Violation(AssertionError):
    pass


def make_table(rng: random.Random, seats: int, chips: list[int], sb: int, bb: int) -> Table:
    players = [Player(name=f"P{i}", seat=i, chips=chips[i]) for i in range(seats)]
    table = Table(players, sb, bb, seed=rng.randrange(1 << 30))
    for player in players:
        player.seated = True
    return table


def total_chips(table: Table) -> int:
    return sum(p.chips for p in table.players) + table.pot


def check_temporal(table: Table, expected_total: int, where: str) -> None:
    # 1. Chips are never created or destroyed.
    got = total_chips(table)
    if got != expected_total:
        raise Violation(f"{where}: chip total {got} != {expected_total}")

    # 2. No negative chips, and nobody bet more than they had.
    for p in table.players:
        if p.chips < 0:
            raise Violation(f"{where}: {p.name} has {p.chips} chips")
        if p.street_bet < 0:
            raise Violation(f"{where}: {p.name} street_bet {p.street_bet}")
        contributed = table.hand_contributions.get(p.seat, 0)
        if contributed < 0:
            raise Violation(f"{where}: {p.name} contributed {contributed}")
        if p.status is PlayerStatus.SITTING_OUT and contributed:
            raise Violation(f"{where}: sitting-out {p.name} contributed {contributed}")

    # 3. An all-in player has no chips behind.
    for p in table.players:
        if p.status is PlayerStatus.ALL_IN and p.chips not in (0,):
            raise Violation(f"{where}: all-in {p.name} still holds {p.chips}")

    # 4. A folded player cannot owe chips on this street.
    for p in table.players:
        if p.status is PlayerStatus.FOLDED and p.street_bet:
            # Folding does not take back chips already in, but it must not leave
            # a player "owing" in a way that keeps them actionable.
            if table.actor == p.seat:
                raise Violation(f"{where}: folded {p.name} is the actor")

    # 5. The player to act must be in the hand and able to act.
    if table.actor is not None and not table.is_hand_over and table.street is not Street.COMPLETE:
        actor = table.players[table.actor]
        if actor.status in (PlayerStatus.FOLDED, PlayerStatus.SITTING_OUT, PlayerStatus.ALL_IN):
            raise Violation(f"{where}: actor {actor.name} is {actor.status.value}")

    # 6. No street may be left hanging: if the street is not complete and there
    #    is no actor, either the hand is over or someone is all-in.
    if (
        not table.is_hand_over
        and table.street not in (Street.COMPLETE, Street.SHOWDOWN)
        and table.actor is None
    ):
        raise Violation(f"{where}: no actor on {table.street.value} and hand not over")


def random_illegal_action(rng: random.Random, table: Table, seat: int) -> Action | None:
    """An action the engine has *not* authorised.  May be None (skip)."""
    legal = table.legal_actions(seat)
    player = table.players[seat]
    choices: list[Action] = []

    # Fold when there is nothing to fold to.
    if not legal.can_fold:
        choices.append(Action(ActionType.FOLD))
    # Check when facing a bet.
    if not legal.can_check:
        choices.append(Action(ActionType.CHECK))
    # Call when there is nothing to call.
    if not legal.can_call:
        choices.append(Action(ActionType.CALL, amount=0))
    # Raise below the legal minimum.
    if legal.min_raise_to > 0:
        choices.append(Action(ActionType.RAISE, amount=max(0, legal.min_raise_to - 10)))
    # Raise above the legal maximum.
    if legal.max_raise_to > 0:
        choices.append(Action(ActionType.RAISE, amount=legal.max_raise_to + 1000))
    # Bet more chips than the player has.
    choices.append(Action(ActionType.RAISE, amount=player.chips + 5000))
    # A negative amount.
    choices.append(Action(ActionType.RAISE, amount=-50))
    # Call a negative amount.
    choices.append(Action(ActionType.CALL, amount=-10))

    return rng.choice(choices) if choices else None


def play_hand(table: Table, rng: random.Random, expected_total: int, tamper_rate: float) -> str:
    """Play one hand.  Returns why it ended."""
    table.start_hand()
    check_temporal(table, expected_total, "after start_hand")

    steps = 0
    while not table.is_hand_over:
        steps += 1
        if steps > 500:
            raise Violation("hand did not terminate within 500 actions")
        seat = table.actor
        if seat is None:
            raise Violation(f"stalled on {table.street.value} with no actor")

        # Once in a while, try to cheat: a model can return any nonsense.
        if rng.random() < tamper_rate:
            bad = random_illegal_action(rng, table, seat)
            if bad is not None:
                before = (
                    total_chips(table),
                    table.pot,
                    table.street,
                    table.current_bet,
                    table.actor,
                    tuple(p.chips for p in table.players),
                    tuple(p.status for p in table.players),
                )
                rejected = False
                try:
                    table.apply_action(seat, bad)
                except Exception:
                    rejected = True  # rejecting is the correct response

                if not rejected:
                    # Accepting is only allowed when the engine *normalises* it
                    # into an equal-or-better action, e.g. a CALL whose amount
                    # is ignored in favour of the real price.  Anything that
                    # changes chips, the pot or the betting level is corruption.
                    after = (
                        total_chips(table),
                        table.pot,
                        table.street,
                        table.current_bet,
                        table.actor,
                        tuple(p.chips for p in table.players),
                        tuple(p.status for p in table.players),
                    )
                    if after != before:
                        raise Violation(
                            f"illegal {bad.type.value} amount={bad.amount} was accepted "
                            f"and changed state:\n    before={before}\n    after ={after}"
                        )

                check_temporal(table, expected_total, f"after illegal {bad.type.value}")
                if table.is_hand_over:
                    return "over"
                if table.actor is None:
                    raise Violation("illegal action left the table with no actor")
                continue

        legal = table.legal_actions(seat)
        action = pick_legal(rng, table, seat, legal)
        try:
            table.apply_action(seat, action)
        except Violation:
            raise
        except Exception as exc:
            # An action built *from* legal_actions must never be rejected.
            raise Violation(
                f"legal_actions offered {action.type.value} amount={action.amount} "
                f"for seat {seat} but apply_action refused it: {exc}\n"
                f"    menu: {legal.summary()}\n"
                f"    state: street={table.street.value} current_bet={table.current_bet} "
                f"player street_bet={table.players[seat].street_bet} "
                f"chips={table.players[seat].chips} min_raise={table.last_full_raise_to}"
            ) from exc
        check_temporal(table, expected_total, f"after {action.type.value} by seat {seat}")

    return "complete"


def pick_legal(rng: random.Random, table: Table, seat: int, legal) -> Action:
    """A legal action, chosen with a bias toward action so hands stay lively."""
    player = table.players[seat]
    options = []
    if legal.can_fold:
        options.append(("fold", 3))
    if legal.can_check:
        options.append(("check", 6))
    if legal.can_call:
        options.append(("call", 6))
    if legal.can_raise or legal.can_bet:
        options.append(("raise", 5))
    if legal.all_in_to is not None:
        options.append(("allin", 1))

    total = sum(w for _, w in options)
    roll = rng.uniform(0, total)
    upto = 0
    choice = options[-1][0]
    for name, weight in options:
        upto += weight
        if roll <= upto:
            choice = name
            break

    if choice == "fold":
        return Action(ActionType.FOLD)
    if choice == "check":
        return Action(ActionType.CHECK)
    if choice == "call":
        return Action(ActionType.CALL, amount=legal.call_cost)
    if choice == "allin":
        return Action(ActionType.ALL_IN)
    low = legal.min_bet_to if legal.can_bet else legal.min_raise_to
    high = legal.max_bet_to if legal.can_bet else legal.max_raise_to
    amount = rng.choice([low, min(high, low * 2), min(high, low * 3), high])
    return Action(ActionType.RAISE, amount=int(amount))


def main() -> int:
    hands = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(20240607)
    blown = 0

    for hand_no in range(hands):
        seats = rng.randint(2, 9)
        starting = rng.choice([20, 40, 80, 200, 500, 1000, 5000])
        chips = [
            max(1, starting + rng.choice([-int(starting * 0.9), 0, 0, int(starting)]))
            for _ in range(seats)
        ]
        sb = rng.choice([1, 5, 10, 25])
        bb = sb * 2
        table = make_table(rng, seats, chips, sb, bb)
        expected = total_chips(table)
        tamper = rng.choice([0.0, 0.15, 0.4])

        try:
            play_hand(table, rng, expected, tamper)
        except Violation as exc:
            blown += 1
            print(f"VIOLATION hand {hand_no}: {exc}")
            print(f"  seats={seats} chips={chips} blinds={sb}/{bb} tamper={tamper}")
            if blown >= 5:
                break
            continue

        # After the hand: the pot must be empty and chips conserved.
        if table.pot != 0:
            blown += 1
            print(f"VIOLATION hand {hand_no}: pot left over at end: {table.pot}")
        if total_chips(table) != expected:
            blown += 1
            print(f"VIOLATION hand {hand_no}: chips after payout {total_chips(table)} != {expected}")

    print(f"\nfuzzed {hands} hands: {blown} violation(s)")
    return 1 if blown else 0


if __name__ == "__main__":
    sys.exit(main())
