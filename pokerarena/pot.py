"""Pot management: main pot, side pots and split-pot payouts.

The original engine simply added every bet into one number and gave the whole
thing to one player.  That is wrong the moment two players are all-in for
different amounts.  This module implements the standard layered-pot model:

    contribution[i] = total chips player i put in this hand

Pots are built by peeling contribution layers from the smallest all-in upward.
Each layer is a separate pot with its own set of *eligible* players (those who
contributed at least up to that layer and have not folded).  A layer is awarded
to the best eligible hand(s); ties split the layer, with odd chips going to the
first winner in clockwise order after the button.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pot:
    """One layer of the pot."""

    amount: int
    eligible: set[int]
    """Player seats that may win this layer."""

    is_side_pot: bool = False

    @property
    def name(self) -> str:
        return "side pot" if self.is_side_pot else "main pot"


@dataclass
class Payout:
    seat: int
    amount: int
    pot_index: int
    split: bool = False


@dataclass
class PotResult:
    pots: list[Pot] = field(default_factory=list)
    payouts: list[Payout] = field(default_factory=list)
    refunds: list[Payout] = field(default_factory=list)
    """Chips returned to a player who bet more than anyone could match."""

    @property
    def total(self) -> int:
        return sum(p.amount for p in self.pots) + sum(r.amount for r in self.refunds)

    def amount_for(self, seat: int) -> int:
        return sum(p.amount for p in self.payouts if p.seat == seat) + sum(
            r.amount for r in self.refunds if r.seat == seat
        )


def build_pots(
    contributions: dict[int, int], folded: set[int]
) -> tuple[list[Pot], list[Payout]]:
    """Layer contributions into a main pot plus side pots.

    ``contributions`` maps seat -> chips committed this hand.  ``folded`` lists
    seats that folded; their chips stay in the pot as dead money but they can
    never win it.

    Returns ``(pots, refunds)``.  Everything is derived from the contributions
    by walking upward through the distinct commitment levels.  A level whose
    participants include two or more contributors is a pot layer, even if only
    one remains eligible to win it. A sole contributor's unmatched layer is
    returned. Folded players' matched chips stay in pots as dead money.
    """
    contributions = {
        seat: amount for seat, amount in contributions.items() if amount > 0
    }
    if not contributions:
        return [], []

    refunds: list[Payout] = []
    pots: list[Pot] = []
    previous = 0

    for level in sorted(set(contributions.values())):
        participants = {seat for seat, total in contributions.items() if total >= level}
        eligible = participants - folded
        layer = level - previous
        previous = level
        amount = layer * len(participants)
        if not amount:
            continue

        if len(participants) == 1:
            owner = next(iter(participants))
            refunds.append(Payout(seat=owner, amount=amount, pot_index=-1))
        elif eligible:
            # Matched money is a pot, even with only one eligible winner.
            # Folded contribution levels do not create another side pot when
            # the live eligibility is unchanged. Splitting such layers apart
            # would award multiple odd chips to the same tied player.
            if pots and pots[-1].eligible == eligible:
                pots[-1].amount += amount
            else:
                pots.append(Pot(amount=amount, eligible=eligible, is_side_pot=bool(pots)))
        elif pots:
            # Matched folded chips remain dead money for the lower live pot.
            # This mostly protects imported histories with unusual fold order.
            pots[-1].amount += amount
        else:
            # Every participant folded.  Return each share to its contributor
            # rather than vaporising the chips.
            for seat in participants:
                refunds.append(Payout(seat=seat, amount=layer, pot_index=-1))

    return pots, refunds


def settle(
    contributions: dict[int, int],
    folded: set[int],
    scores: dict[int, int],
    *,
    button_seat: int = 0,
    seat_order: list[int] | None = None,
) -> PotResult:
    """Distribute every pot to the best eligible hand.

    ``scores`` maps seat -> evaluator score for each player still in the hand.
    Seats missing from ``scores`` are treated as losers of every pot.
    """
    result = PotResult()
    if not contributions:
        return result

    if not (set(contributions) - folded):
        # Everybody folded: hand each stack back rather than vaporising chips.
        for seat, amount in sorted(contributions.items()):
            if amount:
                result.refunds.append(Payout(seat=seat, amount=amount, pot_index=-1))
        return result

    pots, refunds = build_pots(contributions, folded)
    result.pots = pots
    result.refunds.extend(refunds)

    order = seat_order if seat_order is not None else sorted(contributions)
    for index, pot in enumerate(pots):
        contenders = [seat for seat in pot.eligible if seat in scores]
        if not contenders:
            # No live hand can claim this pot (should be rare); refund pro rata
            # to contributors so chips are never destroyed.
            payers = [s for s in order if contributions.get(s, 0) > 0]
            if not payers:
                continue
            share = pot.amount // len(payers)
            remainder = pot.amount - share * len(payers)
            for position, seat in enumerate(payers):
                amount = share + (1 if position < remainder else 0)
                result.payouts.append(Payout(seat=seat, amount=amount, pot_index=index))
            continue

        best = max(scores[seat] for seat in contenders)
        winners = [seat for seat in contenders if scores[seat] == best]
        winners.sort(key=lambda seat: _clockwise_rank(seat, button_seat, order))

        if len(winners) == 1:
            result.payouts.append(
                Payout(seat=winners[0], amount=pot.amount, pot_index=index)
            )
            continue

        share, remainder = divmod(pot.amount, len(winners))
        for position, seat in enumerate(winners):
            # Odd chips go to the earliest winner clockwise from the button.
            amount = share + (1 if position < remainder else 0)
            result.payouts.append(
                Payout(seat=seat, amount=amount, pot_index=index, split=True)
            )

    return result


def _clockwise_rank(seat: int, button_seat: int, order: list[int]) -> int:
    """Sort key placing seats in clockwise order starting after the button."""
    if seat not in order:  # pragma: no cover - defensive
        return len(order) + seat
    n = len(order)
    try:
        button_index = order.index(button_seat)
    except ValueError:  # pragma: no cover - defensive
        button_index = -1
    seat_index = order.index(seat)
    return (seat_index - button_index - 1) % n
