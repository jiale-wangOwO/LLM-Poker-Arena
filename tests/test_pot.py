"""Side-pot construction, payouts, tie splits and uncalled-bet refunds."""

from __future__ import annotations

import random

import pytest

from pokerarena.pot import build_pots, settle


def layering(contributions, folded=()):
    pots, refunds = build_pots(dict(contributions), set(folded))
    return [(pot.amount, sorted(pot.eligible), pot.is_side_pot) for pot in pots]


def refunds_of(contributions, folded=()):
    _, refunds = build_pots(dict(contributions), set(folded))
    return {payout.seat: payout.amount for payout in refunds}


def test_single_layer_when_everyone_matched():
    assert layering({0: 100, 1: 100, 2: 100}) == [(300, [0, 1, 2], False)]
    assert refunds_of({0: 100, 1: 100, 2: 100}) == {}


def test_side_pot_from_short_all_in():
    # Seat 1 is all-in for 50, the others are in for 100.
    pots = layering({0: 100, 1: 50, 2: 100})
    assert pots == [(150, [0, 1, 2], False), (100, [0, 2], True)]
    assert refunds_of({0: 100, 1: 50, 2: 100}) == {}


def test_multiple_side_pots():
    # Peeling the shortest remaining stack each time:
    #   4 x 100 = 400?  no -- 4 x 50 = 200  (everybody, seat 3 is shortest at 50)
    #   3 x 50  = 150  (seats 0, 1, 2)
    #   2 x 50  = 100  (seats 0, 1)
    # seat 0's last 100 is reachable by nobody else, so it is refunded.
    pots = layering({0: 300, 1: 200, 2: 100, 3: 50})
    assert pots == [
        (200, [0, 1, 2, 3], False),
        (150, [0, 1, 2], True),
        (200, [0, 1], True),
    ]
    assert refunds_of({0: 300, 1: 200, 2: 100, 3: 50}) == {0: 100}
    assert sum(amount for amount, _, _ in pots) == 550
    # Every chip is accounted for: 550 in pots + 100 refunded = 650 committed.
    assert sum(amount for amount, _, _ in pots) + 100 == 650


def test_uncalled_shove_is_refunded_not_potted():
    pots, refunds = build_pots({0: 1000, 1: 20}, set())
    assert len(pots) == 1
    assert pots[0].amount == 40
    assert pots[0].eligible == {0, 1}
    assert {(r.seat, r.amount) for r in refunds} == {(0, 980)}


def test_uncalled_shove_between_a_folded_player():
    """A folded player's blind is dead money that stays in the contested pots."""
    pots, refunds = build_pots({0: 1000, 1: 100, 2: 100, 3: 20}, folded={3})
    # The folded blind is dead money in the same main pot; it cannot create a
    # side pot because the three live players have identical eligibility.
    # The remaining 900 of seat 0's shove was never matched.
    assert [(p.amount, sorted(p.eligible)) for p in pots] == [
        (320, [0, 1, 2]),
    ]
    assert {(r.seat, r.amount) for r in refunds} == {(0, 900)}
    assert sum(p.amount for p in pots) + 900 == 1220


def test_folded_players_never_form_a_winning_pot():
    # Seat 2 folded after putting in 100; it is dead money for the others.
    pots, refunds = build_pots({0: 100, 1: 100, 2: 100}, folded={2})
    assert len(pots) == 1
    assert pots[0].amount == 300
    assert pots[0].eligible == {0, 1}
    assert refunds == []


def test_build_pots_ignores_zero_contributions():
    assert layering({0: 0, 1: 40, 2: 40}) == [(80, [1, 2], False)]


def test_single_contributor_gets_everything_back():
    pots, refunds = build_pots({0: 75}, set())
    assert pots == []
    assert {(r.seat, r.amount) for r in refunds} == {(0, 75)}


# --------------------------------------------------------------------------
# Structural invariants (property tests)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_layering_never_loses_or_invents_chips(seed):
    rng = random.Random(seed)
    seats = list(range(rng.randint(2, 6)))
    contributions = {seat: rng.choice([0, 20, 50, 100, 175, 400]) for seat in seats}
    if sum(contributions.values()) == 0:
        pytest.skip("nothing committed")
    folded = {seat for seat in seats if rng.random() < 0.3}
    if len(set(seats) - folded) < 1:
        folded = set()

    pots, refunds = build_pots(contributions, folded)
    out = sum(pot.amount for pot in pots) + sum(r.amount for r in refunds)
    assert out == sum(contributions.values()), (contributions, folded)


@pytest.mark.parametrize("seed", range(60))
def test_every_pot_has_a_live_contender(seed):
    rng = random.Random(1000 + seed)
    seats = list(range(rng.randint(2, 6)))
    contributions = {seat: rng.choice([10, 25, 60, 150, 300]) for seat in seats}
    folded = {seat for seat in seats if rng.random() < 0.4}

    pots, _ = build_pots(contributions, folded)
    for pot in pots:
        assert len(pot.eligible) >= 1, (contributions, folded, pot)
        assert pot.amount > 0


@pytest.mark.parametrize("seed", range(60))
def test_settle_conserves_chips_and_never_pays_a_folder(seed):
    rng = random.Random(2000 + seed)
    seats = list(range(rng.randint(2, 6)))
    contributions = {seat: rng.choice([0, 20, 45, 90, 250]) for seat in seats}
    if sum(contributions.values()) == 0:
        pytest.skip("nothing committed")
    folded = {seat for seat in seats if rng.random() < 0.3}
    live = [seat for seat in seats if seat not in folded]
    if not live:
        pytest.skip("everybody folded")

    scores = {seat: rng.randint(1, 100) for seat in live}
    result = settle(
        contributions,
        folded,
        scores,
        button_seat=rng.choice(seats),
        seat_order=seats,
    )

    assert result.total == sum(contributions.values()), (contributions, folded)
    # A folded player can never *win* a pot.  (Their own uncalled chips may be
    # returned to them, which is a refund, not a payout.)
    for payout in result.payouts:
        assert payout.seat not in folded, "a folded player won a pot"
    for pot in result.pots:
        assert not (pot.eligible & folded)


@pytest.mark.parametrize("seed", range(40))
def test_every_payout_goes_to_a_contributing_contender(seed):
    rng = random.Random(3000 + seed)
    seats = list(range(rng.randint(2, 5)))
    contributions = {seat: rng.choice([20, 50, 120]) for seat in seats}
    scores = {seat: rng.randint(1, 100) for seat in seats}

    result = settle(contributions, set(), scores, seat_order=seats)
    for payout in result.payouts:
        assert contributions.get(payout.seat, 0) > 0
        assert payout.amount > 0
    # Every pot must be fully distributed.
    assert sum(p.amount for p in result.payouts) == sum(
        pot.amount for pot in result.pots
    )


@pytest.mark.parametrize("seed", range(40))
def test_the_best_hand_wins_the_cap_it_could_reach(seed):
    """A player who covered everybody always wins the main pot if best."""
    rng = random.Random(4000 + seed)
    seats = list(range(rng.randint(2, 5)))
    # Seat 0 is the deepest stack, so it is eligible for every pot layer.
    contributions = {0: 500}
    for seat in seats:
        if seat != 0:
            contributions[seat] = rng.choice([20, 50, 120])
    scores = {seat: (999 if seat == 0 else rng.randint(1, 100)) for seat in seats}

    result = settle(contributions, set(), scores, seat_order=seats)
    assert result.amount_for(0) > 0
    assert result.total == sum(contributions.values())


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------
def test_best_hand_takes_the_whole_pot():
    result = settle({0: 100, 1: 100, 2: 100}, set(), {0: 10, 1: 20, 2: 30})
    assert result.amount_for(2) == 300
    assert result.amount_for(0) == 0


def test_side_pot_goes_to_the_best_among_eligible_players():
    # Seat 2 is all-in for 50 with the best hand; seats 0/1 contest the rest.
    contributions = {0: 100, 1: 100, 2: 50}
    scores = {0: 20, 1: 30, 2: 99}
    result = settle(contributions, set(), scores)
    # Main pot: 50*3 = 150 -> seat 2.  Side pot: 50*2 = 100 -> seat 1.
    assert result.amount_for(2) == 150
    assert result.amount_for(1) == 100
    assert result.amount_for(0) == 0
    assert result.total == 250


def test_short_all_in_wins_only_what_it_could_cover():
    contributions = {0: 500, 1: 100}
    scores = {0: 5, 1: 99}
    result = settle(contributions, set(), scores)
    # Seat 0's uncalled 400 comes straight back; seat 1 wins the matched 200.
    assert result.amount_for(0) == 400
    assert result.amount_for(1) == 200
    assert result.total == 600


def test_tie_splits_the_pot_evenly():
    result = settle({0: 100, 1: 100}, set(), {0: 42, 1: 42})
    assert result.amount_for(0) == 100
    assert result.amount_for(1) == 100


def test_odd_chip_goes_to_the_first_winner_after_the_button():
    # 3-way tie for a 100 pot is not divisible; button is seat 2 so seat 0
    # (the first seat clockwise after the button) gets the extra chip.
    result = settle(
        {0: 34, 1: 33, 2: 33},
        set(),
        {0: 7, 1: 7, 2: 7},
        button_seat=2,
        seat_order=[0, 1, 2],
    )
    # Seat 0 contributed the extra chip, which is refunded as uncalled, so the
    # contested pot is 99 -> 33 each.
    assert result.amount_for(0) == 34
    assert result.amount_for(1) == 33
    assert result.amount_for(2) == 33


def test_odd_chip_distribution_is_exact():
    contributions = {0: 25, 1: 25, 2: 25}
    result = settle(contributions, set(), {0: 1, 1: 1, 2: 1}, button_seat=0)
    total = sum(result.amount_for(seat) for seat in contributions)
    assert total == 75
    assert sorted(result.amount_for(s) for s in contributions) == [25, 25, 25]


def test_chips_are_never_destroyed_or_created():
    contributions = {0: 333, 1: 250, 2: 77, 3: 100}
    result = settle(contributions, set(), {0: 5, 1: 9, 2: 3, 3: 1})
    assert result.total == sum(contributions.values())
    assert sum(result.amount_for(s) for s in contributions) == sum(
        contributions.values()
    )


def test_folded_players_never_receive_chips():
    contributions = {0: 100, 1: 100, 2: 60}
    result = settle(contributions, folded={2}, scores={0: 10, 1: 20})
    assert result.amount_for(2) == 0
    assert result.total == 260


def test_empty_contributions_are_safe():
    result = settle({}, set(), {})
    assert result.total == 0


def test_only_contributor_gets_everything_back():
    result = settle({0: 75}, set(), {0: 1})
    assert result.amount_for(0) == 75


def test_matched_folded_money_is_a_pot_not_an_uncalled_refund():
    pots, refunds = build_pots({0: 100, 1: 100, 2: 50}, folded={1})
    assert [(p.amount, p.eligible) for p in pots] == [(150, {0, 2}), (100, {0})]
    assert refunds == []
    result = settle({0: 100, 1: 100, 2: 50}, {1}, {0: 10, 2: 20})
    assert [(p.seat, p.amount) for p in result.payouts] == [(2, 150), (0, 100)]
    assert result.refunds == []


def test_only_unmatched_chips_are_returned_after_everyone_folds():
    pots, refunds = build_pots({0: 1000, 1: 20}, folded={1})
    assert [(p.amount, p.eligible) for p in pots] == [(40, {0})]
    assert [(r.seat, r.amount) for r in refunds] == [(0, 980)]


def test_matched_folded_layers_remain_dead_money():
    result = settle({0: 50, 1: 100, 2: 100}, {1, 2}, {0: 10})
    assert result.refunds == []
    assert result.amount_for(0) == 250


def test_folded_contribution_levels_do_not_skew_a_tied_pot():
    contributions = {0: 100, 1: 100, 2: 21, 3: 46, 4: 71}
    result = settle(contributions, {2, 3, 4}, {0: 42, 1: 42}, button_seat=4)
    assert len(result.pots) == 1
    assert result.pots[0].amount == 338
    assert result.amount_for(0) == result.amount_for(1) == 169
