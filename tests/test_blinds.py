"""Blind/ante escalation tests -- the mechanism that makes games actually end.

Background: with fixed blinds a bot can fold every hand forever while the others
trade a single big blind back and forth, so the game reaches the hand cap with
several players still holding chips.  Before escalation was added, 11 of 12
seeded games ran to the cap; with it, all of them reach a single winner.
"""

from __future__ import annotations

import pytest

from pokerarena.arena import Arena, AISpec, build_players
from pokerarena.config import ArenaConfig
from pokerarena.engine import Player, Table


def make_table(seats=4, chips=400, small=10, big=20):
    players = [Player(name=f"P{i}", seat=i, chips=chips) for i in range(seats)]
    return Table(players, small, big, seed=1)


# --------------------------------------------------------------------------
# raise_blinds / set_ante
# --------------------------------------------------------------------------
def test_raise_blinds_scales_and_reports_the_level():
    table = make_table()
    assert table.blind_level == 1
    assert table.raise_blinds(1.5, max_level=20) is True
    assert table.blind_level == 2
    assert (table.small_blind, table.big_blind) == (15, 30)
    assert table.raise_blinds(1.5, max_level=20) is True
    assert (table.small_blind, table.big_blind) == (23, 45)
    # Blinds stay whole chips and always keep small < big.
    assert table.small_blind < table.big_blind


def test_raise_blinds_emits_an_event():
    table = make_table()
    table.raise_blinds(1.5, max_level=20)
    kinds = [event.kind for event in table.drain_events()]
    assert "blinds_increased" in kinds


def test_raise_blinds_stops_at_the_level_cap():
    table = make_table()
    for _ in range(3):
        table.raise_blinds(1.5, max_level=3)
    assert table.blind_level == 3
    assert table.raise_blinds(1.5, max_level=3) is False
    assert table.blind_level == 3


def test_escalation_stops_once_the_blind_covers_the_biggest_stack():
    """Past this point a blind commits everyone and the level is meaningless."""
    table = make_table(seats=4, chips=100)
    table.big_blind = 100  # already >= the biggest stack
    assert table.raise_blinds(1.5, max_level=99) is False


def test_escalation_keeps_going_while_the_leader_can_still_fold():
    table = make_table(seats=3, chips=1000)
    table.big_blind = 400  # less than the 1000 stacks
    assert table.raise_blinds(1.5, max_level=99) is True


def test_set_ante_clamps_negative_values():
    table = make_table()
    table.set_ante(-50)
    assert table.ante == 0
    table.set_ante(25)
    assert table.ante == 25


# --------------------------------------------------------------------------
# Antes in the hand
# --------------------------------------------------------------------------
def test_antes_are_collected_from_everyone_and_land_in_the_pot():
    table = make_table(seats=4, chips=400)
    table.set_ante(25)
    table.start_hand()
    # 4 antes + small blind + big blind
    assert table.pot_total == 25 * 4 + 10 + 20
    for player in table.players:
        assert player.hand_contribution >= 25


def test_antes_do_not_count_as_a_street_bet():
    """Otherwise the first player would wrongly owe less than the big blind."""
    table = make_table(seats=4, chips=400)
    table.set_ante(25)
    table.start_hand()
    for player in table.players:
        expected = 0
        if player.seat == table.blind_seats()[0]:
            expected = table.small_blind
        if player.seat == table.blind_seats()[1]:
            expected = table.big_blind
        assert player.street_bet == expected
    # Facing the big blind, a non-blind seat still owes the full amount.
    assert table.legal_actions(table.actor).call_cost == table.big_blind


def test_short_stack_ante_goes_all_in_without_breaking_conservation():
    table = make_table(seats=3, chips=400)
    table.players[1].chips = 5
    table.set_ante(25)
    total_before = sum(p.chips for p in table.players)
    table.start_hand()
    assert table.players[1].chips == 0
    assert sum(p.chips for p in table.players) + table.pot_total == total_before
    assert table.players[1].status.value == "all_in"


def test_hand_start_reports_the_blind_level_and_ante():
    table = make_table()
    table.set_ante(30)
    table.raise_blinds(1.5, max_level=20)
    table.start_hand()
    starts = [e for e in table.drain_events() if e.kind == "hand_start"]
    assert starts and starts[0].data["ante"] == 30
    assert starts[0].data["blind_level"] == 2


# --------------------------------------------------------------------------
# End-to-end: games actually finish
# --------------------------------------------------------------------------
LINEUPS = {
    "mixed": [
        ("Ironside", "calculating"),
        ("Vega", "maniac"),
        ("Granite", "rock"),
        ("Kitsune", "trickster"),
    ],
    "tight": [
        ("Ironside", "calculating"),
        ("Granite", "rock"),
        ("Meridian", "pro"),
    ],
    "loose": [
        ("Vega", "maniac"),
        ("Kitsune", "trickster"),
        ("Barnacle", "calling_station"),
    ],
}


@pytest.mark.parametrize("lineup", sorted(LINEUPS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_games_reach_a_single_winner(lineup, seed):
    """Regression: without escalation 11/12 seeded games hit the hand cap."""
    specs = [
        AISpec(name=name, persona_key=key, model="")
        for name, key in LINEUPS[lineup]
    ]
    players, _ = build_players(specs, starting_chips=400)
    total = sum(p.chips for p in players)
    arena = Arena(players, config=ArenaConfig(seed=seed, max_rounds=400))
    winner = arena.play_game()

    assert arena.table.is_game_over(), (
        f"{lineup}/seed {seed} stalled with "
        f"{[p.chips for p in arena.table.players]} at blinds "
        f"{arena.table.small_blind}/{arena.table.big_blind}"
    )
    assert winner
    assert sum(p.chips for p in arena.table.players) == total
    assert arena.table.seated[0].chips == total


def test_blinds_eventually_escalate_in_a_real_game():
    specs = [
        AISpec(name="A", persona_key="rock", model=""),
        AISpec(name="B", persona_key="rock", model=""),
        AISpec(name="C", persona_key="pro", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(
        players,
        config=ArenaConfig(seed=5, max_rounds=400, blind_increase_every=10),
    )
    arena.play_game()
    assert arena.table.blind_level > 1, "the clock never advanced"
    assert arena.table.ante > 0, "antes should kick in with the levels"


def test_escalation_can_be_disabled():
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="pro", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(
        players,
        config=ArenaConfig(seed=2, max_rounds=30, blind_increase_every=0),
    )
    arena.play_game()
    assert arena.table.blind_level == 1
    assert arena.table.ante == 0
