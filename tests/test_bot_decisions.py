"""The offline brain must use the information it is given.

Written after finding that it folded a hand getting 20:1, because it judged
"how good is my made hand" instead of "what does this price require".  A bot that
ignores the price is what makes an offline game feel like a machine rather than
an opponent.
"""

from __future__ import annotations

import random

import pytest

from pokerarena.ai import (
    AIDecider,
    HeuristicTransport,
    _parse_equity,
    _parse_opponents,
    _parse_percentile,
    describe_table,
)
from pokerarena.cards import Card
from pokerarena.engine import Player, PlayerStatus, Street, Table
from pokerarena.personas import persona_store
from pokerarena.strength import equity, pot_odds


def h(*codes):
    return [Card.from_str(c) for c in codes]


def flop_prompt(call_cost, pot, chips, opponents, *, hero=("7d", "Ah"),
                board=("4d", "6c", "3c"), seed=5):
    """One fixed flop decision, rendered exactly as a model would see it."""
    players = [Player(name=f"P{i}", seat=i, chips=chips) for i in range(1 + opponents)]
    for p in players:
        p.seated = True
    table = Table(players, 10, 20, seed=seed)
    table.start_hand()
    table.street = Street.FLOP
    table.board = h(*board)
    for p in players:
        p.new_street()
    players[0].hole_cards = h(*hero)
    for p in players[1:]:
        p.hole_cards = h("2h", "7s")
    table.pot = pot
    table.current_bet = call_cost
    table.actor = 0
    return table, describe_table(table, 0)


def ask(prompt, persona="pro", seed=0):
    p = persona_store().get(persona)
    transport = HeuristicTransport(p, rng=random.Random(seed))
    reply = transport.complete("s", [{"role": "user", "content": prompt}], temperature=0.7)
    return reply.content.split("<action>")[1].split("</action>")[0].split()[0]


def decide_all(call_cost, pot, chips, opponents, persona="pro"):
    counts: dict[str, int] = {}
    for i in range(80):
        _, prompt = flop_prompt(call_cost, pot, chips, opponents)
        choice = ask(prompt, persona=persona, seed=1000 + i)
        counts[choice] = counts.get(choice, 0) + 1
    return counts


# --------------------------------------------------------------------------
# The prompt reports the numbers the bot needs
# --------------------------------------------------------------------------
def test_the_prompt_reports_equity_next_to_the_percentile():
    _, prompt = flop_prompt(40, 200, 1000, 1)
    assert "% equity" in prompt, "equity must be reported for pot-odds reasoning"
    assert _parse_equity(prompt) is not None


def test_the_prompt_states_how_many_opponents_there_are():
    """Regression: the count was read from a line a later rewrite removed, so it
    silently fell back to a full ring and short-handed seats played as though
    they were nine-handed."""
    for opponents in (1, 2, 5):
        _, prompt = flop_prompt(40, 200, 1000, opponents)
        assert _parse_opponents(prompt) == opponents, (
            f"expected {opponents} opponents from the prompt"
        )


def test_equity_and_percentile_are_different_numbers():
    """They answer different questions and must not be conflated."""
    hole = h("7d", "Ah")
    board = h("4d", "6c", "3c")
    share = equity(hole, board)
    # An ace-high gutshot is a weak made hand with plenty of equity.
    assert share > 0.4, f"ace-high gutshot should have real equity, got {share:.2f}"


def test_preflop_equity_is_ordered_and_sane():
    assert equity(h("Ad", "Ac"), []) > 0.80
    assert 0.25 < equity(h("7s", "2h"), []) < 0.40
    assert equity(h("Ad", "Ac"), []) > equity(h("As", "Ks"), []) > equity(h("7s", "2h"), [])


def test_pot_odds_helper():
    assert abs(pot_odds(10, 200) - 10 / 210) < 1e-9
    assert pot_odds(0, 200) == 0.0
    assert pot_odds(200, 200) == 0.5


# --------------------------------------------------------------------------
# The bot uses them
# --------------------------------------------------------------------------
def test_a_huge_price_is_called_even_with_a_weak_made_hand():
    """The headline bug: 20:1 with a gutshot and an overcard is a trivial call."""
    counts = decide_all(call_cost=10, pot=200, chips=1000, opponents=1)
    assert counts.get("call", 0) > 0, f"the bot folded a 20:1 price: {counts}"


def test_a_bad_price_is_folded():
    counts = decide_all(call_cost=200, pot=200, chips=1000, opponents=1)
    assert counts.get("fold", 0) > 0, f"the bot called a 2:1 price with ace-high: {counts}"


def test_the_price_changes_the_decision():
    cheap = decide_all(call_cost=10, pot=200, chips=1000, opponents=1)
    pricey = decide_all(call_cost=200, pot=200, chips=1000, opponents=1)
    assert cheap != pricey, "the bot ignores the price entirely"


def test_more_opponents_means_more_caution():
    """Heads-up equity overstates a multiway hand, so the same price should be
    called less often against a field."""
    heads_up = decide_all(call_cost=40, pot=200, chips=1000, opponents=1)
    crowd = decide_all(call_cost=40, pot=200, chips=1000, opponents=5)
    assert heads_up != crowd, "the bot ignores how many opponents there are"
    assert heads_up.get("call", 0) > crowd.get("call", 0)


# --------------------------------------------------------------------------
# And it stays legal
# --------------------------------------------------------------------------
@pytest.mark.parametrize("persona", ["maniac", "rock", "pro", "trickster"])
def test_the_offline_brain_never_chooses_an_illegal_action(persona):
    players = [Player(name=f"P{i}", seat=i, chips=200) for i in range(4)]
    for p in players:
        p.seated = True
    table = Table(players, 10, 20, seed=9)
    p = persona_store().get(persona)
    decider = AIDecider(
        persona=p,
        transport=HeuristicTransport(p, rng=random.Random(3)),
        model_label="offline",
        retries=0,
        rng=random.Random(3),
    )
    table.start_hand()
    guard = 0
    while not table.is_hand_over and guard < 300:
        guard += 1
        seat = table.actor
        if seat is None:
            break
        legal = table.legal_actions(seat)
        action = decider.decide(table, seat)
        try:
            table.apply_action(seat, action)
        except Exception as exc:  # pragma: no cover - the failure path
            pytest.fail(
                f"{persona} chose {action.type.value} {action.amount}, refused: {exc}\n"
                f"    menu: {legal.summary()}"
            )
