"""Per-player context and the player-detail API.

Each AI keeps its own private conversation: the first turn carries the full
table picture, later turns are compact records of what it said and did.  That is
what makes a seat behave like a character with a memory instead of a stateless
oracle, so these tests guard the thread's shape and its bounds.
"""

from __future__ import annotations

import random

import pytest

from pokerarena.ai import (
    AIDecider,
    HeuristicTransport,
    ScriptedTransport,
    compact_state_line,
    compact_turn_record,
)
from pokerarena.arena import Arena, AISpec, build_players
from pokerarena.cards import Card, format_cards
from pokerarena.config import ArenaConfig
from pokerarena.engine import Action, ActionType, Player, Table
from pokerarena.personas import get_persona
from pokerarena.web import create_app


def make_table(stacks=(1000, 1000, 1000), seed=1, button=0, players=3):
    stacks = tuple(stacks)[:players] if len(stacks) >= players else tuple(stacks) + (1000,) * (players - len(stacks))
    people = [Player(name=f"P{i}", seat=i, chips=c) for i, c in enumerate(stacks)]
    table = Table(people, 10, 20, seed=seed)
    table.button = button
    return table


# --------------------------------------------------------------------------
# Compact records
# --------------------------------------------------------------------------
def test_compact_state_line_is_short_and_informative():
    table = make_table(seed=2)
    table.start_hand()
    line = compact_state_line(
        hand=3,
        street="Flop",
        pot=120,
        board=table.board,
        hole=[__import__("pokerarena.cards", fromlist=["Card"]).Card.from_str("As"),
              __import__("pokerarena.cards", fromlist=["Card"]).Card.from_str("Ks")],
    )
    assert "Hand 3" in line
    assert "Flop" in line
    assert "pot 120" in line
    # Cards render as "A\u2660 K\u2660"; check the ranks, not the exact glyphs.
    assert "A" in line and "K" in line
    assert "your cards:" in line
    assert len(line) < 220, "context records must stay compact"


def test_compact_turn_record_captures_speech_and_reasoning():
    table = make_table(seed=2)
    table.start_hand()
    record = compact_turn_record(
        "P0",
        Action(
            ActionType.RAISE,
            amount=200,
            thought="He limped twice, so his range is capped.",
            speech="Time to find out.",
        ),
        hand=4,
        street="Pre-Flop",
        pot=90,
        board=[],
        hole=table.players[0].hole_cards,
    )
    assert "raise to 200" in record
    assert "Time to find out." in record
    assert "range is capped" in record


# --------------------------------------------------------------------------
# The conversation thread itself
# --------------------------------------------------------------------------
def test_first_decision_sends_only_the_full_prompt():
    table = make_table(seed=2)
    table.start_hand()
    decider = AIDecider(get_persona("pro"), ScriptedTransport([]))
    messages = decider._build_messages(table, table.actor)
    assert len(messages) == 1
    assert "YOUR LEGAL ACTIONS" in messages[0]["content"]
    # The opening prompt is remembered for reuse on later turns.
    assert table.players[table.actor].opening_prompt


def test_every_decision_carries_the_whole_public_picture():
    """Each decision sends the full context, not a "pot / board / my cards" line.

    A poker decision depends on the betting that produced the situation, so the
    current situation must always arrive with: position, the ordered betting
    history of this hand, the other players' stacks, and the recent hands.
    """
    table = make_table(seed=2)
    table.start_hand()
    decider = AIDecider(
        get_persona("pro"),
        ScriptedTransport(
            ["<action>call</action><say>I'm in.</say><thought>cheap</thought>"] * 4
        ),
        memory_turns=4,
    )
    seat = table.actor
    decider.decide(table, seat)
    player = table.players[seat]
    assert len(player.conversation) >= 2, "the decision must be remembered"

    messages = decider._build_messages(table, seat)
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"

    latest = messages[-1]["content"]
    for section in (
        "YOUR LEGAL ACTIONS",          # the menu
        "BETTING SO FAR THIS HAND",    # who did what, in order
        "Your position",               # button / blinds / who acts behind
        "THE PLAYERS",                 # stacks and commitments
    ):
        assert section in latest, f"the latest prompt is missing {section!r}"


def test_the_betting_history_is_in_the_prompt():
    """A three-bet pot and a limped pot must not look identical to a seat."""
    table = make_table(seed=2, players=4)
    table.start_hand()
    decider = AIDecider(
        get_persona("pro"),
        ScriptedTransport(["<action>call</action><thought>x</thought>"] * 12),
    )
    # Everyone limps/calls, then one player raises.
    for _ in range(3):
        if table.is_hand_over or table.actor is None:
            break
        seat = table.actor
        decider.decide(table, seat)
        table.apply_action(seat, _last_action(decider))

    prompt = decider._build_messages(table, table.actor)[-1]["content"]
    assert "BETTING SO FAR THIS HAND" in prompt
    assert "posts the small blind" in prompt, "blinds belong to the history"
    assert "posts the big blind" in prompt
    # The order matters, so the log must be chronological.
    log = prompt.split("BETTING SO FAR THIS HAND")[1]
    blinds = log.index("posts the small blind")
    big = log.index("posts the big blind")
    assert blinds < big, "the log must read in betting order"


def test_recent_hands_appear_in_the_prompt():
    """A seat should be able to remember who has been winning and what showed."""
    from pokerarena.arena import Arena, build_players, heuristic_transports
    from pokerarena.config import ArenaConfig

    specs = [
        AISpec(name="A", persona_key="maniac", model=""),
        AISpec(name="B", persona_key="rock", model=""),
    ]
    players, _ = build_players(specs, starting_chips=200)
    arena = Arena(
        players,
        config=ArenaConfig(seed=5, max_rounds=4),
        transports=heuristic_transports(players, 5),
    )
    arena.play_game()
    assert arena.hand_history, "the game should have played some hands"

    arena.table.start_hand()
    seat = arena.table.actor
    decider = arena.deciders[seat]
    decider.hand_history = arena.hand_history
    prompt = decider._build_messages(arena.table, seat)[-1]["content"]
    assert "RECENT HANDS AT THIS TABLE" in prompt
    assert "won" in prompt, "the summary should say who won"


def _last_action(decider: AIDecider):
    return decider.traces[-1].action


def decider_seat(arena, decider) -> int:
    for seat, candidate in arena.deciders.items():
        if candidate is decider:
            return seat
    raise AssertionError("decider is not seated at this arena")


def test_full_record_is_kept_but_only_recent_turns_are_sent():
    """The session record is complete; the prompt is bounded.

    Keeping everything is what makes a god's-eye review possible, while sending
    only the last few turns keeps a long session from growing without limit.
    """
    table = make_table(seed=2)
    table.start_hand()
    seat = table.actor
    decider = AIDecider(
        get_persona("maniac"),
        ScriptedTransport(["<action>call</action><thought>x</thought>"] * 40),
        memory_turns=2,
    )
    from pokerarena.engine import PlayerStatus

    for _ in range(20):
        table.actor = seat
        table.players[seat].status = PlayerStatus.ACTIVE
        try:
            decider.decide(table, seat)
        except Exception:
            pass

    player = table.players[seat]
    # Nothing was discarded from the record.
    assert len(player.conversation) > 1 + 2 * 2, "the full record must be kept"
    # But only memory_turns exchanges plus the opening turn are sent.
    sent = decider._sent_messages(player)
    assert len(sent) <= 1 + 2 * 2


def test_memory_disabled_sends_no_history():
    table = make_table(seed=2)
    table.start_hand()
    seat = table.actor
    decider = AIDecider(
        get_persona("pro"),
        ScriptedTransport(["<action>call</action><thought>x</thought>"] * 4),
        memory_turns=0,
    )
    from pokerarena.engine import PlayerStatus

    for _ in range(3):
        table.actor = seat
        table.players[seat].status = PlayerStatus.ACTIVE
        try:
            decider.decide(table, seat)
        except Exception:
            pass
    player = table.players[seat]
    # Only the opening prompt is sent; no history turns.
    assert len(decider._sent_messages(player)) == 1


def test_opening_prompts_never_contain_opponent_hole_cards():
    """The full prompt is the one place opponents' cards could leak.

    Every seat's opening prompt must contain its own hand and **nothing** about
    anybody else's -- including the new sections (betting log, stacks, recent
    hands), which is where a careless addition would leak.
    """
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="rock", model=""),
        AISpec(name="C", persona_key="maniac", model=""),
    ]
    players, _ = build_players(specs, starting_chips=400)
    arena = Arena(players, config=ArenaConfig(seed=3, max_rounds=6))

    # Capture each seat's own cards at the moment its prompt was written.
    captured: dict[str, tuple[str, set[str]]] = {}
    for _ in range(25):
        arena.play_hand()
        for player in arena.table.players:
            if player.opening_prompt and player.name not in captured:
                captured[player.name] = (
                    player.opening_prompt,
                    {c.label for c in player.hole_cards},
                )
        if len(captured) == len(arena.table.players):
            break
        arena.table.rotate_button()

    assert len(captured) == len(arena.table.players), "not every seat acted"
    for name, (prompt, own_cards) in captured.items():
        assert "HAND #" in prompt
        for card in own_cards:
            assert card in prompt, f"{name}'s prompt should show its own {card}"
        # Stronger than "not the opponents' current cards" -- which drift between
        # hands.  A prompt may contain *only* its owner's two cards.
        foreign = [
            card.label
            for card in _every_card()
            if card.label not in own_cards and card.label in prompt
        ]
        assert not foreign, f"{name}'s prompt leaked card(s): {foreign}"


def _every_card():
    """All 52 cards, in the same rendering the prompt uses."""
    return [Card(rank, suit) for rank in range(13) for suit in range(4)]


def _play_until_everyone_has_acted(arena, max_hands: int = 30) -> None:
    """Play hands until every seat has been consulted at least once.

    A single hand may fold out before some seats act -- that is poker, and it is
    exactly what these tests must not assume away.  The check goes through each
    seat's conversation because per-hand ``history`` is reset every hand.
    """
    for _ in range(max_hands):
        arena.play_hand()
        if all(p.conversation or not p.is_ai for p in arena.table.players):
            return
        arena.table.rotate_button()
    raise AssertionError("some seat never acted across many hands")


def _play_until_everyone_acts_in_one_hand(arena, max_hands: int = 40) -> None:
    """Play hands until a single hand had every seat make a decision.

    Needed wherever the assertion is about per-hand state, which is discarded
    when the next hand starts.
    """
    for _ in range(max_hands):
        arena.play_hand()
        if all(p.history for p in arena.table.players if p.is_ai):
            return
        arena.table.rotate_button()
    raise AssertionError("no hand ever had every seat act")


# --------------------------------------------------------------------------
# Per-seat history
# --------------------------------------------------------------------------
def test_every_decision_is_recorded_on_the_player():
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="rock", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(players, config=ArenaConfig(seed=4, max_rounds=8))
    _play_until_everyone_acts_in_one_hand(arena)
    for player in arena.table.players:
        assert player.history, f"{player.name} has no decision history"
        for entry in player.history:
            assert entry["hand"] >= 1
            assert entry["action"]
            assert "street" in entry and "pot" in entry


def test_history_is_per_hand_not_cumulative():
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="rock", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(players, config=ArenaConfig(seed=4, max_rounds=8))
    _play_until_everyone_acts_in_one_hand(arena)
    first = {p.seat: len(p.history) for p in arena.table.players}
    assert all(count > 0 for count in first.values())
    arena.table.rotate_button()
    arena.play_hand()
    # The engine resets per-hand history, while the conversation survives.
    for player in arena.table.players:
        assert len(player.history) <= 12, "per-hand history should reset"
        assert player.conversation, "conversation must survive between hands"


def test_conversation_survives_across_hands():
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="rock", model=""),
    ]
    players, _ = build_players(specs, starting_chips=400)
    arena = Arena(players, config=ArenaConfig(seed=6, max_rounds=10))
    arena.play_hand()
    sizes = [len(p.conversation) for p in arena.table.players if p.is_ai]
    assert max(sizes) > 1
    arena.table.rotate_button()
    arena.play_hand()
    after = [len(p.conversation) for p in arena.table.players if p.is_ai]
    assert max(after) >= max(sizes), "memory should carry over between hands"


# --------------------------------------------------------------------------
# Web endpoint
# --------------------------------------------------------------------------
@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _offline_game(client, personas=("maniac", "rock"), hands=12, seed=2):
    response = client.post(
        "/api/game",
        json={
            "personas": list(personas),
            "with_human": False,
            "offline": True,
            "speed_seconds": 0.0,
            "starting_chips": 300,
            "max_hands": hands,
            "seed": seed,
        },
    )
    return response.get_json()["session"]["session"]


def _wait_for_history(client, session_id, timeout=25.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        for seat in range(3):
            response = client.get(f"/api/game/{session_id}/player/{seat}")
            if response.status_code == 200:
                detail = response.get_json()["player"]
                if detail["history"]:
                    return detail
        time.sleep(0.05)
    raise AssertionError("no player history appeared in time")


def test_player_endpoint_returns_persona_stats_and_history(client):
    session_id = _offline_game(client)
    detail = _wait_for_history(client, session_id)

    assert detail["persona"] is not None
    assert set(detail["persona"]) >= {
        "name",
        "tagline",
        "style",
        "voice",
        "aggression",
        "tightness",
    }
    assert detail["model"]
    stats = detail["stats"]
    assert set(stats) >= {"decisions", "fold", "call", "raise", "vpip"}
    assert stats["decisions"] >= 1
    entry = detail["history"][-1]
    assert entry["action"]
    assert entry["pot"] is not None, "the pot must be recorded for the UI"
    assert "hand" in entry and "street" in entry


def test_player_endpoint_reveals_cards_in_god_view(client):
    """The browser is a spectator: every hole card is visible.

    This is display only.  The prompts sent to the models never contain another
    seat's cards (see test_context_does_not_leak_other_players_cards).
    """
    session_id = _offline_game(client)
    detail = _wait_for_history(client, session_id)
    assert "cards_hidden" not in detail, "the god view never hides cards"
    assert detail["hole_cards"], "a seated player should have visible cards"


def test_player_endpoint_exposes_the_private_context(client):
    session_id = _offline_game(client)
    _wait_for_history(client, session_id)
    plain = client.get(f"/api/game/{session_id}/player/0").get_json()["player"]
    assert "context" not in plain, "the raw context is opt-in (it is large)"

    detailed = client.get(
        f"/api/game/{session_id}/player/0?context=1"
    ).get_json()["player"]
    context = detailed["context"]
    assert context, "the seat should be carrying a private thread"
    assert context[0]["role"] == "user"
    assert "YOUR LEGAL ACTIONS" in context[0]["content"]
    assert detailed["memory"]["messages_recorded"] == len(context) - 1, (
        "the recorded turns plus the opening prompt make up the whole thread"
    )
    # The record is complete but only a few turns are charged to the prompt.
    assert detailed["memory"]["messages_sent"] <= 1 + 2 * detailed["memory"]["turns_kept"]


def test_player_endpoint_rejects_an_unknown_seat(client):
    session_id = _offline_game(client)
    assert client.get(f"/api/game/{session_id}/player/99").status_code == 404
    assert client.get("/api/game/nope/player/0").status_code == 404


def test_context_only_contains_legal_information(client):
    """A seat's private context must never leak another seat's *hidden* cards.

    The UI shows everything, but the *prompt* must not: being able to see the
    context tab is exactly how we prove the models are playing honestly.

    Two things are legitimately public and must not be mistaken for a leak:

    * cards revealed at a **past showdown** (everyone saw them), and
    * a card code recurring in a later hand, since the deck is reshuffled.

    So the check is scoped to the current hand: one decision, taken while the
    opponents' cards are still unknown.
    """
    session_id = _offline_game(client, personas=("maniac", "rock", "pro"), hands=1, seed=2)
    _wait_for_history(client, session_id)

    for seat in range(3):
        context = client.get(
            f"/api/game/{session_id}/player/{seat}?context=1"
        ).get_json()["player"].get("context", [])
        assert context, f"seat {seat} produced no context"
        # Only the opening decision, before any showdown has been recorded.
        opening = context[0]["content"]
        assert "RECENT HANDS" not in opening, "hand 1 has no history to show"

        snapshot = client.get(f"/api/game/{session_id}").get_json()["session"]
        own = set(next(p for p in snapshot["players"] if p["seat"] == seat)["hole_cards"] or [])
        assert own, "the seat should have been dealt in"
        foreign = [
            card
            for other in snapshot["players"]
            if other["seat"] != seat and other["seated"]
            for card in (other["hole_cards"] or [])
            if card not in own and card in opening
        ]
        assert not foreign, f"seat {seat} context leaked {foreign}"
