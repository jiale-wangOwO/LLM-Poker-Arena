"""Complete public hand narratives survive result holds and subsequent deals."""

from copy import deepcopy
import json

import pytest

from pokerarena.cards import Card
from pokerarena.engine import Action, ActionType
from pokerarena.web_state import GameSession, SeatSpec
from tests.helpers import session_config


def session(*, human=False, god=False, **options):
    return GameSession(session_config(
        ["rock", "calculating"] if human else ["rock", "calculating", "maniac"],
        with_human=human, offline=True, reveal_all=god, seed=31,
        speed_seconds=0, hand_result_seconds=0, **options,
    ))


def start(game):
    game.arena.table.start_hand()
    game.arena._flush()


def apply(game, action):
    table = game.arena.table
    seat = table.actor
    applied = table.apply_action(seat, action)
    game.arena.control.record_action(table, seat, applied)
    game.arena._flush()


def passive(game):
    table = game.arena.table
    legal = table.legal_actions(table.actor)
    apply(game, Action(ActionType.CHECK if legal.can_check else ActionType.CALL))


def complete(game):
    while not game.arena.table.is_hand_over:
        passive(game)


def test_hand_is_visible_before_first_decision_with_forced_bets_and_stacks():
    game = session(human=True)
    start(game)
    snapshot = game.snapshot()
    assert snapshot["actions"] == []
    hand, = snapshot["hands"]
    assert hand["hand_number"] == 1
    assert hand["status"] == "in_progress"
    assert hand["result"] is None
    assert hand["button"] == 0
    assert hand["small_blind"] == 10
    assert hand["big_blind"] == 20
    assert hand["ante"] == 0
    assert [player["position"] for player in hand["players"]] == ["BTN", "SB", "BB"]
    assert [player["starting_chips"] for player in hand["players"]] == [1000] * 3
    assert hand["players"][0]["name"] == "You"
    assert hand["entries"][0] == {"kind": "street", "street": "preflop", "label": "Pre-Flop", "board": [], "pot": 0}
    blinds = [entry for entry in hand["entries"] if entry["kind"] == "blind"]
    assert [(entry["role"], entry["seat"], entry["amount"], entry["nominal_amount"]) for entry in blinds] == [
        ("small", 1, 10, 10), ("big", 2, 20, 20),
    ]
    assert not any(entry["all_in"] for entry in blinds)


def test_history_preserves_each_decision_street_and_global_replay_sequence():
    game = session(human=True)
    start(game)
    apply(game, Action(ActionType.CALL, speech="I'm staying in.", source="human"))
    complete(game)
    snapshot = game.snapshot()
    hand, = snapshot["hands"]
    actions = [entry for entry in hand["entries"] if entry["kind"] == "action"]
    assert [entry["seq"] for entry in actions] == [entry["seq"] for entry in snapshot["actions"]]
    assert actions[0]["name"] == "You"
    assert actions[0]["speech"] == "I'm staying in."
    assert actions[0]["source"] == "human"
    streets = [entry for entry in hand["entries"] if entry["kind"] == "street"]
    assert [entry["street"] for entry in streets] == ["preflop", "flop", "turn", "river"]
    assert [len(entry["board"]) for entry in streets] == [0, 3, 4, 5]
    current = "preflop"
    for entry in hand["entries"]:
        if entry["kind"] == "street":
            current = entry["street"]
        elif entry["kind"] == "action":
            assert entry["street"] == current
    assert all("replay" not in entry and "thought" not in entry for entry in actions)


def test_settlement_is_complete_before_arena_appends_record_and_survives_next_deal():
    game = session()
    start(game)
    while not game.arena.table.is_hand_over:
        apply(game, Action(ActionType.FOLD))
    assert game.arena.hand_history == []
    state = game.snapshot()
    completed, = state["hands"]
    assert completed["status"] == "completed"
    result = completed["result"]
    assert result == state["hand_result"]
    assert result["pot"] == 20
    assert result["board"] == []
    assert not result["reached_showdown"]
    assert result["showdown"] == []
    assert result["winners"] == [game.arena.table.players[2].name]
    assert result["deltas"] == {0: 0, 1: -10, 2: 10}
    assert {(payout["amount"], payout["reason"]) for payout in result["payouts"]} == {
        (10, "uncalled_bet_returned"), (20, "all_folded"),
    }
    captured = deepcopy(completed)
    game.arena.table.rotate_button()
    start(game)
    passive(game)
    assert game.snapshot()["hands"][0] == captured
    assert game.snapshot()["hands"][1]["status"] == "in_progress"
    assert game.snapshot()["hands"][1]["button"] != captured["button"]


def test_new_settlement_does_not_show_an_older_arena_result_during_publication_gap():
    game = session()
    game.arena.decide = lambda table, seat: Action(ActionType.FOLD)
    game.arena.play_hand()
    assert len(game.arena.hand_history) == 1
    game.arena.table.rotate_button()
    start(game)
    while not game.arena.table.is_hand_over:
        apply(game, Action(ActionType.FOLD))
    state = game.snapshot()
    assert state["hand_result"]["hand"] == 2
    assert state["hands"][-1]["result"] == state["hand_result"]


def test_antes_and_short_blinds_show_actual_posted_amounts():
    game = session()
    table = game.arena.table
    table.players[1].chips = 12
    table.players[2].chips = 8
    table.set_ante(5)
    start(game)
    hand, = game.snapshot()["hands"]
    assert [player["starting_chips"] for player in hand["players"]] == [1000, 12, 8]
    antes = [entry for entry in hand["entries"] if entry["kind"] == "ante"]
    assert [entry["amount"] for entry in antes] == [5, 5, 5]
    blinds = [entry for entry in hand["entries"] if entry["kind"] == "blind"]
    assert [(entry["amount"], entry["nominal_amount"], entry["all_in"]) for entry in blinds] == [
        (7, 10, True), (3, 20, True),
    ]
    assert next(event for event in game.snapshot()["events"] if event["kind"] == "antes_posted")["paid"] == {0: 5, 1: 5, 2: 5}


def test_forced_bets_only_hand_has_complete_result_and_original_starting_stacks():
    game = session(seats=[
        SeatSpec(seat=0, kind="ai", persona="rock"),
        SeatSpec(seat=1, kind="ai", persona="calculating"),
    ])
    table = game.arena.table
    table.players[0].chips = 5
    table.players[1].chips = 10
    start(game)
    assert table.is_hand_over
    state = game.snapshot()
    assert state["actions"] == []
    hand, = state["hands"]
    assert hand["status"] == "completed"
    assert [player["starting_chips"] for player in hand["players"]] == [5, 10]
    assert not any(entry["kind"] == "action" for entry in hand["entries"])
    assert [entry["role"] for entry in hand["entries"] if entry["kind"] == "blind"] == ["small", "big"]
    result = hand["result"]
    assert len(result["board"]) == 5
    assert result["pot"] == 10
    assert sum(result["deltas"].values()) == 0
    assert sum(payout["amount"] for payout in result["payouts"]) == 15
    assert len(result["showdown"]) == 2


def test_history_result_includes_side_pots_refunds_and_correct_net_change():
    game = session()
    table = game.arena.table
    for player, chips in zip(table.players[:3], (400, 100, 200)):
        player.chips = chips
    start(game)
    while not table.is_hand_over:
        apply(game, Action(ActionType.ALL_IN))
    hand, = game.snapshot()["hands"]
    result = hand["result"]
    assert sum(result["deltas"].values()) == 0
    assert [pot["amount"] for pot in result["pots"]] == [300, 200]
    assert result["pot"] == 500
    assert sum(payout["amount"] for payout in result["payouts"]) == 700
    assert any(payout["reason"] == "uncalled_bet_returned" and payout["amount"] == 200 for payout in result["payouts"])
    assert {entry["seat"] for entry in result["showdown"]} == {0, 1, 2}


def test_zero_profit_split_keeps_every_winner_and_board():
    game = session()
    start(game)
    table = game.arena.table
    table.board = [Card.from_str(code) for code in ("As", "Ks", "Qs", "Js", "Ts")]
    for player, codes in zip(table.players[:3], [("2c", "3c"), ("2d", "3d"), ("2h", "3h")]):
        player.hole_cards = [Card.from_str(code) for code in codes]
        player.chips = 980
        player.street_bet = player.hand_contribution = 20
    table.pot = 60
    table.hand_contributions = {0: 20, 1: 20, 2: 20}
    table.to_showdown()
    game.arena._flush()
    result = game.snapshot()["hands"][0]["result"]
    assert result["deltas"] == {0: 0, 1: 0, 2: 0}
    assert len(result["winners"]) == 3
    assert len(result["showdown"]) == 3
    assert all(payout["split"] for payout in result["payouts"])


def test_history_never_copies_private_thoughts_or_folded_cards_even_in_god_mode():
    game = session(god=True)
    start(game)
    folded = game.arena.table.actor
    cards = [card.code for card in game.arena.table.players[folded].hole_cards]
    apply(game, Action(ActionType.FOLD, thought="private fold inference", speech="See you next hand.", source="llm"))
    complete(game)
    private = game.snapshot()
    assert private["actions"][0]["thought"] == "private fold inference"
    hand, = private["hands"]
    assert "private fold inference" not in json.dumps(hand)
    assert "hole_cards" not in json.dumps(hand["players"])
    assert all(entry["seat"] != folded for entry in hand["result"]["showdown"])
    game.set_god_mode(False)
    public = game.snapshot()
    assert public["hands"] == private["hands"]
    assert public["players"][folded]["hole_cards"] is None
    assert public["actions"][0]["thought"] == ""
    assert all(entry.get("cards") != cards for entry in hand["result"]["showdown"])


def test_hand_history_survives_eviction_of_live_events_and_replay_frames(monkeypatch):
    monkeypatch.setattr("pokerarena.web_state.MAX_EVENT_LOG", 2)
    monkeypatch.setattr("pokerarena.web_state.MAX_REPLAY_FRAMES", 1)
    game = session()
    start(game)
    complete(game)
    captured = deepcopy(game.snapshot()["hands"][0])
    start(game)
    passive(game)
    state = game.snapshot()
    assert state["hands"][0] == captured
    assert len(state["events"]) == 2
    assert state["replay_first_seq"] > 1
    assert state["hands"][0]["entries"][1]["kind"] == "blind"
    assert state["hands"][0]["result"]["winners"]


def test_retention_evicts_complete_oldest_hands_and_reports_where_history_starts(monkeypatch):
    monkeypatch.setattr("pokerarena.web_state.MAX_HISTORY_HANDS", 2)
    game = session()
    for _ in range(3):
        start(game)
        while not game.arena.table.is_hand_over:
            apply(game, Action(ActionType.FOLD))
    state = game.snapshot()
    assert state["history_truncated"]
    assert state["history_first_hand"] == 2
    assert [hand["hand_number"] for hand in state["hands"]] == [2, 3]
    assert all(hand["status"] == "completed" and hand["result"] and not hand["truncated"] for hand in state["hands"])


def test_action_retention_discards_whole_oldest_hand_or_marks_single_partial(monkeypatch):
    monkeypatch.setattr("pokerarena.web_state.MAX_HISTORY_ACTIONS", 3)
    game = session()
    start(game)
    while not game.arena.table.is_hand_over:
        apply(game, Action(ActionType.FOLD))
    start(game)
    passive(game)
    passive(game)
    state = game.snapshot()
    assert [hand["hand_number"] for hand in state["hands"]] == [2]
    assert state["history_truncated"]
    passive(game)
    passive(game)
    hand, = game.snapshot()["hands"]
    assert hand["truncated"]
    actions = [entry for entry in hand["entries"] if entry["kind"] == "action"]
    assert len(actions) == 3
    assert [entry["seq"] for entry in actions] == [4, 5, 6]
    assert any(entry["kind"] == "blind" for entry in hand["entries"])


def test_http_history_contract_and_snapshot_copies_are_independent():
    from pokerarena.web import create_app

    game = session()
    start(game)
    app = create_app()
    app.extensions["poker_manager"].sessions[game.id] = game
    with app.test_client() as client:
        state = client.get(f"/api/game/{game.id}?replay_after=0").get_json()["session"]
    assert state["hands"][0]["hand_number"] == 1
    assert state["history_first_hand"] == 1
    assert state["history_truncated"] is False
    assert state["hands"][0]["players"][0]["starting_chips"] == 1000
    state["hands"][0]["entries"][1]["amount"] = 999
    assert game.snapshot()["hands"][0]["entries"][1]["amount"] == 10


def test_incremental_history_returns_new_and_live_hands_then_empty_at_completed_cursor():
    game = session()
    start(game)
    complete(game)
    first = deepcopy(game.snapshot()["hands"][0])
    # Arena's completed-record append must not invalidate a cached narrative.
    game.arena.hand_history.append(game.arena._hand_record({0: 1000, 1: 1000, 2: 1000}))
    assert game.snapshot()["hands"][0] == first
    start(game)
    state = game.snapshot(history_after=1)
    assert [hand["hand_number"] for hand in state["hands"]] == [2]
    assert state["hands"][0]["status"] == "in_progress"
    assert not any(entry["kind"] == "action" for entry in state["hands"][0]["entries"])
    passive(game)
    assert any(entry["kind"] == "action" for entry in game.snapshot(history_after=1)["hands"][0]["entries"])
    complete(game)
    final = game.snapshot(history_after=1)
    assert final["hands"][0]["status"] == "completed"
    assert final["hands"][0]["result"]["hand"] == 2
    assert game.snapshot(history_after=2)["hands"] == []
    assert game.snapshot(history_after=0)["hands"] == game.snapshot()["hands"]


def test_incremental_history_keeps_retention_floor_on_empty_delta(monkeypatch):
    monkeypatch.setattr("pokerarena.web_state.MAX_HISTORY_HANDS", 2)
    game = session()
    for _ in range(3):
        start(game)
        complete(game)
    delta = game.snapshot(history_after=3)
    assert delta["hands"] == []
    assert delta["history_first_hand"] == 2
    assert delta["history_truncated"]
    assert [hand["hand_number"] for hand in game.snapshot(history_after=0)["hands"]] == [2, 3]


@pytest.mark.parametrize("path", ["/api/game", "/api/game/{session_id}"])
def test_http_incremental_hand_history_supports_both_routes_and_cursor_reset(path):
    from pokerarena.web import create_app

    game = session()
    start(game)
    complete(game)
    start(game)
    passive(game)
    app = create_app()
    manager = app.extensions["poker_manager"]
    manager.sessions[game.id] = game
    manager.order.append(game.id)
    path = path.format(session_id=game.id)
    with app.test_client() as client:
        full = client.get(path).get_json()["session"]
        delta = client.get(path + "?history_after=1&replay_after=1").get_json()["session"]
        assert [hand["hand_number"] for hand in full["hands"]] == [1, 2]
        assert [hand["hand_number"] for hand in delta["hands"]] == [2]
        assert delta["history_first_hand"] == full["history_first_hand"]
        assert delta["history_truncated"] == full["history_truncated"]
        assert [action["seq"] for action in delta["actions"]] == [action["seq"] for action in full["actions"]]
        reset = client.get(path + "?history_after=0").get_json()["session"]
        assert reset["hands"] == full["hands"]


@pytest.mark.parametrize("cursor", ["-1", "invalid", "1.5", "", "null"])
def test_http_rejects_invalid_history_cursor(cursor):
    from pokerarena.web import create_app

    game = session()
    app = create_app()
    app.extensions["poker_manager"].sessions[game.id] = game
    with app.test_client() as client:
        response = client.get(f"/api/game/{game.id}?history_after={cursor}")
    assert response.status_code == 400
    assert response.get_json()["error"] == "history_after must be a non-negative hand number"
