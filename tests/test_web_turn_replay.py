"""Public replay and browser input preserve the state of a real decision."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time

import pytest

from pokerarena.engine import Action, ActionType, IllegalAction
from pokerarena.web_state import GameSession
from tests.helpers import session_config


def session(*, human=False, god=False):
    return GameSession(session_config(
        ["rock", "calculating"] if human else ["rock", "calculating", "maniac"],
        with_human=human, offline=True, reveal_all=god, seed=31,
        speed_seconds=0, hand_result_seconds=0,
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


def test_duplicate_browser_clicks_enqueue_exactly_one_action():
    game = session(human=True)
    start(game)
    game.awaiting_human = True
    token = game.snapshot()["turn_token"]

    def click():
        try:
            return game.submit_action("call", turn_token=token)["queued"]
        except IllegalAction:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: click(), range(2)))
    assert sorted(results) == [False, True]
    assert game.pending_action.qsize() == 1
    assert game.awaiting_human is False
    assert game.snapshot()["turn_token"] is None
    assert game.snapshot()["legal"] is None


def test_previous_turn_token_cannot_act_in_a_new_hand_and_legacy_still_works():
    game = session(human=True)
    start(game)
    game.awaiting_human = True
    old_token = game.snapshot()["turn_token"]
    assert game.submit_action("fold", turn_token=old_token)["queued"]
    game.pending_action.get_nowait()
    start(game)
    game.awaiting_human = True
    with pytest.raises(IllegalAction, match="turn has changed"):
        game.submit_action("call", turn_token=old_token)
    assert game.pending_action.empty()
    assert game.submit_action("call")["queued"]
    with pytest.raises(IllegalAction, match="not your turn"):
        game.submit_action("call")


def test_invalid_action_does_not_consume_the_human_turn():
    game = session(human=True)
    start(game)
    game.awaiting_human = True
    token = game.snapshot()["turn_token"]
    with pytest.raises(IllegalAction):
        game.submit_action("check", turn_token=token)
    assert game.awaiting_human
    assert game.pending_action.empty()
    assert game.submit_action("call", turn_token=token)["queued"]


def test_replay_keeps_stacks_cards_and_board_after_another_hand_is_dealt():
    game = session(god=True)
    start(game)
    passive(game)
    original = deepcopy(game.snapshot()["actions"][0]["replay"])
    assert original["hand_number"] == 1
    assert original["board"] == []
    assert original["legal"] is None
    assert original["is_replay"]
    while not game.arena.table.is_hand_over:
        passive(game)
    start(game)
    passive(game)
    assert game.snapshot()["hand_number"] == 2
    assert game.snapshot()["actions"][0]["replay"] == original
    assert original["players"][0]["hole_cards"] != game.snapshot()["players"][0]["hole_cards"]


def test_replay_cannot_see_a_future_showdown_or_an_opponents_private_deal_event():
    game = session()
    start(game)
    passive(game)
    first = game.snapshot()["actions"][0]
    assert all(player["hole_cards"] is None for player in first["replay"]["players"])
    deal = next(event for event in game.snapshot()["events"] if event["kind"] == "hole_cards_dealt")
    assert deal["players"] == {}
    while not game.arena.table.is_hand_over:
        passive(game)
    game.finished = True
    current = game.snapshot()
    assert all(player["hole_cards"] for player in current["players"] if player["seated"])
    historical = current["actions"][0]["replay"]
    assert historical["showdown"] == []
    assert historical["hand_result"] is None
    assert all(player["hole_cards"] is None for player in historical["players"])
    assert all(event["kind"] != "showdown" for event in historical["events"])


def test_folded_hands_stay_secret_and_showdown_never_reveals_private_thoughts():
    game = session()
    start(game)
    folded = game.arena.table.actor
    apply(game, Action(ActionType.FOLD, thought="a private bluff read", source="llm"))
    while not game.arena.table.is_hand_over:
        passive(game)
    game.finished = True
    game.arena.transcript.append({"seat": folded, "hand": 1, "action": "fold",
                                  "thought": "private thought", "reasoning": "private reasoning"})
    player = game.snapshot()["players"][folded]
    assert player["hole_cards"] is None
    assert game.snapshot()["actions"][0]["thought"] == ""
    detail = game.player_detail(folded, include_prompt=True)
    assert detail["hand_strength"] is None
    assert detail["reasoning_visible"] is False
    assert detail["history"][0]["thought"] == ""
    assert "reasoning" not in detail["history"][0]
    for entry in game.arena.table.showdown_results:
        revealed = game.player_detail(entry["seat"], include_prompt=True)
        assert revealed["hole_cards"]
        assert revealed["reasoning_visible"] is False
        assert revealed["context"] == []
    game.set_god_mode(True)
    assert game.player_detail(folded)["hole_cards"]
    assert game.snapshot()["actions"][0]["replay"]["players"][folded]["hole_cards"]
    game.set_god_mode(False)
    assert game.snapshot()["actions"][0]["replay"]["players"][folded]["hole_cards"] is None


def test_action_that_closes_street_records_its_original_street():
    game = session()
    start(game)
    while game.arena.table.street.value == "preflop":
        passive(game)
    action = game.snapshot()["actions"][-1]
    assert action["street"] == "preflop"
    assert action["replay"]["street"] == "flop"
    assert len(action["replay"]["board"]) == 3


def test_all_in_players_are_counted_in_the_live_hand():
    game = session()
    start(game)
    actor = game.arena.table.actor
    apply(game, Action(ActionType.ALL_IN))
    state = game.snapshot()
    assert state["players"][actor]["chips"] == 0
    assert state["players"][actor]["is_all_in"] is True
    assert state["players"][actor]["in_hand"] is True
    assert state["active_players"] == 3
    assert {player["position"] for player in state["players"] if player["seated"]} == {"BTN", "SB", "BB"}


def test_detailed_replay_frames_are_bounded_but_action_rows_remain(monkeypatch):
    monkeypatch.setattr("pokerarena.web_state.MAX_REPLAY_FRAMES", 2)
    game = session()
    start(game)
    for _ in range(4):
        passive(game)
    actions = game.snapshot()["actions"]
    assert len(actions) == 4
    assert [action["seq"] for action in actions if "replay" in action] == [3, 4]
    assert game.snapshot()["replay_first_seq"] == 3
    assert game.snapshot()["replay_last_seq"] == 4


def test_incremental_snapshots_send_only_new_replay_frames():
    game = session()
    start(game)
    passive(game)
    passive(game)
    partial = game.snapshot(replay_after=1)
    assert [action["seq"] for action in partial["actions"]] == [1, 2]
    assert [action["seq"] for action in partial["actions"] if "replay" in action] == [2]
    assert partial["replay_first_seq"] == 1
    assert partial["replay_last_seq"] == 2
    assert not any("replay" in action for action in game.snapshot(replay_after=2)["actions"])


def test_http_turn_token_and_incremental_replay_contract():
    from pokerarena.web import create_app

    app = create_app()
    game = session(human=True)
    start(game)
    game.awaiting_human = True
    app.extensions["poker_manager"].sessions[game.id] = game
    with app.test_client() as client:
        path = f"/api/game/{game.id}"
        state = client.get(path).get_json()["session"]
        assert client.get(path + "?replay_after=invalid").status_code == 400
        stale = client.post(path + "/action", json={"action": "call", "turn_token": "previous-turn"})
        assert stale.status_code == 409
        accepted = client.post(path + "/action", json={"action": "call", "turn_token": state["turn_token"]})
        assert accepted.status_code == 200
        assert client.post(path + "/action", json={"action": "call", "turn_token": state["turn_token"]}).status_code == 409


def test_human_legal_price_uses_only_the_pot_their_stack_can_win():
    game = session(human=True)
    start(game)
    table = game.arena.table
    human = table.players[0]
    human.chips = 100
    human.street_bet = human.hand_contribution = 0
    for player in table.players[1:3]:
        player.street_bet = player.hand_contribution = 400
    table.pot = 800
    table.current_bet = 400
    table.hand_contributions = {0: 0, 1: 400, 2: 400}
    game.awaiting_human = True
    legal = game.snapshot()["legal"]
    assert legal["call_cost"] == 100
    assert legal["eligible_pot_after_call"] == 300
    assert legal["call_equity_required"] == pytest.approx(1 / 3)
    assert legal["ineligible_pot"] == 600


def test_stop_releases_a_waiting_human_without_carrying_input():
    game = session(human=True)
    start(game)
    answered = []
    worker = threading.Thread(target=lambda: answered.append(game._human_decider(game.arena.table, 0)))
    worker.start()
    deadline = time.monotonic() + 2
    while not game.awaiting_human and time.monotonic() < deadline:
        time.sleep(0.01)
    assert game.awaiting_human
    assert game.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(answered) == 1
    state = game.snapshot()
    assert state["stopped"] and state["finished"]
    assert state["winner"] is None
    assert state["turn_token"] is None
    assert not state["awaiting_human"]


def test_stop_cancels_pause_and_finished_hand_hold():
    game = session()
    game.paused = True
    game.hold_until = time.time() + 20
    game.hold_total = 20
    workers = [threading.Thread(target=game.arena.control.wait_if_paused),
               threading.Thread(target=lambda: game.arena.control.between_hands(game.arena.table))]
    for worker in workers:
        worker.start()
    game.stop()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert game.arena.control.should_stop()
    assert not game.paused
    assert not game.snapshot()["holding_result"]


def test_stop_discards_an_inflight_model_reply_and_starts_no_other_calls():
    from pokerarena.ai import ChatResult

    requested = threading.Event()
    complete = threading.Event()
    calls = []

    class BlockingTransport:
        def complete(self, *args, **kwargs):
            calls.append(1)
            requested.set()
            assert complete.wait(timeout=3)
            return ChatResult(content="<action>call</action><thought>private</thought><say></say>")

    game = session()
    for decider in game.arena.deciders.values():
        decider.transport = BlockingTransport()
    game.arena.save_history = lambda: None
    game.start()
    assert requested.wait(timeout=2)
    game.stop()
    complete.set()
    game._thread.join(timeout=3)
    assert not game._thread.is_alive()
    assert len(calls) == 1
    assert game.actions == []
    assert game.arena.hand_history == []
    assert game.error is None
    assert game.winner is None
    assert not any(event["kind"] == "game_end" for event in game.snapshot()["events"])


def test_http_stop_is_explicit_and_idempotent():
    from pokerarena.web import create_app

    app = create_app()
    game = session()
    app.extensions["poker_manager"].sessions[game.id] = game
    with app.test_client() as client:
        for _ in range(2):
            result = client.post(f"/api/game/{game.id}/control", json={"action": "stop"})
            assert result.status_code == 200
            assert result.get_json()["stopped"]
            assert result.get_json()["finished"]


def test_replacement_validates_first_then_stops_previous_session(monkeypatch):
    from pokerarena.web_state import SessionManager, SessionConfig

    monkeypatch.setattr(GameSession, "start", lambda self: None)
    manager = SessionManager()
    previous = manager.create(session().config)
    with pytest.raises(ValueError):
        manager.create(SessionConfig(), replace_session=previous.id)
    assert not previous.snapshot()["stopped"]
    replacement = manager.create(session().config, replace_session=previous.id)
    assert previous.snapshot()["stopped"]
    assert not replacement.snapshot()["stopped"]
    with pytest.raises(ValueError, match="unknown replacement"):
        manager.create(session().config, replace_session="missing")
    assert len(manager.sessions) == 2


def test_evicted_sessions_are_stopped(monkeypatch):
    from pokerarena.web_state import SessionManager

    monkeypatch.setattr(GameSession, "start", lambda self: None)
    manager = SessionManager(max_sessions=1)
    previous = manager.create(session().config)
    current = manager.create(session().config)
    assert previous.snapshot()["stopped"]
    assert manager.get(previous.id) is None
    assert not current.snapshot()["stopped"]


def test_player_history_uses_unique_global_sequences_and_keeps_fallbacks():
    game = session(god=True)
    start(game)
    actor = game.arena.table.actor
    apply(game, Action(ActionType.CALL, source="fallback", thought="provider unavailable"))
    while not game.arena.table.is_hand_over:
        passive(game)
    detail = game.player_detail(actor)
    expected = [action["seq"] for action in game.actions if action["seat"] == actor]
    assert [entry["seq"] for entry in detail["history"]] == expected
    assert len(set(expected)) == len(expected)
    assert detail["history"][0]["source"] == "fallback"
    assert {entry["street"] for entry in detail["history"]} == {"preflop", "flop", "turn", "river"}


def test_hand_cap_returns_chip_leader_and_marks_the_tournament_unfinished():
    game = session()
    game.arena.config.max_rounds = 1
    game.arena.decide = lambda table, seat: Action(ActionType.FOLD)
    game.arena.save_history = lambda: None
    game._run()
    table = game.arena.table
    leader = table.players[2].name  # Big blind wins the matched blinds.
    assert not table.is_game_over()
    state = game.snapshot()
    assert state["winner"] == leader
    assert state["chip_leaders"] == [leader]
    assert state["completion_reason"] == "hand_cap"
    assert state["button"] == 0  # Keep the final hand's dealer during its result.
    ending = next(event for event in state["events"] if event["kind"] == "game_end")
    assert ending["data"]["reason"] == "hand_cap"


def test_split_pot_records_winners_even_when_every_net_chip_change_is_zero():
    from pokerarena.cards import Card

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
    record = game.arena._hand_record({0: 1000, 1: 1000, 2: 1000})
    assert all(record["deltas"][seat] == 0 for seat in range(3))
    assert set(record["winners"]) == {player.name for player in table.players[:3]}
