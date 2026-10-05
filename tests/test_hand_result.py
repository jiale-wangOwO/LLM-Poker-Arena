"""End-of-hand presentation: the result the table shows, and the hold.

Two player-facing behaviours that were missing:

* after a hand ends the table now **holds the result** for a moment, instead of
  dealing the next hand instantly (you could not see who won, let alone what
  they held);
* that result is exposed as structured data -- winner, pot, board, and the
  hands that were shown -- so the UI can present it.
"""

from __future__ import annotations

import time

import pytest

from pokerarena.providers import ProviderStore
from pokerarena.personas import PersonaStore
from pokerarena.web import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    providers = ProviderStore(path=tmp_path / "providers.json")
    personas = PersonaStore(path=tmp_path / "personas.json")
    monkeypatch.setattr("pokerarena.web.PROVIDERS", providers)
    monkeypatch.setattr("pokerarena.web.PERSONA_STORE", personas)
    monkeypatch.setattr("pokerarena.web_state._PROVIDER_STORE", providers)
    monkeypatch.setattr("pokerarena.web_state._PERSONA_STORE", personas)
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def start(client, **overrides):
    payload = {
        "personas": ["maniac", "rock", "pro"],
        "with_human": False,
        "offline": True,
        "speed_seconds": 0.0,
        "starting_chips": 120,
        "max_hands": 60,
        "seed": 9,
        # Most tests are about the shape of the data, not the pause, so keep the
        # suite fast; the hold itself is covered by its own test.
        "hand_result_seconds": 0,
    }
    payload.update(overrides)
    response = client.post("/api/game", json=payload)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["session"]["session"]


def snap(client, session_id):
    return client.get(f"/api/game/{session_id}").get_json()["session"]


def test_the_result_is_exposed_and_shaped_for_display(client):
    session_id = start(client)
    deadline = time.time() + 30
    result = None
    while time.time() < deadline:
        state = snap(client, session_id)
        if state.get("hand_result") and state["hand_result"]["winners"]:
            result = state["hand_result"]
            break
        if state["finished"]:
            break
        time.sleep(0.03)
    assert result is not None, "no completed hand was ever reported"

    assert set(result) >= {
        "hand", "board", "winners", "pot", "reached_showdown", "showdown", "deltas",
    }
    assert result["hand"] >= 1
    assert result["winners"], "a hand must have a winner"
    assert isinstance(result["board"], list)


def test_a_showdown_reports_one_entry_per_player(client):
    """A seat involved in two pots must not be listed twice."""
    session_id = start(client, max_hands=80, seed=4)
    deadline = time.time() + 40
    seen: set[int] = set()
    showdowns = 0
    while time.time() < deadline and showdowns < 3:
        state = snap(client, session_id)
        result = state.get("hand_result")
        if result and result["hand"] not in seen:
            seen.add(result["hand"])
            if result["reached_showdown"]:
                showdowns += 1
                seats = [entry["seat"] for entry in result["showdown"]]
                assert len(seats) == len(set(seats)), (
                    f"hand {result['hand']} listed a seat twice: {seats}"
                )
                for entry in result["showdown"]:
                    assert entry["name"]
                    assert entry["cards"], "a revealed hand must have cards"
                    assert entry["hand_name"], "a revealed hand must be named"
        if state["finished"]:
            break
        time.sleep(0.03)
    assert showdowns, "no showdown happened to check"


def test_the_hand_is_held_before_the_next_one_is_dealt(client):
    """The gap between hands is what lets a spectator see the result."""
    session_id = start(client, speed_seconds=0.0, hand_result_seconds=1.0)
    started = time.time()
    transitions: list[float] = []
    last_hand = None
    held = 0
    while time.time() - started < 30 and len(transitions) < 4:
        state = snap(client, session_id)
        if state.get("holding_result"):
            held += 1
        hand = state["hand_number"]
        if last_hand is not None and hand != last_hand:
            transitions.append(time.time())
        last_hand = hand
        if state["finished"]:
            break
        time.sleep(0.02)

    assert transitions, "the game never advanced a hand"
    assert held, "the result was never held"


def test_hold_can_be_switched_off():
    """`hand_result_seconds = 0` must deal straight on."""
    from pokerarena.providers import ProviderStore
    from pokerarena.personas import PersonaStore
    from pokerarena.web_state import SeatSpec, SessionConfig, GameSession
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    config = SessionConfig(
        seats=[
            SeatSpec(seat=0, kind="ai", persona="maniac"),
            SeatSpec(seat=1, kind="ai", persona="rock"),
        ],
        offline=True,
        speed_seconds=0.0,
        starting_chips=200,
        max_hands=3,
        seed=2,
        hand_result_seconds=0.0,
    )
    session = GameSession(config)
    session.start()
    deadline = time.time() + 30
    while not session.finished and time.time() < deadline:
        time.sleep(0.05)
    assert session.finished
    # With the hold disabled the flag is never raised.
    assert session.snapshot()["holding_result"] is False


def test_skip_hold_is_accepted(client):
    session_id = start(client, speed_seconds=0.2)
    response = client.post(
        f"/api/game/{session_id}/control", json={"action": "skip-hold"}
    )
    assert response.status_code == 200
    assert "error" not in response.get_json()


@pytest.mark.parametrize("seed", [2, 4, 7, 11, 13])
def test_every_hand_reports_the_pot_that_was_won(seed):
    """An uncontested hand is still played for chips, and must say how many.

    The pot is zeroed when it is paid out, so the size has to be captured at
    that moment -- otherwise the result card reads "won 0".

    Checked over the whole hand history rather than by polling, which misses
    hands.  Several seeds are used because a game can legitimately end in a
    single hand (one player stacking everyone), so no individual seed is
    guaranteed to produce an uncontested pot to compare against.
    """
    from pokerarena.web_state import GameSession
    from tests.helpers import session_config

    session = GameSession(
        session_config(
            ["maniac", "rock", "pro"],
            with_human=False,
            offline=True,
            speed_seconds=0.0,
            starting_chips=1000,
            max_hands=60,
            seed=seed,
            hand_result_seconds=0.0,
        )
    )
    session.start()
    deadline = time.time() + 60
    while not session.finished and time.time() < deadline:
        time.sleep(0.05)
    assert session.finished, "the game did not finish"

    history = session.arena.hand_history
    assert history, "no hands were played"
    for record in history:
        assert record["winners"], f"hand {record['hand_number']} has no winner"
        assert record["pot"] > 0, (
            f"hand {record['hand_number']} reports a pot of {record['pot']}"
        )


def test_an_uncontested_pot_keeps_its_size():
    """Directly pin the case that used to report zero: everyone folds."""
    from pokerarena.engine import Action, ActionType, Player, Table

    players = [Player(name=f"P{i}", seat=i, chips=1000) for i in range(3)]
    for p in players:
        p.seated = True
    table = Table(players, 10, 20, seed=1)
    table.start_hand()
    # Fold everyone around to the big blind.
    while not table.is_hand_over:
        seat = table.actor
        if seat is None:
            break
        table.apply_action(seat, Action(ActionType.FOLD))
    assert table.hand_number == 1
    assert table.pots_snapshot, "an uncontested hand must still record a pot"
    total = sum(pot["amount"] for pot in table.pots_snapshot)
    assert total == 30, f"expected the blinds (30) to be the whole pot, got {total}"
    assert table.pot == 0, "the pot must be paid out"


def test_the_hold_is_announced_so_the_ui_can_count_it_down(client):
    session_id = start(client, speed_seconds=0.0, hand_result_seconds=3.0)
    deadline = time.time() + 30
    while time.time() < deadline:
        state = snap(client, session_id)
        if state["holding_result"]:
            assert state["hold_total"] > 0, "the hold length must be announced"
            assert 0 <= state["hold_remaining"] <= state["hold_total"] + 0.1
            return
        if state["finished"]:
            break
        time.sleep(0.02)
    pytest.fail("the game never held a result")


def test_one_player_wins_the_whole_game(client):
    """A game must be able to finish; the outcome is exposed for the header."""
    session_id = start(client, max_hands=200, starting_chips=200, seed=7)
    deadline = time.time() + 60
    while time.time() < deadline:
        state = snap(client, session_id)
        if state["finished"]:
            assert state["winner"], "a finished game must name a winner"
            return
        time.sleep(0.05)
    pytest.fail("the game never finished")


def test_control_still_rejects_an_unknown_action(client):
    session_id = start(client)
    response = client.post(
        f"/api/game/{session_id}/control", json={"action": "teleport"}
    )
    assert response.status_code == 400
    assert "skip-hold" in response.get_json()["error"]
