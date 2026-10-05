"""Web layer tests: API contract, snapshot hygiene, and human-action validation."""

from __future__ import annotations

import time

import pytest

from pokerarena.engine import IllegalAction
from pokerarena.web import create_app
from pokerarena.web_state import SessionManager, available_personas
from tests.helpers import session_config, seats_for


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client with its own provider store.

    Without this, a test that adds a provider would write it into the real
    ``providers.json`` and pollute the next run.
    """
    from pokerarena.providers import ProviderStore

    store = ProviderStore(path=tmp_path / "providers.json")
    monkeypatch.setattr("pokerarena.web.PROVIDERS", store)
    monkeypatch.setattr("pokerarena.web_state._PROVIDER_STORE", store)
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def offline_payload(**overrides):
    payload = {
        "personas": ["maniac", "rock"],
        "with_human": False,
        "offline": True,
        "speed_seconds": 0.0,
        "starting_chips": 300,
        "small_blind": 10,
        "big_blind": 20,
        "max_hands": 60,
        "seed": 5,
        # No pause between hands: these tests are about the game, not the
        # presentation, and a whole game would otherwise take 20s of waiting.
        "hand_result_seconds": 0,
    }
    payload.update(overrides)
    return payload


def wait_for(client, session_id, predicate, timeout=20.0):
    """Poll the snapshot until ``predicate`` holds (or time out)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = client.get(f"/api/game/{session_id}")
        assert response.status_code == 200
        last = response.get_json()["session"]
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"condition not met in time; last snapshot: {last}")


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------
def test_index_serves_the_table_page(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # The UI is English-only.
    assert "Poker" in body and "Arena" in body
    assert "/api/game" in body
    assert "New game" in body
    # No Chinese text should remain anywhere in the page.
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in body), (
        "the interface should be English only"
    )


def test_health_lists_personas_and_providers(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    keys = {p["key"] for p in data["personas"]}
    assert "maniac" in keys
    for persona in data["personas"]:
        # The seat editor needs all of these to render a persona.
        assert set(persona) >= {"key", "name", "tagline", "short", "style", "voice"}
        assert persona["short"], "every persona needs a compact label"
        assert len(persona["short"]) <= 22
    assert data["default_provider"]


def test_personas_endpoint_matches_health(client):
    from_health = client.get("/api/health").get_json()["personas"]
    from_endpoint = client.get("/api/personas").get_json()["personas"]
    assert [p["key"] for p in from_endpoint] == [p["key"] for p in from_health]


def test_latest_game_is_404_before_any_game_starts(client):
    assert client.get("/api/game").status_code == 404


def test_unknown_session_is_404(client):
    assert client.get("/api/game/doesnotexist").status_code == 404
    response = client.post("/api/game/doesnotexist/action", json={"action": "fold"})
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Creating games
# --------------------------------------------------------------------------
def test_create_game_returns_a_snapshot(client):
    response = client.post("/api/game", json=offline_payload())
    assert response.status_code == 201
    snapshot = response.get_json()["session"]
    assert snapshot["session"]
    # Every chair is rendered so the felt keeps its shape; unoccupied ones are
    # flagged `seated: false` and hold no cards.
    assert len(snapshot["players"]) == 6
    assert sum(1 for p in snapshot["players"] if p["seated"]) == 2
    assert [p["seat"] for p in snapshot["players"]] == list(range(6))
    assert snapshot["human_seat"] is None
    # The game thread starts immediately, so do not assume a street: just check
    # the board is a well-formed list of card codes.
    assert isinstance(snapshot["board"], list)
    assert len(snapshot["board"]) <= 5
    for code in snapshot["board"]:
        assert len(code) == 2
    # The snapshot must expose exactly the keys the UI reads.
    for key in (
        "hand_number",
        "street",
        "street_label",
        "pot",
        "board",
        "players",
        "events",
        "thoughts",
        "legal",
        "awaiting_human",
        "finished",
    ):
        assert key in snapshot, key


def test_snapshot_never_contains_api_keys(client):
    response = client.post("/api/game", json=offline_payload())
    body = response.get_data(as_text=True).lower()
    assert "api_key" not in body
    assert "sk-" not in body
    snapshot = client.get("/api/game").get_json()["session"]
    # Only the fields the UI needs are exposed.
    assert set(snapshot) >= {"players", "board", "pot", "events", "legal"}


def test_create_game_rejects_unknown_persona(client):
    response = client.post("/api/game", json=offline_payload(personas=["nope"]))
    assert response.status_code == 400
    assert "unknown persona" in response.get_json()["error"]


def test_create_game_rejects_unknown_model(client):
    response = client.post(
        "/api/game", json=offline_payload(models=["not-a-model"], offline=False)
    )
    assert response.status_code == 400
    assert "unknown provider" in response.get_json()["error"]


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"small_blind": 100, "big_blind": 20}, "small_blind"),
        ({"starting_chips": 0}, "starting_chips"),
        ({"max_hands": 0}, "max_hands"),
        ({"big_blind": "abc"}, "big_blind"),
        ({"personas": "maniac"}, "personas must be a list"),
    ],
)
def test_create_game_validates_input(client, override, needle):
    response = client.post("/api/game", json=offline_payload(**override))
    assert response.status_code == 400
    assert needle in response.get_json()["error"]


def test_too_many_seats_is_rejected(client):
    """A seventh player has nowhere to sit: the felt holds six."""
    response = client.post(
        "/api/game",
        json=offline_payload(personas=["maniac"] * 7, with_human=False),
    )
    assert response.status_code == 400
    assert "seat" in response.get_json()["error"].lower()


def test_six_handed_table_is_accepted(client):
    """Five AI seats plus the human is the layout maximum."""
    response = client.post(
        "/api/game",
        json=offline_payload(
            personas=["maniac", "rock", "trickster", "shark", "pro"],
            with_human=True,
        ),
    )
    assert response.status_code == 201
    snapshot = response.get_json()["session"]
    assert len(snapshot["players"]) == 6
    assert all(p["seated"] for p in snapshot["players"])


def test_spectator_game_needs_two_seats(client):
    response = client.post(
        "/api/game", json=offline_payload(personas=["maniac"], with_human=False)
    )
    assert response.status_code == 400
    assert "at least two players" in response.get_json()["error"]


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------
def test_offline_game_runs_to_completion_and_conserves_chips(client):
    response = client.post("/api/game", json=offline_payload(seed=11, max_hands=80))
    session_id = response.get_json()["session"]["session"]
    snapshot = wait_for(client, session_id, lambda s: s["finished"], timeout=60)
    assert snapshot["winner"]
    assert sum(p["chips"] for p in snapshot["players"]) == 600


def test_snapshot_exposes_legal_actions_only_on_the_human_turn(client):
    response = client.post(
        "/api/game",
        json=offline_payload(
            personas=["rock", "calculating"],
            with_human=True,
            seed=3,
            speed_seconds=0.0,
        ),
    )
    session_id = response.get_json()["session"]["session"]
    snapshot = wait_for(client, session_id, lambda s: s["awaiting_human"], timeout=20)
    assert snapshot["actor"] == snapshot["human_seat"] == 0
    assert snapshot["legal"] is not None
    assert snapshot["legal"]["can_fold"] is True


def test_human_action_must_be_legal(client):
    response = client.post(
        "/api/game",
        json=offline_payload(
            personas=["rock", "calculating"], with_human=True, seed=3
        ),
    )
    session_id = response.get_json()["session"]["session"]
    wait_for(client, session_id, lambda s: s["awaiting_human"], timeout=20)

    # Nonsense is refused.
    bad = client.post(f"/api/game/{session_id}/action", json={"action": "teleport"})
    assert bad.status_code == 409

    # So is a sub-minimum raise: the engine gate rejects it even though the
    # grammar parsed it fine.
    tiny = client.post(f"/api/game/{session_id}/action", json={"action": "raise 1"})
    assert tiny.status_code == 409
    assert "raise" in tiny.get_json()["error"].lower()

    # Checking while facing a bet is refused too.
    snapshot = client.get(f"/api/game/{session_id}").get_json()["session"]
    legal = snapshot["legal"]
    if not legal["can_check"]:
        wrong = client.post(f"/api/game/{session_id}/action", json={"action": "check"})
        assert wrong.status_code == 409

    # A legal action is accepted and the game moves on.
    action = "check" if legal["can_check"] else "call"
    good = client.post(f"/api/game/{session_id}/action", json={"action": action})
    assert good.status_code == 200
    assert good.get_json()["result"]["queued"] is True

    # The game must keep running: more events, no deadlock, no error.
    progressed = wait_for(
        client,
        session_id,
        lambda s: s["finished"] or len(s["events"]) > len(snapshot["events"]),
        timeout=30,
    )
    assert progressed["error"] is None


def test_action_outside_your_turn_is_refused():
    """The turn guard rejects a submission when the human is not on the clock.

    Driven directly rather than through the HTTP layer so it is deterministic:
    the background thread would otherwise race the assertion.
    """
    manager = SessionManager()
    session = manager.create(
        session_config(
            ["rock", "calculating"],
            with_human=True,
            offline=True,
            speed_seconds=0.0,
            starting_chips=300,
            small_blind=10,
            big_blind=20,
            max_hands=5,
            seed=3,
        )
    )
    with session.lock:
        session.awaiting_human = False

    with pytest.raises(IllegalAction) as excinfo:
        session.submit_action("fold")
    assert "not your turn" in str(excinfo.value)


def test_submit_action_refuses_when_no_human_seat():
    manager = SessionManager()
    session = manager.create(
        session_config(
            ["rock", "calculating"],
            with_human=False,
            offline=True,
            speed_seconds=0.0,
            starting_chips=300,
            max_hands=2,
            seed=3,
        )
    )
    with pytest.raises(IllegalAction) as excinfo:
        session.submit_action("fold")
    assert "no human seat" in str(excinfo.value)


# --------------------------------------------------------------------------
# Session registry
# --------------------------------------------------------------------------
def test_session_manager_caps_retained_sessions():
    manager = SessionManager(max_sessions=2)
    created = []
    for _ in range(4):
        session = manager.create(
            session_config(
                ["maniac", "rock"],
                with_human=False,
                offline=True,
                speed_seconds=0.0,
                starting_chips=200,
                max_hands=5,
            )
        )
        created.append(session)
    listed = manager.list()
    assert len(listed) == 2
    assert manager.get(created[0].id) is None
    assert manager.get(created[-1].id) is created[-1]


def test_persona_catalogue_is_serialisable():
    personas = available_personas()
    assert len(personas) >= 6
    for persona in personas:
        assert set(persona) >= {"key", "name", "tagline", "aggression", "tightness"}
        assert 0.0 <= persona["aggression"] <= 1.0


# --------------------------------------------------------------------------
# Table layout (the browser page's seat geometry)
# --------------------------------------------------------------------------
def _seat_positions() -> list[dict[str, float]]:
    """Parse SEAT_POS out of the template so the layout is testable."""
    import re
    from pathlib import Path

    template = Path("pokerarena/templates/table.html").read_text(encoding="utf-8")
    block = re.search(r"const SEAT_POS = \[(.*?)\];", template, re.DOTALL)
    assert block, "SEAT_POS not found in the template"
    return [
        {"left": float(left) / 100, "top": float(top) / 100}
        for left, top in re.findall(r'left:\s*"([\d.]+)%",\s*top:\s*"([\d.]+)%"', block.group(1))
    ]


def test_template_is_a_real_file_and_renders():
    response_status = create_app().test_client().get("/").status_code
    assert response_status == 200


def test_seat_layout_has_enough_slots():
    positions = _seat_positions()
    assert len(positions) >= 6, "the API allows 6 players per table"


def test_seat_boxes_never_overlap_each_other():
    """The felt fits exactly six 176x110 seat boxes; the API caps tables there.

    This is the invariant that actually matters: overlapping seats look broken,
    and it is why the web layer rejects a seventh player.
    """
    felt_w, felt_h = 965.0, 603.0
    seat_w, seat_h = 176.0, 110.0
    positions = _seat_positions()

    for count in range(2, 7):
        for a in range(count):
            for b in range(a + 1, count):
                ax, ay = positions[a]["left"] * felt_w, positions[a]["top"] * felt_h
                bx, by = positions[b]["left"] * felt_w, positions[b]["top"] * felt_h
                overlap_x = abs(ax - bx) < seat_w
                overlap_y = abs(ay - by) < seat_h
                assert not (overlap_x and overlap_y), (
                    f"seats {a} and {b} overlap in a {count}-handed table"
                )


def test_every_seat_box_fits_inside_the_felt():
    """Regression: with 3-4 players a seat used to be drawn off the felt rail."""
    felt_w, felt_h = 965.0, 603.0
    seat_w, seat_h = 176.0, 110.0
    positions = _seat_positions()

    for count in range(2, 7):
        for seat in range(count):
            pos = positions[seat]
            cx, cy = pos["left"] * felt_w, pos["top"] * felt_h
            assert cx - seat_w / 2 >= 60, f"seat {seat} ({count}-handed) clips left"
            assert cx + seat_w / 2 <= felt_w - 60, f"seat {seat} ({count}-handed) clips right"
            assert cy - seat_h / 2 >= 6, f"seat {seat} ({count}-handed) clips top"
            assert cy + seat_h / 2 <= felt_h - 6, f"seat {seat} ({count}-handed) clips bottom"


def test_seats_do_not_cover_the_board_and_pot():
    """The centre panel holds the community cards, street label and pot.

    Measured from a headless-Chrome screenshot at 1600x920: the felt renders at
    ~965x603 and the five-card board row is ~320px wide and ~78px tall.  The
    #centre element is wider than its content, so using its box here would
    report a false overlap.
    """
    felt_w, felt_h = 965.0, 603.0
    seat_w, seat_h = 176.0, 110.0
    centre_cx, centre_cy = 0.5 * felt_w, 0.45 * felt_h  # #centre top: 45%
    board_w, board_h = 320.0, 78.0
    pot_h = 60.0  # the "POT" label sits directly under the board

    for count in range(2, 7):
        positions = _seat_positions()
        for seat in range(count):
            pos = positions[seat]
            cx, cy = pos["left"] * felt_w, pos["top"] * felt_h
            # #centre is a vertical stack: board above, pot below.
            vertical_gap = abs(cy - centre_cy)
            in_band = vertical_gap < (board_h + pot_h + seat_h) / 2
            horizontal_gap = abs(cx - centre_cx)
            in_column = horizontal_gap < (board_w + seat_w) / 2
            assert not (in_band and in_column), (
                f"seat {seat} ({count}-handed) covers the board/pot area"
            )


def test_snapshot_exposes_model_labels(client):
    response = client.post("/api/game", json=offline_payload())
    snapshot = response.get_json()["session"]
    for player in snapshot["players"]:
        assert "model" in player
        # Empty chairs have no brain at all, so only seated AI seats get a label.
        if player["is_ai"] and player["seated"]:
            assert player["model"] == "local bot"
