"""God mode and operator-defined personas.

God mode is a spectator aid that must never influence play:

* it is a **live switch**, changeable while a game runs;
* it gates **both** hole cards and reasoning, so it is one coherent idea;
* with it off the view must be what a player at the table would see.

Personas are the character layer.  The seven built-ins are templates; the
operator can clone or write new ones, and a custom persona has to work exactly
like a built-in one, including driving the offline bot.
"""

from __future__ import annotations

import time

import pytest

from pokerarena.personas import PersonaStore
from pokerarena.providers import ProviderStore
from pokerarena.web import create_app
from tests.helpers import session_config

PERSONAS = ["maniac", "rock", "trickster"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated provider and persona stores, so tests never touch the real files."""
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


def start(client, *, reveal_all=False, hands=400, speed=0.0, seed=7):
    response = client.post(
        "/api/game",
        json={
            "personas": PERSONAS,
            "with_human": False,
            "offline": True,
            "speed_seconds": speed,
            "starting_chips": 1000,
            "max_hands": hands,
            "seed": seed,
            "reveal_all": reveal_all,
        },
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["session"]["session"]


def snapshot(client, session_id):
    return client.get(f"/api/game/{session_id}").get_json()["session"]


def wait_for_actions(client, session_id, count=6, timeout=25.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = snapshot(client, session_id)
        if len(last["actions"]) >= count or last["finished"]:
            return last
        time.sleep(0.05)
    raise AssertionError(f"only {len(last['actions']) if last else 0} actions in {timeout}s")


def set_god(client, session_id, enabled):
    response = client.post(
        f"/api/game/{session_id}/control",
        json={"action": "god-mode", "enabled": enabled},
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def seated_ai(snap):
    return [p for p in snap["players"] if p["seated"] and p["is_ai"]]


# --------------------------------------------------------------------------
# God mode
# --------------------------------------------------------------------------
def test_god_mode_off_hides_cards_and_reasoning(client):
    session_id = start(client, reveal_all=False, speed=0.2)
    snap = wait_for_actions(client, session_id)
    if snap["finished"]:
        pytest.skip("game finished before it could be inspected")

    assert snap["god_mode"] is False
    assert snap["reveal_all"] is False
    for player in seated_ai(snap):
        assert player["hole_cards"] is None, f"{player['name']} leaked hole cards"
    assert snap["thoughts"] == [], "reasoning leaked into the thought feed"
    assert all(a["thought"] == "" for a in snap["actions"]), "reasoning leaked into the rail"


def test_god_mode_on_shows_cards_and_reasoning(client):
    session_id = start(client, reveal_all=True, speed=0.2)
    snap = wait_for_actions(client, session_id)
    if snap["finished"]:
        pytest.skip("game finished before it could be inspected")

    assert snap["god_mode"] is True
    for player in seated_ai(snap):
        assert player["hole_cards"], f"{player['name']} should be visible"
    assert snap["thoughts"], "no reasoning captured"
    assert any(a["thought"] for a in snap["actions"]), "no reasoning on the rail"


def test_a_pure_spectator_sees_nothing_with_god_mode_off(client):
    """With no human seat there is no hand that is "yours".

    Treating every non-AI seat as the viewer's own is how an all-AI table ended
    up broadcasting every hole card in the clear while god mode was off.
    """
    session_id = start(client, reveal_all=False, speed=0.2)
    snap = wait_for_actions(client, session_id)
    if snap["finished"]:
        pytest.skip("game finished before it could be inspected")

    assert snap["human_seat"] is None, "this table should have no human seat"
    assert snap["god_mode"] is False
    for player in seated_ai(snap):
        assert player["hole_cards"] is None, (
            f"{player['name']}'s cards leaked to a spectator: {player['hole_cards']}"
        )


def test_a_spectator_sees_nothing_through_the_player_panel_either(client):
    """The detail endpoint is not a back door."""
    session_id = start(client, reveal_all=False, speed=0.2)
    wait_for_actions(client, session_id)
    for seat in range(3):
        detail = client.get(f"/api/game/{session_id}/player/{seat}").get_json()["player"]
        if not detail["seated"]:
            continue
        assert detail["hole_cards"] is None, (
            f"seat {seat} leaked cards through player_detail"
        )
        assert detail["reasoning_visible"] is False


def test_a_seated_human_still_sees_their_own_hand(client):
    """Turning god mode off must not blind the player to their own cards."""
    from pokerarena.web_state import GameSession
    from tests.helpers import session_config

    session = GameSession(
        session_config(
            ["rock", "maniac"],
            with_human=True,
            offline=True,
            speed_seconds=0.0,
            starting_chips=400,
            max_hands=200,
            seed=3,
            reveal_all=False,
        )
    )
    session.arena.table.start_hand()
    snap = session.snapshot()
    human = next(p for p in snap["players"] if p["seat"] == snap["human_seat"])
    assert human["seated"]
    assert human["hole_cards"], "a player must always see their own hand"
    for player in snap["players"]:
        if player["seated"] and player["seat"] != snap["human_seat"]:
            assert player["hole_cards"] is None


def test_god_mode_reveals_a_spectator_table_too(client):
    """The spectator can still turn god mode on and watch everything."""
    session_id = start(client, reveal_all=True, speed=0.2)
    snap = wait_for_actions(client, session_id)
    if snap["finished"]:
        pytest.skip("game finished before it could be inspected")
    assert snap["human_seat"] is None
    for player in seated_ai(snap):
        assert player["hole_cards"], f"{player['name']} should be visible"


def test_god_mode_can_be_toggled_while_the_game_runs(client):
    """The whole point: it is a switch, not a setting fixed at kickoff."""
    session_id = start(client, reveal_all=False, speed=0.2)
    snap = wait_for_actions(client, session_id)
    if snap["finished"]:
        pytest.skip("game finished before it could be inspected")
    assert snap["god_mode"] is False

    assert set_god(client, session_id, True)["god_mode"] is True
    on = wait_for_actions(client, session_id, len(snap["actions"]) + 3)
    assert on["god_mode"] is True
    for player in seated_ai(on):
        assert player["hole_cards"]
    assert on["thoughts"]

    assert set_god(client, session_id, False)["god_mode"] is False
    off = wait_for_actions(client, session_id, len(on["actions"]) + 3)
    assert off["god_mode"] is False
    for player in seated_ai(off):
        assert player["hole_cards"] is None
    assert off["thoughts"] == []


def test_player_detail_respects_god_mode(client):
    """The floating card must not be a back door to hidden information."""
    session_id = start(client, reveal_all=False, speed=0.2)
    wait_for_actions(client, session_id)

    detail = client.get(f"/api/game/{session_id}/player/1?context=1").get_json()["player"]
    assert detail["hole_cards"] is None
    assert detail["reasoning_visible"] is False
    assert detail["context"] == [], "the private thread leaked"
    assert all(h["thought"] == "" for h in detail["history"])

    set_god(client, session_id, True)
    detail = client.get(f"/api/game/{session_id}/player/1?context=1").get_json()["player"]
    assert detail["hole_cards"], "god mode should reveal the cards"
    assert detail["reasoning_visible"] is True
    assert detail["context"], "god mode should reveal the private thread"


def test_own_cards_are_always_visible_to_the_human(client):
    """God mode is about *other* players; you always see your own hand."""
    config = session_config(
        ["rock", "maniac"],
        with_human=True,
        offline=True,
        speed_seconds=0.0,
        starting_chips=400,
        max_hands=200,
        seed=3,
        reveal_all=False,
    )
    from pokerarena.web_state import GameSession

    session = GameSession(config)
    session.arena.table.start_hand()
    snap = session.snapshot()
    human = next(p for p in snap["players"] if p["seat"] == snap["human_seat"])
    assert human["seated"]
    # The human's own cards are theirs to see regardless of god mode.
    assert human["hole_cards"], "a player must always see their own hand"
    for player in snap["players"]:
        if player["seated"] and player["is_ai"]:
            assert player["hole_cards"] is None


def test_god_mode_does_not_change_what_the_models_are_told(client):
    """The firewall: display settings must never reach a prompt."""
    session_id = start(client, reveal_all=True, speed=0.0)
    wait_for_actions(client, session_id, 8)
    detail = client.get(f"/api/game/{session_id}/player/1?context=1").get_json()["player"]
    thread = " ".join(entry["content"] for entry in detail["context"])

    # The seat's own cards appear in its prompt; nobody else's may.
    others = [
        p["hole_cards"]
        for p in snapshot(client, session_id)["players"]
        if p["seat"] != 1 and p["seated"] and p["hole_cards"]
    ]
    for cards in others:
        for code in cards:
            # A shared card is possible in principle (same rank+suit cannot be
            # dealt twice), so a hit means another seat's card was leaked.
            assert code not in thread, f"another seat's card {code} leaked into the prompt"


# --------------------------------------------------------------------------
# Personas: custom definitions
# --------------------------------------------------------------------------
def test_catalogue_starts_with_the_seven_templates(client):
    personas = client.get("/api/personas").get_json()["personas"]
    assert len(personas) == 7
    assert {p["key"] for p in personas} == {
        "calculating", "maniac", "rock", "trickster",
        "calling_station", "pro", "shark",
    }
    assert all(p["custom"] is False for p in personas)


def test_create_a_custom_persona_from_scratch(client):
    response = client.post(
        "/api/personas",
        json={
            "name": "Vera",
            "tagline": "patient trapper",
            "short": "trapper",
            "style": "You trap with monsters and let opponents bluff into you.",
            "voice": "Quiet and dry.",
            "aggression": 0.2,
            "tightness": 0.8,
            "bluff_frequency": 0.05,
            "temperature": 0.7,
        },
    )
    assert response.status_code == 201, response.get_json()
    persona = response.get_json()["persona"]
    assert persona["key"] == "vera"
    assert persona["custom"] is True
    assert persona["tightness"] == 0.8
    assert persona["short"] == "trapper"

    catalogue = client.get("/api/personas").get_json()["personas"]
    assert len(catalogue) == 8
    assert any(p["name"] == "Vera" and p["custom"] for p in catalogue)


def test_clone_a_template_and_edit_it_independently(client):
    cloned = client.post(
        "/api/personas/shark/clone", json={"name": "Max II"}
    )
    assert cloned.status_code == 201, cloned.get_json()
    clone = cloned.get_json()["persona"]
    assert clone["template"] == "shark"
    assert clone["aggression"] == 0.80, "a clone starts from the template's numbers"

    edited = client.patch(
        f"/api/personas/{clone['key']}",
        json={"aggression": 0.1, "style": "You play like a rock.", "name": "Max III"},
    )
    assert edited.status_code == 200
    body = edited.get_json()["persona"]
    assert body["aggression"] == 0.1
    assert body["style"] == "You play like a rock."
    assert body["name"] == "Max III"

    # The template is untouched.
    shark = next(
        p for p in client.get("/api/personas").get_json()["personas"] if p["key"] == "shark"
    )
    assert shark["aggression"] == 0.80
    assert "rock" not in shark["style"]


def test_built_in_templates_are_read_only(client):
    response = client.patch("/api/personas/pro", json={"name": "Hacked"})
    assert response.status_code == 400
    assert "built-in" in response.get_json()["error"]
    assert client.delete("/api/personas/pro").status_code == 400
    # Duplicating is the supported route.
    assert client.post("/api/personas/pro/clone", json={}).status_code == 201


def test_persona_delete_and_unknown_ids(client):
    created = client.post("/api/personas", json={"name": "Temp", "style": "x"})
    key = created.get_json()["persona"]["key"]
    assert client.delete(f"/api/personas/{key}").status_code == 200
    assert client.delete(f"/api/personas/{key}").status_code == 404
    assert client.patch("/api/personas/ghost", json={"name": "x"}).status_code == 404
    assert client.post("/api/personas/ghost/clone", json={}).status_code == 404


def test_persona_fields_are_clamped(client):
    created = client.post(
        "/api/personas",
        json={"name": "Extreme", "style": "x", "aggression": 5, "tightness": -3,
              "temperature": 9},
    )
    persona = created.get_json()["persona"]
    assert persona["aggression"] == 1.0
    assert persona["tightness"] == 0.0
    assert persona["temperature"] == 2.0


def test_a_custom_persona_can_be_seated_and_plays(client):
    """A custom persona must be a first-class citizen: seatable and functional."""
    created = client.post(
        "/api/personas",
        json={
            "name": "Vera",
            "tagline": "patient trapper",
            "style": "You trap with monsters.",
            "voice": "Quiet.",
            "aggression": 0.3,
            "tightness": 0.7,
            "bluff_frequency": 0.1,
        },
    )
    key = created.get_json()["persona"]["key"]

    response = client.post(
        "/api/game",
        json={
            "offline": True,
            "speed_seconds": 0.0,
            "starting_chips": 500,
            "max_hands": 6,
            "seed": 5,
            "seats": [
                {"seat": 0, "kind": "ai", "persona": key},
                {"seat": 1, "kind": "ai", "persona": "rock"},
            ],
        },
    )
    assert response.status_code == 201, response.get_json()
    session = response.get_json()["session"]
    names = {p["name"] for p in session["players"] if p["seated"]}
    assert "Vera" in names, f"the custom persona's name was not used: {names}"

    # And it actually plays.
    snap = wait_for_actions(client, session["session"], 4)
    assert len(snap["actions"]) >= 4


def test_the_store_persists_custom_personas(tmp_path):
    path = tmp_path / "personas.json"
    first = PersonaStore(path=path)
    first.add(name="Persisted", style="You persist.", tightness=0.4)
    second = PersonaStore(path=path)
    found = [p for p in second.all() if p.name == "Persisted"]
    assert len(found) == 1
    assert found[0].custom is True
    assert found[0].tightness == 0.4
    # The templates are always there too.
    assert len([p for p in second.all() if not p.custom]) == 7


def test_a_missing_personas_file_is_fine(tmp_path):
    store = PersonaStore(path=tmp_path / "nope.json")
    assert len(store.all()) == 7
    assert store.get("rock").name == "Tom"


def test_duplicate_names_get_distinct_keys(client):
    a = client.post("/api/personas", json={"name": "Twin", "style": "x"})
    b = client.post("/api/personas", json={"name": "Twin", "style": "y"})
    assert a.status_code == 201 and b.status_code == 201
    assert a.get_json()["persona"]["key"] != b.get_json()["persona"]["key"]
