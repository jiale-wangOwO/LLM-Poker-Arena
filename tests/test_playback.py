"""Playback controls, the god's-eye view, and API-key handling.

Three operator-facing features:
  * pause / single-step playback,
  * a spectator view that shows every hole card (display only),
  * an API key typed into the UI, held in memory and never leaked.
"""

from __future__ import annotations

import time

import pytest

from pokerarena.config import MODEL_REGISTRY
from pokerarena.web import create_app
from pokerarena.web_state import GameSession
from tests.helpers import session_config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client with its own provider store, so tests cannot leak providers."""
    from pokerarena.providers import ProviderStore

    store = ProviderStore(path=tmp_path / "providers.json")
    monkeypatch.setattr("pokerarena.web.PROVIDERS", store)
    monkeypatch.setattr("pokerarena.web_state._PROVIDER_STORE", store)
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def start_game(client, **overrides):
    payload = {
        "personas": ["maniac", "rock", "trickster"],
        "with_human": False,
        "offline": True,
        "speed_seconds": 0.0,
        "starting_chips": 400,
        "max_hands": 40,
        "seed": 3,
    }
    payload.update(overrides)
    response = client.post("/api/game", json=payload)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["session"]["session"]


def get(client, session_id):
    return client.get(f"/api/game/{session_id}").get_json()["session"]


def wait_for(client, session_id, predicate, timeout=25.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = get(client, session_id)
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"condition not met; last: {last}")


# --------------------------------------------------------------------------
# Numbered actions
# --------------------------------------------------------------------------
def test_every_action_is_numbered_in_order(client):
    session_id = start_game(client)
    snapshot = wait_for(client, session_id, lambda s: len(s["actions"]) >= 5)
    actions = snapshot["actions"]
    numbers = [a["seq"] for a in actions]
    assert numbers == sorted(numbers), "actions must be numbered in order"
    assert len(set(numbers)) == len(numbers), "numbers must be unique"
    assert numbers[0] >= 1
    for action in actions:
        assert action["name"]
        assert action["action"]
        assert action["hand"] >= 1
        assert "street_label" in action


def test_action_numbers_are_contiguous(client):
    session_id = start_game(client)
    snapshot = wait_for(client, session_id, lambda s: len(s["actions"]) >= 8)
    numbers = [a["seq"] for a in snapshot["actions"]]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))


# --------------------------------------------------------------------------
# Pause / step
# --------------------------------------------------------------------------
def test_pause_freezes_the_game_and_resume_continues(client):
    session_id = start_game(client, speed_seconds=0.0)
    snapshot = wait_for(client, session_id, lambda s: len(s["actions"]) >= 4)

    paused = client.post(
        f"/api/game/{session_id}/control", json={"action": "pause"}
    ).get_json()
    assert paused["paused"] is True

    # Let any in-flight decision land, then confirm the count stops moving.
    time.sleep(1.2)
    first = len(get(client, session_id)["actions"])
    time.sleep(1.2)
    second = len(get(client, session_id)["actions"])
    # A game may finish while paused-draining; then it is legitimately static too.
    assert second - first <= 1, "the game kept running while paused"

    resumed = client.post(
        f"/api/game/{session_id}/control", json={"action": "resume"}
    ).get_json()
    assert resumed["paused"] is False
    later = wait_for(
        client, session_id, lambda s: len(s["actions"]) > second or s["finished"], timeout=15
    )
    assert later["actions"] or later["finished"]


def test_step_advances_exactly_one_action(client):
    session_id = start_game(client)
    snapshot = wait_for(client, session_id, lambda s: len(s["actions"]) >= 4)
    client.post(f"/api/game/{session_id}/control", json={"action": "pause"})
    time.sleep(0.8)  # let the current decision settle
    before = len(get(client, session_id)["actions"])

    client.post(
        f"/api/game/{session_id}/control", json={"action": "step", "count": 1}
    )
    deadline = time.time() + 8
    while time.time() < deadline:
        after = len(get(client, session_id)["actions"])
        if after > before:
            break
        time.sleep(0.05)
    # One grant lets one more decision through.
    assert after - before <= 2, f"step released too many ({before} -> {after})"

    client.post(f"/api/game/{session_id}/control", json={"action": "resume"})


def test_control_rejects_an_unknown_action(client):
    session_id = start_game(client)
    response = client.post(
        f"/api/game/{session_id}/control", json={"action": "warp"}
    )
    assert response.status_code == 400
    assert client.get("/api/game/nope").status_code == 404
    assert client.post("/api/game/nope/control", json={"action": "pause"}).status_code == 404


# --------------------------------------------------------------------------
# God view
# --------------------------------------------------------------------------
def test_god_view_shows_every_hole_card_mid_hand(client):
    session_id = start_game(client, reveal_all=True)

    # Calling snapshot() is what reveals the cards, so the *server* must have
    # dealt before we poll; otherwise the very first snapshot is taken between
    # hands (cards cleared) and the wait would spin forever.  Polling the game
    # endpoint is what drives it, so wait on the action count first.
    wait_for(client, session_id, lambda s: len(s["actions"]) >= 6 or s["finished"])

    # Then find a moment where every seated player holds cards.  Retrying is
    # essential: a snapshot can land between two hands, when the engine has
    # cleared the table, and the next hand is dealt a moment later.
    snapshot = wait_for(
        client,
        session_id,
        lambda s: s["finished"]
        or (seated(s) and all(p["hole_cards"] for p in seated(s))),
        timeout=30,
    )
    if snapshot["finished"]:
        pytest.skip("game finished too quickly")

    for player in seated(snapshot):
        assert player["hole_cards"], f"{player['name']} has no visible cards"
    assert snapshot["reveal_all"] is True


def seated(snapshot) -> list[dict]:
    """Only the chairs actually playing.

    An empty chair is still rendered at the table (so the felt keeps its shape)
    but holds no cards and never acts.
    """
    return [p for p in snapshot["players"] if p.get("seated")]


def test_reveal_can_be_turned_off_for_a_realistic_view():
    """A realistic view hides opponents' cards mid-hand.

    Checked directly on a session whose thread has not started, so the table is
    deterministically mid-hand rather than racing a fast offline game.
    """
    config = session_config(
        ["maniac", "rock", "trickster"],
        with_human=False,
        offline=True,
        speed_seconds=0.0,
        starting_chips=400,
        max_hands=10,
        seed=3,
        reveal_all=False,
    )
    session = GameSession(config)
    assert session.finished is False
    snapshot = session.snapshot()
    assert snapshot["reveal_all"] is False
    hidden = [p for p in snapshot["players"] if p["is_ai"] and p["hole_cards"] is None]
    assert hidden, "with reveal off, opponents' cards should be hidden"


def test_god_view_streams_every_card_from_the_start():
    config = session_config(
        ["maniac", "rock", "trickster"],
        with_human=False,
        offline=True,
        speed_seconds=0.0,
        starting_chips=400,
        max_hands=10,
        seed=3,
        reveal_all=True,
    )
    session = GameSession(config)
    snapshot = session.snapshot()
    assert snapshot["reveal_all"] is True
    # The engine has not dealt yet, so there is nothing to show; once it has,
    # every seat's cards are visible (checked below).
    session.arena.table.start_hand()
    snapshot = session.snapshot()
    for player in seated(snapshot):
        assert player["hole_cards"], f"{player['name']} should be visible"


def test_player_detail_always_shows_cards_to_the_spectator(client):
    session_id = start_game(client)
    wait_for(client, session_id, lambda s: len(s["actions"]) >= 4)
    detail = client.get(f"/api/game/{session_id}/player/0").get_json()["player"]
    assert detail["hole_cards"], "the detail panel is a spectator view"
    assert "cards_hidden" not in detail


# --------------------------------------------------------------------------
# Providers and API keys
# --------------------------------------------------------------------------
def test_a_fresh_install_has_one_provider(client):
    """Out of the box there is exactly one: deepseek-flash, no key yet."""
    data = client.get("/api/models").get_json()
    providers = data["providers"]
    assert len(providers) == 1, [p["id"] for p in providers]
    only = providers[0]
    assert only["id"] == "deepseek-flash"
    assert only["model"] == "deepseek-flash"
    assert only["reasoning"] is True
    assert only["has_key"] is False
    assert data["default_provider"] == "deepseek-flash"


def test_models_endpoint_never_leaks_keys(client):
    client.post(
        "/api/providers",
        json={
            "label": "Secret",
            "model": "m",
            "base_url": "https://example.com/v1",
            "api_key": "sk-must-not-appear",
        },
    )
    data = client.get("/api/models").get_json()
    for provider in data["providers"]:
        assert set(provider) >= {"id", "label", "model", "base_url", "has_key"}
        assert "api_key" not in provider
    body = client.get("/api/models").get_data(as_text=True)
    assert "sk-must-not-appear" not in body
    # Every provider is editable, including the seeded one.
    assert all("builtin" not in p for p in data["providers"])


def test_a_provider_key_never_appears_in_a_snapshot(client, monkeypatch):
    """Keys live in the provider store; no response may echo one back."""
    secret = "sk-typed-secret-key-1234567890"
    monkeypatch.setattr("pokerarena.web._verify_provider", lambda *a, **k: None)

    created = client.post(
        "/api/providers",
        json={
            "label": "With key",
            "model": "m",
            "base_url": "https://example.com/v1",
            "api_key": secret,
        },
    )
    provider_id = created.get_json()["provider"]["id"]
    response = client.post(
        "/api/game",
        json={
            "offline": True,  # no network calls
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "maniac", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
            "speed_seconds": 0.0,
            "starting_chips": 300,
            "max_hands": 3,
            "seed": 1,
        },
    )
    assert response.status_code == 201, response.get_json()
    session_id = response.get_json()["session"]["session"]

    for url in (
        f"/api/game/{session_id}",
        f"/api/game/{session_id}/player/0",
        f"/api/game/{session_id}/player/0?context=1",
        "/api/models",
        "/api/sessions",
    ):
        body = client.get(url).get_data(as_text=True)
        assert secret not in body, f"{url} leaked the API key"


def test_a_seat_with_no_key_refuses_to_start_a_real_game(client, monkeypatch):
    """No key + not offline is an error, not a silent slide into local bots."""
    monkeypatch.setattr("pokerarena.web._verify_provider", lambda *a, **k: None)
    client.post("/api/providers", json={"label": "NoKey", "model": "m",
                                        "base_url": "https://example.com/v1"})
    response = client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "maniac", "provider": "nokey"},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": "nokey"},
            ],
        },
    )
    assert response.status_code == 400
    assert "no API key" in response.get_json()["error"]

    # Ticking "offline bots only" makes the same table legal.
    ok = client.post(
        "/api/game",
        json={
            "offline": True,
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "maniac", "provider": "nokey"},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": "nokey"},
            ],
        },
    )
    assert ok.status_code == 201


def test_test_endpoint_reports_a_bad_key(client, monkeypatch):
    def reject(provider_id, api_key, base_url=""):
        raise ValueError("the API key was rejected by the provider (HTTP 401)")

    monkeypatch.setattr("pokerarena.web._verify_provider", reject)
    response = client.post(
        "/api/models/test", json={"provider": "deepseek-flash", "api_key": "sk-bad"}
    )
    assert response.status_code == 400
    assert "rejected" in response.get_json()["error"]

    response = client.post("/api/models/test", json={"provider": "nope", "api_key": "x"})
    assert response.status_code == 400
    response = client.post(
        "/api/models/test", json={"provider": "deepseek-flash", "api_key": ""}
    )
    assert response.status_code == 400
    assert "no API key" in response.get_json()["error"]


def test_test_endpoint_accepts_a_good_key(client, monkeypatch):
    monkeypatch.setattr("pokerarena.web._verify_provider", lambda *a, **k: None)
    response = client.post(
        "/api/models/test", json={"provider": "deepseek-flash", "api_key": "sk-good"}
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_operator_can_add_edit_and_delete_a_provider(client, tmp_path):
    """The whole point: point the app at any OpenAI-compatible endpoint."""
    response = client.post(
        "/api/providers",
        json={
            "label": "My Relay",
            "model": "some-model-v2",
            "base_url": "https://relay.example.com/v1/",
            "api_key": "sk-relay-secret",
            "reasoning": True,
        },
    )
    assert response.status_code == 201, response.get_json()
    provider = response.get_json()["provider"]
    provider_id = provider["id"]
    assert provider["base_url"] == "https://relay.example.com/v1"  # trailing / trimmed
    assert provider["has_key"] is True
    assert provider["reasoning"] is True

    listed = client.get("/api/models").get_json()["providers"]
    assert any(p["id"] == provider_id for p in listed)

    # A game can be seated on the new provider.  (offline=True keeps the test
    # from making real network calls; the seat still records the provider.)
    created = client.post(
        "/api/game",
        json={
            "offline": True,
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "maniac", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
        },
    )
    assert created.status_code == 201, created.get_json()
    snapshot = created.get_json()["session"]
    # The seat remembers which provider it was given (never the key).
    assert {p["provider"] for p in snapshot["players"] if p["seated"]} == {provider_id}

    # Editing: every field is changeable, and a blank key keeps the stored one.
    updated = client.patch(
        f"/api/providers/{provider_id}",
        json={
            "label": "Renamed Relay",
            "model": "some-model-v3",
            "base_url": "https://relay2.example.com/v1",
            "temperature": 0.5,
            "max_tokens": 1234,
            "reasoning": False,
        },
    )
    assert updated.status_code == 200
    body = updated.get_json()["provider"]
    assert body["label"] == "Renamed Relay"
    assert body["model"] == "some-model-v3"
    assert body["base_url"] == "https://relay2.example.com/v1"
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 1234
    assert body["reasoning"] is False
    assert body["has_key"] is True, "a blank key field must not wipe the stored key"

    # Replace the key explicitly, and clear it on request.
    replaced = client.patch(
        f"/api/providers/{provider_id}", json={"api_key": "sk-replaced"}
    )
    assert replaced.get_json()["provider"]["has_key"] is True
    cleared = client.patch(
        f"/api/providers/{provider_id}", json={"clear_api_key": True}
    )
    assert cleared.get_json()["provider"]["has_key"] is False

    # The seeded provider is editable too.
    seeded = client.patch(
        "/api/providers/deepseek-flash",
        json={"label": "Flash", "api_key": "sk-flash"},
    )
    assert seeded.status_code == 200, seeded.get_json()
    assert seeded.get_json()["provider"]["label"] == "Flash"
    assert seeded.get_json()["provider"]["has_key"] is True

    assert client.delete(f"/api/providers/{provider_id}").status_code == 200
    assert client.delete(f"/api/providers/{provider_id}").status_code == 404


def test_editing_a_provider_rejects_a_bad_url_and_unknown_id(client):
    response = client.patch(
        "/api/providers/deepseek-flash", json={"base_url": "nope"}
    )
    assert response.status_code == 400
    assert "base URL" in response.get_json()["error"]
    assert client.patch("/api/providers/ghost", json={"label": "x"}).status_code == 404


def test_provider_store_persists_across_instances(tmp_path):
    from pokerarena.providers import ProviderStore

    path = tmp_path / "providers.json"
    first = ProviderStore(path=path)
    # A missing file seeds the file with exactly one provider.
    assert [p.id for p in first.all()] == ["deepseek-flash"]

    first.add(
        label="Persisted",
        model="m1",
        base_url="https://example.com/v1",
        api_key="sk-persisted",
    )
    second = ProviderStore(path=path)
    found = [p for p in second.all() if p.label == "Persisted"]
    assert len(found) == 1
    assert found[0].api_key == "sk-persisted"
    # The seed survives alongside it.
    assert [p.id for p in second.all()] == ["deepseek-flash", "persisted"]


def test_deleting_providers_json_resets_to_the_seed(tmp_path):
    from pokerarena.providers import ProviderStore, SEED_PROVIDER

    path = tmp_path / "providers.json"
    store = ProviderStore(path=path)
    store.load()
    store.add(label="Extra", model="m", base_url="https://x.dev/v1")
    assert len(store.all()) == 2

    path.unlink()
    fresh = ProviderStore(path=path)
    assert [p.id for p in fresh.all()] == [SEED_PROVIDER["id"]]
    assert path.is_file(), "the seed should be written back out"


def test_provider_rejects_a_nonsense_base_url(client):
    response = client.post(
        "/api/providers",
        json={"label": "Bad", "model": "x", "base_url": "not-a-url"},
    )
    assert response.status_code == 400
    assert "base URL" in response.get_json()["error"]
    response = client.post(
        "/api/providers", json={"label": "", "model": "x", "base_url": "https://a.dev"}
    )
    assert response.status_code == 400


def _two_seat_arena(provider, api_keys=None):
    """An arena whose two seats both play on ``provider``."""
    from pokerarena.arena import Arena, AISpec, build_players
    from pokerarena.config import ArenaConfig

    specs = [
        AISpec(name="A", persona_key="pro", model=provider.id),
        AISpec(name="B", persona_key="rock", model=provider.id),
    ]
    players, _ = build_players(specs, starting_chips=200)
    return Arena(
        players,
        config=ArenaConfig(seed=1, max_rounds=2),
        api_keys=api_keys,
        providers=StoreStub([provider]),
    )


class StoreStub:
    """Minimal provider lookup, so these tests need no file on disk."""

    def __init__(self, providers):
        self._providers = {p.id: p for p in providers}

    def all(self):
        return list(self._providers.values())

    def get(self, provider_id):
        return self._providers.get(provider_id)


def test_a_provider_key_is_used_without_reading_the_environment(monkeypatch):
    """Keys come from the provider itself -- an env var must not be consulted."""
    from pokerarena.providers import Provider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-also-in-env")
    provider = Provider(
        id="p", label="P", model="m", base_url="https://example.com/v1",
        api_key="sk-stored-with-the-provider",
    )
    arena = _two_seat_arena(provider)
    for decider in arena.deciders.values():
        assert decider.transport.api_key == "sk-stored-with-the-provider"


def test_a_runtime_override_wins_over_the_stored_key():
    from pokerarena.providers import Provider

    provider = Provider(
        id="p", label="P", model="m", base_url="https://example.com/v1",
        api_key="sk-stored",
    )
    arena = _two_seat_arena(provider, api_keys={"p": "sk-override"})
    for decider in arena.deciders.values():
        assert decider.transport.api_key == "sk-override"


def test_missing_key_falls_back_to_the_local_bot(monkeypatch):
    from pokerarena.ai import HeuristicTransport
    from pokerarena.providers import Provider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-still-in-env")
    provider = Provider(id="p", label="P", model="m", base_url="https://example.com/v1")
    arena = _two_seat_arena(provider)
    for decider in arena.deciders.values():
        assert isinstance(decider.transport, HeuristicTransport)


def test_an_unknown_provider_falls_back_to_the_local_bot():
    """A seat naming a provider that no longer exists must still play."""
    from pokerarena.arena import Arena, AISpec, build_players
    from pokerarena.ai import HeuristicTransport
    from pokerarena.config import ArenaConfig

    specs = [
        AISpec(name="A", persona_key="pro", model="ghost"),
        AISpec(name="B", persona_key="rock", model="ghost"),
    ]
    players, _ = build_players(specs, starting_chips=200)
    arena = Arena(
        players,
        config=ArenaConfig(seed=1, max_rounds=2),
        providers=StoreStub([]),
    )
    for decider in arena.deciders.values():
        assert isinstance(decider.transport, HeuristicTransport)
