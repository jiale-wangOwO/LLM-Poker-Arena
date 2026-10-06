"""Token usage per provider, shown in Settings → Providers.

The counters are deliberately session-only: they are not written to
``providers.json``, because that file holds API keys and should not churn with
runtime data. Every test here resets them first so they do not leak into each
other.
"""

from __future__ import annotations

import pytest

from pokerarena.providers import ProviderStore, record_usage, reset_usage, usage_for
from pokerarena.web import create_app


@pytest.fixture(autouse=True)
def clean_counters():
    reset_usage()
    yield
    reset_usage()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store = ProviderStore(path=tmp_path / "providers.json")
    monkeypatch.setattr("pokerarena.web.PROVIDERS", store)
    monkeypatch.setattr("pokerarena.web_state._PROVIDER_STORE", store)
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


# --------------------------------------------------------------------------
# The counter
# --------------------------------------------------------------------------
def test_a_provider_starts_at_zero():
    totals = usage_for("deepseek-flash")
    assert totals == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_usage_accumulates_across_calls():
    record_usage("p", {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140})
    record_usage("p", {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260})
    totals = usage_for("p")
    assert totals["calls"] == 2
    assert totals["prompt_tokens"] == 300
    assert totals["completion_tokens"] == 100
    assert totals["total_tokens"] == 400


def test_providers_are_counted_separately():
    record_usage("a", {"total_tokens": 10})
    record_usage("b", {"total_tokens": 25})
    assert usage_for("a")["total_tokens"] == 10
    assert usage_for("b")["total_tokens"] == 25


def test_a_missing_or_partial_usage_report_is_ignored_gracefully():
    """Not every endpoint reports usage, and some report only some fields."""
    record_usage("p", None)
    record_usage("p", {})
    record_usage("", {"total_tokens": 999})          # no provider id
    record_usage("p", {"total_tokens": 50})          # only one field
    record_usage("p", {"prompt_tokens": None})       # present but null
    totals = usage_for("p")
    assert totals["calls"] == 2
    assert totals["total_tokens"] == 50
    assert totals["prompt_tokens"] == 0


def test_reset_clears_one_or_all():
    record_usage("a", {"total_tokens": 10})
    record_usage("b", {"total_tokens": 20})
    reset_usage("a")
    assert usage_for("a")["calls"] == 0
    assert usage_for("b")["calls"] == 1
    reset_usage()
    assert usage_for("b")["calls"] == 0


# --------------------------------------------------------------------------
# Over the API
# --------------------------------------------------------------------------
def test_the_provider_list_reports_usage(client):
    record_usage("deepseek-flash", {"prompt_tokens": 120, "completion_tokens": 30,
                                   "total_tokens": 150})
    payload = client.get("/api/models").get_json()
    entry = next(p for p in payload["providers"] if p["id"] == "deepseek-flash")
    assert entry["usage"]["calls"] == 1
    assert entry["usage"]["total_tokens"] == 150


def test_every_provider_entry_carries_usage_even_when_unused(client):
    payload = client.get("/api/models").get_json()
    assert payload["providers"], "the seed provider should be listed"
    for entry in payload["providers"]:
        assert "usage" in entry
        assert entry["usage"]["calls"] == 0


def test_usage_can_be_reset_through_the_api(client):
    record_usage("deepseek-flash", {"total_tokens": 500})
    assert client.post("/api/models/usage/reset", json={}).status_code == 200
    payload = client.get("/api/models").get_json()
    entry = next(p for p in payload["providers"] if p["id"] == "deepseek-flash")
    assert entry["usage"]["calls"] == 0


def test_usage_is_never_written_to_the_provider_file(client, tmp_path):
    """Counters are runtime data; providers.json holds keys and config."""
    path = tmp_path / "providers.json"
    # The store writes lazily on first use, so touch it through the API first.
    client.get("/api/models")
    assert path.is_file(), "the store should have written its file"
    before = path.read_text(encoding="utf-8")

    record_usage("deepseek-flash", {"total_tokens": 12345})
    client.get("/api/models")
    client.post("/api/models/usage/reset", json={})

    assert path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# Attribution: the seat's provider is the one that gets charged
# --------------------------------------------------------------------------
def test_the_transport_reports_usage_to_the_callback():
    from pokerarena.ai import OpenAITransport

    seen: list[dict] = []
    transport = OpenAITransport(
        model="m", base_url="http://x", api_key="k", on_usage=seen.append
    )
    assert transport.on_usage is not None
    # Directly exercise the callback contract the arena relies on.
    transport.on_usage({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert seen == [{"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}]


def test_a_seat_charges_the_provider_it_is_configured_with(tmp_path, monkeypatch):
    """The arena must attribute spend to the seat's own provider id."""
    from pokerarena.ai import OpenAITransport
    from pokerarena.arena import Arena, AISpec, build_players
    from pokerarena.config import ArenaConfig
    from pokerarena.personas import get_persona

    store = ProviderStore(path=tmp_path / "providers.json")
    store.update("deepseek-flash", api_key="sk-test-key-value")

    specs = [
        AISpec(name="A", persona_key="pro", model="deepseek-flash"),
        AISpec(name="B", persona_key="rock", model="deepseek-flash"),
    ]
    players, _ = build_players(specs, starting_chips=1000)
    arena = Arena(players, config=ArenaConfig(seed=1), providers=store)

    transport = arena._default_transport(players[0], get_persona(players[0].persona))
    assert isinstance(transport, OpenAITransport)
    assert transport.on_usage is not None

    # Stand in for the API response.
    transport.on_usage({"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1000})
    assert usage_for("deepseek-flash")["total_tokens"] == 1000


def test_a_keyless_seat_falls_back_and_charges_nothing(tmp_path):
    from pokerarena.ai import HeuristicTransport
    from pokerarena.arena import Arena, AISpec, build_players
    from pokerarena.config import ArenaConfig
    from pokerarena.personas import get_persona

    store = ProviderStore(path=tmp_path / "providers.json")
    store.update("deepseek-flash", api_key="")

    specs = [
        AISpec(name="A", persona_key="pro", model="deepseek-flash"),
        AISpec(name="B", persona_key="rock", model="deepseek-flash"),
    ]
    players, _ = build_players(specs, starting_chips=1000)
    arena = Arena(players, config=ArenaConfig(seed=1), providers=store)

    transport = arena._default_transport(players[0], get_persona(players[0].persona))
    assert isinstance(transport, HeuristicTransport)
    assert usage_for("deepseek-flash")["calls"] == 0
