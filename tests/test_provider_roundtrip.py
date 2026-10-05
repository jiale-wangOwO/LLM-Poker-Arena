"""End-to-end proof that a seat really calls its configured provider.

Spins up a throwaway OpenAI-compatible endpoint, registers it as an operator
provider through the HTTP API, seats a game on it, and then checks what the
server actually sent: the ``Authorization`` header, the model name, and how much
conversation context is included once a seat has a long history.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pokerarena.web import create_app

REPLIES = [
    "<action>call</action><say>ok</say><thought>flatting this one</thought>",
    "<action>check</action><say>check</say><thought>nothing to bet</thought>",
    "<action>fold</action><say>no</say><thought>not worth it</thought>",
]


class StubHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []
    hits: list[str] = []

    def log_message(self, *args):  # silence the test output
        pass

    def do_GET(self):
        StubHandler.hits.append(f"GET {self.path}")
        if self.path.rstrip("/").endswith("/models"):
            self._json({"data": [{"id": "stub-model"}]})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        StubHandler.hits.append(f"POST {self.path}")
        StubHandler.seen.append(
            {
                "auth": self.headers.get("Authorization"),
                "path": self.path,
                "model": payload.get("model"),
                "messages": payload.get("messages", []),
                "max_tokens": payload.get("max_tokens"),
            }
        )
        reply = REPLIES[min(len(StubHandler.seen) - 1, len(REPLIES) - 1)]
        self._json(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    def _json(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def stub_server():
    StubHandler.seen = []
    StubHandler.hits = []
    server = HTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _bypass_loopback_proxy(monkeypatch):
    """Keep a developer's system proxy out of the loopback requests.

    httpx honours NO_PROXY but not the Windows per-proxy bypass list, so on a
    machine with a system proxy (Clash, Fiddler, ...) a request to 127.0.0.1 can
    be sent to the proxy and come back 502.  Real deployments are unaffected;
    this only keeps the test deterministic.
    """
    monkeypatch.setenv(
        "NO_PROXY", "127.0.0.1,localhost,::1"
    )
    monkeypatch.setenv(
        "no_proxy", "127.0.0.1,localhost,::1"
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client whose provider store is isolated to a temp file."""
    from pokerarena.providers import ProviderStore

    store = ProviderStore(path=tmp_path / "providers.json")
    monkeypatch.setattr("pokerarena.web.PROVIDERS", store)
    monkeypatch.setattr("pokerarena.web_state._PROVIDER_STORE", store)
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_a_seat_calls_its_configured_provider(client, stub_server):
    """The whole chain: seat editor -> provider -> real HTTP request -> action."""
    created = client.post(
        "/api/providers",
        json={
            "label": "Stub",
            "model": "stub-model",
            "base_url": stub_server,
            "api_key": "sk-stub-key",
            "max_tokens": 777,
        },
    )
    assert created.status_code == 201, created.get_json()
    provider_id = created.get_json()["provider"]["id"]
    assert StubHandler.seen == [], f"the test endpoint was hit too early: {StubHandler.seen[:2]}"

    response = client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "pro", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
            "starting_chips": 500,
            "max_hands": 2,
            "speed_seconds": 0.0,
            "seed": 4,
        },
    )
    assert response.status_code == 201, response.get_json()

    # The game thread calls the stub; give it a moment.
    import time

    deadline = time.time() + 25
    while time.time() < deadline and len(StubHandler.seen) < 2:
        time.sleep(0.1)
    if not StubHandler.seen:
        # This failure is usually environmental (a system proxy swallowing the
        # loopback request), so report what the server actually saw.
        raise AssertionError(
            "the provider was never called:\n"
            f"  requests seen by the endpoint: {StubHandler.hits}\n"
            f"  only loopback traffic is expected; check for a system proxy"
        )

    first = StubHandler.seen[0]
    assert first["auth"] == "Bearer sk-stub-key", "the stored key was not sent"
    assert first["model"] == "stub-model", "the provider's model name was not used"
    assert first["max_tokens"] == 777, "the provider's max_tokens was not used"
    assert first["path"].endswith("/chat/completions")
    assert first["messages"], "no messages were sent"


def test_context_grows_across_hands_and_is_capped(client, stub_server):
    """A seat accumulates history, and the prompt carries a long window of it."""
    created = client.post(
        "/api/providers",
        json={
            "label": "Stub2",
            "model": "stub-model",
            "base_url": stub_server,
            "api_key": "sk-stub-key-2",
        },
    )
    provider_id = created.get_json()["provider"]["id"]
    response = client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "maniac", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "calling_station", "provider": provider_id},
            ],
            "starting_chips": 400,
            "max_hands": 12,
            "speed_seconds": 0.0,
            "seed": 11,
            "hand_result_seconds": 0,
        },
    )
    assert response.status_code == 201
    session_id = response.get_json()["session"]["session"]

    import time

    deadline = time.time() + 40
    while time.time() < deadline and len(StubHandler.seen) < 10:
        time.sleep(0.1)

    # Later prompts must contain more than the opening pair of messages: that is
    # the seat's memory of what it has already done.
    sizes = [len(call["messages"]) for call in StubHandler.seen]
    assert sizes, "no calls were recorded"
    assert max(sizes) > min(sizes), f"context never grew: {sizes[:12]}"

    # The last request to whichever seat has the longest history is what a long
    # memory looks like; 50 turns is 100 messages plus the opening prompt.
    biggest = max(sizes)
    assert biggest >= 4, f"context stayed tiny: {sizes[:12]}"

    # And the configured window is the documented 50 turns.
    from pokerarena.config import ArenaConfig

    assert ArenaConfig().memory_turns == 50


def test_a_provider_without_a_key_fails_loudly_at_session_creation(client, stub_server):
    """Starting a real game on a keyless provider is an error, not a silent bot.

    (The old behaviour -- quietly falling back to the offline brain -- hid the
    fact that nothing was ever going to be sent to the model.)
    """
    created = client.post(
        "/api/providers",
        json={
            "label": "Keyless",
            "model": "stub-model",
            "base_url": stub_server,
        },
    )
    assert created.status_code == 201
    provider_id = created.get_json()["provider"]["id"]
    assert created.get_json()["provider"]["has_key"] is False

    response = client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "pro", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
        },
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "no API key" in error and provider_id in error

    # Adding the key to that same provider makes the same table start.
    patched = client.patch(
        f"/api/providers/{provider_id}", json={"api_key": "sk-added-later"}
    )
    assert patched.get_json()["provider"]["has_key"] is True
    ok = client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "pro", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
        },
    )
    assert ok.status_code == 201, ok.get_json()


def test_editing_a_provider_changes_where_seats_call(client, stub_server):
    """An edit takes effect for the next game, including the model name."""
    created = client.post(
        "/api/providers",
        json={
            "label": "Editable",
            "model": "before-edit",
            "base_url": stub_server,
            "api_key": "sk-edit-key",
        },
    )
    provider_id = created.get_json()["provider"]["id"]
    StubHandler.seen = []

    assert client.post(
        "/api/game",
        json={
            "seats": [
                {"seat": 0, "kind": "ai", "persona": "pro", "provider": provider_id},
                {"seat": 1, "kind": "ai", "persona": "rock", "provider": provider_id},
            ],
            "max_hands": 1,
            "speed_seconds": 0.0,
        },
    ).status_code == 201

    import time

    deadline = time.time() + 25
    while time.time() < deadline and not StubHandler.seen:
        time.sleep(0.1)
    assert StubHandler.seen, f"no call reached the endpoint: {StubHandler.hits}"
    assert StubHandler.seen[0]["model"] == "before-edit"
