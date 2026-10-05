"""Flask app serving the poker table to a browser.

Run with::

    python -m pokerarena.web                 # http://127.0.0.1:5000

The server binds to localhost by default.  It never serialises API keys, and
every human action is validated by the engine before it can change the game.

The page itself lives in ``pokerarena/templates/table.html`` so it can be edited
without touching Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .config import ArenaConfig
from .engine import IllegalAction
from .personas import STORE as PERSONA_STORE
from .providers import STORE as PROVIDERS
from .providers import default_provider_id
from .web_state import (
    MAX_SEATS,
    SeatSpec,
    SessionConfig,
    SessionManager,
    available_personas,
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _provider_has_key(provider_id: str) -> bool:
    provider = PROVIDERS.get(provider_id)
    return bool(provider and provider.has_key)


def _verify_provider(provider_id: str, api_key: str, base_url: str = "") -> None:
    """Fail fast on a key the endpoint rejects, instead of silently going offline.

    Uses the models listing endpoint, which costs no tokens.  A network error is
    *not* fatal (the key may still be fine); an HTTP 401/403 is.
    """
    import urllib.error
    import urllib.request

    provider = PROVIDERS.get(provider_id)
    if provider is None:
        return
    url = (base_url.strip().rstrip("/") or provider.base_url).rstrip("/") + "/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=12):
            return
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError(
                "the API key was rejected by the provider "
                f"(HTTP {exc.code}); check the key and the base URL"
            ) from None
        # Other HTTP errors (e.g. a relay with no /models route) are not proof
        # the key is bad, so let the game try anyway.
    except Exception:
        # Timeout / DNS / offline: not proof of a bad key.
        pass


def _verify_api_key(model_key: str, api_key: str, base_url: str) -> None:
    """Backwards-compatible shim used by older call sites and tests."""
    _verify_provider(model_key, api_key, base_url)


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    app.config["JSON_SORT_KEYS"] = False
    manager = SessionManager()
    app.extensions["poker_manager"] = manager

    # -- pages -------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template(
            "table.html",
            personas=available_personas(),
            default_max_hands=ArenaConfig.DEFAULT_MAX_HANDS,
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            personas=available_personas(),
            default_provider=default_provider_id(),
        )

    @app.get("/api/personas")
    def list_personas():
        """The persona catalogue the seat editor renders."""
        return jsonify(personas=available_personas())

    @app.post("/api/personas")
    def add_persona():
        """Create a custom persona, optionally starting from a template."""
        payload = request.get_json(silent=True) or {}
        fields = {k: v for k, v in payload.items() if k not in {"name", "template", "clone"}}
        try:
            if payload.get("clone"):
                persona = PERSONA_STORE.clone(
                    str(payload["clone"]), str(payload.get("name", ""))
                )
            else:
                persona = PERSONA_STORE.add(
                    name=str(payload.get("name", "")),
                    template=str(payload.get("template", "")),
                    **fields,
                )
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(persona=persona.public()), 201

    @app.patch("/api/personas/<persona_key>")
    def update_persona(persona_key: str):
        payload = request.get_json(silent=True) or {}
        try:
            persona = PERSONA_STORE.update(
                persona_key,
                **{k: v for k, v in payload.items() if k != "key"},
            )
        except KeyError:
            return jsonify(error="unknown persona"), 404
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(persona=persona.public())

    @app.post("/api/personas/<persona_key>/clone")
    def clone_persona(persona_key: str):
        payload = request.get_json(silent=True) or {}
        try:
            persona = PERSONA_STORE.clone(persona_key, str(payload.get("name", "")))
        except KeyError:
            return jsonify(error="unknown persona"), 404
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(persona=persona.public()), 201

    @app.delete("/api/personas/<persona_key>")
    def delete_persona(persona_key: str):
        try:
            removed = PERSONA_STORE.remove(persona_key)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not removed:
            return jsonify(error="unknown persona"), 404
        return jsonify(removed=persona_key)

    @app.get("/api/models")
    def list_models():
        """Providers available to the seat editor.

        Keys are never returned; ``has_key`` only reports whether one exists.
        """
        store = PROVIDERS
        return jsonify(
            default_provider=default_provider_id(),
            max_seats=MAX_SEATS,
            providers=[p.public() for p in store.all()],
        )

    @app.post("/api/providers")
    def add_provider():
        payload = request.get_json(silent=True) or {}
        try:
            provider = PROVIDERS.add(
                label=str(payload.get("label", "")),
                model=str(payload.get("model", "")),
                base_url=str(payload.get("base_url", "")),
                api_key=str(payload.get("api_key", "")),
                temperature=payload.get("temperature", 0.95),
                max_tokens=payload.get("max_tokens", 2500),
                reasoning=bool(payload.get("reasoning", False)),
            )
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(provider=provider.public()), 201

    @app.patch("/api/providers/<provider_id>")
    def update_provider(provider_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            provider = PROVIDERS.update(provider_id, **payload)
        except KeyError:
            return jsonify(error="unknown provider"), 404
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(provider=provider.public())

    @app.delete("/api/providers/<provider_id>")
    def delete_provider(provider_id: str):
        if not PROVIDERS.remove(provider_id):
            return jsonify(error="unknown or built-in provider"), 404
        return jsonify(removed=provider_id)

    @app.post("/api/models/test")
    def test_model():
        """Pre-flight check: is this key accepted by the provider?"""
        payload = request.get_json(silent=True) or {}
        provider_id = str(payload.get("provider") or payload.get("model") or "").strip()
        api_key = str(payload.get("api_key", "")).strip()
        base_url = str(payload.get("base_url", "")).strip()
        provider = PROVIDERS.get(provider_id)
        if provider is None:
            return jsonify(ok=False, error="unknown provider"), 400
        api_key = api_key or provider.api_key.strip()
        if not api_key:
            return jsonify(ok=False, error="no API key supplied"), 400
        try:
            _verify_provider(provider_id, api_key, base_url)
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, model=provider.model)

    @app.get("/api/sessions")
    def list_sessions():
        return jsonify(sessions=manager.list())

    # -- game control ------------------------------------------------------
    @app.post("/api/game")
    def new_game():
        payload = request.get_json(silent=True) or {}
        try:
            config = _config_from_payload(payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        session = manager.create(config)
        return jsonify(session=session.snapshot()), 201

    @app.get("/api/game/<session_id>")
    def get_game(session_id: str):
        session = manager.get(session_id)
        if session is None:
            return jsonify(error="unknown session"), 404
        return jsonify(session=session.snapshot())

    @app.get("/api/game")
    def get_latest_game():
        session = manager.latest()
        if session is None:
            return jsonify(error="no game in progress"), 404
        return jsonify(session=session.snapshot())

    @app.post("/api/game/<session_id>/action")
    def post_action(session_id: str):
        session = manager.get(session_id)
        if session is None:
            return jsonify(error="unknown session"), 404
        payload = request.get_json(silent=True) or {}
        raw = payload.get("action", "")
        try:
            result = session.submit_action(raw)
        except IllegalAction as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(result=result)

    @app.get("/api/game/<session_id>/player/<int:seat>")
    def get_player(session_id: str, seat: int):
        """Per-seat detail: persona, stats, decision history, private context."""
        session = manager.get(session_id)
        if session is None:
            return jsonify(error="unknown session"), 404
        include_prompt = request.args.get("context") in {"1", "true", "yes"}
        try:
            detail = session.player_detail(seat, include_prompt=include_prompt)
        except KeyError:
            return jsonify(error=f"no seat {seat}"), 404
        return jsonify(player=detail)

    @app.post("/api/game/<session_id>/control")
    def control(session_id: str):
        """Pause, resume, single-step, or toggle god mode on a running game."""
        session = manager.get(session_id)
        if session is None:
            return jsonify(error="unknown session"), 404
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action", "")).strip().lower()
        god_mode = session.god_mode()
        if action == "pause":
            paused = session.set_paused(True)
        elif action == "resume":
            paused = session.set_paused(False)
        elif action == "step":
            session.step(int(payload.get("count", 1) or 1))
            paused = True
        elif action == "god-mode":
            god_mode = session.set_god_mode(bool(payload.get("enabled", True)))
            paused = session.paused
        elif action == "skip-hold":
            # "Deal the next hand now" -- for a spectator who has seen enough.
            with session.lock:
                session.skip_hold = True
            paused = session.paused
        else:
            return jsonify(
                error="action must be pause, resume, step, god-mode or skip-hold"
            ), 400
        return jsonify(
            paused=paused, finished=session.finished, god_mode=god_mode
        )

    return app


def _seats_from_payload(payload: dict) -> list:
    """Normalise the seat list.

    The seat editor sends ``seats``.  A ``personas`` list (the earlier shape, and
    what scripts and tests tend to use) is translated into seats for
    convenience: the human, if any, takes seat 0 and the AIs follow.
    """
    raw = payload.get("seats")
    if isinstance(raw, list):
        return raw

    personas = payload.get("personas")
    if personas is None:
        raise ValueError("seats is required: seat the table in the UI first")
    if not isinstance(personas, list):
        raise ValueError("personas must be a list of persona keys")
    personas = [str(p).strip() for p in personas if str(p).strip()]
    unknown = [p for p in personas if not PERSONA_STORE.has(p)]
    if unknown:
        raise ValueError(f"unknown persona(s): {', '.join(unknown)}")

    providers = payload.get("models") or []
    if not isinstance(providers, list):
        raise ValueError("models must be a list of provider ids")

    seats: list[dict] = []
    index = 0
    if bool(payload.get("with_human", True)):
        seats.append({"seat": index, "kind": "human"})
        index += 1
    for offset, key in enumerate(personas):
        seats.append(
            {
                "seat": index,
                "kind": "ai",
                "persona": key,
                "provider": str(providers[offset % len(providers)])
                if providers
                else "",
            }
        )
        index += 1
    return seats


def _config_from_payload(payload: dict) -> SessionConfig:
    """Build a session config from the seat editor's payload."""
    raw_seats = _seats_from_payload(payload)
    if not isinstance(raw_seats, list):
        raise ValueError("seats must be a list")

    seats: list[SeatSpec] = []
    seen: set[int] = set()
    for entry in raw_seats:
        if not isinstance(entry, dict):
            raise ValueError("each seat must be an object")
        try:
            index = int(entry.get("seat", -1))
        except (TypeError, ValueError):
            raise ValueError("seat index must be an integer") from None
        if not 0 <= index < MAX_SEATS:
            raise ValueError(f"seat index must be 0..{MAX_SEATS - 1}")
        if index in seen:
            raise ValueError(f"seat {index} listed twice")
        seen.add(index)

        kind = str(entry.get("kind", "empty")).strip().lower()
        if kind not in {"empty", "human", "ai"}:
            raise ValueError(f"seat {index}: kind must be empty, human or ai")
        persona = str(entry.get("persona", "")).strip()
        provider = str(entry.get("provider", "")).strip()
        name = str(entry.get("name", "")).strip()[:24]
        if kind == "ai":
            if not PERSONA_STORE.has(persona):
                raise ValueError(f"seat {index}: unknown persona {persona!r}")
            if provider and PROVIDERS.get(provider) is None:
                raise ValueError(f"seat {index}: unknown provider {provider!r}")
        seats.append(
            SeatSpec(
                seat=index,
                kind=kind,
                persona=persona if kind == "ai" else "",
                provider=provider if kind == "ai" else "",
                name=name,
            )
        )

    occupied = [s for s in seats if not s.is_empty]
    if len(occupied) < 2:
        raise ValueError("seat at least two players")
    if sum(1 for s in occupied if s.kind == "human") > 1:
        raise ValueError("only one human seat is supported")

    def number(key: str, default: int, low: int, high: int) -> int:
        value = payload.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    seed = payload.get("seed")
    if seed in ("", None):
        seed = None
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer") from None

    speed = payload.get("speed_seconds", 0.0)
    try:
        speed = max(0.0, min(5.0, float(speed)))
    except (TypeError, ValueError):
        raise ValueError("speed_seconds must be a number") from None

    hold = payload.get("hand_result_seconds", None)
    if hold is not None:
        try:
            hold = max(0.0, min(30.0, float(hold)))
        except (TypeError, ValueError):
            raise ValueError("hand_result_seconds must be a number") from None

    offline = bool(payload.get("offline", False))

    # Fail loudly when a seated provider has no key, rather than starting a game
    # that is quietly played by local bots.
    if not offline:
        missing = sorted(
            {
                s.provider
                for s in occupied
                if s.kind == "ai" and s.provider and not _provider_has_key(s.provider)
            }
        )
        if missing:
            raise ValueError(
                "no API key for "
                + ", ".join(missing)
                + ": add one in Settings, or switch on 'Offline bots only'"
            )

    config = SessionConfig(
        seats=seats,
        starting_chips=number("starting_chips", 1000, 100, 1_000_000),
        small_blind=number("small_blind", 10, 1, 100_000),
        big_blind=number("big_blind", 20, 1, 200_000),
        max_hands=number("max_hands", ArenaConfig.DEFAULT_MAX_HANDS, 1, 5000),
        seed=seed,
        speed_seconds=speed,
        offline=offline,
        reveal_all=bool(payload.get("reveal_all", True)),
        **({"hand_result_seconds": hold} if hold is not None else {}),
    )
    if config.small_blind > config.big_blind:
        raise ValueError("small_blind cannot exceed big_blind")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerarena.web")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    app = create_app()
    providers = PROVIDERS.all()
    print(f"LLM Poker Arena web UI -> http://{args.host}:{args.port}")
    if providers:
        for provider in providers:
            state = "key set" if provider.has_key else "no key (will use offline bots)"
            print(f"  {provider.label}: {provider.model} @ {provider.base_url} [{state}]")
    else:
        print("  no providers configured; add one in Settings")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
