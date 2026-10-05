"""Smoke-test real LLM seats: play a few hands and print what the model said.

Usage:
    python tools/llm_smoketest.py --hands 3                    # default provider
    python tools/llm_smoketest.py --provider deepseek-flash
    python tools/llm_smoketest.py --model deepseek             # env-var registry

``--provider`` reads ``providers.json`` (the same list the web UI edits, key
included).  ``--model`` keeps the older environment-variable path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerarena.ai import OpenAITransport  # noqa: E402
from pokerarena.arena import Arena, AISpec, build_players  # noqa: E402
from pokerarena.config import ArenaConfig  # noqa: E402
from pokerarena.engine import Event, Table  # noqa: E402
from pokerarena.personas import get_persona  # noqa: E402
from pokerarena.providers import STORE as PROVIDERS  # noqa: E402


def resolve_transport_target(args) -> tuple[str, str, str, int, float] | None:
    """``(model, base_url, api_key, max_tokens, timeout)`` or None."""
    if args.model:
        from pokerarena.config import resolve_model

        config = resolve_model(args.model)
        if not config.available:
            print(f"ERROR: {config.api_key_env} is not set; cannot test real LLM seats.")
            return None
        return (
            config.model,
            config.base_url,
            config.api_key or "",
            config.max_tokens,
            config.timeout,
        )

    provider = PROVIDERS.get(args.provider) if args.provider else PROVIDERS.default()
    if provider is None:
        print("ERROR: no providers configured; add one in the web UI Settings.")
        print("       (or pass --model <registry key> with an environment key)")
        return None
    if not provider.has_key:
        print(f"ERROR: provider '{provider.id}' has no API key stored.")
        print("       Add it in Settings → Providers, or use --model.")
        return None
    return (
        provider.model,
        provider.base_url,
        provider.api_key,
        provider.max_tokens,
        provider.timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument(
        "--provider", default="", help="provider id from providers.json (default: the first with a key)"
    )
    parser.add_argument("--model", default="", help="legacy: a config registry key + env key")
    parser.add_argument("--lineup", default="maniac,rock,trickster")
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    target = resolve_transport_target(args)
    if target is None:
        return 2
    model_name, base_url, api_key, max_tokens, timeout = target
    print(f"model={model_name}  base_url={base_url}")

    persona_keys = [k.strip() for k in args.lineup.split(",") if k.strip()]
    specs = [
        AISpec(name=get_persona(key).name, persona_key=key, model=args.provider or args.model)
        for key in persona_keys
    ]
    players, _ = build_players(specs, starting_chips=args.chips)
    arena_config = ArenaConfig(
        starting_chips=args.chips,
        small_blind=10,
        big_blind=20,
        max_rounds=args.hands,
        seed=args.seed,
        llm_retries=1,
    )

    # Force the real transport for every seat, even if keys look missing.
    transports = {
        player.seat: OpenAITransport(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        for player in players
        if player.is_ai
    }

    errors: list[str] = []
    calls = {"n": 0}

    def on_event(table: Table, events: list[Event]) -> None:
        for event in events:
            if event.kind == "action_taken":
                data = event.data
                calls["n"] += 1
                tag = "LLM " if data.get("source") == "llm" else data.get("source", "?")
                print(
                    f"  [{tag:>8}] {data['name']:<10} {data['action']:<7} "
                    f"{data['amount']:>5}  {data.get('speech') or ''}"
                )
                thought = (data.get("thought") or "").strip()
                if thought:
                    print(f"             thought: {thought[:300]}")
            elif event.kind == "showdown":
                for entry in event.data["players"]:
                    print(f"  showdown: {entry['name']} {entry['hole_cards']} -> {entry['hand_name']}")
            elif event.kind == "pot_awarded":
                print(f"  -> {event.data['name']} wins {event.data['amount']}")

    arena = Arena(players, config=arena_config, transports=transports, on_event=on_event)

    started = time.time()
    for hand in range(args.hands):
        if arena.table.is_game_over():
            break
        print(f"\n=== hand {hand + 1} ===")
        try:
            arena.play_hand()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            print(f"  !! hand failed: {exc}")
            break
        arena.table.eliminate_broke_players()
        arena.table.rotate_button()

    elapsed = time.time() - started
    print("\n" + "=" * 60)
    print(f"actions resolved: {calls['n']}   elapsed: {elapsed:.1f}s")
    for decider in arena.deciders.values():
        for trace in decider.traces:
            if trace.fell_back:
                print(f"  FALLBACK: {trace.name} ({trace.model}) -> {trace.error}")
    print(f"errors: {len(errors)}")
    for error in errors:
        print("  ", error)
    print("stacks:", {p.name: p.chips for p in arena.table.players})
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
