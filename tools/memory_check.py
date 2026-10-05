"""Verify per-player context works against a real model.

Plays a few hands with real LLM seats and reports, per seat:
  * how its private context grows,
  * whether the model's replies reference earlier decisions,
  * how many input tokens each call costs (memory is not free).

Usage: python tools/memory_check.py --hands 4 --model deepseek-flash
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
from pokerarena.config import ArenaConfig, resolve_model  # noqa: E402
from pokerarena.personas import get_persona  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--lineup", default="maniac,rock")
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--memory", type=int, default=4)
    args = parser.parse_args()

    config = resolve_model(args.model)
    if not config.available:
        print(f"ERROR: {config.api_key_env} not set")
        return 2
    print(f"model={config.model}  memory_turns={args.memory}\n")

    keys = [k.strip() for k in args.lineup.split(",") if k.strip()]
    specs = [
        AISpec(name=get_persona(k).name, persona_key=k, model=args.model) for k in keys
    ]
    players, _ = build_players(specs, starting_chips=args.chips)
    transports = {
        p.seat: OpenAITransport(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key or "",
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        for p in players
        if p.is_ai
    }
    arena = Arena(
        players,
        config=ArenaConfig(
            starting_chips=args.chips,
            max_rounds=args.hands,
            seed=args.seed,
            llm_retries=1,
            memory_turns=args.memory,
        ),
        transports=transports,
    )

    started = time.time()
    for hand in range(args.hands):
        if arena.table.is_game_over():
            break
        print(f"=== hand {hand + 1} ===")
        arena.play_hand()
        arena.table.eliminate_broke_players()
        arena.table.rotate_button()

    elapsed = time.time() - started
    print(f"\nelapsed {elapsed:.0f}s\n")

    for player in arena.table.players:
        if not player.is_ai:
            continue
        print(f"--- {player.name} ({player.persona}) ---")
        print(f"  context messages kept: {len(player.conversation)}")
        for message in player.conversation:
            role = message["role"]
            body = " ".join(message["content"].split())[:150]
            print(f"    [{role:<9}] {body}")
        print()

    # Token cost of the memory window, per seat.
    for seat, decider in arena.deciders.items():
        usages = [t.usage for t in decider.traces if t.usage]
        if not usages:
            continue
        last = usages[-1]
        print(
            f"seat {seat} {arena.table.players[seat].name}: "
            f"last call prompt={last.get('prompt_tokens')} "
            f"completion={last.get('completion_tokens')} "
            f"decisions={len(decider.traces)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
