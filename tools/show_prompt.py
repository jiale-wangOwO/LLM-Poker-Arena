"""Print the exact prompt a seat receives mid-hand, to eyeball the context.

    python tools/show_prompt.py [--hand 3] [--seed 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerarena.ai import recent_hand_summaries, describe_table  # noqa: E402
from pokerarena.arena import AISpec, Arena, build_players, heuristic_transports  # noqa: E402
from pokerarena.config import ArenaConfig  # noqa: E402
from pokerarena.personas import get_persona  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=int, default=3, help="which hand to dump")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--chips", type=int, default=1000)
    args = parser.parse_args()

    specs = [
        AISpec(name=get_persona(k).name, persona_key=k)
        for k in ("maniac", "rock", "trickster", "pro")
    ]
    players, _ = build_players(specs, starting_chips=args.chips)
    arena = Arena(
        players,
        config=ArenaConfig(seed=args.seed, max_rounds=args.hand),
        transports=heuristic_transports(players, args.seed),
    )
    for _ in range(args.hand - 1):
        arena.play_hand()

    # Dump the prompt for the seat that is about to act in a fresh hand.
    arena.table.start_hand()
    for decider in arena.deciders.values():
        decider.hand_history = arena.hand_history
        decider._log_cursor = 0
    # Play a couple of actions so the betting log has content.
    for _ in range(3):
        if arena.table.is_hand_over or arena.table.actor is None:
            break
        seat = arena.table.actor
        arena.table.apply_action(seat, arena.deciders[seat].decide(arena.table, seat))

    seat = arena.table.actor
    if seat is None:
        print("hand ended before anyone could act")
        return 0
    print(f"===== PROMPT FOR SEAT {seat} ({arena.table.players[seat].name}) =====")
    print(
        describe_table(
            arena.table,
            seat,
            recent_hands=recent_hand_summaries(arena.hand_history),
        )
    )
    print()
    print("===== SYSTEM PROMPT =====")
    print(arena.deciders[seat].system_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
