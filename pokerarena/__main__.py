"""Command-line entry point for LLM Poker Arena.

Examples
--------
Offline demo with differentiated personalities (no API keys needed)::

    python -m pokerarena --demo

Play against three LLM personas (needs DEEPSEEK_API_KEY etc. in the env or a
``.env`` file)::

    python -m pokerarena --me --lineup maniac,rock,trickster

Check which models are actually configured::

    python -m pokerarena --models
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from .arena import Arena, AISpec, build_players, heuristic_transports
from .config import DEFAULT_MODEL, MODEL_REGISTRY, ArenaConfig
from .personas import PERSONAS, default_lineup, persona_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pokerarena",
        description="Texas Hold'em for LLM-driven AI players with distinct personalities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--me",
        action="store_true",
        help="sit at the table yourself (interactive).",
    )
    parser.add_argument(
        "--lineup",
        default=None,
        help=(
            "comma-separated persona keys for the AI seats. "
            f"Known: {', '.join(persona_keys())}"
        ),
    )
    parser.add_argument(
        "--ais",
        type=int,
        default=3,
        help="how many AI opponents when --lineup is not given (default 3).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "comma-separated model keys, assigned to AI seats in order. "
            f"Known: {', '.join(sorted(MODEL_REGISTRY))}"
        ),
    )
    parser.add_argument(
        "--chips", type=int, default=1000, help="starting stack (default 1000)."
    )
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument(
        "--hands",
        type=int,
        default=ArenaConfig.DEFAULT_MAX_HANDS,
        help=(
            "stop after this many hands (safety cap; default "
            f"{ArenaConfig.DEFAULT_MAX_HANDS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="fix the RNG.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="force offline heuristic brains for every AI (no network, no keys).",
    )
    parser.add_argument(
        "--models-report",
        action="store_true",
        dest="models_report",
        help="print the model configuration and exit.",
    )
    parser.add_argument(
        "--no-thoughts",
        action="store_true",
        help="hide the AI reasoning panel.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to pause between events (useful to watch a demo).",
    )
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="show every AI's hole cards (debugging / spectating).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="no live table; print a compact result summary instead.",
    )
    return parser


def parse_lineup(args, console: Console) -> list[str]:
    known = persona_keys()
    if args.lineup:
        keys = [key.strip() for key in args.lineup.split(",") if key.strip()]
        unknown = [key for key in keys if key not in PERSONAS]
        if unknown:
            console.print(
                f"[red]Unknown persona(s): {', '.join(unknown)}. "
                f"Known: {', '.join(known)}[/red]"
            )
            raise SystemExit(2)
        return keys
    return default_lineup(args.ais)


def parse_models(args, count: int) -> list[str]:
    if args.demo:
        return [""] * count
    if not args.models:
        # Prefer the configured default model when it has a key, then any other
        # configured model.  Without a key the arena degrades to local brains.
        if MODEL_REGISTRY.get(DEFAULT_MODEL) and MODEL_REGISTRY[DEFAULT_MODEL].available:
            preferred = DEFAULT_MODEL
        else:
            preferred = next(
                (key for key, cfg in MODEL_REGISTRY.items() if cfg.available), ""
            )
        return [preferred] * count
    keys = [key.strip() for key in args.models.split(",") if key.strip()]
    for key in keys:
        if key not in MODEL_REGISTRY:
            raise SystemExit(
                f"Unknown model '{key}'. Known: {', '.join(sorted(MODEL_REGISTRY))}"
            )
    # Cycle the requested models across the seats.
    return [keys[index % len(keys)] for index in range(count)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    if args.models_report:
        from .cli import print_llm_config

        print_llm_config(console)
        return 0

    if args.small_blind > args.big_blind:
        console.print("[red]--small-blind cannot exceed --big-blind[/red]")
        return 2

    persona_keys_list = parse_lineup(args, console)
    model_keys = parse_models(args, len(persona_keys_list))

    specs = []
    for index, persona_key in enumerate(persona_keys_list):
        persona = PERSONAS[persona_key]
        specs.append(
            AISpec(
                name=persona.name,
                persona_key=persona_key,
                model=model_keys[index],
            )
        )

    players, human_seat = build_players(
        specs, starting_chips=args.chips, with_human=args.me
    )
    config = ArenaConfig(
        starting_chips=args.chips,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        max_rounds=args.hands,
        seed=args.seed,
    )

    transports = None
    if args.demo or args.quiet:
        # `--quiet` also goes offline unless real keys were explicitly requested,
        # so batch runs never accidentally fire hundreds of API calls.
        if args.demo or not args.models:
            transports = heuristic_transports(players, args.seed)

    if args.quiet:
        return _run_quiet(players, config, transports, human_seat, args, console)

    from .cli import make_human_decider, run_cli

    arena = Arena(
        players,
        config=config,
        transports=transports,
        human_seat=human_seat,
    )
    if human_seat is not None:
        from .cli import TableView

        view = TableView(console)
        arena.human_decider = make_human_decider(view, console)

    run_cli(
        arena,
        console=console,
        show_thoughts=not args.no_thoughts,
        delay=args.delay,
        reveal_all=args.reveal,
    )
    return 0


def _run_quiet(players, config, transports, human_seat, args, console) -> int:
    """Headless run: no live table, just the outcome and the reasoning log."""
    if human_seat is not None:
        console.print("[red]--quiet cannot be combined with --me[/red]")
        return 2

    arena = Arena(players, config=config, transports=transports)
    winner = arena.play_game()
    console.rule(f"[bold green]Winner: {winner}[/bold green]")

    from rich.table import Table as RichTable

    table = RichTable(title="Results")
    table.add_column("Player")
    table.add_column("Persona")
    table.add_column("Chips", justify="right")
    for player in arena.table.players:
        table.add_row(player.name, player.persona, str(player.chips))
    console.print(table)

    if arena.transcript:
        sample = arena.transcript[:12]
        reasoning = RichTable(title="Sample reasoning")
        reasoning.add_column("Hand", justify="right")
        reasoning.add_column("Player")
        reasoning.add_column("Action")
        reasoning.add_column("Thought", overflow="fold")
        for entry in sample:
            reasoning.add_row(
                str(entry["hand"]),
                entry["name"],
                entry["action"],
                entry["thought"][:150],
            )
        console.print(reasoning)

    path = arena.save_history()
    if path:
        console.print(f"[dim]hand history written to {path}[/dim]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
