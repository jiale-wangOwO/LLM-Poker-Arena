"""Generate an equity-based preflop hand ordering and embed it in strength.py.

Run offline (not at import time): computes each of the 169 starting hands'
all-in equity against a uniformly random opponent hand by Monte Carlo, sorts by
equity, and writes the result into ``pokerarena/strength.py`` between the
markers.  That gives a defensible, self-contained percentile table with no
runtime cost.

Usage:  python tools/gen_preflop_ranking.py [--samples 4000]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerarena.cards import RANKS, Card, evaluate  # noqa: E402

RANK_CHARS = RANKS

RANK_INDEX = {c: i for i, c in enumerate(RANK_CHARS)}
SUITS = "cdhs"

BEGIN = "# --- BEGIN GENERATED PREFLOP ORDER ---"
END = "# --- END GENERATED PREFLOP ORDER ---"


def all_starting_hands() -> list[tuple[str, str, bool]]:
    out: list[tuple[str, str, bool]] = []
    for i, a in enumerate(RANK_CHARS):
        for b in RANK_CHARS[i:]:
            high, low = (a, b) if RANK_INDEX[a] >= RANK_INDEX[b] else (b, a)
            if high == low:
                out.append((high, low, False))
            else:
                out.append((high, low, True))
                out.append((high, low, False))
    return out


def make_cards(high: str, low: str, suited: bool) -> list[Card]:
    if high == low:
        # Pick two different suits so the pair is legal.
        return [Card(RANK_INDEX[high], 0), Card(RANK_INDEX[low], 1)]
    if suited:
        return [Card(RANK_INDEX[high], 0), Card(RANK_INDEX[low], 0)]
    return [Card(RANK_INDEX[high], 0), Card(RANK_INDEX[low], 1)]


def equity(hand: list[Card], samples: int, rng: random.Random) -> float:
    """All-in equity of ``hand`` against one uniformly random hand."""
    deck = [Card(r, s) for r in range(13) for s in range(4)]
    blocked = {(c.rank, c.suit) for c in hand}
    available = [c for c in deck if (c.rank, c.suit) not in blocked]

    wins = ties = 0
    for _ in range(samples):
        draw = rng.sample(available, 2 + 5)
        opp = draw[:2]
        board = draw[2:]
        mine = evaluate(hand + board)
        theirs = evaluate(opp + board)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            ties += 1
    return (wins + ties / 2) / samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    hands = all_starting_hands()
    assert len(hands) == 169, len(hands)

    scored = []
    for index, (high, low, suited) in enumerate(hands):
        cards = make_cards(high, low, suited)
        eq = equity(cards, args.samples, rng)
        scored.append((eq, high, low, suited))
        print(
            f"\r{index + 1:>3}/169  {high}{low}{'s' if suited else 'o' if high != low else ''}"
            f"  {eq:.4f}",
            end="",
            flush=True,
        )
    print()

    # Strongest first.
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    tokens = []
    for _, high, low, suited in scored:
        if high == low:
            tokens.append(f"{high}{low}")
        else:
            tokens.append(f"{high}{low}{'s' if suited else 'o'}")
    body = ",\n    ".join(
        ", ".join(f'"{t}"' for t in tokens[i : i + 8])
        for i in range(0, len(tokens), 8)
    )

    table = (
        f"{BEGIN}\n"
        "#: Every distinct starting hand, strongest first, ranked by Monte-Carlo\n"
        "#: all-in equity against a uniformly random opponent hand\n"
        f"#: ({args.samples} samples/hand, seed {args.seed}).\n"
        "#: Regenerate with: python tools/gen_preflop_ranking.py\n"
        "EQUITY_ORDER: tuple[str, ...] = (\n"
        f"    {body},\n"
        ")\n"
        f"{END}"
    )

    target = ROOT / "pokerarena" / "strength.py"
    source = target.read_text(encoding="utf-8")
    if BEGIN in source and END in source:
        source = re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END), table, source, flags=re.DOTALL
        )
    else:
        source = source.rstrip() + "\n\n\n" + table + "\n"
    target.write_text(source, encoding="utf-8")

    print(f"\nWrote ranking into {target}")
    print("Strongest 10:", tokens[:10])
    print("Weakest 10:", tokens[-10:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
