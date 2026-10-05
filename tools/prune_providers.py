"""Tidy providers.json: drop anything matching a label prefix.

Used to clear test residue like the 'My Relay' entries left by API smoke tests.
    python tools/prune_providers.py --label "My Relay" [--label Probe] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerarena.providers import PROVIDERS_FILE, ProviderStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", action="append", default=[],
                        help="remove providers whose label starts with this (repeatable)")
    parser.add_argument("--id", action="append", default=[],
                        help="remove providers with exactly this id (repeatable)")
    parser.add_argument("--list", action="store_true", help="only list what is stored")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    store = ProviderStore(PROVIDERS_FILE)
    custom = store.custom()

    if args.list or not (args.label or args.id):
        if not custom:
            print(f"{PROVIDERS_FILE.name}: no operator-defined providers")
        for provider in custom:
            print(f"  {provider.id:<28} {provider.label:<24} {provider.model:<28} "
                  f"key={'yes' if provider.api_key else 'no'}")
        return 0

    keep, drop = [], []
    for provider in custom:
        matches = provider.id in args.id or any(
            provider.label.startswith(prefix) for prefix in args.label
        )
        (drop if matches else keep).append(provider)

    for provider in drop:
        print(f"remove  {provider.id:<28} {provider.label}")
    if not drop:
        print("nothing matched")
        return 0
    if args.dry_run:
        print("(dry run: providers.json unchanged)")
        return 0

    store._custom = keep  # noqa: SLF001 - the store owns the list
    store.save()
    print(f"kept {len(keep)} provider(s); providers.json updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
