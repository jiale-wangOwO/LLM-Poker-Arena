"""Give the landing page's images a content-hash query so updates show at once.

GitHub Pages serves assets with `Cache-Control: max-age=600`, so replacing an
image leaves browsers showing the old one for ten minutes -- which looks exactly
like the deploy failed. Appending a short hash of the file makes the URL change
whenever the bytes change, so a new image is a new URL and caches cannot lie.

Run after replacing any screenshot:  python tools/stamp_assets.py
"""

from __future__ import annotations

import hashlib
import pathlib
import re

DOCS = pathlib.Path("docs")
PAGE = DOCS / "index.html"
STAMP = re.compile(r"(?P<name>[\w.-]+\.(?:png|jpg|jpeg))(?:\?v=[0-9a-f]+)?")


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main() -> int:
    html = PAGE.read_text(encoding="utf-8")
    known = {p.name: p for p in DOCS.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    changed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        asset = known.get(name)
        if asset is None:
            return match.group(0)
        stamped = f"{name}?v={digest(asset)}"
        if stamped != match.group(0):
            changed.append(f"{name} -> {stamped}")
        return stamped

    updated = STAMP.sub(replace, html)
    if updated != html:
        PAGE.write_text(updated, encoding="utf-8")

    # Report what every reference now points at, and whether the file exists.
    print(f"assets in docs/: {', '.join(sorted(known))}")
    for match in re.finditer(r'(?:src|href)="([^"]+\.(?:png|jpg|jpeg)(?:\?v=[0-9a-f]+)?)"', updated):
        ref = match.group(1)
        name = ref.split("?")[0]
        ok = name in known
        print(f"  {'ok  ' if ok else 'MISS'} {ref}")

    if changed:
        print(f"\nstamped {len(changed)} reference(s):")
        for c in changed:
            print(f"  {c}")
    else:
        print("\nalready up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
