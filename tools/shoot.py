"""Screenshot helper: start a game, render the page in headless Chrome, save PNG.

Usage: python tools/shoot.py [--players 4] [--wait 6] [--out _shot.png]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--wait", type=float, default=6.0)
    parser.add_argument("--out", default="_shot.png")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=920)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url", default="/")
    args = parser.parse_args()

    lineup = ["maniac", "rock", "trickster", "shark", "calculating", "pro"][
        : args.players
    ]
    session = post(
        f"{args.base}/api/game",
        {
            "personas": lineup,
            "with_human": True,
            "offline": True,
            "speed_seconds": args.speed,
            "starting_chips": 1000,
            "small_blind": 10,
            "big_blind": 20,
            "max_hands": 60,
            "seed": args.seed,
        },
    )["session"]
    print(f"session {session['session']}  players={len(session['players'])}")

    # Let a few actions accumulate so the log and reasoning panels are populated.
    time.sleep(min(args.wait, 5))

    out = ROOT / args.out
    out.unlink(missing_ok=True)
    # Chrome caches aggressively between runs; use a throwaway profile and an
    # explicit cache dir so every screenshot reflects the current template.
    profile = ROOT / ".chrome_shot_profile"
    profile.mkdir(exist_ok=True)
    url = f"{args.base}{args.url}"
    if "?" in url:
        url += f"&_={int(time.time())}"
    else:
        url += f"?_={int(time.time())}"
    result = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-application-cache",
            "--hide-scrollbars",
            f"--user-data-dir={profile}",
            f"--disk-cache-dir={profile / 'cache'}",
            f"--window-size={args.width},{args.height}",
            f"--screenshot={out}",
            f"--virtual-time-budget={int(args.wait * 1000)}",
            url,
        ],
        check=False,
        capture_output=True,
    )
    if out.exists():
        print(f"screenshot -> {out}  ({out.stat().st_size} bytes)")
        return 0
    print("FAILED: no screenshot produced")
    print(result.stderr.decode(errors="replace")[-800:])
    return 1


if __name__ == "__main__":
    sys.exit(main())
