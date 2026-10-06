"""Chrome DevTools Protocol helper for verifying the web UI.

Launches headless Chrome with remote debugging, then evaluates JavaScript in the
page so tests and manual checks can assert on real rendered state (not just the
served HTML) and capture screenshots.

Usage:
    python tools/cdp.py eval "document.title"
    python tools/cdp.py errors            # console errors + page JS exceptions
    python tools/cdp.py shot out.png --wait 5
    python tools/cdp.py script --file check.js
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("chromium") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("no Chrome/Edge found")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Browser:
    def __init__(self, url: str, *, width: int = 1600, height: int = 940, wait: float = 3.0):
        self.port = free_port()
        self.profile = Path(tempfile.mkdtemp(prefix="cdp-"))
        self.url = url
        self.proc = subprocess.Popen(
            [
                find_chrome(),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "--mute-audio",
                f"--remote-debugging-port={self.port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={self.profile}",
                f"--window-size={width},{height}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._ws = None
        self._ws_module = None
        self.target = self._wait_for_target(timeout=25)
        self._connect()
        time.sleep(wait)

    # -- CDP plumbing ------------------------------------------------------
    def _wait_for_target(self, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/list", timeout=2
                ) as response:
                    targets = json.load(response)
                for target in targets:
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        return target
            except Exception:
                time.sleep(0.2)
        raise SystemExit("Chrome did not expose a page target")

    def _connect(self) -> None:
        try:
            from websocket import create_connection  # type: ignore
        except ImportError:
            raise SystemExit(
                "the 'websocket-client' package is required for the CDP helper:\n"
                "  pip install websocket-client"
            ) from None
        self._ws = create_connection(self.target["webSocketDebuggerUrl"], timeout=240)
        self._id = 0

    def send(self, method: str, **params):
        self._id += 1
        message_id = self._id
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            raw = json.loads(self._ws.recv())
            if raw.get("id") == message_id:
                if "error" in raw:
                    raise RuntimeError(f"{method}: {raw['error']}")
                return raw.get("result", {})

    # -- high level --------------------------------------------------------
    def eval(self, expression: str, *, await_promise: bool = False):
        result = self.send(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=await_promise,
        )
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            raise RuntimeError(detail.get("exception", {}).get("description") or detail.get("text", "JS error"))
        return result.get("result", {}).get("value")

    def screenshot(self, path: Path) -> None:
        data = self.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(data["data"]))

    def console_errors(self) -> list[str]:
        """Re-run the page and collect console errors / uncaught exceptions."""
        self.send("Runtime.enable")
        self.send("Log.enable")
        self.send("Page.enable")
        self.send("Page.reload", ignoreCache=True)
        errors: list[str] = []
        deadline = time.time() + 14
        self._ws.settimeout(1.0)
        while time.time() < deadline:
            try:
                raw = json.loads(self._ws.recv())
            except Exception:
                continue
            method = raw.get("method")
            if method == "Runtime.exceptionThrown":
                detail = raw["params"]["exceptionDetails"]
                errors.append(f"exception: {detail.get('text')} {detail.get('exception', {}).get('description', '')}".strip())
            elif method == "Log.entryAdded":
                entry = raw["params"]["entry"]
                if entry.get("level") in {"error", "warning"}:
                    errors.append(f"{entry['level']}: {entry.get('text')}")
            elif method == "Runtime.consoleAPICalled":
                if raw["params"].get("type") == "error":
                    parts = [str(a.get("value", a.get("description", ""))) for a in raw["params"].get("args", [])]
                    errors.append("console.error: " + " ".join(parts))
        return errors

    def close(self) -> None:
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


def main() -> int:
    # Windows consoles often default to a legacy code page (GBK/CP1252) that
    # cannot encode the suit glyphs the page contains.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["eval", "errors", "shot", "script"])
    parser.add_argument("arg", nargs="?", default="")
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--wait", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=940)
    parser.add_argument("--file", default="")
    parser.add_argument("--shot", default="", help="with 'script': also save a screenshot")
    args = parser.parse_args()

    browser = Browser(args.url, width=args.width, height=args.height, wait=args.wait)
    try:
        if args.command == "eval":
            print(json.dumps(browser.eval(args.arg), indent=2, ensure_ascii=False))
        elif args.command == "script":
            code = Path(args.file).read_text(encoding="utf-8")
            value = browser.eval(code, await_promise=True)
            print(json.dumps(value, indent=2, ensure_ascii=False))
            if args.shot:
                out = Path(args.shot).resolve()
                time.sleep(0.4)
                browser.screenshot(out)
                print(f"screenshot: {out} ({out.stat().st_size} bytes)")
        elif args.command == "errors":
            errors = browser.console_errors()
            print("\n".join(errors) if errors else "no console errors")
        elif args.command == "shot":
            out = Path(args.arg or "shot.png").resolve()
            browser.screenshot(out)
            print(f"{out} ({out.stat().st_size} bytes)")
    finally:
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
