"""Scan the staged Git content for secrets and other things that must not ship.

Run before a public push.  Reads `git show :<path>` so it inspects exactly what
would be committed, not what happens to sit in the working tree.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Patterns that would indicate a real credential.  Ordered roughly by severity.
PATTERNS = [
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    ("Ark/Volcengine key", re.compile(r"\b(?:ARK|VOLC)[A-Z_]*=\s*[\"']?[A-Za-z0-9\-_]{20,}")),
    ("Google key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token literal", re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{30,}")),
    ("assigned api key", re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"'][^\"'\s]{20,}[\"']")),
]

# Files that legitimately contain key-shaped text (placeholders, docs, tests).
# Keep this list *narrow*: every entry is a string that this scanner would
# otherwise flag, so adding to it weakens the check.
ALLOW_SUBSTRINGS = (
    "sk-typed-in-ui",
    "sk-from-environment",
    "sk-from-the-environment",
    "sk-typed-secret-key-1234567890",
    "sk-ant-test-placeholder",
    "sk-must-not-appear",
    "sk-stub-key",
    "sk-persisted",
    "sk-relay-secret",
    "sk-added-later",
    "sk-edit-key",
    "sk-good",
    "sk-bad",
    "sk-will-be-rejected",
    "sk-override",
    "sk-stored",
    "sk-placeholder",
    "<your",
    "your-key",
    "REDACTED",
    "example",
)


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def staged_content(path: str) -> str:
    out = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8", "replace")


def main() -> int:
    files = staged_files()
    if not files:
        print("nothing staged")
        return 1

    findings: list[tuple[str, int, str, str]] = []
    for path in files:
        try:
            text = staged_content(path)
        except subprocess.CalledProcessError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(allowed in line for allowed in ALLOW_SUBSTRINGS):
                continue
            for label, pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append((path, lineno, label, match.group(0)[:60]))

    print(f"scanned {len(files)} staged files")
    if findings:
        print(f"\n{len(findings)} potential secret(s):")
        for path, lineno, label, hit in findings:
            print(f"  {path}:{lineno}  {label}: {hit}")
        return 1

    print("no secrets found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
