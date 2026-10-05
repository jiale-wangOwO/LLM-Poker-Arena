"""Scan the whole git history for secrets before making the repo public.

Differs from tools/scan_secrets.py, which only checks what is staged: publishing
means *every* blob that has ever been committed becomes readable, so a key that
was committed and later deleted is still exposed.
"""

from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [
    ("OpenAI/DeepSeek-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("Google key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{30,}")),
    ("assigned key", re.compile(r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*[\"'][^\"'\s]{24,}[\"']")),
    ("email with a real domain", re.compile(r"[\w\.\-]+@(?!example|test|localhost|your)[\w\-]+\.[a-z]{2,}")),
    ("absolute Windows user path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._\-]+")),
    ("home directory path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._\-]+/")),
]

# Strings that are obviously placeholders or documentation.
ALLOW = (
    "example", "placeholder", "your-", "yourkey", "your_key", "REDACTED",
    "sk-typed-in-ui", "sk-from-environment", "sk-from-the-environment",
    "sk-typed-secret-key-1234567890", "sk-ant-test-placeholder",
    "sk-must-not-appear", "sk-stub-key", "sk-persisted", "sk-relay-secret",
    "sk-added-later", "sk-edit-key", "sk-good", "sk-bad", "sk-will-be-rejected",
    "sk-override", "sk-stored", "sk-placeholder",
    # The scanner's own pattern definitions.
    "sk-[A-Za-z0-9", "gh[pousr]_", "AIza", "AKIA", "xox[baprs]",
    "BEGIN [A-Z ]*PRIVATE KEY",
    # Legitimate project contacts.
    "noreply@", "users.noreply.github.com",
)


def blobs() -> list[tuple[str, str]]:
    """Every (path, blob-hash) pair that has ever existed."""
    out = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        capture_output=True, text=True, check=True,
    ).stdout
    pairs = []
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            pairs.append((parts[1].strip(), parts[0]))
    return pairs


def main() -> int:
    seen: set[str] = set()
    findings: list[tuple[str, str, str]] = []
    scanned = 0

    for path, sha in blobs():
        if sha in seen:
            continue
        seen.add(sha)
        try:
            content = subprocess.run(
                ["git", "cat-file", "-p", sha],
                capture_output=True, check=True,
            ).stdout.decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            continue
        if "\x00" in content[:1000]:  # binary
            continue
        scanned += 1
        for lineno, line in enumerate(content.splitlines(), 1):
            if any(a in line for a in ALLOW):
                continue
            for label, pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append((path, f"{label}: {match.group(0)[:70]}", f"line {lineno}"))

    print(f"scanned {scanned} distinct blobs across the whole history")
    if findings:
        print(f"\n{len(findings)} finding(s):")
        for path, what, where in findings[:40]:
            print(f"  {path}  [{where}]  {what}")
        return 1
    print("no secrets or personal paths found in any committed blob")
    return 0


if __name__ == "__main__":
    sys.exit(main())
