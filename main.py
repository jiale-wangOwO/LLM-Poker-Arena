"""LLM Poker Arena -- project entry point.

This delegates to :mod:`pokerarena.__main__`.  Run ``python main.py --help`` for
the full option list, or use ``python -m pokerarena`` from anywhere.

Quick starts
------------
Offline demo (no API keys required)::

    python main.py --demo

Play against three LLM personas (set DEEPSEEK_API_KEY / OPENAI_API_KEY / ...)::

    python main.py --me --lineup maniac,rock,trickster --models deepseek

Web UI (browser table)::

    python -m pokerarena.web
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pokerarena.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
