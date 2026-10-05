"""Runtime configuration for LLM Poker Arena.

All configuration is loaded from environment variables, optionally seeded from a
``.env`` file at the project root.  Nothing here reaches out to the network, so
importing this module is always safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal ``.env`` loader (no third-party dependency).

    Existing environment variables always win, so an explicitly exported key is
    never clobbered by a stale file.
    """
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """An OpenAI-compatible chat-completions endpoint."""

    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.9
    max_tokens: int = 2500
    """Generous by default: reasoning models (e.g. deepseek-flash) spend part of
    this budget on hidden chain-of-thought before emitting any content."""

    timeout: float = 120.0
    reasoning_model: bool = False
    """True when the endpoint returns a separate ``reasoning_content`` field."""

    @property
    def api_key(self) -> str | None:
        value = os.getenv(self.api_key_env)
        return value.strip() if value and value.strip() else None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def key_from(self, override: str | None) -> str | None:
        """Resolve the key to use: an explicit override first, then the env.

        The web UI passes a key typed into its settings dialog, so a session can
        use a model without anything being written to disk or exported.
        """
        if override and override.strip():
            return override.strip()
        return self.api_key


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# ``base_url`` may be overridden per-model through an env var of the form
# ``<API_KEY_ENV stem>_BASE_URL``.  OPENAI_BASE_URL is honoured globally as a
# convenience for relay/proxy setups.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "deepseek": ModelConfig(
        name="deepseek",
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.95,
        reasoning_model=False,
    ),
    "deepseek-flash": ModelConfig(
        name="deepseek-flash",
        model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.95,
        max_tokens=4000,
        timeout=180.0,
        reasoning_model=True,
    ),
    "deepseek-pro": ModelConfig(
        name="deepseek-pro",
        model=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.95,
        max_tokens=4000,
        timeout=180.0,
        reasoning_model=True,
    ),
    "ark-deepseek": ModelConfig(
        name="ark-deepseek",
        model=os.getenv("ARK_MODEL", "deepseek-v3-250324"),
        base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        api_key_env="ARK_API_KEY",
        temperature=0.95,
    ),
    "doubao": ModelConfig(
        name="doubao",
        model=os.getenv("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115"),
        base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        api_key_env="ARK_API_KEY",
        temperature=1.0,
    ),
    "openai": ModelConfig(
        name="openai",
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url=os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        api_key_env="OPENAI_API_KEY",
        temperature=0.9,
    ),
    "claude": ModelConfig(
        name="claude",
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.9,
    ),
}

#: Suggested default for new games, preferring a reasoning model when present.
DEFAULT_MODEL = "deepseek-flash"


@dataclass
class ArenaConfig:
    """Table-level configuration for a single game."""

    #: Hands to play before the safety valve stops the game.  This is the single
    #: source of truth for the cap; the CLI flag, the web dialog and the web API
    #: fallback all default to this value.
    DEFAULT_MAX_HANDS = 200

    starting_chips: int = 1000
    small_blind: int = 10
    big_blind: int = 20
    max_rounds: int = DEFAULT_MAX_HANDS
    """Safety valve: hard stop after this many hands."""

    seed: int | None = None
    """Fix the RNG for reproducible games (used heavily by the test suite)."""

    llm_retries: int = 2
    """How many extra attempts an AI gets after an illegal/unparseable answer."""

    memory_turns: int = 50
    """How many previous decisions a seat keeps *in the prompt it sends*.

    Every seat's full record is retained for the whole session regardless (the UI
    shows all of it); this only bounds how much is charged to each call.  Fifty
    turns is generous -- the compact records are small, so a long memory costs
    far less than replaying full prompts would.
    """

    blind_increase_every: int = 15
    """Raise the blinds every N hands.  0 disables escalation.

    Escalation is what ends a game.  With fixed blinds a short stack can fold
    every hand forever while the others trade a single big blind back and forth,
    so a game runs to the hand cap with no winner -- which is exactly what
    happens at 0.  Fifteen hands per level gets the big blind past a typical
    stack by around hand 100, which forces all-in confrontations.
    """

    blind_increase_factor: float = 1.5
    max_blind_level: int = 20

    ante_fraction: float = 0.25
    """Ante as a fraction of the big blind, applied from level 2 onward.

    Antes are the real fix for a passive table: they make folding cost
    something, so there is always a reason to contest the pot.  A quarter of the
    big blind is the standard tournament ante.
    """

    hand_history_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    verbose_llm: bool = True

    def __post_init__(self) -> None:
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.small_blind > self.big_blind:
            raise ValueError("Small blind cannot exceed big blind")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.memory_turns < 0:
            raise ValueError("memory_turns cannot be negative")


def resolve_model(spec: str) -> ModelConfig:
    """Map a CLI/registry name onto a :class:`ModelConfig`."""
    if spec in MODEL_REGISTRY:
        return MODEL_REGISTRY[spec]
    known = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(f"Unknown model '{spec}'. Known models: {known}")
