"""Model providers: the OpenAI-compatible endpoints this app can play on.

One file, one list.  ``providers.json`` starts with a single ``deepseek-flash``
entry and everything else is whatever the operator adds in the web UI
(`Settings → Providers`), where each one can also be edited afterwards.

Keys live in that file, which is why it is gitignored.  Nothing here reads an
API key from the environment: a provider is self-contained, so what you see in
the settings list is exactly what a seat will use.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROVIDERS_FILE = Path(__file__).resolve().parent.parent / "providers.json"
_LOCK = threading.RLock()

#: The one provider a fresh install starts with.
SEED_PROVIDER = {
    "id": "deepseek-flash",
    "label": "DeepSeek Flash",
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.95,
    "max_tokens": 4000,
    "timeout": 180.0,
    "reasoning": True,
}


@dataclass
class Provider:
    """A selectable model endpoint."""

    id: str
    label: str
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.95
    max_tokens: int = 2500
    timeout: float = 120.0
    reasoning: bool = False
    """True when the endpoint returns a hidden ``reasoning_content`` field."""

    @property
    def has_key(self) -> bool:
        return bool(self.api_key.strip())

    def public(self) -> dict:
        """Serialisable form for the UI -- never includes the key itself."""
        return {
            "id": self.id,
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "has_key": self.has_key,
        }


@dataclass
class ProviderStore:
    """The provider list, persisted to :data:`PROVIDERS_FILE`."""

    path: Path = field(default_factory=lambda: PROVIDERS_FILE)
    _items: list[Provider] = field(default_factory=list, repr=False)
    _loaded: bool = field(default=False, repr=False)

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        with _LOCK:
            if self._loaded:
                return
            self._loaded = True
            raw = None
            if self.path.is_file():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = None
            entries = (raw or {}).get("providers") if raw else None
            if not entries:
                # First run (or a corrupt / emptied file): start from the seed.
                # Deleting providers.json is therefore also the way to reset.
                self._items = [Provider(**SEED_PROVIDER)]
                self.save()
                return
            for entry in entries:
                try:
                    self._items.append(
                        Provider(
                            id=str(entry["id"]),
                            label=str(entry.get("label") or entry["id"]),
                            model=str(entry["model"]),
                            base_url=str(entry["base_url"]).rstrip("/"),
                            api_key=str(entry.get("api_key", "")),
                            temperature=float(entry.get("temperature", 0.95)),
                            max_tokens=int(entry.get("max_tokens", 2500)),
                            timeout=float(entry.get("timeout", 120.0)),
                            reasoning=bool(entry.get("reasoning", False)),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    def save(self) -> None:
        with _LOCK:
            payload = {"providers": [asdict(p) for p in self._items]}
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # -- access ------------------------------------------------------------
    def all(self) -> list[Provider]:
        self.load()
        with _LOCK:
            return list(self._items)

    def get(self, provider_id: str) -> Provider | None:
        for provider in self.all():
            if provider.id == provider_id:
                return provider
        return None

    def default(self) -> Provider | None:
        """The provider a new seat should start on.

        Prefers one with a key; falls back to whatever was added first.
        """
        items = self.all()
        for provider in items:
            if provider.has_key:
                return provider
        return items[0] if items else None

    def default_id(self) -> str:
        provider = self.default()
        return provider.id if provider else ""

    # -- mutation ----------------------------------------------------------
    def add(
        self,
        *,
        label: str,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.95,
        max_tokens: int = 2500,
        timeout: float = 120.0,
        reasoning: bool = False,
        provider_id: str = "",
    ) -> Provider:
        label, model, base_url = label.strip(), model.strip(), base_url.strip().rstrip("/")
        if not label:
            raise ValueError("a label is required")
        if not model:
            raise ValueError("a model name is required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base URL must start with http:// or https://")

        slug = provider_id.strip().lower() or _slugify(label)
        self.load()
        with _LOCK:
            if any(p.id == slug for p in self._items):
                slug = f"{slug}-{uuid.uuid4().hex[:4]}"
            provider = Provider(
                id=slug,
                label=label,
                model=model,
                base_url=base_url,
                api_key=api_key.strip(),
                temperature=_clamp(temperature, 0.0, 2.0),
                max_tokens=max(64, int(max_tokens)),
                timeout=max(5.0, float(timeout)),
                reasoning=bool(reasoning),
            )
            self._items.append(provider)
            self.save()
            return provider

    def update(self, provider_id: str, **changes) -> Provider:
        self.load()
        with _LOCK:
            for provider in self._items:
                if provider.id != provider_id:
                    continue
                if changes.get("label"):
                    provider.label = str(changes["label"]).strip()
                if changes.get("model"):
                    provider.model = str(changes["model"]).strip()
                if changes.get("base_url"):
                    url = str(changes["base_url"]).strip().rstrip("/")
                    if not url.startswith(("http://", "https://")):
                        raise ValueError("base URL must start with http:// or https://")
                    provider.base_url = url
                # An empty key means "leave it alone", so the UI never has to
                # echo a stored secret back to the browser.
                if changes.get("api_key"):
                    provider.api_key = str(changes["api_key"]).strip()
                if changes.get("clear_api_key"):
                    provider.api_key = ""
                if "temperature" in changes:
                    provider.temperature = _clamp(changes["temperature"], 0.0, 2.0)
                if "max_tokens" in changes:
                    provider.max_tokens = max(64, int(changes["max_tokens"]))
                if "timeout" in changes:
                    provider.timeout = max(5.0, float(changes["timeout"]))
                if "reasoning" in changes:
                    provider.reasoning = bool(changes["reasoning"])
                self.save()
                return provider
        raise KeyError(f"unknown provider {provider_id!r}")

    def remove(self, provider_id: str) -> bool:
        self.load()
        with _LOCK:
            before = len(self._items)
            self._items = [p for p in self._items if p.id != provider_id]
            if len(self._items) != before:
                self.save()
                return True
            return False


def _clamp(value, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:38] or f"provider-{uuid.uuid4().hex[:4]}"


#: Process-wide store.  The web app shares one instance.
STORE = ProviderStore()


# --------------------------------------------------------------------------
# Token usage
# --------------------------------------------------------------------------
#: Totals per provider id, kept in memory only.  Deliberately not persisted:
#: these are counters for the current session, and writing them to
#: providers.json would put runtime churn into a config file that holds keys.
_USAGE: dict[str, dict[str, int]] = {
    # provider_id: {"calls", "prompt_tokens", "completion_tokens", "total_tokens"}
}
_USAGE_LOCK = threading.Lock()


def record_usage(provider_id: str, usage: dict | None) -> None:
    """Add one API call's reported token usage to a provider's running total."""
    if not provider_id or not usage:
        return
    with _USAGE_LOCK:
        totals = _USAGE.setdefault(
            provider_id,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        totals["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value


def usage_for(provider_id: str) -> dict[str, int]:
    """This session's totals for one provider (zeros if it has not been used)."""
    with _USAGE_LOCK:
        return dict(
            _USAGE.get(
                provider_id,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        )


def reset_usage(provider_id: str | None = None) -> None:
    """Clear the totals, for one provider or for all of them."""
    with _USAGE_LOCK:
        if provider_id is None:
            _USAGE.clear()
        else:
            _USAGE.pop(provider_id, None)


def provider_store() -> ProviderStore:
    return STORE


def default_provider_id() -> str:
    """The provider to preselect in the seat editor."""
    return STORE.default_id()
