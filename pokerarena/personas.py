"""AI player personas: the built-in templates plus operator-defined ones.

A persona is what makes a seat feel like a person rather than a solver.  It has
two halves:

* **behaviour** -- ``aggression``, ``tightness`` and ``bluff_frequency`` drive the
  offline brain directly, and are described to a model in the system prompt;
* **character** -- ``style`` (the strategy brief), ``voice`` (how it talks),
  ``name`` and ``tagline``.

The seven built-ins ship with the app.  Everything else lives in
``personas.json`` and is created in the web UI (`Settings → Personas`), where a
custom persona can either be written from scratch or cloned from a template and
edited.  A cloned persona is fully independent: editing the template later does
not change the clone.

The engine never trusts a persona to keep the game legal -- the LLM layer filters
every decision through :meth:`pokerarena.engine.Table.legal_actions` -- so a
persona can be as wild as we like without risking an illegal bet.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

PERSONAS_FILE = Path(__file__).resolve().parent.parent / "personas.json"
_LOCK = threading.RLock()


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    tagline: str
    """One-line summary shown in the UI."""

    style: str
    """Poker-strategy brief injected into the system prompt."""

    voice: str
    """How they talk at the table (drives the ``say`` field)."""

    aggression: float
    """0.0 = passive calling station, 1.0 = relentless pressure."""

    tightness: float
    """0.0 = plays everything, 1.0 = only premium hands."""

    bluff_frequency: float
    """How often they are willing to represent a hand they do not have."""

    short: str = ""
    """Two or three words for tight spaces (the seat box at the table).

    Left blank in the catalogue, where it is derived from the tagline.
    """

    temperature: float = 0.9
    """Sampling temperature for the model call."""

    avatar: str = "?"
    color: str = "white"
    custom: bool = False
    """True for operator-defined personas (these can be edited and deleted)."""

    template: str = ""
    """For a custom persona, the template it was cloned from (may be empty)."""

    def short_style(self) -> str:
        """A compact style label, e.g. ``"ultra-tight"``."""
        if self.short:
            return self.short
        # "ultra-tight, waits for the nuts" -> "ultra-tight"
        head = self.tagline.split(",")[0].strip()
        return head[:22] if head else self.name

    def describe(self) -> str:
        return f"{self.name} -- {self.tagline}"

    def public(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "tagline": self.tagline,
            "short": self.short_style(),
            "style": self.style,
            "voice": self.voice,
            "aggression": self.aggression,
            "tightness": self.tightness,
            "bluff_frequency": self.bluff_frequency,
            "temperature": self.temperature,
            "avatar": self.avatar,
            "custom": self.custom,
            "template": self.template,
        }


# --------------------------------------------------------------------------
# The built-in roster
# --------------------------------------------------------------------------
PERSONAS: dict[str, Persona] = {
    "calculating": Persona(
        key="calculating",
        name="Alice",
        tagline="cold, maths-first, zero ego",
        style=(
            "You play a tight-aggressive, mathematics-first game. You think in "
            "terms of ranges, pot odds and equity, and you never let a bad beat "
            "or a hot streak change your decisions. You fold marginal hands "
            "without regret, value-bet thinly when you believe you are ahead, "
            "and you are utterly unemotional about the result of any single hand."
        ),
        voice=(
            "Speak in short, clipped, analytical sentences. Occasionally cite a "
            "number (pot odds, outs, equity). Never boast and never complain."
        ),
        aggression=0.55,
        tightness=0.62,
        bluff_frequency=0.10,
        temperature=0.65,
        avatar="A",
        color="cyan",
    ),
    "maniac": Persona(
        key="maniac",
        name="Rex",
        tagline="hyper-aggressive pressure machine",
        style=(
            "You are a hyper-aggressive maniac. You raise far more than is "
            "reasonable, you three-bet light, you barrel multiple streets and "
            "you are completely unafraid of variance. You would rather lose a "
            "big pot swinging than win a small one passively. Your opponents "
            "never know whether you have the nuts or nothing at all."
        ),
        voice=(
            "Loud, brash, trash-talking. Use short exclamations. Taunt the table "
            "after a big pot. Never sound nervous."
        ),
        aggression=0.95,
        tightness=0.18,
        bluff_frequency=0.50,
        temperature=1.05,
        avatar="R",
        color="red",
    ),
    "rock": Persona(
        key="rock",
        name="Tom",
        tagline="ultra-tight, waits for the nuts",
        style=(
            "You are an extremely tight, patient nit. You fold almost "
            "everything and only continue with genuinely premium holdings. When "
            "you finally raise, your opponents should be afraid, because you "
            "simply do not raise light. You are content to bleed blinds for "
            "hours waiting for the right spot."
        ),
        voice=(
            "Terse and unbothered. Grudging. Occasionally mutter about patience "
            "and waiting for a real hand."
        ),
        aggression=0.30,
        tightness=0.92,
        bluff_frequency=0.04,
        temperature=0.60,
        avatar="T",
        color="white",
    ),
    "trickster": Persona(
        key="trickster",
        name="Zoe",
        tagline="deceptive, unbalanced, impossible to read",
        style=(
            "You are a deceptive trickster. You deliberately mix your play so "
            "that nothing you do can be read: you slow-play monsters, you bluff "
            "with air, you sometimes fold a decent hand to keep opponents "
            "guessing. You care more about being unreadable than about any single "
            "pot, and you enjoy making strong players uncomfortable."
        ),
        voice=(
            "Playful and slippery. Ask leading questions. Say things that could "
            "mean anything. Never confirm what you had."
        ),
        aggression=0.70,
        tightness=0.40,
        bluff_frequency=0.65,
        temperature=1.0,
        avatar="Z",
        color="magenta",
    ),
    "calling_station": Persona(
        key="calling_station",
        name="Bob",
        tagline="sticky, curious, hates folding",
        style=(
            "You are a calling station. You hate folding and you call far too "
            "much because you always want to see what happens next. You rarely "
            "raise, but you also rarely believe anyone, so you keep opponents "
            "honest by paying them off. You are here to see flops."
        ),
        voice=(
            "Easy-going and chatty. Friendly, slightly oblivious. Say you just "
            "want to see one more card."
        ),
        aggression=0.15,
        tightness=0.12,
        bluff_frequency=0.06,
        temperature=0.85,
        avatar="B",
        color="green",
    ),
    "pro": Persona(
        key="pro",
        name="Sam",
        tagline="balanced modern solver-style pro",
        style=(
            "You are a modern, solver-influenced professional. You play a "
            "balanced strategy with well-chosen bet sizings, you defend your "
            "blinds appropriately, you apply pressure in position and you give "
            "up gracefully when the story does not add up. You adjust to your "
            "opponents: you value-bet more against stations and bluff more "
            "against nits."
        ),
        voice=(
            "Calm, professional, economical. Occasional dry observation about "
            "the table. No trash talk."
        ),
        aggression=0.65,
        tightness=0.50,
        bluff_frequency=0.30,
        temperature=0.85,
        avatar="S",
        color="blue",
    ),
    "shark": Persona(
        key="shark",
        name="Max",
        tagline="exploitative, punishes every weakness",
        style=(
            "You are a ruthless exploitative shark. You watch every opponent "
            "closely, identify their leaks within a few hands, and then attack "
            "those leaks relentlessly: you bluff the players who fold too much, "
            "you value-bet the players who call too much, and you avoid the ones "
            "who fight back. Every decision is aimed at a specific opponent."
        ),
        voice=(
            "Confident and slightly predatory. Comment on what you have noticed "
            "about the other players. Stay polite but menacing."
        ),
        aggression=0.80,
        tightness=0.45,
        bluff_frequency=0.40,
        temperature=0.90,
        avatar="M",
        color="yellow",
    ),
}

DEFAULT_PERSONA = "pro"

#: Fields a caller may set on a persona.  ``key``/``custom``/``template`` are
#: managed by the store, not typed in by hand.
EDITABLE_FIELDS = (
    "name",
    "tagline",
    "short",
    "style",
    "voice",
    "aggression",
    "tightness",
    "bluff_frequency",
    "temperature",
    "avatar",
)


class PersonaStore:
    """The built-in templates plus whatever the operator has defined."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else PERSONAS_FILE
        self._custom: list[Persona] = []
        self._loaded = False

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        with _LOCK:
            if self._loaded:
                return
            self._loaded = True
            if not self.path.is_file():
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            for entry in raw.get("personas", []):
                try:
                    self._custom.append(_persona_from_dict(entry, custom=True))
                except (KeyError, TypeError, ValueError):
                    continue

    def save(self) -> None:
        with _LOCK:
            payload = {
                "personas": [
                    {k: v for k, v in asdict(p).items() if k not in {"custom"}}
                    for p in self._custom
                ]
            }
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # -- access ------------------------------------------------------------
    def all(self) -> list[Persona]:
        self.load()
        with _LOCK:
            return [*PERSONAS.values(), *self._custom]

    def custom(self) -> list[Persona]:
        self.load()
        with _LOCK:
            return list(self._custom)

    def get(self, key: str) -> Persona:
        self.load()
        with _LOCK:
            if key in PERSONAS:
                return PERSONAS[key]
            for persona in self._custom:
                if persona.key == key:
                    return persona
        known = ", ".join(sorted(p.key for p in self.all()))
        raise KeyError(f"Unknown persona '{key}'. Known personas: {known}")

    def has(self, key: str) -> bool:
        try:
            self.get(key)
        except KeyError:
            return False
        return True

    def keys(self) -> list[str]:
        return sorted(p.key for p in self.all())

    # -- mutation ----------------------------------------------------------
    def add(self, *, name: str = "", template: str = "", **fields) -> Persona:
        """Create a custom persona.

        ``template`` names a persona to start from (built-in or custom);
        ``fields`` overrides individual attributes on top of it.
        """
        base = PERSONAS.get(template) or next(
            (p for p in self.custom() if p.key == template), None
        )
        if base is None:
            base = DEFAULT_PERSONA_TEMPLATE
        base = replace(base, custom=False, template="")

        name = str(name or "").strip() or "New persona"
        key = self._unique_key(_slugify(name))
        explicit = _clean_fields(fields)
        # An explicit name wins; otherwise fall back to the template's own name
        # (a clone of Sam is called Sam until you rename it).
        if not explicit.get("name"):
            explicit.pop("name", None)
        template_key = template if self.has(template) else ""
        persona = replace(
            base,
            key=key,
            custom=True,
            template=template_key,
            name=name if name != "New persona" or "name" in explicit else base.name,
            **explicit,
        )
        self.load()
        with _LOCK:
            self._custom.append(persona)
            self.save()
            return persona

    def update(self, key: str, **fields) -> Persona:
        """Edit a custom persona.  Built-in templates are read-only."""
        self.load()
        with _LOCK:
            for index, persona in enumerate(self._custom):
                if persona.key != key:
                    continue
                changes = _clean_fields(fields)
                if "name" in changes and not changes["name"]:
                    changes.pop("name")
                updated = replace(persona, **changes)
                self._custom[index] = updated
                self.save()
                return updated
        if key in PERSONAS:
            raise ValueError(
                f"'{key}' is a built-in template; clone it to make an editable copy"
            )
        raise KeyError(f"unknown persona {key!r}")

    def clone(self, key: str, name: str = "") -> Persona:
        """Copy a built-in template (or another custom persona) and make it editable."""
        source = self.get(key)
        new_name = str(name or "").strip() or f"{source.name} (copy)"
        return self.add(
            name=new_name,
            template=key,
            tagline=source.tagline,
            short=source.short,
            style=source.style,
            voice=source.voice,
            aggression=source.aggression,
            tightness=source.tightness,
            bluff_frequency=source.bluff_frequency,
            temperature=source.temperature,
            avatar=source.avatar,
        )

    def remove(self, key: str) -> bool:
        self.load()
        with _LOCK:
            before = len(self._custom)
            self._custom = [p for p in self._custom if p.key != key]
            if len(self._custom) != before:
                self.save()
                return True
        if key in PERSONAS:
            raise ValueError(f"'{key}' is a built-in template and cannot be deleted")
        return False

    def _unique_key(self, slug: str) -> str:
        taken = {p.key for p in self.all()}
        if slug not in taken:
            return slug
        for _ in range(50):
            candidate = f"{slug}-{uuid.uuid4().hex[:4]}"
            if candidate not in taken:
                return candidate
        raise ValueError("could not allocate a persona key")


def _persona_from_dict(entry: dict, *, custom: bool) -> Persona:
    known = {f.name for f in Persona.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    payload = {k: v for k, v in entry.items() if k in known}
    payload["custom"] = custom
    return Persona(**payload)


def _clean_fields(fields: dict) -> dict:
    """Coerce and validate the fields a caller may set."""
    out: dict = {}
    for name, value in fields.items():
        if name not in EDITABLE_FIELDS or value is None:
            continue
        if name in {"aggression", "tightness", "bluff_frequency"}:
            out[name] = _clamp(value, 0.0, 1.0)
        elif name == "temperature":
            out[name] = _clamp(value, 0.0, 2.0)
        else:
            text = str(value).strip()
            if name in {"name", "tagline", "short", "avatar"}:
                text = text[:120]
            out[name] = text
    return out


def _clamp(value, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:38] or f"persona-{uuid.uuid4().hex[:4]}"


#: Used when a caller supplies an unknown template.
DEFAULT_PERSONA_TEMPLATE = PERSONAS[DEFAULT_PERSONA]

#: Process-wide store.  The web app shares one instance.
STORE = PersonaStore()


def persona_store() -> PersonaStore:
    return STORE


# --------------------------------------------------------------------------
# Module-level helpers kept for the CLI and the offline brain
# --------------------------------------------------------------------------
def get_persona(key: str) -> Persona:
    return STORE.get(key)


def persona_keys() -> list[str]:
    return STORE.keys()


def default_lineup(count: int = 3) -> list[str]:
    """A varied, well-differentiated default lineup."""
    order = ["maniac", "rock", "trickster", "calculating", "shark", "calling_station", "pro"]
    if count <= len(order):
        return order[:count]
    lineup = list(order)
    while len(lineup) < count:
        lineup.append(order[len(lineup) % len(order)])
    return lineup
