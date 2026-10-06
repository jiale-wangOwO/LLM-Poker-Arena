"""Ask each persona to react to the same moment, for the landing page.

Offline bots say nothing, so real table talk needs the real model. The key is
read from providers.json exactly as the web UI does -- nothing is printed and
nothing is written anywhere.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

# The console on Windows defaults to GBK, which mangles the em-dashes the models
# like to use.  Samples are meant to be quoted verbatim, so force UTF-8 out.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pokerarena.ai import OpenAITransport, SYSTEM_TEMPLATE
from pokerarena.personas import persona_store

STORE = pathlib.Path("providers.json")
if not STORE.exists():
    sys.exit("no providers.json")

data = json.loads(STORE.read_text(encoding="utf-8"))
providers = data.get("providers") or data
default = data.get("default") or (providers[0]["id"] if providers else None)
provider = next((p for p in providers if p.get("id") == default), providers[0])
print(f"provider: {provider.get('label')}  model={provider['model']}  base={provider['base_url']}")
key = (provider.get("api_key") or "").strip()
if not key:
    sys.exit("provider has no key")

transport = OpenAITransport(
    model=provider["model"],
    base_url=provider["base_url"],
    api_key=key,
    # This model spends most of its budget on hidden reasoning before it answers,
    # so give it plenty or the reply comes back empty.
    max_tokens=6000,
    timeout=120,
)

# One shared situation, described exactly as the game would describe it, so the
# only thing that varies between replies is the personality.
SITUATION = """You are playing No-Limit Texas Hold'em, 4-handed.

Blinds 10/20.   You are on the button.
Pot: 260
Board: Kh 7c 2d  -- 9s   (turn)
Your hole cards: Ks Qh -- One Pair (strong, ~61th percentile), 71% equity heads-up (a typical range)
Effective stacks: you 1,340, the big blind 1,180.
Highest bet on this street: 180   Already in from you: 0   To call: 180

BETTING SO FAR THIS HAND (in order)
  #1 Alice folds
  #2 Rex raises to 60
  #3 Tom folds
  #4 You call 60
  FLOP
  #5 Rex bets 85
  #6 You call 85
  TURN
  #7 Rex bets 180

Rex has raised or bet every street. You have top pair, good kicker.

Answer with a short in-character line of table talk and your action. You may
bluff, but never state what you hold.
"""

# A few seconds of thought is plenty for one line.
for key_name in ("calculating", "maniac", "rock", "trickster", "calling_station", "shark"):
    persona = persona_store().get(key_name)
    system = SYSTEM_TEMPLATE.format(
        name=persona.name,
        style=persona.style,
        voice=persona.voice,
    )
    started = time.time()
    try:
        reply = transport.complete(
            system, [{"role": "user", "content": SITUATION}], temperature=persona.temperature
        )
    except Exception as exc:
        print(f"\n--- {persona.name} ({key_name}) FAILED: {exc}")
        continue
    text = reply.content or ""
    say = text.split("<say>")[1].split("</say>")[0].strip() if "<say>" in text else ""
    action = text.split("<action>")[1].split("</action>")[0].strip() if "<action>" in text else ""
    thought = text.split("<thought>")[1].split("</thought>")[0].strip() if "<thought>" in text else ""
    print(f"\n=== {persona.name}  ({persona.tagline})  [{time.time() - started:.1f}s]")
    print(f"    says  : {say!r}")
    print(f"    does  : {action!r}")
    if thought:
        print(f"    thinks: {thought[:220]}")
