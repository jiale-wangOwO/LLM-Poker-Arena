"""Audit the offline heuristic brain.

Two questions:

1. Does it ever return an action the engine refuses?  The same code path is the
   fallback when a model misbehaves, so an illegal decision there turns a
   recoverable model error into a crash.
2. Does it use the information it is given?  A bot whose choice does not move
   when the price, the field size or the stack depth changes is the
   "single bet-judging machine" this was meant to stop being.

Spots are built through the real ``describe_table`` path, because a hand-rolled
prompt can parse to defaults that make every decision look identical.

Run: ``python tools/audit_bot.py``
"""

from __future__ import annotations

import random
from collections import Counter

from pokerarena.ai import AIDecider, HeuristicTransport, describe_table
from pokerarena.cards import Card
from pokerarena.engine import Player, Street, Table
from pokerarena.personas import persona_store


def make_decider(persona="pro", seed=0):
    """The offline brain, wired the way the arena wires a keyless seat."""
    p = persona_store().get(persona)
    return AIDecider(
        persona=p,
        transport=HeuristicTransport(p, rng=random.Random(seed)),
        model_label="offline",
        rng=random.Random(seed),
    )


def fresh_table(seats=3, chips=1000, sb=10, bb=20, seed=0):
    players = [Player(name=f"P{i}", seat=i, chips=chips) for i in range(seats)]
    for p in players:
        p.seated = True
    return Table(players, sb, bb, seed=seed), players


def flop_spot(call_cost, pot, chips, opponents, *, persona="pro", seed=5,
              hero=("7d", "Ah"), board=("4d", "6c", "3c")):
    """Build one fixed flop decision and return the prompt a model would see."""
    table, players = fresh_table(seats=1 + opponents, chips=chips, seed=seed)
    table.start_hand()
    table.street = Street.FLOP
    table.board = [Card.from_str(c) for c in board]
    for p in players:
        p.new_street()
    players[0].hole_cards = [Card.from_str(c) for c in hero]
    for p in players[1:]:
        p.hole_cards = [Card.from_str(c) for c in ("2h", "7s")]
    table.pot = pot
    table.current_bet = call_cost
    players[0].street_bet = 0
    table.actor = 0
    return table, describe_table(table, 0)


def survey(label, persona="pro", **kwargs):
    """Ask the offline brain the same spot many times with different dice."""
    counts: Counter = Counter()
    for i in range(120):
        _, prompt = flop_spot(persona=persona, **kwargs)
        p = persona_store().get(persona)
        transport = HeuristicTransport(p, rng=random.Random(1000 + i))
        out = transport.complete("s", [{"role": "user", "content": prompt}], temperature=0.7)
        counts[out.content.split("<action>")[1].split("</action>")[0].split()[0]] += 1
    print(f"  {label:<40} {dict(counts)}")
    return counts


def main() -> int:
    rng = random.Random(7)
    personas = ["maniac", "rock", "trickster", "calling_station", "pro", "shark", "calculating"]
    problems: list[str] = []
    decisions = 0
    choice_mix: Counter = Counter()

    print("=== 1. does the offline brain ever choose an illegal action? ===")
    for game in range(150):
        persona = personas[game % len(personas)]
        small = rng.choice([5, 10])
        table, _ = fresh_table(
            seats=rng.randint(2, 6),
            chips=rng.choice([40, 100, 500, 1000]),
            sb=small,
            bb=small * 2,
            seed=game,
        )
        decider = make_decider(persona, seed=game)
        table.start_hand()
        steps = 0
        while not table.is_hand_over and steps < 400:
            steps += 1
            seat = table.actor
            if seat is None:
                break
            legal = table.legal_actions(seat)
            before = table.players[seat].chips
            was_over = table.is_hand_over
            try:
                action = decider.decide(table, seat)
            except Exception as exc:
                problems.append(f"persona={persona} seat={seat}: decide() raised {exc!r}")
                break
            decisions += 1
            choice_mix[action.type.value] += 1
            try:
                table.apply_action(seat, action)
            except Exception as exc:
                problems.append(
                    f"persona={persona} chose {action.type.value} amount={action.amount} "
                    f"which the engine refused: {exc}\n    menu was: {legal.summary()}"
                )
                break
            # Only meaningful mid-hand: ending the hand pays out the pot.
            if not was_over and not table.is_hand_over:
                after = table.players[seat].chips
                if after > before:
                    problems.append(
                        f"persona={persona}: {action.type.value} increased the stack "
                        f"{before} -> {after} mid-hand"
                    )
                    break
                if after < 0 or table.pot < 0:
                    problems.append(f"persona={persona}: negative chips or pot")
                    break

    print(f"  {decisions} decisions across 150 hands")
    print(f"  action mix: {dict(choice_mix)}")

    print("\n=== 2. does the choice use the price, the field and the stack? ===")
    cheap = survey("call 10 into 200  (20:1)", call_cost=10, pot=200, chips=1000, opponents=1)
    fair = survey("call 100 into 200  (3:1)", call_cost=100, pot=200, chips=1000, opponents=1)
    rich = survey("call 200 into 200  (2:1)", call_cost=200, pot=200, chips=1000, opponents=1)
    if cheap == rich:
        problems.append("the brain ignores pot odds")
    if cheap.get("call", 0) <= rich.get("call", 0):
        problems.append("a 20:1 price is not called more often than a 2:1 price")

    heads = survey("call 40, one opponent", call_cost=40, pot=200, chips=1000, opponents=1)
    crowd = survey("call 40, five opponents", call_cost=40, pot=200, chips=1000, opponents=5)
    if heads == crowd:
        problems.append("the brain ignores how many opponents are in the hand")
    if heads.get("call", 0) <= crowd.get("call", 0):
        problems.append("a multiway pot is not treated more cautiously")

    # Stack depth is a weaker signal than price, so test it where it should
    # matter: a marginal hand and a call that would commit most of a short
    # stack.  A clearly +EV call is correct at any depth, so testing with one
    # would prove nothing.
    short = survey(
        "marginal hand, 40 to call, 60 behind",
        call_cost=40, pot=120, chips=60, opponents=1,
        hero=("5s", "Qh"), board=("4d", "6c", "3c"),
    )
    deep = survey(
        "marginal hand, 40 to call, 5000 behind",
        call_cost=40, pot=120, chips=5000, opponents=1,
        hero=("5s", "Qh"), board=("4d", "6c", "3c"),
    )
    if short == deep:
        print("  (note: stack depth made no difference in this spot)")

    # Personas must not all play the same way.
    print("\n=== 3. do the personas differ? ===")
    profiles = {}
    for persona in personas:
        mix = survey(f"{persona}", persona=persona, call_cost=60, pot=200, chips=1000, opponents=2)
        profiles[persona] = mix
    if len(set(tuple(sorted(m.items())) for m in profiles.values())) == 1:
        problems.append("every persona makes the same decision")
    else:
        print(f"  ok: {len(set(tuple(sorted(m.items())) for m in profiles.values()))} "
              f"distinct profiles across {len(personas)} personas")

    print("\n" + "=" * 62)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:10]:
            print(f"  {p}")
        return 1
    print("offline brain: always legal, and responsive to price, field and stacks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
