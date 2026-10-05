"""Measure behavioural differences between personas (offline heuristics)."""
import random

from pokerarena.ai import AIDecider, HeuristicTransport
from pokerarena.engine import ActionType, Player, Table
from pokerarena.personas import get_persona


def measure(persona_key, hands=400, players=6, seed=0):
    """Measure one persona's behaviour in a full-ring game.

    Seat 0 is the persona under test; the others are a fixed pro so the numbers
    describe the persona rather than table composition.
    """
    rng = random.Random(seed)
    persona = get_persona(persona_key)
    hands_played = 0
    vpip = 0          # voluntarily put money in preflop
    pfr = 0           # preflop raise
    folds = 0
    raises = 0
    calls = 0
    checks = 0
    allins = 0
    decisions = 0
    preflop_decisions = 0
    preflop_raises = 0
    postflop_decisions = 0
    postflop_raises = 0

    for h in range(hands):
        ps = [Player(name=f"P{i}", seat=i, chips=1000,
                     persona=persona_key if i == 0 else "pro") for i in range(players)]
        t = Table(ps, 10, 20, seed=rng.randrange(10**6))
        t.start_hand()
        d = AIDecider(persona, HeuristicTransport(persona, random.Random(rng.random())), retries=0)
        voluntary = False
        raised_pre = False
        steps = 0
        while not t.is_hand_over and steps < 200:
            steps += 1
            seat = t.actor
            street = t.street.value
            action = d.decide(t, seat)
            t.apply_action(seat, action)
            if seat == 0:
                is_raise = action.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)
                decisions += 1
                if street == "preflop":
                    preflop_decisions += 1
                    if is_raise:
                        preflop_raises += 1
                else:
                    postflop_decisions += 1
                    if is_raise:
                        postflop_raises += 1
                if action.type is ActionType.FOLD:
                    folds += 1
                elif is_raise:
                    raises += 1
                    if street == "preflop":
                        raised_pre = True
                elif action.type is ActionType.CALL:
                    calls += 1
                elif action.type is ActionType.CHECK:
                    checks += 1
                if action.type is ActionType.ALL_IN:
                    allins += 1
                if street == "preflop" and action.type in (
                    ActionType.CALL, ActionType.RAISE, ActionType.BET, ActionType.ALL_IN
                ):
                    voluntary = True
        hands_played += 1
        if voluntary:
            vpip += 1
        if raised_pre:
            pfr += 1

    return {
        "persona": persona_key,
        "vpip%": round(100 * vpip / hands_played, 1),
        "pfr%": round(100 * pfr / hands_played, 1),
        "preRaise%": round(100 * preflop_raises / max(preflop_decisions, 1), 1),
        "postRaise%": round(100 * postflop_raises / max(postflop_decisions, 1), 1),
        "fold%": round(100 * folds / max(decisions, 1), 1),
        "call%": round(100 * calls / max(decisions, 1), 1),
        "allin%": round(100 * allins / max(decisions, 1), 1),
    }


keys = ["rock", "calculating", "pro", "shark", "trickster", "maniac", "calling_station"]
rows = [measure(k, hands=300, seed=7) for k in keys]
header = (f"{'persona':<16}{'VPIP%':>7}{'PFR%':>7}{'preRaise%':>11}"
          f"{'postRaise%':>12}{'fold%':>7}{'call%':>7}{'allin%':>8}")
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['persona']:<16}{r['vpip%']:>7}{r['pfr%']:>7}{r['preRaise%']:>11}"
          f"{r['postRaise%']:>12}{r['fold%']:>7}{r['call%']:>7}{r['allin%']:>8}")
