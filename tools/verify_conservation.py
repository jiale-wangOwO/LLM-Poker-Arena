"""Verify chip conservation hand-by-hand through the Arena/CLI event path."""
from pokerarena.arena import Arena, AISpec, build_players
from pokerarena.config import ArenaConfig
from pokerarena.engine import HAND_END, HAND_START, POT_AWARDED, SHOWDOWN, Event, Table

problems = []


def run(seed, hands=60, stacks=1000, count=4):
    specs = [
        AISpec(name=n, persona_key=p, model="")
        for n, p in [
            ("Vega", "maniac"),
            ("Granite", "rock"),
            ("Kitsune", "trickster"),
            ("Ironside", "calculating"),
        ][:count]
    ]
    players, _ = build_players(specs, starting_chips=stacks)
    total = sum(p.chips for p in players)
    arena = Arena(players, config=ArenaConfig(seed=seed, max_rounds=hands))

    def check(stage, table):
        live = sum(p.chips for p in table.players) + table.pot
        if live != total:
            problems.append((seed, stage, live, total, table.hand_number))
            return False
        return True

    def on_event(table: Table, events: list[Event]):
        for event in events:
            if event.kind == HAND_START:
                check("hand_start", table)
            elif event.kind == POT_AWARDED:
                check(f"payout seat={event.data.get('seat')}", table)
            elif event.kind == SHOWDOWN:
                check("showdown", table)
            elif event.kind == HAND_END:
                check("hand_end", table)

    arena.on_event = on_event
    winner = arena.play_game()
    final = sum(p.chips for p in arena.table.players)
    if final != total:
        problems.append((seed, "FINAL", final, total, len(arena.hand_history)))
    return winner, final, total, len(arena.hand_history)


for seed in range(25):
    winner, final, total, hands = run(seed)
    print(f"seed {seed:>3}: winner={winner:<10} hands={hands:>3} chips={final}/{total}")

print()
if problems:
    print("PROBLEMS FOUND:")
    for problem in problems[:20]:
        print("  ", problem)
else:
    print("No chip-accounting problems in 25 full games.")
