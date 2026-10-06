# What the rewrite fixed

This started as a half-finished project whose poker was wrong in ways that
quietly corrupted games. The rules layer was rewritten from scratch.

The table below is the record of what was wrong and what the rule is now. It is
kept because several of these are subtle — and because a rules engine that
quietly gets side pots wrong is worse than one that refuses to run.

| # | Bug in the original | Consequence | Fix |
|---|---|---|---|
| 1 | `betting_round` reset `current_bet` on the flop but `setup_round` posted blinds **before** the reset on pre-flop | blinds were wiped, so the pot under-counted | blinds are posted after street state is initialised |
| 2 | `raise N` added `N` to the pot but set `current_bet = player.current_bet + N` | raise sizes were wrong, and a raise could fail to raise the price at all | raise amounts are **total street commitments**; the last full raise size drives the minimum |
| 3 | The betting loop only re-checked `active_players` computed once before the loop | a player who called could be skipped, or the loop could spin | action closure: everyone still owing chips or holding an unspent action gets a turn |
| 4 | No minimum-raise tracking | unlimited under-raises | min raise = current bet + last full raise; a sub-minimum **all-in** is still allowed but does not reopen the action |
| 5 | `showdown` picked `best_player` by pairwise comparison and never handled ties | the first player in seat order won split pots | pots split evenly; odd chips go to the first winner clockwise from the button |
| 6 | **No side pots at all** | an all-in player could win (or lose) chips they never matched | layered pots with per-layer eligibility |
| 7 | Uncalled bets were never returned | a shove nobody could call vanished into the pot | uncalled excess is refunded |
| 8 | Pot awarded but `pot` never cleared | the awarded chips were counted again | explicit pot balance, zeroed on award |
| 9 | Heads-up posted blinds the 6-max way | button acted last pre-flop; button posted the big blind | heads-up: button posts the small blind and acts first pre-flop, last post-flop |
| 10 | `rotate_dealer` used stale indices while `remove_broke_players` shrank the list | the button could point at, or skip, the wrong player | seats are stable; busted players are marked `SITTING_OUT`, never removed |
| 11 | `best_hand` / `tiebreaker` had fragile edge cases | Royal Flush was unreachable, `is_straight` compared ints to a string | bitmask evaluator, cross-checked against an independent reference over random hands |
| 12 | `test/test_compare.py` imported `Hand` from `card`, which never had it | the whole test suite failed at collection | evaluator is one module; the old suite now runs against a compatibility shim |
| 13 | AI decisions were injected straight into the game | any illegal model output aborted the hand | a legality firewall: models choose only from the engine-computed legal menu |
| 14 | A crashed/absent model API killed the game | no API key meant no game at all | retry with the rejection reason, then a legal fallback; missing keys degrade to a local bot |

---

Back to the [README](../README.md).
