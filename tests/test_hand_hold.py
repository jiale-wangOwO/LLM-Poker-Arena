"""Check that the end-of-hand hold scales with how much there is to read.

A flat pause is either too short for a four-way river showdown or a pointless
wait after a pre-flop fold-out.  These tests pin the scaling and the escape
hatches (the browser can skip; 0 disables).
"""

from __future__ import annotations

import time

import pytest

from pokerarena.web_state import PlaybackControl


class FakeTable:
    """Only the attributes ``_result_weight`` looks at."""

    def __init__(self, revealed=0, board=0, pots=1):
        self.showdown_results = [
            {"seat": i, "name": f"P{i}", "hole_cards": [], "hand_name": "One Pair"}
            for i in range(revealed)
        ]
        self.board = list(range(board))
        self.pots_snapshot = [{"amount": 100, "eligible": [], "side": bool(i)} for i in range(pots)]


def weight(**kwargs) -> float:
    return PlaybackControl._result_weight(FakeTable(**kwargs))


def test_a_fold_out_flashes_past():
    """Nothing was shown and there is no board: barely a pause."""
    assert weight(revealed=0, board=0) <= 0.4


def test_a_won_without_showdown_is_still_brief():
    assert weight(revealed=0, board=5) <= 0.4


def test_a_showdown_is_worth_waiting_for():
    small = weight(revealed=2, board=3)
    assert small >= 1.0, f"a two-way showdown got only {small:.2f}x"


def test_more_revealed_hands_means_a_longer_hold():
    two = weight(revealed=2, board=5)
    four = weight(revealed=4, board=5)
    assert four > two, "a four-way showdown needs longer than a heads-up one"


def test_a_full_board_means_a_longer_hold():
    flop = weight(revealed=2, board=3)
    river = weight(revealed=2, board=5)
    assert river > flop


def test_a_split_pot_gets_extra_time():
    single = weight(revealed=2, board=5, pots=1)
    split = weight(revealed=2, board=5, pots=2)
    assert split > single, "a split pot needs explaining"


def test_a_big_showdown_is_capped():
    """Never hold for so long that the table feels stuck."""
    biggest = weight(revealed=6, board=5, pots=3)
    assert biggest <= 1.9, f"uncapped hold would be {biggest:.2f}x the base"


def test_a_fold_out_never_gets_a_full_pause():
    """A flat multi-second wait for "everyone folded" feels broken."""
    session = FakeSession(4.5)
    control = PlaybackControl(session)
    announced: list[float] = []

    import threading

    def stop():
        for _ in range(400):
            if session.hold_total:
                announced.append(session.hold_total)
                with session.lock:
                    session.skip_hold = True
                return
            time.sleep(0.003)

    watcher = threading.Thread(target=stop, daemon=True)
    watcher.start()
    control.between_hands(FakeTable(revealed=0, board=0))
    watcher.join(timeout=2)

    assert announced, "the hand was never held at all"
    assert announced[0] <= 1.2, f"a fold-out was held for {announced[0]:.2f}s"


# --------------------------------------------------------------------------
# The hold itself
# --------------------------------------------------------------------------
class FakeSession:
    def __init__(self, base):
        import threading

        self.config = type("C", (), {"hand_result_seconds": base})()
        self.lock = threading.RLock()
        self.finished = False
        self.skip_hold = False
        self.hold_until = 0.0
        self.hold_total = 0.0


def hold_seconds(base, **table_kwargs) -> tuple[float, float]:
    """Run a hold, returning (announced length, result weight).

    The wait is interrupted as soon as it begins -- what matters here is the
    length the control *chose*, not how long the test sat waiting for it.
    """
    import threading

    session = FakeSession(base)
    control = PlaybackControl(session)
    table = FakeTable(**table_kwargs)
    announced: list[float] = []

    def stop():
        for _ in range(400):
            if session.hold_total:
                announced.append(session.hold_total)
                with session.lock:
                    session.skip_hold = True
                return
            time.sleep(0.003)

    watcher = threading.Thread(target=stop, daemon=True)
    watcher.start()
    control.between_hands(table)
    watcher.join(timeout=2)
    return (announced[0] if announced else 0.0), PlaybackControl._result_weight(table)


def test_the_hold_actually_lasts_as_long_as_it_announces():
    """The UI draws a countdown from `hold_total`, so the two must agree."""
    session = FakeSession(1.0)
    control = PlaybackControl(session)
    table = FakeTable(revealed=3, board=5)
    seen: list[float] = []

    def sample():
        for _ in range(400):
            if session.hold_total:
                seen.append(session.hold_total)
                time.sleep(0.05)
                seen.append(session.hold_until - time.time())
                return
            time.sleep(0.005)

    import threading

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    control.between_hands(table)
    watcher.join(timeout=1)

    assert seen, "the hold never announced a length"
    announced = seen[0]
    remaining = seen[1]
    assert announced > 0
    # What the UI is told at the start must match the time actually spent.
    assert remaining <= announced, "the countdown started larger than the hold"
    assert remaining >= announced - 0.3, (
        f"announced {announced:.2f}s but only {remaining:.2f}s was left almost at once"
    )


def test_a_river_showdown_holds_longer_than_a_fold_out():
    quiet, _ = hold_seconds(1.0, revealed=0, board=0)
    loud, _ = hold_seconds(1.0, revealed=3, board=5)
    assert loud > quiet * 1.5, f"fold-out {quiet:.2f}s vs showdown {loud:.2f}s"
    assert quiet > 0, "even a fold-out should register briefly"


def test_zero_disables_the_hold():
    """`hand_result_seconds = 0` must return immediately and block nothing."""
    session = FakeSession(0.0)
    control = PlaybackControl(session)
    started = time.time()
    control.between_hands(FakeTable(revealed=3, board=5))
    elapsed = time.time() - started
    assert elapsed < 0.1, f"took {elapsed:.2f}s with the hold disabled"
    assert session.hold_until == 0.0
    assert session.hold_total == 0.0


def test_the_browser_can_skip_the_hold():
    import threading

    session = FakeSession(5.0)
    control = PlaybackControl(session)
    table = FakeTable(revealed=3, board=5)

    def skipper():
        time.sleep(0.15)
        with session.lock:
            session.skip_hold = True

    threading.Thread(target=skipper, daemon=True).start()
    started = time.time()
    control.between_hands(table)
    elapsed = time.time() - started
    assert elapsed < 2.0, f"skip took {elapsed:.2f}s"
    assert session.skip_hold is False, "the skip flag must be consumed"
