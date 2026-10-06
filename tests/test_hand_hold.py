"""Fixed result timing and Skip/Stop boundaries, without 20-second test waits."""

from __future__ import annotations

import threading
import time

import pytest

from pokerarena.ai import ChatResult
from pokerarena.web_state import GameSession, PlaybackControl, SessionConfig
from tests.helpers import seats_for


class FakeTable:
    def __init__(self, revealed=0, board=0, pots=1, hand=7):
        self.hand_number = hand
        self.showdown_results = [{"seat": i} for i in range(revealed)]
        self.board = list(range(board))
        self.pots_snapshot = [{"amount": 100} for _ in range(pots)]


class FakeSession:
    def __init__(self, seconds=20):
        self.config = SessionConfig(hand_result_seconds=seconds)
        self.lock = threading.RLock()
        self.finished = False
        self.paused = False
        self._stop_requested = threading.Event()
        self.skip_hold = False
        self.hold_until = self.hold_total = self._hold_deadline = 0.0
        self.hold_hand = None


class VirtualClock:
    """Advance only the owner thread; existing session workers use real time."""

    def __init__(self, callback=None):
        self.elapsed = 0.0
        self.wall_jump = 0.0
        self.callback = callback
        self.sleeps = []
        self.owner = threading.get_ident()

    def time(self):
        if threading.get_ident() != self.owner:
            return time.time()
        return 1000.0 + self.elapsed + self.wall_jump

    def monotonic(self):
        if threading.get_ident() != self.owner:
            return time.monotonic()
        return 1000.0 + self.elapsed

    def sleep(self, seconds):
        if threading.get_ident() != self.owner:
            time.sleep(seconds)
            return
        self.sleeps.append(seconds)
        if self.callback:
            self.callback(self)
        self.elapsed += seconds


def test_virtual_clock_leaves_other_session_workers_on_real_time():
    callbacks = []
    clock = VirtualClock(lambda current: callbacks.append(current.elapsed))
    readings = []

    def worker():
        readings.append((clock.time(), clock.monotonic()))
        clock.sleep(0.001)

    before_wall, before_monotonic = time.time(), time.monotonic()
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    wall, monotonic = readings[0]
    assert before_wall <= wall <= time.time()
    assert before_monotonic <= monotonic <= time.monotonic()
    assert clock.elapsed == 0 and not clock.sleeps and not callbacks
    clock.sleep(0.25)
    assert clock.elapsed == 0.25 and callbacks == [0]


class FoldTransport:
    def complete(self, *args, **kwargs):
        return ChatResult(content="<action>fold</action><thought></thought><say></say>")


def game_session(*, hands=2, seconds=20):
    game = GameSession(SessionConfig(
        seats=seats_for(["rock", "maniac"]), offline=True,
        starting_chips=1000, speed_seconds=0, max_hands=hands, seed=3,
        hand_result_seconds=seconds,
    ))
    game.arena.save_history = lambda: None
    for decider in game.arena.deciders.values():
        decider.transport = FoldTransport()
    return game


def wait_for(game, predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = game.snapshot()
        if predicate(state):
            return state
        time.sleep(0.005)
    raise AssertionError(f"condition not reached: {game.snapshot()}")


def test_default_is_twenty_seconds():
    assert SessionConfig().hand_result_seconds == 20.0


@pytest.mark.parametrize("table", [
    FakeTable(revealed=0, board=0),
    FakeTable(revealed=0, board=5),
    FakeTable(revealed=2, board=3),
    FakeTable(revealed=4, board=5),
    FakeTable(revealed=6, board=5, pots=3),
], ids=["preflop-fold", "river-fold", "flop-showdown", "river-showdown", "side-pots"])
def test_every_result_lasts_exactly_twenty_seconds(monkeypatch, table):
    session = FakeSession()
    announced = []

    def inspect(clock):
        with session.lock:
            announced.append((session.hold_total, session.hold_hand,
                              session._hold_deadline - clock.monotonic()))

    clock = VirtualClock(inspect)
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    PlaybackControl(session).between_hands(table)

    assert clock.elapsed == pytest.approx(20, abs=1e-9)
    assert announced[0] == (20, table.hand_number, 20)
    assert all(total == 20 and hand == table.hand_number and 0 < left <= 20
               for total, hand, left in announced)
    assert session.hold_until == session.hold_total == session._hold_deadline == 0
    assert session.hold_hand is None
    assert not session.skip_hold


@pytest.mark.parametrize("seconds", [0, 0.125, 3.75])
def test_explicit_headless_or_test_duration_is_supported(monkeypatch, seconds):
    session = FakeSession(seconds)
    clock = VirtualClock()
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    PlaybackControl(session).between_hands(FakeTable(revealed=6, board=5, pots=3))
    assert clock.elapsed == pytest.approx(seconds)
    assert session.hold_total == session.hold_until == 0
    assert session.hold_hand is None


def test_system_clock_adjustment_does_not_change_result_duration(monkeypatch):
    session = FakeSession()

    def change_wall_clock(clock):
        clock.wall_jump = 3600 if clock.elapsed < 10 else -3600

    clock = VirtualClock(change_wall_clock)
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    PlaybackControl(session).between_hands(FakeTable())
    assert clock.elapsed == pytest.approx(20)


def test_pausing_does_not_extend_twenty_second_countdown(monkeypatch):
    session = FakeSession()
    session.paused = True
    clock = VirtualClock()
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    PlaybackControl(session).between_hands(FakeTable())
    assert clock.elapsed == pytest.approx(20)
    assert session.paused


def test_auto_advance_holds_every_hand_including_the_final_one(monkeypatch):
    game = game_session(hands=2)
    samples = []

    def inspect(clock):
        state = game.snapshot()
        samples.append((clock.elapsed, state))
        assert not state["finished"]
        assert state["holding_result"] and state["hand_over"]
        assert state["hand_result"]["hand"] == state["hold_hand"] == state["hand_number"]
        assert state["hold_total"] == 20
        assert 0 <= state["hold_remaining"] <= 20

    clock = VirtualClock(inspect)
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    game._run()

    assert game.error is None, game.error
    assert clock.elapsed == pytest.approx(40)
    assert {state["hand_number"] for _, state in samples} == {1, 2}
    assert min(at for at, state in samples if state["hand_number"] == 2) == pytest.approx(20)
    final = game.snapshot()
    assert final["finished"] and final["completion_reason"] == "hand_cap"
    assert final["hand_result"]["hand"] == 2
    assert not final["holding_result"]
    assert final["hold_total"] == final["hold_remaining"] == 0
    assert final["hold_hand"] is None


def test_skip_advances_to_next_hand_without_skipping_its_result():
    game = game_session(hands=2)
    try:
        assert not game.skip_result(), "Skip while playing must not be carried forward"
        game.start()
        first = wait_for(game, lambda state: state["holding_result"])
        assert first["hold_hand"] == first["hand_result"]["hand"] == 1
        assert not game.skip_result(hand=0)
        assert game.snapshot()["holding_result"]
        assert game.skip_result(hand=1)

        second = wait_for(game, lambda state: state["holding_result"] and state["hold_hand"] == 2)
        assert second["hold_total"] == 20
        assert second["hold_remaining"] > 19
        assert not second["finished"], "The final hand must be shown before game completion"
        assert not game.skip_result(hand=1), "A delayed click must not skip a newer result"
        assert game.snapshot()["holding_result"]
        assert game.skip_result(hand=2)
        finished = wait_for(game, lambda state: state["finished"])
        assert not finished["holding_result"]
        assert finished["hand_result"]["hand"] == 2
        assert not game.skip_result(hand=2)
    finally:
        game.stop()
        game._thread.join(timeout=2)
    assert not game._thread.is_alive()


def test_skip_while_paused_keeps_the_next_decision_paused():
    game = game_session(hands=2)
    try:
        game.start()
        first = wait_for(game, lambda state: state["holding_result"])
        game.set_paused(True)
        last_action = first["seq"]
        assert game.skip_result(hand=first["hold_hand"])
        next_hand = wait_for(game, lambda state: state["hand_number"] == 2)
        time.sleep(0.06)
        after = game.snapshot()
        assert next_hand["paused"] and after["paused"]
        assert after["seq"] == last_action
        assert not after["holding_result"]
        game.set_paused(False)
        wait_for(game, lambda state: state["holding_result"] and state["hold_hand"] == 2)
    finally:
        game.stop()
        game._thread.join(timeout=2)
    assert not game._thread.is_alive()


def test_stop_immediately_releases_the_result_and_worker():
    game = game_session(hands=2)
    game.start()
    wait_for(game, lambda state: state["holding_result"])
    assert game.stop()
    stopped = game.snapshot()
    assert stopped["stopped"] and stopped["finished"]
    assert not stopped["holding_result"]
    assert stopped["hold_remaining"] == stopped["hold_total"] == 0
    assert stopped["hold_hand"] is None
    game._thread.join(timeout=2)
    assert not game._thread.is_alive()
    assert not game.skip_result()


def test_skip_after_countdown_expiry_is_not_sticky(monkeypatch):
    game = game_session()
    clock = VirtualClock()
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    game._hold_deadline = clock.monotonic()
    game.hold_until = clock.time()
    game.hold_total = 20
    game.hold_hand = 1
    assert not game.skip_result(hand=1)
    assert not game.skip_hold
    game.arena.control.between_hands(FakeTable(hand=2))
    assert clock.elapsed == pytest.approx(20)


def test_finished_session_never_announces_a_new_hold(monkeypatch):
    session = FakeSession()
    session.finished = True
    clock = VirtualClock()
    monkeypatch.setattr("pokerarena.web_state.time", clock)
    PlaybackControl(session).between_hands(FakeTable())
    assert not clock.sleeps
    assert session.hold_total == session.hold_until == 0
