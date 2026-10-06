"""Web-facing game session: runs a game in a thread and exposes a JSON snapshot.

The web layer deliberately reuses the *same* engine and AI stack as the CLI.
The only thing added is:

* a background thread so the HTTP handlers never block on a model call;
* a queue that lets a browser seat submit actions to a waiting game thread;
* a snapshot method that serialises the table into something a browser can draw.

Security posture: bound to localhost by default, one game per session, no API
keys are ever serialised into a snapshot, and every human action is validated
through :meth:`pokerarena.engine.Table.legal_actions` before it is applied.
"""

from __future__ import annotations

import queue
from copy import deepcopy
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .ai import Action, calling_price
from .arena import Arena, ArenaStopped, AISpec, heuristic_transports
from .config import ArenaConfig
from .engine import (
    ACTION_TAKEN,
    ANTES_POSTED,
    BLINDS_POSTED,
    BOARD_DEALT,
    HAND_END,
    HAND_START,
    HOLE_CARDS_DEALT,
    MESSAGE,
    PLAYER_ELIMINATED,
    POT_AWARDED,
    SHOWDOWN,
    STREET_START,
    ActionType,
    Event,
    IllegalAction,
    Player,
    Table,
)
from .personas import persona_store
from .providers import STORE as _PROVIDER_STORE

#: Indirection so tests can point the web layer at a scratch file.
_PERSONA_STORE = None


def provider_store():
    """The process-wide provider store (built-ins + operator-defined)."""
    return _PROVIDER_STORE


def personas():
    """The process-wide persona store (built-in templates + custom ones)."""
    if _PERSONA_STORE is not None:
        return _PERSONA_STORE
    return persona_store()


if TYPE_CHECKING:  # pragma: no cover
    from .engine import LegalActions

MAX_EVENT_LOG = 400
MAX_THOUGHTS = 200
MAX_HISTORY_HANDS = 200
MAX_HISTORY_ACTIONS = 4000
"""Retain complete recent hands, independently of the short event/replay logs."""


MAX_SEATS = 6
"""Chips seats the felt can render without the boxes colliding."""


@dataclass
class SeatSpec:
    """One chair at the table, as configured before the game starts."""

    seat: int
    kind: str = "empty"
    """``empty`` | ``human`` | ``ai``."""

    persona: str = ""
    provider: str = ""
    name: str = ""

    @property
    def is_empty(self) -> bool:
        return self.kind == "empty"

    def display_name(self) -> str:
        if self.name.strip():
            return self.name.strip()
        if self.kind == "human":
            return "You"
        if self.kind == "ai":
            try:
                return personas().get(self.persona).name
            except KeyError:
                pass
        return f"Seat {self.seat + 1}"


@dataclass
class SessionConfig:
    """Everything needed to seat a table and start a game."""

    seats: list[SeatSpec] = field(default_factory=list)
    starting_chips: int = 1000
    small_blind: int = 10
    big_blind: int = 20
    max_hands: int = 200
    seed: int | None = None
    speed_seconds: float = 0.55
    """Pause between AI actions, purely so a browser can follow the game."""

    offline: bool = False
    """Force heuristic brains (used by the web demo mode)."""

    reveal_all: bool = True
    """Start with god mode on.

    God mode is a spectator aid with two effects, both display-only:

    * every seat's hole cards are visible, and
    * every seat's reasoning (thoughts, private context) is readable.

    It can be switched at any time while a game runs.  It never changes what the
    models are told, so turning it on does not let anyone play better.
    """

    hand_result_seconds: float = 20.0
    """Fixed time every finished hand stays visible, unless explicitly skipped.

    Fold-outs, showdowns, side pots and the final hand use the same duration.
    Set to 0 to disable presentation pacing for headless/test sessions.
    """

    def seated(self) -> list[SeatSpec]:
        return [s for s in self.seats if not s.is_empty]

    def human_seat(self) -> int | None:
        for seat in self.seats:
            if seat.kind == "human":
                return seat.seat
        return None


class PlaybackControl:
    """Pause / single-step / numbered-action bookkeeping for one session.

    The arena calls :meth:`wait_if_paused` before every decision and
    :meth:`record_action` after every applied action, which is all the state a
    replay scrubber needs.
    """

    def __init__(self, session: "GameSession"):
        self.session = session

    def should_stop(self) -> bool:
        """The arena checks this before a call and before applying its reply."""
        return self.session._stop_requested.is_set()

    def action_guard(self):
        """Keep accepting Stop and applying a move as one atomic boundary."""
        return self.session.lock

    # -- pacing ------------------------------------------------------------
    def wait_if_paused(self) -> None:
        """Block the game thread while paused, or until one step is granted."""
        session = self.session
        while True:
            with session.lock:
                if session.finished:
                    return
                if not session.paused:
                    return
                if session.step_budget > 0:
                    session.step_budget -= 1
                    return
            time.sleep(0.03)

    def between_hands(self, table: Table) -> None:
        """Hold the finished hand on screen before the next one is dealt.

        The arena calls this after a hand ends and before the next
        ``start_hand``, which is the one moment the engine still has the
        showdown, the board and the payouts available.  Pausing here is what
        lets a spectator see who won and what they held.

        Every hand uses the configured duration, including the final hand.
        The wait is interruptible by Skip or Stop. Pausing decisions does not
        extend the countdown, and skipping does not resume a paused game.
        """
        session = self.session
        hold = session.config.hand_result_seconds
        with session.lock:
            session.skip_hold = False
            if hold <= 0 or session.finished or self.should_stop():
                return
            session.hold_until = time.time() + hold
            session._hold_deadline = time.monotonic() + hold
            session.hold_total = hold
            session.hold_hand = table.hand_number
        try:
            while True:
                with session.lock:
                    if session.finished or self.should_stop() or session.skip_hold:
                        break
                    remaining = session._hold_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                time.sleep(min(0.03, remaining))
        finally:
            with session.lock:
                session.skip_hold = False
                session.hold_until = 0.0
                session._hold_deadline = 0.0
                session.hold_total = 0.0
                session.hold_hand = None

    # -- numbering ---------------------------------------------------------
    def record_action(self, table: Table, seat: int, action: Action) -> None:
        session = self.session
        with session.lock:
            session.seq += 1
            player = table.players[seat]
            # Applying an action may immediately deal another street or pay
            # the pot. The betting log retains the actual decision street.
            betting = table.hand_log[-1] if table.hand_log else {}
            session.actions.append(
                {
                    "seq": session.seq,
                    "hand": table.hand_number,
                    "street": betting.get("street", table.street.value),
                    "street_label": betting.get("street_label", table.street.label),
                    "seat": seat,
                    "name": player.name,
                    "action": action.type.value,
                    "action_label": action.describe(),
                    "amount": action.amount,
                    "pot": betting.get("pot", table.pot_total),
                    "source": action.source,
                    "speech": action.speech,
                    # Inline reasoning for the action log.
                    "thought": " ".join((action.thought or "").split())[:400],
                    "replay": session._table_frame(table),
                }
            )
            del session.actions[:-MAX_ACTIONS]
            if len(session.actions) > MAX_REPLAY_FRAMES:
                session.actions[-MAX_REPLAY_FRAMES - 1].pop("replay", None)
            # Everything the engine emits from here belongs to this action.
            session._action_index = len(session.actions) - 1


MAX_ACTIONS = 4000
"""Upper bound on the replay log (a full game is a few hundred actions)."""

MAX_REPLAY_FRAMES = 400
"""Detailed recent frames; older compact action rows remain readable."""

MAX_DECISIONS = 6000
"""Upper bound on the full decision log."""


class GameSession:
    """One game, running in its own thread."""

    def __init__(self, config: SessionConfig):
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.created_at = datetime.now()
        self.lock = threading.RLock()

        if not config.seated():
            raise ValueError("seat at least two players before starting")

        # Build one Player per chair so seat indices line up with the felt.  An
        # empty chair is a real Player marked unseated: it holds its place in the
        # rotation but is dealt nothing and never acts.
        players: list[Player] = []
        for index in range(MAX_SEATS):
            spec = next((s for s in config.seats if s.seat == index), None)
            if spec is None or spec.is_empty:
                players.append(
                    Player(
                        name=spec.display_name() if spec else f"Seat {index + 1}",
                        seat=index,
                        chips=0,
                        is_ai=True,
                        persona="",
                        model="",
                        seated=False,
                    )
                )
                continue
            is_ai = spec.kind == "ai"
            players.append(
                Player(
                    name=spec.display_name(),
                    seat=index,
                    chips=config.starting_chips,
                    is_ai=is_ai,
                    persona=spec.persona if is_ai else "human",
                    # The chosen provider is recorded even in offline mode, so
                    # the UI still shows what each seat is set to; offline only
                    # overrides *which brain runs*.
                    model=spec.provider if is_ai else "",
                    seated=True,
                )
            )

        human_seat = config.human_seat()
        specs = [
            AISpec(
                name=p.name,
                persona_key=p.persona,
                model=p.model,
            )
            for p in players
            if p.is_ai and p.seated
        ]

        arena_config = ArenaConfig(
            starting_chips=config.starting_chips,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            max_rounds=config.max_hands,
            seed=config.seed,
        )
        transports = None
        if config.offline:
            # Offline mode always uses the local brains, whatever provider each
            # seat names.  No network calls, no key needed.
            transports = heuristic_transports(
                players, config.seed, personas=personas()
            )

        self.arena = Arena(
            players,
            config=arena_config,
            transports=transports,
            human_seat=human_seat,
            human_decider=self._human_decider if human_seat is not None else None,
            control=PlaybackControl(self),
            providers=provider_store(),
            personas=personas(),
        )
        self.arena.on_event = self._on_event

        self.event_log: list[dict] = []
        self.thoughts: list[dict] = []
        self.pending_action: queue.Queue[tuple[str, Action]] = queue.Queue(maxsize=1)
        self._stop_requested = threading.Event()
        self.awaiting_human = False
        self._human_turn = 0
        self._human_token: str | None = None
        self.winner: str | None = None
        self.completion_reason: str | None = None
        self.error: str | None = None
        self.finished = False
        self.started_at: datetime | None = None
        self._thread: threading.Thread | None = None

        # -- playback control -------------------------------------------------
        self.paused = False
        self.step_budget = 0
        """One-shot permits consumed by the game thread when paused."""
        self.hold_until = 0.0
        """Wall-clock time until which the finished hand is held on screen."""
        self.hold_total = 0.0
        """Length of the current hold, so the UI can draw a countdown."""
        self._hold_deadline = 0.0
        """Monotonic deadline; system-clock adjustments cannot extend a hold."""
        self.hold_hand: int | None = None
        """Hand number the active result countdown belongs to."""
        self.skip_hold = False
        """Set by the UI to jump straight to the next hand."""
        self._god_mode = bool(config.reveal_all)
        """Live god-mode switch; see :meth:`god_mode`."""
        self.seq = 0
        """Monotonic action counter, so every decision is numbered in order."""
        self.actions: list[dict] = []
        """Compact, ordered action log used by the replay scrubber."""

        self.decisions: list[dict] = []
        """Full session decision log (every action, every seat, in order).

        The engine's per-hand history resets each hand, so this is what makes
        game-wide statistics and the complete context view possible.
        """

        self._action_index = -1
        """Index into :attr:`actions` that the game has currently reached."""
        self._hands: list[dict] = []
        """Public hand narratives: forced bets, streets, decisions and results.

        Capturing engine events keeps the narrative intact after the engine
        resets its current hand or the short live event feed rolls over. Hole
        cards and private reasoning are deliberately never stored here.
        """
        self._history_truncated = False
        self._live_frame = self._table_frame(self.arena.table)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_at = datetime.now()
        self._thread = threading.Thread(target=self._run, name=f"game-{self.id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            winner = self.arena.play_game()
            with self.lock:
                if not self._stop_requested.is_set():
                    self.winner = winner
                    self.completion_reason = "last_player" if self.arena.table.is_game_over() else "hand_cap"
        except ArenaStopped:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"{type(exc).__name__}: {exc}"
            self.completion_reason = "error"
        finally:
            with self.lock:
                self.finished = True
                self.awaiting_human = False
                self._human_token = None

    def stop(self) -> bool:
        """Stop at the next decision boundary, discarding an in-flight reply.

        Network requests already sent may finish and count toward provider
        usage. The arena never applies them or starts another model call.
        """
        with self.lock:
            if self.finished and not self._stop_requested.is_set():
                return False
            self._stop_requested.set()
            self.finished = True
            self.awaiting_human = False
            self._human_token = None
            self.paused = False
            self.step_budget = 0
            self.skip_hold = False
            self.hold_until = 0.0
            self._hold_deadline = 0.0
            self.hold_total = 0.0
            self.hold_hand = None
            self.winner = None
            self.completion_reason = "stopped"
            return True

    # -- human input -------------------------------------------------------
    def _human_decider(self, table: Table, seat: int) -> Action:
        """Block the game thread until the browser submits a legal action."""
        with self.lock:
            if self._stop_requested.is_set():
                from .ai import safe_fallback
                return safe_fallback(table, seat, table.legal_actions(seat))
            # Browser input belongs to a single decision and must never carry
            # into a later betting round.
            while not self.pending_action.empty():
                self.pending_action.get_nowait()
            self._human_turn += 1
            self._human_token = self._turn_token(table, seat)
            self.awaiting_human = True
            self._live_frame = self._table_frame(table)
        # No hard deadline while paused: a spectator may study the spot as long
        # as they like, and the session's own watchdog is the pause control.
        deadline = time.time() + 300
        while True:
            try:
                token, action = self.pending_action.get(timeout=0.4)
            except queue.Empty:
                with self.lock:
                    paused = self.paused
                    finished = self.finished
                if finished:
                    break
                if not paused and time.time() > deadline:
                    break
                continue
            with self.lock:
                if token != self._human_token:
                    continue
                self.awaiting_human = False
                self._human_token = None
            return action
        with self.lock:
            self.awaiting_human = False
            self._human_token = None
        # Nobody answered: keep the game alive with a legal non-committal move.
        from .ai import safe_fallback

        return safe_fallback(table, seat, table.legal_actions(seat))

    def _turn_token(self, table: Table, seat: int) -> str:
        return f"{self.id}:{table.hand_number}:{table.street.value}:{seat}:{self.seq}:{self._human_turn}"

    def submit_action(self, raw: str, turn_token: str | None = None) -> dict:
        """Validate and queue a human action.  Returns a status dict.

        The action is normalised through the engine *before* it is queued, so a
        hand-crafted HTTP request cannot smuggle an illegal move into the game.
        """
        if self.arena.human_seat is None:
            raise IllegalAction("This table has no human seat")
        table = self.arena.table
        seat = self.arena.human_seat
        with self.lock:
            current_token = self._human_token or self._turn_token(table, seat)
            if turn_token is not None and turn_token != current_token:
                raise IllegalAction("This turn has changed; refresh the table before acting")
            if not self.awaiting_human:
                raise IllegalAction("It is not your turn")
            if table.actor != seat:
                raise IllegalAction("It is not your turn")
            legal = table.legal_actions(seat)
            validated = table.normalize_action(seat, self._parse(raw, legal))
            try:
                self.pending_action.put_nowait((current_token, validated))
            except queue.Full:
                raise IllegalAction("Another action is already queued") from None
            # Consume the turn under the same lock used to validate and queue.
            self.awaiting_human = False
            return {"queued": True, "action": validated.describe(), "turn_token": current_token}

    @staticmethod
    def _parse(raw: str, legal: "LegalActions") -> Action:
        if not isinstance(raw, str):
            raise IllegalAction("action must be a string")
        text = (raw or "").strip().lower()
        if text in {"f", "fold"}:
            return Action(ActionType.FOLD, source="human")
        if text in {"k", "check", "x"}:
            return Action(ActionType.CHECK, source="human")
        if text in {"c", "call"}:
            return Action(ActionType.CALL, source="human")
        if text in {"a", "allin", "all-in", "all in", "shove", "jam"}:
            return Action(ActionType.ALL_IN, source="human")
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return Action(ActionType.RAISE, amount=int(digits), source="human")
        raise IllegalAction(f"Unrecognised action {raw!r}. Legal: {legal.summary()}")

    # -- events ------------------------------------------------------------
    # -- playback control --------------------------------------------------
    def set_paused(self, paused: bool) -> bool:
        """Pause or resume the game thread.  Returns the new state."""
        with self.lock:
            if self.finished:
                return False
            self.paused = bool(paused)
            if not paused:
                self.step_budget = 0
            return self.paused

    def skip_result(self, hand: int | None = None) -> bool:
        """Skip only the currently displayed result, never a later hand.

        A hand number binds delayed browser requests to their result. Calls
        while playing, after expiry, or after Stop leave no sticky skip flag.
        """
        with self.lock:
            if (self.finished or self._hold_deadline <= time.monotonic()
                    or self.hold_total <= 0 or (hand is not None and hand != self.hold_hand)):
                return False
            self.skip_hold = True
            self.hold_until = 0.0
            self._hold_deadline = 0.0
            self.hold_total = 0.0
            self.hold_hand = None
            return True

    def god_mode(self) -> bool:
        """Whether the browser may see everyone's cards and reasoning."""
        with self.lock:
            return bool(self._god_mode)

    def set_god_mode(self, enabled: bool) -> bool:
        """Switch god mode while the game runs.  Display only, never the prompts."""
        with self.lock:
            self._god_mode = bool(enabled)
            return self._god_mode

    def reveals_cards(self, seat: int, *, hand_over: bool = False) -> bool:
        """Whether the browser may show ``seat``'s hole cards right now.

        One rule, used by both the table snapshot and the player detail panel so
        the two can never disagree:

        * god mode shows everything;
        * showdown exposes the hands actually tabled, never a folded hand;
        * otherwise a *seated human* sees their own hand and nothing else;
        * a pure spectator (no human seat at all) sees nothing, because there is
          no hand that is "theirs".  Treating every non-AI seat as "yours" is how
          an all-AI table ended up broadcasting every hand in the clear.
        """
        if self.god_mode():
            return True
        human = self.arena.human_seat
        return (human is not None and seat == human) or any(
            entry.get("seat") == seat for entry in self.arena.table.showdown_results
        )

    def step(self, count: int = 1) -> bool:
        """While paused, allow ``count`` more decisions to run."""
        with self.lock:
            if self.finished:
                return False
            self.paused = True
            self.step_budget += max(1, int(count))
            return True

    # -- per-player detail -------------------------------------------------
    def player_detail(self, seat: int, include_prompt: bool = False) -> dict:
        """Everything known about one seat, for the UI's player panel.

        Includes that seat's *own* decision history and (optionally) the private
        conversation context it is carrying, which is what makes the AI feel like
        a character rather than a function.
        """
        table = self.arena.table
        if not 0 <= seat < len(table.players):
            raise KeyError(f"no seat {seat}")
        player = table.players[seat]
        decider = self.arena.deciders.get(seat)
        persona = None
        if player.is_ai:
            try:
                persona = personas().get(player.persona).public()
            except KeyError:
                persona = None

        from .strength import describe_strength

        strength_note = (
            describe_strength(player.hole_cards, table.board)
            if player.hole_cards
            else None
        )
        # Reasoning follows the same switch as cards: it is private until god
        # mode (or the end of the game) makes it readable.
        reveal = self.reveals_cards(seat, hand_over=table.is_hand_over)
        reasoning_visible = self.god_mode() or (self.arena.human_seat is not None
                                              and seat == self.arena.human_seat)

        with self.lock:
            # Every decision this seat has made, newest last, across the game.
            # The ``seq`` is looked up from the decision log so the panel can
            # show the same number as the action rail.
            history = [dict(entry) for entry in self.decisions if entry.get("seat") == seat]
            # Stats over the whole game for this seat.
            stats = self._seat_stats(seat)
            conversation_size = len(player.conversation)
            memory_turns = getattr(decider, "memory_turns", 0)
            model = model_label(player, decider)
            messages_sent = (len(decider._sent_messages(player))
                             if player.opening_prompt and decider is not None else 1)
            turns_recorded = conversation_size // 2
            hand_history = [
                entry
                for entry in self.arena.hand_history
                if player.seat in entry.get("deltas", {})
            ]

        detail = {
            "seat": seat,
            "name": player.name,
            "is_ai": player.is_ai,
            "seated": player.seated,
            "persona": persona,
            "persona_name": _persona_name(player),
            "model": model,
            "chips": player.chips,
            "status": player.status.value,
            "street_bet": player.street_bet,
            "hand_contribution": player.hand_contribution,
            # God mode reveals this seat's cards.  Display only -- the model
            # prompts never include another seat's cards.
            "hole_cards": (
                [c.code for c in player.hole_cards]
                if (reveal and player.hole_cards)
                else None
            ),
            "reasoning_visible": reasoning_visible,
            "hand_strength": strength_note if reveal else None,
            "persona_profile": (
                {
                    "style": persona["style"],
                    "voice": persona["voice"],
                    "aggression": persona["aggression"],
                    "tightness": persona["tightness"],
                    "bluff_frequency": persona["bluff_frequency"],
                    "temperature": persona["temperature"],
                }
                if persona
                else None
            ),
            "last_action": player.last_action.describe() if player.last_action else None,
            "history": history,
            "hand_history": hand_history,
            "stats": stats,
            "memory": {
                "turns_kept": memory_turns,
                "turns_recorded": turns_recorded,
                "messages_recorded": len(player.conversation),
                "messages_sent": messages_sent,
                "enabled": memory_turns > 0,
            },
            "hand_number": table.hand_number,
            "street": table.street.label,
        }
        if include_prompt and decider is not None:
            # The complete private thread: the opening full prompt, then every
            # exchange recorded.  Marked with ``sent`` so the UI can show which
            # part is actually charged to the prompt versus kept for review.
            sent_roles = decider._sent_messages(player)
            sent_count = len(sent_roles)
            thread: list[dict] = []
            if player.opening_prompt:
                thread.append(
                    {
                        "role": "user",
                        "content": player.opening_prompt,
                        "kind": "opening",
                    }
                )
            for message in player.conversation:
                thread.append(
                    {
                        "role": message.get("role"),
                        "content": message.get("content", ""),
                        "hand": message.get("hand"),
                        "street": message.get("street"),
                        "action": message.get("action"),
                        "kind": "turn",
                    }
                )
            # The last ``sent_count`` entries are what the next call will send.
            for index, entry in enumerate(thread):
                entry["sent"] = index >= len(thread) - sent_count
            detail["context"] = thread if reasoning_visible else []
        if not reasoning_visible:
            # God mode off: this seat's thoughts are its own business.  Keep the
            # actions (they are public) but drop the reasoning behind them.
            for entry in detail["history"]:
                entry["thought"] = ""
                entry.pop("reasoning", None)
                entry.pop("raw_response", None)
            detail["thoughts"] = []
        return detail

    def _seat_stats(self, seat: int) -> dict:
        """Aggregate this seat's behaviour over the game so far.

        Built from the engine's per-hand history, which records *every* decision
        (the transcript only keeps the ones with something to say).
        """
        player = self.arena.table.players[seat]
        entries = list(player.history)
        # A seat's per-hand history resets each hand, so accumulate the game
        # totals by replaying the arena's full decision log for this seat.
        all_entries = [
            e for e in self._decision_log() if e.get("seat") == seat
        ] or entries
        counts: dict[str, int] = {}
        for entry in all_entries:
            key = entry.get("action_type") or (entry.get("action") or "").split()[:1]
            if isinstance(key, list):
                key = key[0] if key else "?"
            counts[key] = counts.get(key, 0) + 1

        hands_seen = len(
            {e.get("hand") for e in all_entries if e.get("hand") is not None}
        )
        voluntary: set = set()
        for entry in all_entries:
            if entry.get("street") != "preflop":
                continue
            key = entry.get("action_type") or (entry.get("action") or "").split()[0]
            if key in {"call", "bet", "raise", "all_in", "all-in"}:
                voluntary.add(entry.get("hand"))
        return {
            "decisions": len(all_entries),
            "hands": hands_seen,
            "fold": counts.get("fold", 0),
            "check": counts.get("check", 0),
            "call": counts.get("call", 0),
            "bet": counts.get("bet", 0),
            "raise": counts.get("raise", 0),
            "all_in": counts.get("all_in", 0) + counts.get("all-in", 0),
            "voluntary_hands": len(voluntary),
            "vpip": (
                round(100 * len(voluntary) / hands_seen, 1) if hands_seen else 0.0
            ),
            "speech": sum(1 for e in all_entries if e.get("speech")),
        }

    def _record_decision(self, entry: dict) -> None:
        """Keep a durable, per-hand-proof record of every action."""
        player = self.arena.table.players[entry["seat"]]
        record = {
            "seq": self.seq,
            "seat": entry["seat"],
            "name": entry.get("name"),
            "hand": self.arena.table.hand_number,
            "street": self.arena.table.street.value,
            "action": entry.get("action"),
            "action_type": entry.get("action"),
            "amount": entry.get("amount"),
            "pot": entry.get("pot"),
            "speech": entry.get("speech", ""),
            "thought": entry.get("thought", ""),
            "source": entry.get("source", ""),
        }
        if self.actions:
            record["street"] = self.actions[-1]["street"]
            record["street_label"] = self.actions[-1]["street_label"]
            record["action"] = self.actions[-1]["action_label"]
        # Prefer the engine's richer record for this decision when available.
        if player.history:
            latest = player.history[-1]
            for key in ("street", "street_label", "thought", "reasoning",
                        "speech", "model", "persona", "action_type"):
                if latest.get(key) is not None:
                    record[key] = latest[key]
        self.decisions.append(record)
        del self.decisions[:-MAX_DECISIONS]
        # Advance the cursor: subsequent non-action events belong to this action.
        self._action_index = len(self.actions) - 1 if self.actions else 0

    def _decision_log(self) -> list[dict]:
        """Every decision by every seat, in order (full session record).

        Each entry carries its action ``seq`` so a player's history can be read
        against the shared timeline in the action rail.
        """
        return self.decisions

    def _recent_thoughts(self, seat: int) -> list[dict]:
        """That seat's decision log, newest first, for the detail panel."""
        return [d for d in reversed(self.decisions) if d.get("seat") == seat]

    def _seq_for(self, entry: dict) -> int | None:
        """Find the action number for a transcript entry.

        The transcript stores the decorated text (``"raise to 197"``) while the
        timeline stores the action type (``"raise"``), so compare on the leading
        word and match the first unclaimed number in the same hand.
        """
        hand = entry.get("hand")
        seat = entry.get("seat")
        action = str(entry.get("action") or "").split(" ")[0]
        amount = entry.get("amount")
        for candidate in self.actions:
            if candidate.get("seat") != seat or candidate.get("hand") != hand:
                continue
            if candidate.get("action") != action:
                continue
            # Prefer an exact amount match when the transcript recorded one.
            if amount is None or candidate.get("amount") in (None, amount):
                return candidate["seq"]
        return None

    def _on_event(self, table: Table, events: list[Event]) -> None:
        with self.lock:
            self._record_hand_events(table, events)
            for event in events:
                entry = self._serialise_event(event)
                if entry is None:
                    continue
                thought = entry.pop("thought", None)
                if entry.get("kind") == ACTION_TAKEN:
                    self._record_decision(entry)
                # Tag every entry with the action it belongs to, so the replay
                # scrubber can reconstruct the table at any numbered action.
                entry["_ai"] = self._action_index
                entry["hand_number"] = table.hand_number
                entry["seq"] = self.seq
                self.event_log.append(entry)
                del self.event_log[:-MAX_EVENT_LOG]
                if thought:
                    self.thoughts.append(
                        {
                            "seat": entry.get("seat"),
                            "name": entry.get("name"),
                            "hand": table.hand_number,
                            "street": self.actions[-1]["street"] if self.actions else table.street.value,
                            "action": entry.get("action"),
                            "speech": entry.get("speech", ""),
                            "thought": thought,
                            "source": entry.get("source", ""),
                            "model": self._model_for_seat(entry.get("seat")),
                        }
                    )
                    del self.thoughts[:-MAX_THOUGHTS]
                if event.kind == BLINDS_POSTED:
                    self._blind_hand = table.hand_number
                    self._blind_seats = (event.data["small"]["seat"], event.data["big"]["seat"])
            self._live_frame = self._table_frame(table)
            if any(event.kind == ACTION_TAKEN for event in events) and self.actions:
                # Include the action's complete public narration (new street,
                # payout, showdown) without touching older captured frames.
                frame = self.actions[-1].get("replay")
                if frame is not None:
                    frame["events"] = deepcopy(self._live_frame["events"])
                    frame["hand_result"] = deepcopy(self._live_frame["hand_result"])
        # Slow the game down so a browser can follow it.
        if self.config.speed_seconds and not table.is_hand_over:
            time.sleep(self.config.speed_seconds)

    def _model_for_seat(self, seat) -> str:
        if seat is None:
            return ""
        player = self.arena.table.players[seat] if seat < len(self.arena.table.players) else None
        if player is None:
            return ""
        decider = self.arena.deciders.get(seat)
        return model_label(player, decider)

    def _serialise_event(self, event: Event) -> dict | None:
        data = event.data
        kind = event.kind
        if kind == ACTION_TAKEN:
            base = {
                "kind": kind,
                "seat": data["seat"],
                "name": data["name"],
                "action": data["action"],
                "amount": data["amount"],
                "pot": data["pot"],
                "chips": data["chips"],
                "street_bet": data["street_bet"],
                "speech": data.get("speech", ""),
                "source": data.get("source", ""),
                "full_raise": data.get("full_raise", False),
            }
            if data.get("thought"):
                base["thought"] = data["thought"]
            return base
        if kind == HAND_START:
            return {
                "kind": kind,
                "hand_number": data["hand_number"],
                "button": data["button"],
                "button_name": data["button_name"],
            }
        if kind == HOLE_CARDS_DEALT:
            return {"kind": kind, "players": data["players"]}
        if kind == BLINDS_POSTED:
            return {"kind": kind, "small": data["small"], "big": data["big"]}
        if kind == ANTES_POSTED:
            return {"kind": kind, "ante": data["ante"], "paid": data["paid"]}
        if kind == STREET_START:
            return {
                "kind": kind,
                "street": data["street"],
                "label": data["label"],
                "board": data["board"],
                "pot": data["pot"],
            }
        if kind == BOARD_DEALT:
            return {"kind": kind, "cards": data["cards"], "board": data["board"]}
        if kind == SHOWDOWN:
            return {"kind": kind, "players": data["players"]}
        if kind == POT_AWARDED:
            return {
                "kind": kind,
                "seat": data["seat"],
                "name": data["name"],
                "amount": data["amount"],
                "reason": data.get("reason", ""),
                "hand": data.get("hand"),
                "pot_label": data.get("pot_label"),
                "split": data.get("split", False),
            }
        if kind == PLAYER_ELIMINATED:
            return {"kind": kind, "seat": data["seat"], "name": data["name"]}
        if kind == HAND_END:
            return {"kind": kind, "reason": data.get("reason"), "pot": data.get("pot")}
        if kind == MESSAGE:
            return {"kind": kind, "text": data.get("text", "")}
        return {"kind": kind, "data": {k: v for k, v in data.items() if k != "thought"}}

    # -- durable public hand history ---------------------------------------
    def _record_hand_events(self, table: Table, events: list[Event]) -> None:
        """Capture a complete public narrative at the engine publication boundary.

        A single publication can include the deal, all forced bets and even a
        complete all-in runout. Recover starting stacks from contributions and
        payouts, so those hands remain accurate without a voluntary decision.
        """
        for event in events:
            data = event.data
            if event.kind == HAND_START:
                paid_out: dict[int, int] = {}
                for award in events:
                    if award.kind == POT_AWARDED:
                        seat = award.data["seat"]
                        paid_out[seat] = paid_out.get(seat, 0) + award.data["amount"]
                participants = [player for player in table.players if player.hole_cards]
                order = sorted((player.seat for player in participants),
                               key=lambda seat: (seat - data["button"]) % len(table.players))
                labels = {2: ["BTN / SB", "BB"], 3: ["BTN", "SB", "BB"],
                          4: ["BTN", "SB", "BB", "CO"],
                          5: ["BTN", "SB", "BB", "UTG", "CO"],
                          6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"]}
                positions = dict(zip(order, labels.get(len(order), [])))
                self._hands.append({
                    "hand_number": data["hand_number"], "status": "in_progress",
                    "button": data["button"], "button_name": data["button_name"],
                    "small_blind": data.get("small_blind", table.small_blind),
                    "big_blind": data.get("big_blind", table.big_blind),
                    "ante": data.get("ante", table.ante),
                    "blind_level": data.get("blind_level", table.blind_level),
                    "players": [
                        {"seat": player.seat, "name": player.name,
                         "position": positions.get(player.seat, ""),
                         "starting_chips": player.chips + player.hand_contribution
                                           - paid_out.get(player.seat, 0)}
                        for player in participants
                    ],
                    "entries": [], "result": None, "truncated": False,
                    "_payouts": [],
                })
            if not self._hands or self._hands[-1]["hand_number"] != table.hand_number:
                continue
            hand = self._hands[-1]
            entries = hand["entries"]
            if event.kind == STREET_START:
                entries.append({"kind": "street", "street": data["street"],
                                "label": data["label"], "board": list(data["board"]),
                                "pot": data["pot"]})
            elif event.kind == ANTES_POSTED:
                starting = {player["seat"]: player["starting_chips"] for player in hand["players"]}
                for seat, amount in data["paid"].items():
                    entries.append({"kind": "ante", "seat": seat,
                                    "name": table.players[seat].name,
                                    "amount": amount, "nominal_amount": data["ante"],
                                    "street": "preflop", "all_in": amount == starting.get(seat)})
            elif event.kind == BLINDS_POSTED:
                starting = {player["seat"]: player["starting_chips"] for player in hand["players"]}
                ante_paid = {entry["seat"]: entry["amount"] for entry in entries if entry["kind"] == "ante"}
                for role in ("small", "big"):
                    blind = data[role]
                    entries.append({"kind": "blind", "role": role,
                                    "seat": blind["seat"], "name": blind["name"],
                                    "amount": blind["amount"], "nominal_amount": hand[f"{role}_blind"],
                                    "street": "preflop",
                                    "all_in": blind["amount"] + ante_paid.get(blind["seat"], 0)
                                              == starting.get(blind["seat"])})
            elif event.kind == ACTION_TAKEN:
                # record_action precedes event publication, preserving its
                # original street even if applying it dealt the next board.
                action = self.actions[-1] if self.actions and self.actions[-1]["hand"] == table.hand_number else None
                entries.append({"kind": "action", "seq": action["seq"] if action else self.seq,
                                "seat": data["seat"], "name": data["name"],
                                "action": data["action"],
                                "action_label": action["action_label"] if action else data["action"],
                                "amount": data["amount"], "pot": data["pot"],
                                "speech": data.get("speech", ""), "source": data.get("source", ""),
                                "street": action["street"] if action else table.street.value,
                                "street_label": action["street_label"] if action else table.street.label})
            elif event.kind == POT_AWARDED:
                hand["_payouts"].append(dict(data))
        if self._hands and self._hands[-1]["hand_number"] == table.hand_number and table.is_hand_over:
            hand = self._hands[-1]
            hand["status"] = "completed"
            payouts = hand["_payouts"]
            hand["result"] = self._hand_result({
                "hand": table.hand_number, "board": [card.code for card in table.board],
                "showdown": table.showdown_results, "payouts": payouts, "pots": table.pots_snapshot,
                "pot": sum(pot.get("amount", 0) for pot in table.pots_snapshot),
                "winners": list(dict.fromkeys(payout["name"] for payout in payouts
                                               if payout.get("reason") != "uncalled_bet_returned")),
                "deltas": {player["seat"]: table.players[player["seat"]].chips - player["starting_chips"]
                           for player in hand["players"]},
            })
        self._trim_hand_history()

    def _trim_hand_history(self) -> None:
        """Evict whole oldest hands, rather than silently dropping their ending."""
        action_count = sum(entry["kind"] == "action" for hand in self._hands for entry in hand["entries"])
        while len(self._hands) > 1 and (len(self._hands) > MAX_HISTORY_HANDS or action_count > MAX_HISTORY_ACTIONS):
            removed = self._hands.pop(0)
            action_count -= sum(entry["kind"] == "action" for entry in removed["entries"])
            self._history_truncated = True
        # The engine normally caps one hand below the history action budget.
        # Keep an explicit partial flag if a custom engine exceeds that limit.
        if self._hands and action_count > MAX_HISTORY_ACTIONS:
            hand = self._hands[0]
            drop = action_count - MAX_HISTORY_ACTIONS
            kept = []
            for entry in hand["entries"]:
                if entry["kind"] == "action" and drop:
                    drop -= 1
                else:
                    kept.append(entry)
            hand["entries"] = kept
            hand["truncated"] = True
            self._history_truncated = True

    def _public_hand_history(self, *, after: int | None = None) -> list[dict]:
        return [deepcopy({key: value for key, value in hand.items() if not key.startswith("_")})
                for hand in self._hands if after is None or hand["hand_number"] > after]

    # -- snapshot ----------------------------------------------------------
    def _table_frame(self, table: Table) -> dict:
        """Capture table values, never references that another hand can mutate."""
        legal = None
        if (
            self.arena.human_seat is not None
            and table.actor == self.arena.human_seat
            and not table.is_hand_over
        ):
            try:
                legal = table.legal_actions(self.arena.human_seat).to_dict()
                legal.update(calling_price(table, self.arena.human_seat))
            except IllegalAction:  # pragma: no cover - race guard
                legal = None

        try:
            small, big = (self._blind_seats if getattr(self, "_blind_hand", None) == table.hand_number
                          else table.blind_seats())
        except (RuntimeError, ValueError):
            small, big = None, None
        participants = [p.seat for p in table.players if p.seated and
                        (p.hole_cards or p.status.value != "sitting_out")]
        order = sorted(participants, key=lambda seat: (seat - table.button) % len(table.players))
        labels = ({2: ["BTN / SB", "BB"], 3: ["BTN", "SB", "BB"],
                   4: ["BTN", "SB", "BB", "CO"],
                   5: ["BTN", "SB", "BB", "UTG", "CO"],
                   6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"]}.get(len(order), []))
        positions = dict(zip(order, labels))
        players = []
        for player in table.players:
            decider = self.arena.deciders.get(player.seat)
            players.append(
                {
                    "name": player.name,
                    "seat": player.seat,
                    "chips": player.chips,
                    "status": player.status.value,
                    "seated": player.seated,
                    "street_bet": player.street_bet,
                    "hand_contribution": player.hand_contribution,
                    "in_hand": player.in_hand,
                    "can_act": player.is_active,
                    "is_all_in": player.is_all_in,
                    "position": positions.get(player.seat, ""),
                    "is_ai": player.is_ai,
                    "persona": player.persona,
                    "persona_name": _persona_name(player),
                    "model": model_label(player, decider),
                    "provider": player.model,
                    # The private capture is projected at the API boundary.
                    "hole_cards": [c.code for c in player.hole_cards] or None,
                    "last_action": (
                        player.last_action.describe() if player.last_action else None
                    ),
                }
            )

        last_hand = self.arena.hand_history[-1] if self.arena.hand_history else None
        current_result = None
        if table.is_hand_over and table.hand_number:
            payouts = [event.data for event in table.events if event.kind == POT_AWARDED]
            current_result = self._hand_result({
                "hand": table.hand_number, "board": [c.code for c in table.board],
                "showdown": table.showdown_results,
                "pot": sum(pot.get("amount", 0) for pot in table.pots_snapshot),
                "payouts": payouts, "pots": table.pots_snapshot,
                "winners": list(dict.fromkeys(payout.get("name") for payout in payouts
                                               if payout.get("reason") != "uncalled_bet_returned")),
            })
            if last_hand and last_hand.get("hand") == table.hand_number:
                current_result = self._hand_result(last_hand)
            elif not payouts and getattr(self, "_live_frame", {}).get("hand_number") == table.hand_number:
                current_result = deepcopy(self._live_frame.get("hand_result") or current_result)
            if self._hands and self._hands[-1]["hand_number"] == table.hand_number and self._hands[-1]["result"]:
                current_result = deepcopy(self._hands[-1]["result"])
        frame = {
            "session": self.id,
            "hand_number": table.hand_number,
            "street": table.street.value,
            "street_label": table.street.label,
            "pot": table.pot_total,
            "board": [c.code for c in table.board],
            "current_bet": table.current_bet,
            "button": table.button,
            "small_blind_seat": small,
            "big_blind_seat": big,
            "small_blind": table.small_blind,
            "big_blind": table.big_blind,
            "ante": table.ante,
            "blind_level": table.blind_level,
            "actor": table.actor,
            "players": players,
            "human_seat": self.arena.human_seat,
            "legal": legal,
            "events": [entry for entry in self.event_log if entry.get("hand_number") == table.hand_number][-60:],
            "hands_played": len(self.arena.hand_history),
            "hand_cap": self.config.max_hands,
            "showdown": table.showdown_results,
            "pots": table.pots_snapshot,
            "hand_result": current_result,
            "active_players": sum(p.in_hand for p in table.players),
            "seq": self.seq,
            "hand_over": table.is_hand_over,
            "chip_leaders": [player.name for player in table.players if player.seated
                             and player.chips == max((p.chips for p in table.players if p.seated), default=0)],
        }
        return deepcopy(frame)

    def _public_events(self, events: list[dict], god_mode: bool) -> list[dict]:
        events = deepcopy(events)
        if not god_mode:
            for event in events:
                if event.get("kind") == HOLE_CARDS_DEALT:
                    human = self.arena.human_seat
                    event["players"] = {seat: cards for seat, cards in event.get("players", {}).items()
                                        if human is not None and str(seat) == str(human)}
                event.pop("thought", None)
                event.pop("reasoning", None)
        return events

    def _public_frame(self, frame: dict, god_mode: bool, *, replay: bool = False) -> dict:
        frame = deepcopy(frame)
        public_seats = {entry.get("seat") for entry in frame.get("showdown", [])}
        human = self.arena.human_seat
        for player in frame["players"]:
            if not god_mode and player["seat"] != human and player["seat"] not in public_seats:
                player["hole_cards"] = None
        frame["events"] = self._public_events(frame.get("events", []), god_mode)
        frame.update(god_mode=god_mode, reveal_all=god_mode)
        if replay:
            frame.update(is_replay=True, awaiting_human=False, legal=None, turn_token=None,
                         finished=False, winner=None, holding_result=False,
                         hold_remaining=0, hold_total=0, hold_hand=None,
                         thoughts=[], stopped=False, completion_reason=None)
        return frame

    def snapshot(self, *, replay_after: int | None = None, history_after: int | None = None) -> dict:
        """Return live state, optionally with replay frames and hand-history deltas.

        Compact action rows are always present. Clients polling incrementally
        cache frames by action seq and evict below ``replay_first_seq``. A full
        refresh is required after changing god mode so cached privacy matches.

        ``history_after`` is the last completed hand already cached by the
        client. Newer hands are sent in full, including the live hand before its
        first action; completed public records do not change after publication.
        Evict cached hands below ``history_first_hand`` even on an empty delta.
        Omit the cursor (or use 0) to retrieve all retained hand narratives.
        """
        with self.lock:
            god_mode = self._god_mode
            # The worker publishes after complete engine operations. Reading
            # that frame avoids a half-dealt board or half-awarded chip stacks.
            frame = self._live_frame if self._thread is not None else self._table_frame(self.arena.table)
            snapshot = self._public_frame(frame, god_mode)
            actions = []
            for stored in self.actions:
                action = {key: value for key, value in stored.items() if key != "replay"}
                if not god_mode:
                    action["thought"] = ""
                if "replay" in stored and (replay_after is None or stored["seq"] > replay_after):
                    action["replay"] = self._public_frame(stored["replay"], god_mode, replay=True)
                actions.append(action)
            remaining = max(0.0, self._hold_deadline - time.monotonic())
            last_hand = self.arena.hand_history[-1] if self.arena.hand_history else None
            replay_seqs = [action["seq"] for action in self.actions if "replay" in action]
            snapshot.update(
                awaiting_human=self.awaiting_human,
                turn_token=(self._human_token or self._turn_token(self.arena.table, self.arena.human_seat))
                if self.awaiting_human else None,
                events=self._public_events(self.event_log, god_mode),
                thoughts=deepcopy(self.thoughts) if god_mode else [],
                actions=actions, paused=self.paused, finished=self.finished,
                hands=self._public_hand_history(after=history_after),
                history_first_hand=self._hands[0]["hand_number"] if self._hands else None,
                history_truncated=self._history_truncated,
                stopped=self._stop_requested.is_set(),
                completion_reason=self.completion_reason,
                winner=self.winner, error=self.error, hands_played=len(self.arena.hand_history),
                hand_result=(snapshot["hand_result"] if snapshot["hand_over"] and snapshot["hand_result"]
                             else self._hand_result(last_hand) if last_hand else snapshot["hand_result"]),
                holding_result=remaining > 0, hold_remaining=round(remaining, 2),
                hold_total=round(self.hold_total, 2),
                hold_hand=self.hold_hand if remaining > 0 else None,
                replay_first_seq=replay_seqs[0] if replay_seqs else None,
                replay_last_seq=replay_seqs[-1] if replay_seqs else None,
            )
            if not self.awaiting_human:
                snapshot["legal"] = None
            return snapshot

    @staticmethod
    def _hand_result(record: dict) -> dict:
        """The finished hand, reduced to what the table needs to display.

        Deduplicated by seat: a player involved in both the main pot and a side
        pot can appear more than once in the raw showdown list, and showing the
        same hand twice in the result panel just looks broken.
        """
        showdown = record.get("showdown") or []
        by_seat: dict = {}
        for entry in showdown:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            seat = entry.get("seat")
            if seat in by_seat:
                continue
            by_seat[seat] = {
                "seat": seat,
                "name": entry.get("name"),
                "cards": entry.get("hole_cards") or entry.get("cards") or [],
                "hand_name": entry.get("hand_name") or "",
            }
        return {
            "hand": record.get("hand_number") or record.get("hand"),
            "board": record.get("board") or [],
            "winners": record.get("winners") or [],
            "pot": record.get("pot") or 0,
            "reached_showdown": bool(by_seat),
            "showdown": list(by_seat.values()),
            "deltas": record.get("deltas") or {},
            "payouts": [
                {key: entry[key] for key in ("seat", "name", "amount", "reason", "pot_label", "split")
                 if key in entry}
                for entry in record.get("payouts") or [] if isinstance(entry, dict)
            ],
            "pots": [
                {key: entry[key] for key in ("amount", "eligible", "side") if key in entry}
                for entry in record.get("pots") or [] if isinstance(entry, dict)
            ],
        }


def _persona_name(player: Player) -> str:
    if not player.is_ai:
        return "human"
    try:
        return personas().get(player.persona).name
    except KeyError:
        return player.persona or "ai"


def model_label(player: Player, decider) -> str:
    """Human-readable label for the brain behind a seat."""
    if not player.is_ai:
        return "human"
    if not player.seated:
        return "empty"
    if decider is None:
        return "no brain"
    transport = getattr(decider, "transport", None)
    name = type(transport).__name__ if transport is not None else ""
    if name == "OpenAITransport":
        return getattr(transport, "model", "api")
    if name == "HeuristicTransport":
        return "local bot"
    return name or "unknown"


# --------------------------------------------------------------------------
# Session registry
# --------------------------------------------------------------------------


class SessionManager:
    """A bounded registry; explicit replacement stops the previous game."""

    def __init__(self, max_sessions: int = 4):
        self.sessions: dict[str, GameSession] = {}
        self.order: list[str] = []
        self.max_sessions = max_sessions
        self.lock = threading.RLock()

    def create(self, config: SessionConfig, *, replace_session: str | None = None) -> GameSession:
        session = GameSession(config)
        with self.lock:
            previous = self.sessions.get(replace_session) if replace_session else None
            if replace_session and previous is None:
                raise ValueError("unknown replacement session")
            if previous is not None:
                previous.stop()
            self.sessions[session.id] = session
            self.order.append(session.id)
            while len(self.order) > self.max_sessions:
                oldest = self.order.pop(0)
                evicted = self.sessions.pop(oldest, None)
                if evicted is not None:
                    evicted.stop()
            session.start()
        return session

    def get(self, session_id: str) -> GameSession | None:
        with self.lock:
            return self.sessions.get(session_id)

    def latest(self) -> GameSession | None:
        with self.lock:
            if not self.order:
                return None
            return self.sessions.get(self.order[-1])

    def list(self) -> list[dict]:
        with self.lock:
            return [
                {
                    "id": session.id,
                    "created_at": session.created_at.isoformat(timespec="seconds"),
                    "players": [
                        {
                            "seat": p.seat,
                            "name": p.name,
                            "kind": "human" if not p.is_ai else "ai",
                            "persona": p.persona if p.seated else "",
                            "provider": p.model,
                        }
                        for p in session.arena.table.players
                        if p.seated
                    ],
                    "finished": session.finished,
                    "winner": session.winner,
                    "hands_played": len(session.arena.hand_history),
                }
                for session in (self.sessions.get(sid) for sid in self.order)
                if session is not None
            ]


def available_personas() -> list[dict]:
    """The persona catalogue, ready to render in the seat editor.

    Built-in templates come first, in catalogue order, then any custom personas
    the operator has defined.
    """
    return [persona.public() for persona in personas().all()]
