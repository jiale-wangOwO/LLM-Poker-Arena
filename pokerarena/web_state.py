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
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .ai import Action
from .arena import Arena, AISpec, heuristic_transports
from .config import ArenaConfig
from .engine import (
    ACTION_TAKEN,
    BLINDS_POSTED,
    BOARD_DEALT,
    HAND_END,
    HAND_START,
    MESSAGE,
    PLAYER_ELIMINATED,
    POT_AWARDED,
    SHOWDOWN,
    Street,
    STREET_START,
    ActionType,
    Event,
    IllegalAction,
    Player,
    Table,
)
from .personas import PERSONAS, default_lineup, get_persona, persona_store
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

    hand_result_seconds: float = 4.5
    """Base time the finished hand stays on screen before the next is dealt.

    Scaled by what there is to read (see ``PlaybackControl.between_hands``): a
    three-way showdown on the river gets roughly 1.6x this, a won-without-showdown
    hand about a second.  It is only the *overlay* that is timed -- the previous
    hand also stays available in the corner until the next one ends, so a longer
    pause would just be dead time.  Set to 0 to deal straight on.
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

        The length is scaled by how much there is to read rather than being a
        flat number of seconds: a four-way showdown that went to the river needs
        far longer than a hand where everyone folded pre-flop, and a fixed pause
        is either too short to read the first or a pointless wait for the second.
        The wait is always interruptible -- the browser can skip it, and the UI
        shows a countdown so the pause is never a mystery.
        """
        session = self.session
        base = session.config.hand_result_seconds
        if base <= 0:
            return

        weight = self._result_weight(table)
        hold = base * weight
        if weight <= 0.5:
            # Nothing was shown and there is no board: a brief acknowledgement,
            # never a full pause.  A flat 2s wait for "everyone folded" is the
            # kind of pause that makes a table feel broken.
            hold = min(hold, 1.1)
        session.hold_until = time.time() + hold
        session.hold_total = hold
        deadline = session.hold_until
        while time.time() < deadline:
            with session.lock:
                if session.finished or session.skip_hold:
                    session.skip_hold = False
                    break
            time.sleep(0.03)
        session.hold_until = 0.0
        session.hold_total = 0.0

    @staticmethod
    def _result_weight(table: Table) -> float:
        """How long this result deserves, as a multiple of the base hold.

        Driven by what actually has to be read: the number of hands shown, how
        many board cards there are, and whether the pot was contested.  A
        pre-flop fold-out scores near zero and only gets a brief acknowledgement.
        """
        showdown = table.showdown_results or []
        revealed = len({entry.get("seat") for entry in showdown if isinstance(entry, dict)})
        if not revealed:
            # Nobody showed: either everyone folded, or one player took it down.
            return 0.35

        weight = 0.55
        weight += 0.18 * revealed          # one line to read per revealed hand
        weight += 0.06 * len(table.board)  # a full board is more to take in
        if len(table.pots_snapshot) > 1:
            weight += 0.15                 # a split pot needs explaining
        return min(weight, 1.9)

    # -- numbering ---------------------------------------------------------
    def record_action(self, table: Table, seat: int, action: Action) -> None:
        session = self.session
        with session.lock:
            session.seq += 1
            player = table.players[seat]
            session.actions.append(
                {
                    "seq": session.seq,
                    "hand": table.hand_number,
                    "street": table.street.value,
                    "street_label": table.street.label,
                    "seat": seat,
                    "name": player.name,
                    "action": action.type.value,
                    "action_label": action.describe(),
                    "amount": action.amount,
                    "pot": table.pot_total,
                    "source": action.source,
                    "speech": action.speech,
                    # Inline reasoning for the action log.
                    "thought": " ".join((action.thought or "").split())[:400],
                }
            )
            del session.actions[:-MAX_ACTIONS]
            # Everything the engine emits from here belongs to this action.
            session._action_index = len(session.actions) - 1


MAX_ACTIONS = 4000
"""Upper bound on the replay log (a full game is a few hundred actions)."""

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
        self.pending_action: queue.Queue[Action] = queue.Queue(maxsize=4)
        self.awaiting_human = False
        self.winner: str | None = None
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

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_at = datetime.now()
        self._thread = threading.Thread(target=self._run, name=f"game-{self.id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.winner = self.arena.play_game()
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.finished = True
                self.awaiting_human = False

    # -- human input -------------------------------------------------------
    def _human_decider(self, table: Table, seat: int) -> Action:
        """Block the game thread until the browser submits a legal action."""
        with self.lock:
            self.awaiting_human = True
        # No hard deadline while paused: a spectator may study the spot as long
        # as they like, and the session's own watchdog is the pause control.
        deadline = time.time() + 300
        while True:
            try:
                action = self.pending_action.get(timeout=0.4)
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
                self.awaiting_human = False
            return action
        with self.lock:
            self.awaiting_human = False
        # Nobody answered: keep the game alive with a legal non-committal move.
        from .ai import safe_fallback

        return safe_fallback(table, seat, table.legal_actions(seat))

    def submit_action(self, raw: str) -> dict:
        """Validate and queue a human action.  Returns a status dict.

        The action is normalised through the engine *before* it is queued, so a
        hand-crafted HTTP request cannot smuggle an illegal move into the game.
        """
        if self.arena.human_seat is None:
            raise IllegalAction("This table has no human seat")
        table = self.arena.table
        seat = self.arena.human_seat
        with self.lock:
            if not self.awaiting_human:
                raise IllegalAction("It is not your turn")
            if table.actor != seat:
                raise IllegalAction("It is not your turn")
            legal = table.legal_actions(seat)

        # Raises IllegalAction on anything the rules forbid.  This is the same
        # gate the AI path uses.
        validated = table.normalize_action(seat, self._parse(raw, legal))

        try:
            self.pending_action.put_nowait(validated)
        except queue.Full:  # pragma: no cover - defensive
            raise IllegalAction("Another action is already queued")
        return {"queued": True, "action": validated.describe()}

    @staticmethod
    def _parse(raw: str, legal: "LegalActions") -> Action:
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

        * god mode shows everything, and a finished hand is public anyway;
        * otherwise a *seated human* sees their own hand and nothing else;
        * a pure spectator (no human seat at all) sees nothing, because there is
          no hand that is "theirs".  Treating every non-AI seat as "yours" is how
          an all-AI table ended up broadcasting every hand in the clear.
        """
        if self.god_mode() or hand_over or self.finished:
            return True
        human = self.arena.human_seat
        return human is not None and seat == human

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
                persona = get_persona(player.persona).public()
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
        reasoning_visible = reveal or (self.arena.human_seat is not None
                                       and seat == self.arena.human_seat)

        with self.lock:
            # Every decision this seat has made, newest last, across the game.
            # The ``seq`` is looked up from the decision log so the panel can
            # show the same number as the action rail.
            history = [
                {**entry, "seq": self._seq_for(entry)}
                for entry in self.arena.transcript
                if entry.get("seat") == seat
            ]
            # Stats over the whole game for this seat.
            stats = self._seat_stats(seat)
            conversation_size = len(player.conversation)
            memory_turns = getattr(decider, "memory_turns", 0)
            model = model_label(player, decider)
            messages_sent = len(decider._sent_messages(player)) if player.opening_prompt else 1
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
            "hand_strength": strength_note,
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
            "seq": len(self.actions),
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
                self.event_log.append(entry)
                del self.event_log[:-MAX_EVENT_LOG]
                if thought:
                    self.thoughts.append(
                        {
                            "seat": entry.get("seat"),
                            "name": entry.get("name"),
                            "hand": table.hand_number,
                            "street": table.street.value,
                            "action": entry.get("action"),
                            "speech": entry.get("speech", ""),
                            "thought": thought,
                            "source": entry.get("source", ""),
                            "model": self._model_for_seat(entry.get("seat")),
                        }
                    )
                    del self.thoughts[:-MAX_THOUGHTS]
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
        if kind == BLINDS_POSTED:
            return {"kind": kind, "small": data["small"], "big": data["big"]}
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

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict:
        table = self.arena.table
        with self.lock:
            awaiting = self.awaiting_human
            finished = self.finished
            winner = self.winner
            error = self.error
            paused = self.paused
            events = list(self.event_log[-900:])
            thoughts = list(self.thoughts[-MAX_THOUGHTS:])
            actions = list(self.actions[-MAX_ACTIONS:])

        legal: dict | None = None
        if (
            self.arena.human_seat is not None
            and table.actor == self.arena.human_seat
            and not table.is_hand_over
        ):
            try:
                legal = table.legal_actions(self.arena.human_seat).to_dict()
            except IllegalAction:  # pragma: no cover - race guard
                legal = None

        players = []
        # God mode: a spectator sees every hole card.  Display only -- the
        # prompts sent to the models never contain another seat's cards, so the
        # players still play honestly either way.
        god_mode = self.god_mode()
        reveal_all = god_mode or table.is_hand_over or finished
        for player in table.players:
            reveal = reveal_all or (
                self.arena.human_seat is not None
                and player.seat == self.arena.human_seat
            )
            decider = self.arena.deciders.get(player.seat)
            players.append(
                {
                    "name": player.name,
                    "seat": player.seat,
                    "chips": player.chips,
                    "status": player.status.value,
                    "seated": player.seated,
                    "street_bet": player.street_bet,
                    "is_ai": player.is_ai,
                    "persona": player.persona,
                    "persona_name": _persona_name(player),
                    "model": model_label(player, decider),
                    "provider": player.model,
                    # Hidden unless god mode says otherwise (or the hand is over,
                    # which makes every card public anyway).
                    "hole_cards": (
                        [c.code for c in player.hole_cards]
                        if (reveal and player.seated and player.hole_cards)
                        else None
                    ),
                    "last_action": (
                        player.last_action.describe() if player.last_action else None
                    ),
                }
            )

        try:
            small, big = table.blind_seats()
        except Exception:  # pragma: no cover - defensive
            small, big = None, None

        # The result of the hand that just finished, so the browser can show it
        # during the hold instead of jumping straight to a fresh table.
        with self.lock:
            last_hand = self.arena.hand_history[-1] if self.arena.hand_history else None
            remaining = max(0.0, self.hold_until - time.time())
            holding = remaining > 0
            hold_total = self.hold_total

        return {
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
            "awaiting_human": awaiting,
            "human_seat": self.arena.human_seat,
            "legal": legal,
            "events": events,
            # Reasoning is part of god mode: with it off, the action rail shows
            # what happened but not why.
            "thoughts": thoughts if god_mode else [],
            "actions": actions if god_mode else [
                {**action, "thought": ""} for action in actions
            ],
            "paused": paused,
            "finished": finished,
            "winner": winner,
            "error": error,
            "hands_played": len(self.arena.hand_history),
            "hand_cap": self.config.max_hands,
            "showdown": table.showdown_results,
            "pots": table.pots_snapshot,
            "reveal_all": god_mode,
            "god_mode": god_mode,
            # Result of the hand that just finished (shown during the hold).
            "hand_result": self._hand_result(last_hand) if last_hand else None,
            "holding_result": holding,
            "hold_remaining": round(remaining, 2),
            "hold_total": round(hold_total, 2),
        }

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
        }


def _persona_name(player: Player) -> str:
    if not player.is_ai:
        return "human"
    try:
        return get_persona(player.persona).name
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
    """Keeps at most one active session, so the server cannot be made to spin."""

    def __init__(self, max_sessions: int = 4):
        self.sessions: dict[str, GameSession] = {}
        self.order: list[str] = []
        self.max_sessions = max_sessions
        self.lock = threading.RLock()

    def create(self, config: SessionConfig) -> GameSession:
        session = GameSession(config)
        with self.lock:
            self.sessions[session.id] = session
            self.order.append(session.id)
            while len(self.order) > self.max_sessions:
                oldest = self.order.pop(0)
                self.sessions.pop(oldest, None)
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
