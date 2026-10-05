"""Texas Hold'em table engine: blinds, betting order, all-ins, side pots, showdown.

This is a from-scratch replacement for the original ``game.py``.  The design
goals were:

* **Correct betting closure.**  Every player still in the hand must act at least
  once per street, and must act again after any full raise.  Short all-in
  "raises" do not reopen the action for players who already acted.
* **Correct side pots.**  Layered pots with per-layer eligibility, split pots on
  ties, and refunds of uncalled bets.
* **Correct blind/position handling.**  Heads-up uses the inverted blind order
  (button posts the small blind and acts first pre-flop, last post-flop).
* **No deadlocks.**  All-in run-outs, single-player-left, and 200-blind-cap
  situations all terminate.
* **Purity.**  The engine never decides *for* a player; it exposes legal actions
  and applies whatever decision it is given.  That makes AI behaviour testable
  and keeps LLM calls out of the rules code.

Seats are indices into the player list and never change, so the dealer button
and per-hand action order stay stable for the whole game.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from .cards import Card, Deck, evaluate

# --------------------------------------------------------------------------
# Enums / small value types
# --------------------------------------------------------------------------


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"

    @property
    def label(self) -> str:
        return {
            "preflop": "Pre-Flop",
            "flop": "Flop",
            "turn": "Turn",
            "river": "River",
            "showdown": "Showdown",
            "complete": "Complete",
        }[self.value]

    @property
    def board_size(self) -> int:
        return {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(self.value, 5)

    @property
    def next_street(self) -> "Street":
        order = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
        if self in order:
            index = order.index(self)
            return order[index + 1] if index + 1 < len(order) else Street.SHOWDOWN
        return Street.SHOWDOWN


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class PlayerStatus(str, Enum):
    ACTIVE = "active"
    """In the hand and able to act."""

    FOLDED = "folded"
    ALL_IN = "all_in"
    SITTING_OUT = "sitting_out"
    """Out of chips -- spectating until the game ends."""

    @property
    def in_hand(self) -> bool:
        return self in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)


@dataclass(frozen=True)
class Action:
    """A decision made by a player, normalised by the engine."""

    type: ActionType
    amount: int = 0
    """For BET/RAISE this is the *total* street commitment, not the delta."""

    thought: str = ""
    speech: str = ""
    source: str = "engine"
    """``"llm"``, ``"human"``, ``"fallback"`` or ``"engine"``."""

    def describe(self) -> str:
        if self.type is ActionType.BET:
            return f"bet {self.amount}"
        if self.type is ActionType.RAISE:
            return f"raise to {self.amount}"
        if self.type is ActionType.ALL_IN:
            return f"all-in {self.amount}"
        return self.type.value


@dataclass
class LegalActions:
    """The complete menu of legal decisions for one player."""

    seat: int
    can_fold: bool
    can_check: bool
    can_call: bool
    call_cost: int
    call_is_all_in: bool
    can_bet: bool
    can_raise: bool
    min_bet_to: int
    max_bet_to: int
    min_raise_to: int
    max_raise_to: int
    all_in_to: int | None
    """Total street commitment if the player shoves, or None if chips are 0."""

    def to_dict(self) -> dict:
        return {
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "call_cost": self.call_cost,
            "call_is_all_in": self.call_is_all_in,
            "can_bet": self.can_bet,
            "can_raise": self.can_raise,
            "min_bet_to": self.min_bet_to,
            "max_bet_to": self.max_bet_to,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
            "all_in_to": self.all_in_to,
        }

    def summary(self) -> str:
        parts = []
        if self.can_fold:
            parts.append("fold")
        if self.can_check:
            parts.append("check")
        if self.can_call:
            label = f"call {self.call_cost}"
            if self.call_is_all_in:
                label += " (all-in)"
            parts.append(label)
        if self.can_bet:
            parts.append(f"bet {self.min_bet_to}-{self.max_bet_to}")
        if self.can_raise:
            parts.append(f"raise to {self.min_raise_to}-{self.max_raise_to}")
        if self.all_in_to is not None:
            parts.append(f"all-in {self.all_in_to}")
        return ", ".join(parts)


class IllegalAction(ValueError):
    """Raised when an action cannot be applied to the current state."""


@dataclass
class Player:
    """A seat at the table."""

    name: str
    seat: int
    chips: int
    is_ai: bool = True
    persona: str = "balanced"
    model: str | None = None
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: list[Card] = field(default_factory=list)
    street_bet: int = 0
    """Chips committed on the current street."""

    hand_contribution: int = 0
    """Chips committed during the current hand (drives side pots)."""

    has_acted: bool = False

    last_action: Action | None = None
    thoughts: list[str] = field(default_factory=list)

    conversation: list[dict] = field(default_factory=list)
    """This seat's own private context, kept in full for the whole session.

    Every exchange is retained (nothing is trimmed) so the complete history can
    be inspected from a god's-eye view.  Token cost is bounded separately: the
    AI decider only *sends* the most recent turns to the model.
    """

    opening_prompt: str = ""
    """The full first-turn prompt, reused as the head of every request."""

    seated: bool = True
    """False for an empty chair -- shown at the table but dealt no cards."""

    history: list[dict] = field(default_factory=list)
    """A per-seat log of every decision this hand: action, speech, reasoning.

    Used by the UI's player detail panel and written to the hand history.
    """

    # -- bookkeeping -------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self.status is PlayerStatus.ACTIVE

    @property
    def in_hand(self) -> bool:
        return self.status.in_hand

    @property
    def is_all_in(self) -> bool:
        return self.status is PlayerStatus.ALL_IN

    @property
    def stack(self) -> int:
        """Chips still behind (not committed to the current pot)."""
        return self.chips

    def commit(self, amount: int, *, counts_toward_street: bool = True) -> int:
        """Move ``amount`` chips into the pot, capping at the remaining stack.

        ``counts_toward_street=False`` is for antes: they are dead money that
        goes in the pot but does *not* advance the current street's betting
        level, so a player who posted an ante still owes the full big blind.
        """
        if amount < 0:
            raise ValueError("Cannot commit a negative amount")
        actual = min(amount, self.chips)
        self.chips -= actual
        if counts_toward_street:
            self.street_bet += actual
        self.hand_contribution += actual
        if self.chips == 0 and self.status is PlayerStatus.ACTIVE:
            self.status = PlayerStatus.ALL_IN
        return actual

    def new_street(self) -> None:
        self.street_bet = 0
        self.has_acted = False

    def new_hand(self) -> None:
        self.hole_cards = []
        self.street_bet = 0
        self.hand_contribution = 0
        self.has_acted = False
        self.last_action = None
        self.thoughts = []
        # ``conversation`` deliberately survives across hands: a seat remembers
        # what it said and did earlier in the session, like a real player.
        # ``history`` is per-hand and resets.
        self.history = []
        if self.chips > 0 and self.seated:
            self.status = PlayerStatus.ACTIVE
        else:
            self.status = PlayerStatus.SITTING_OUT

    def to_dict(self, reveal: bool = False) -> dict:
        data = {
            "name": self.name,
            "seat": self.seat,
            "chips": self.chips,
            "status": self.status.value,
            "street_bet": self.street_bet,
            "hand_contribution": self.hand_contribution,
            "is_ai": self.is_ai,
            "persona": self.persona,
        }
        if reveal:
            data["hole_cards"] = [card.code for card in self.hole_cards]
        return data


# --------------------------------------------------------------------------
# Events -- the engine's narration, consumed by CLI/web renderers
# --------------------------------------------------------------------------


@dataclass
class Event:
    kind: str
    data: dict = field(default_factory=dict)


HAND_START = "hand_start"
ANTES_POSTED = "antes_posted"
BLINDS_POSTED = "blinds_posted"
HOLE_CARDS_DEALT = "hole_cards_dealt"
STREET_START = "street_start"
BOARD_DEALT = "board_dealt"
ACTION_TAKEN = "action_taken"
POT_UPDATED = "pot_updated"
SHOWDOWN = "showdown"
POT_AWARDED = "pot_awarded"
HAND_END = "hand_end"
PLAYER_ELIMINATED = "player_eliminated"
GAME_END = "game_end"
MESSAGE = "message"


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


class Table:
    """A single Texas Hold'em table with a fixed set of seats."""

    def __init__(
        self,
        players: Sequence[Player],
        small_blind: int = 10,
        big_blind: int = 20,
        *,
        rng: random.Random | None = None,
        seed: int | None = None,
        hand_number_offset: int = 0,
    ):
        if len(players) < 2:
            raise ValueError("A table needs at least two players")
        if small_blind <= 0 or big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if small_blind > big_blind:
            raise ValueError("Small blind cannot exceed big blind")

        self.players: list[Player] = list(players)
        for index, player in enumerate(self.players):
            player.seat = index
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.blind_level = 1
        self.ante = 0
        """Chips every player puts in before the hand, dead in the pot.

        Antes are what stop a table from folding itself to a standstill: they
        make folding cost something, so the pot is worth contesting even when
        nobody holds a premium hand.  ``set_ante`` scales them with the blinds.
        """

        self.rng = rng or random.Random(seed)
        self.deck = Deck(self.rng)

        self.button = 0
        self.hand_number = hand_number_offset
        self.street = Street.COMPLETE

        self.board: list[Card] = []
        self.current_bet = 0
        """Highest street commitment to match."""

        self.last_full_raise_to = 0
        """Size of the last *full* bet/raise, used for min-raise validation."""

        self.actor: int | None = None
        """Seat currently to act, or None between streets."""

        self.events: list[Event] = []
        self.hand_contributions: dict[int, int] = {}
        """Total chips each seat put in this hand -- drives side-pot layering."""

        self.pot: int = 0
        """Chips currently in the middle, awaiting award."""

        self.pots_snapshot: list[dict] = []
        self.showdown_results: list[dict] = []
        self.last_raiser: int | None = None

        self.hand_log: list[dict] = []
        """Every action of the *current* hand, in order.

        Reset by :meth:`start_hand`.  This is the public betting history that a
        player is entitled to -- it is what lets a seat read a three-bet pot as
        opposed to a limped one, notice who has been passive, and reason about
        what the other players are representing.
        """

    # -- helpers -----------------------------------------------------------
    def emit(self, kind: str, **data) -> None:
        self.events.append(Event(kind, data))

    def drain_events(self) -> list[Event]:
        events, self.events = self.events, []
        return events

    @property
    def seated(self) -> list[Player]:
        return [p for p in self.players if p.status is not PlayerStatus.SITTING_OUT]

    @property
    def contenders(self) -> list[Player]:
        """Players still able to win the pot."""
        return [p for p in self.players if p.in_hand]

    @property
    def actable(self) -> list[Player]:
        return [p for p in self.players if p.status is PlayerStatus.ACTIVE]

    @property
    def is_hand_over(self) -> bool:
        return self.street in (Street.SHOWDOWN, Street.COMPLETE)

    @property
    def pot_total(self) -> int:
        """Chips physically in the middle right now."""
        return self.pot

    def player_at(self, seat: int) -> Player:
        return self.players[seat]

    # -- seating / ordering -------------------------------------------------
    def seats_in_hand(self) -> list[int]:
        return [p.seat for p in self.players if p.in_hand]

    def raise_blinds(self, factor: float = 1.5, max_level: int = 12) -> bool:
        """Increase the blinds one level.  Returns True if they went up.

        Without this the game can stalemate forever: a short stack simply folds
        every hand and pays the big blind, so the remaining players trade one big
        blind per hand and nobody ever busts.  Escalating blinds put a clock on
        the game, exactly like a real tournament.

        Escalation stops once the big blind is already larger than the biggest
        remaining stack: past that point a blind commits everyone, the level
        carries no information, and the chip lead can just be handed back and
        forth forever.  The new level starts at the *next hand* (fractions round
        up so the blinds stay whole chips).
        """
        if self.blind_level >= max_level:
            return False
        stacks = [p.chips for p in self.seated if p.chips > 0]
        if stacks and self.big_blind >= max(stacks):
            return False

        self.blind_level += 1
        new_small = max(
            self.base_small_blind + 1,
            math.ceil(self.small_blind * factor),
        )
        new_big = max(new_small + 1, math.ceil(self.big_blind * factor))
        self.small_blind = new_small
        self.big_blind = new_big
        self.emit(
            "blinds_increased",
            level=self.blind_level,
            small_blind=new_small,
            big_blind=new_big,
        )
        return True

    def set_ante(self, ante: int) -> None:
        """Set the per-hand ante (scaled alongside the blinds)."""
        self.ante = max(0, int(ante))

    def _next_seat(self, start: int, predicate: Callable[[Player], bool]) -> int | None:
        """First seat at or after ``start`` (wrapping) satisfying ``predicate``."""
        count = len(self.players)
        for offset in range(count):
            seat = (start + offset) % count
            if predicate(self.players[seat]):
                return seat
        return None

    def _next_active(self, start: int) -> int | None:
        return self._next_seat(
            start, lambda p: p.status is PlayerStatus.ACTIVE
        )

    def blind_seats(self) -> tuple[int, int]:
        """Return ``(small_blind_seat, big_blind_seat)``.

        Heads-up is special: the button posts the small blind.
        """
        if len(self.seated) == 2:
            small = self.button
            big = self._require(self._next_seat((self.button + 1) % len(self.players), lambda p: p.status is not PlayerStatus.SITTING_OUT))
            return small, big
        small = self._require(
            self._next_seat((self.button + 1) % len(self.players), _can_post_blind)
        )
        big = self._require(
            self._next_seat((small + 1) % len(self.players), _can_post_blind)
        )
        return small, big

    def preflop_first_seat(self) -> int | None:
        """Seat that opens pre-flop action (after the big blind)."""
        _, big = self.blind_seats()
        return self._next_active((big + 1) % len(self.players))

    def postflop_first_seat(self) -> int | None:
        """Seat that opens post-flop action (first active left of the button)."""
        if len(self.seated) == 2:
            _, big = self.blind_seats()
            return big if self.players[big].status is PlayerStatus.ACTIVE else None
        return self._next_active((self.button + 1) % len(self.players))

    @staticmethod
    def _require(value: int | None) -> int:
        if value is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("No eligible seat found")
        return value

    # -- hand lifecycle ----------------------------------------------------
    def start_hand(self) -> None:
        """Reset state, shuffle, post blinds and deal hole cards."""
        self.hand_number += 1
        self.deck.reset()
        self.board = []
        self.current_bet = 0
        self.last_full_raise_to = 0
        self.actor = None
        self.showdown_results = []
        self.pots_snapshot = []
        self.last_raiser = None
        self.hand_contributions = {}
        self.pot = 0
        self.events = []
        self.hand_log = []

        for player in self.players:
            player.new_hand()

        for player in self.seated:
            player.hole_cards = self.deck.deal(2)

        self.emit(
            HAND_START,
            hand_number=self.hand_number,
            button=self.button,
            button_name=self.players[self.button].name,
            ante=self.ante,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            blind_level=self.blind_level,
        )
        self.emit(
            HOLE_CARDS_DEALT,
            players={
                p.seat: [c.code for c in p.hole_cards]
                for p in self.seated
            },
        )

        # Order matters: reset street state first, *then* take antes and post
        # blinds, otherwise the reset would wipe them straight back off.
        self._begin_street(Street.PREFLOP)
        self._post_antes()
        self._post_blinds()
        self.actor = self.preflop_first_seat()
        self._maybe_run_out()

    def _post_antes(self) -> None:
        """Take the ante from every seated player.  Dead money in the pot."""
        if self.ante <= 0:
            return
        paid: dict[int, int] = {}
        for player in self.seated:
            amount = player.commit(self.ante, counts_toward_street=False)
            self._record_contribution(player, amount)
            if amount:
                paid[player.seat] = amount
        if paid:
            self.emit("antes_posted", ante=self.ante, paid=paid)

    def _post_blinds(self) -> None:
        small_seat, big_seat = self.blind_seats()
        small_player = self.players[small_seat]
        big_player = self.players[big_seat]

        small_amount = small_player.commit(self.small_blind)
        big_amount = big_player.commit(self.big_blind)

        self.current_bet = max(self.current_bet, big_player.street_bet)
        self.last_full_raise_to = self.big_blind
        self._record_contribution(small_player, small_amount)
        self._record_contribution(big_player, big_amount)

        # Blinds are part of the public betting history: a seat reading the log
        # needs to see them to know the price and who is already invested.
        for player, amount, blind in (
            (small_player, small_amount, "small blind"),
            (big_player, big_amount, "big blind"),
        ):
            if amount:
                self.hand_log.append(
                    {
                        "seat": player.seat,
                        "name": player.name,
                        "street": Street.PREFLOP.value,
                        "street_label": Street.PREFLOP.label,
                        "action": "post_blind",
                        "amount": amount,
                        "described": f"posts the {blind} {amount}",
                        "street_bet": player.street_bet,
                        "pot": self.pot_total,
                        "full_raise": False,
                        "all_in": player.status is PlayerStatus.ALL_IN,
                    }
                )

        self.emit(
            BLINDS_POSTED,
            small={"seat": small_seat, "name": small_player.name, "amount": small_amount},
            big={"seat": big_seat, "name": big_player.name, "amount": big_amount},
        )

    def _record_contribution(self, player: Player, amount: int) -> None:
        if not amount:
            return
        self.hand_contributions[player.seat] = (
            self.hand_contributions.get(player.seat, 0) + amount
        )
        self.pot += amount

    def _begin_street(self, street: Street) -> None:
        """Open a street: reset betting state and reveal its board cards.

        Invariant: after this returns, ``street`` is the live street and
        ``board`` holds exactly that street's cards (0/3/4/5).
        """
        self.street = street
        for player in self.players:
            if player.status is not PlayerStatus.SITTING_OUT:
                player.new_street()

        if street is not Street.PREFLOP:
            self.current_bet = 0
            self.last_full_raise_to = self.big_blind
            self.last_raiser = None
            self._deal_next_street()
        else:
            self.current_bet = 0
            self.last_full_raise_to = self.big_blind

        self.emit(
            STREET_START,
            street=street.value,
            label=street.label,
            board=[c.code for c in self.board],
            pot=self.pot_total,
        )

    def _maybe_run_out(self) -> None:
        """Deal the board out when no further betting is possible.

        Happens when only one player is left (hand ends immediately) or when at
        most one player still has chips to act (everyone else is all-in).
        """
        if self.is_hand_over:
            return
        if len(self.contenders) <= 1:
            self._end_hand_early()
            return
        if len(self.actable) <= 1:
            self._run_out_board()

    def _run_out_board(self) -> None:
        """Deal every remaining street with no betting, then show down."""
        while len(self.board) < 5 and not self.is_hand_over:
            self._advance_street()
        self.to_showdown()

    def _advance_street(self) -> None:
        """Move play to the next street (dealing its board cards)."""
        self._begin_street(self.street.next_street)

    def _deal_next_street(self) -> None:
        """Deal enough cards to fill the current street's board."""
        target = self.street.board_size
        if len(self.board) >= target:
            return
        self.deck.burn(1)
        new_cards = self.deck.deal(target - len(self.board))
        self.board.extend(new_cards)
        self.emit(
            BOARD_DEALT,
            street=self.street.value,
            cards=[c.code for c in new_cards],
            board=[c.code for c in self.board],
        )

    # -- legal actions ------------------------------------------------------
    def legal_actions(self, seat: int) -> LegalActions:
        """Everything ``seat`` may legally do right now."""
        player = self.players[seat]
        if self.actor != seat or player.status is not PlayerStatus.ACTIVE:
            raise IllegalAction(
                f"{player.name} cannot act right now (actor={self.actor}, status={player.status.value})"
            )

        to_call = max(0, self.current_bet - player.street_bet)
        call_cost = min(to_call, player.chips)
        can_call = to_call > 0 and player.chips > 0
        can_check = to_call == 0

        raise_allowed = player.chips > call_cost
        min_raise_to = max(self.current_bet + self.last_full_raise_to, self.big_blind)
        max_raise_to = player.street_bet + player.chips
        can_raise = raise_allowed and max_raise_to > self.current_bet
        # A raise that cannot reach the minimum is still permitted as an all-in
        # shove, but it does not reopen the betting.
        min_raise_to = min(min_raise_to, max_raise_to)

        can_bet = self.current_bet == 0 and can_raise

        return LegalActions(
            seat=seat,
            can_fold=True,
            can_check=can_check,
            can_call=can_call,
            call_cost=call_cost,
            call_is_all_in=can_call and call_cost >= player.chips,
            can_bet=can_bet,
            can_raise=can_raise,
            min_bet_to=self.big_blind if can_bet else 0,
            max_bet_to=max_raise_to if can_bet else 0,
            min_raise_to=min_raise_to if can_raise else 0,
            max_raise_to=max_raise_to if can_raise else 0,
            all_in_to=max_raise_to if player.chips > 0 else None,
        )

    def normalize_action(self, seat: int, action: Action) -> Action:
        """Validate and canonicalise a decision, raising :class:`IllegalAction`."""
        legal = self.legal_actions(seat)
        player = self.players[seat]
        kind = action.type

        if kind is ActionType.FOLD:
            return Action(ActionType.FOLD, thought=action.thought, speech=action.speech, source=action.source)

        if kind is ActionType.CHECK:
            if not legal.can_check:
                raise IllegalAction(
                    f"Cannot check: {legal.call_cost} to call (use 'call')"
                )
            return Action(ActionType.CHECK, thought=action.thought, speech=action.speech, source=action.source)

        if kind is ActionType.CALL:
            if not legal.can_call:
                if legal.can_check:
                    raise IllegalAction("Nothing to call; use 'check'")
                raise IllegalAction("Cannot call with no chips")
            return Action(
                ActionType.CALL,
                amount=legal.call_cost,
                thought=action.thought,
                speech=action.speech,
                source=action.source,
            )

        if kind is ActionType.ALL_IN:
            if legal.all_in_to is None:
                raise IllegalAction("No chips left to shove")
            return Action(
                ActionType.ALL_IN,
                amount=legal.all_in_to,
                thought=action.thought,
                speech=action.speech,
                source=action.source,
            )

        if kind in (ActionType.BET, ActionType.RAISE):
            target = action.amount
            if target is None or target <= 0:
                raise IllegalAction("A bet/raise needs a positive target amount")
            if target > legal.max_raise_to:
                raise IllegalAction(
                    f"Cannot raise to {target}: that is more than your total "
                    f"commitment of {legal.max_raise_to}"
                )
            if target == legal.max_raise_to:
                # Committing the whole stack is simply an all-in.
                return Action(
                    ActionType.ALL_IN,
                    amount=legal.max_raise_to,
                    thought=action.thought,
                    speech=action.speech,
                    source=action.source,
                )
            if target <= self.current_bet:
                raise IllegalAction(
                    f"Raise target {target} must exceed the current bet "
                    f"{self.current_bet}; use 'call' to match"
                )
            if target < legal.min_raise_to:
                # Below the legal minimum and not the whole stack: reject.
                # (A sub-minimum all-in takes the `target == max` branch above
                # and is always permitted.)
                raise IllegalAction(
                    f"Minimum raise is to {legal.min_raise_to} "
                    f"(current bet {self.current_bet} + last raise "
                    f"{self.last_full_raise_to}); to commit everything, use all-in"
                )
            kind = ActionType.BET if self.current_bet == 0 else ActionType.RAISE
            return Action(
                kind,
                amount=target,
                thought=action.thought,
                speech=action.speech,
                source=action.source,
            )

        raise IllegalAction(f"Unknown action type {kind!r}")

    # -- applying actions ---------------------------------------------------
    def apply_action(self, seat: int, action: Action) -> Action:
        """Apply a (validated) action and advance the table state.

        Action closure rule
        -------------------
        A player still owes an action while ``street_bet < current_bet`` or while
        they have not acted yet on this street.  A full raise hands everybody
        else a fresh action; a *short* all-in raise does not, which falls out of
        the same rule for free -- it only pulls in players who were already
        behind the new amount.
        """
        applied = self.normalize_action(seat, action)
        player = self.players[seat]
        previous_bet = self.current_bet
        was_full_raise = False

        if applied.type is ActionType.FOLD:
            player.status = PlayerStatus.FOLDED
        elif applied.type is ActionType.CHECK:
            pass
        elif applied.type is ActionType.CALL:
            committed = player.commit(applied.amount)
            self._record_contribution(player, committed)
        else:  # BET / RAISE / ALL_IN
            target = applied.amount
            committed = player.commit(target - player.street_bet)
            self._record_contribution(player, committed)
            if player.street_bet > previous_bet:
                raise_size = player.street_bet - previous_bet
                self.current_bet = player.street_bet
                was_full_raise = raise_size >= self.last_full_raise_to
                if was_full_raise:
                    self.last_full_raise_to = raise_size
                    self.last_raiser = seat
                    # A full raise reopens the action for everyone else.
                    for other in self.players:
                        if other.seat != seat and other.status is PlayerStatus.ACTIVE:
                            other.has_acted = False

        player.has_acted = True
        player.last_action = applied

        # The ordered record of this hand, which is what makes the betting
        # readable: "who raised, who called, in what order".  ``last_action``
        # only remembers the most recent action per player, so a model given
        # nothing else cannot tell a limped pot from a three-bet pot.
        self.hand_log.append(
            {
                "seat": seat,
                "name": player.name,
                "street": self.street.value,
                "street_label": self.street.label,
                "action": applied.type.value,
                "amount": applied.amount,
                "described": applied.describe(),
                "street_bet": player.street_bet,
                "pot": self.pot_total,
                "full_raise": was_full_raise,
                "all_in": player.status is PlayerStatus.ALL_IN,
            }
        )

        self.emit(
            ACTION_TAKEN,
            seat=seat,
            name=player.name,
            action=applied.type.value,
            amount=applied.amount,
            street_bet=player.street_bet,
            pot=self.pot_total,
            chips=player.chips,
            thought=applied.thought,
            speech=applied.speech,
            source=applied.source,
            full_raise=was_full_raise,
        )

        if len(self.contenders) <= 1:
            self._end_hand_early()
            return applied

        self._advance_after_action(seat)
        return applied

    def _advance_after_action(self, seat: int) -> None:
        """Find the next actor, or close the street when betting is done."""
        if self._street_betting_complete():
            self._close_street()
            return
        nxt = self._next_actor((seat + 1) % len(self.players))
        if nxt is None:
            self._close_street()
            return
        self.actor = nxt

    def _street_betting_complete(self) -> bool:
        """True when nobody is left who owes chips or has an unspent action."""
        for player in self.players:
            if player.status is not PlayerStatus.ACTIVE:
                continue
            if player.street_bet < self.current_bet or not player.has_acted:
                return False
        return True

    def _next_actor(self, start: int) -> int | None:
        count = len(self.players)
        for offset in range(count):
            seat = (start + offset) % count
            player = self.players[seat]
            if player.status is not PlayerStatus.ACTIVE:
                continue
            if not player.has_acted or player.street_bet < self.current_bet:
                return seat
        return None

    def _close_street(self) -> None:
        """Finish the current street and move to the next one."""
        self.actor = None
        if len(self.contenders) <= 1:
            self._end_hand_early()
            return
        if self.street is Street.RIVER:
            self.to_showdown()
            return

        self._advance_street()

        if self.is_hand_over:
            return
        if len(self.contenders) <= 1:
            self._end_hand_early()
            return
        if len(self.actable) <= 1:
            # Everyone else is all-in; run the rest of the board out.
            self.to_showdown()
            return

        self.actor = self.postflop_first_seat()

    def _end_hand_early(self) -> None:
        """Everyone folded but one: award the pot without a showdown."""
        winner = self.contenders[0]
        total = self.pot
        # Record the pot before zeroing it: the hand history reports the size of
        # the pot that was actually won, and without this an uncontested hand
        # looked like it was played for nothing.
        self.pots_snapshot = [{"amount": total, "eligible": [winner.seat], "side": False}]
        self.showdown_results = []
        self.pot = 0
        self.emit(
            POT_AWARDED,
            seat=winner.seat,
            name=winner.name,
            amount=total,
            reason="all_folded",
            hand=None,
        )
        winner.chips += total
        self.emit(HAND_END, reason="all_folded", winner=winner.seat, pot=total)
        self.street = Street.COMPLETE
        self.actor = None

    # -- showdown -----------------------------------------------------------
    def to_showdown(self) -> None:
        self.actor = None
        self.street = Street.SHOWDOWN
        live = self.contenders
        scores: dict[int, int] = {}
        revealed = []
        for player in live:
            result = evaluate(player.hole_cards + self.board)
            scores[player.seat] = result.score
            revealed.append(
                {
                    "seat": player.seat,
                    "name": player.name,
                    "hole_cards": [c.code for c in player.hole_cards],
                    "best_five": None,
                    "hand_name": result.name,
                    "score": result.score,
                }
            )
        self.emit(SHOWDOWN, players=revealed, board=[c.code for c in self.board])
        self._settle_pots(scores, revealed)

    def _settle_pots(self, scores: dict[int, int], revealed: list[dict]) -> None:
        from .pot import settle

        pot_before = self.pot
        folded = {p.seat for p in self.players if p.status is PlayerStatus.FOLDED}
        result = settle(
            dict(self.hand_contributions),
            folded,
            scores,
            button_seat=self.button,
            seat_order=[p.seat for p in self.players],
        )

        self.pots_snapshot = [
            {"amount": pot.amount, "eligible": sorted(pot.eligible), "side": pot.is_side_pot}
            for pot in result.pots
        ]

        for refund in result.refunds:
            player = self.players[refund.seat]
            player.chips += refund.amount
            self.emit(
                POT_AWARDED,
                seat=refund.seat,
                name=player.name,
                amount=refund.amount,
                reason="uncalled_bet_returned",
                pot_index=-1,
            )

        for payout in result.payouts:
            player = self.players[payout.seat]
            player.chips += payout.amount
            label = result.pots[payout.pot_index].name if payout.pot_index >= 0 else "pot"
            self.emit(
                POT_AWARDED,
                seat=payout.seat,
                name=player.name,
                amount=payout.amount,
                reason="showdown",
                pot_index=payout.pot_index,
                pot_label=label,
                split=payout.split,
                hand=next(
                    (r["hand_name"] for r in revealed if r["seat"] == payout.seat), None
                ),
            )

        self.showdown_results = revealed
        # The middle is now empty; pot accounting lives on in hand_contributions
        # for the hand history, but must not be counted as live chips again.
        self.pot = 0
        self.emit(
            HAND_END,
            reason="showdown",
            pot=pot_before,
            results=revealed,
        )
        self.street = Street.COMPLETE

    # -- game progression ---------------------------------------------------
    def rotate_button(self) -> None:
        """Move the button to the next seated player."""
        nxt = self._next_seat(
            (self.button + 1) % len(self.players),
            lambda p: p.status is not PlayerStatus.SITTING_OUT,
        )
        if nxt is not None:
            self.button = nxt

    def eliminate_broke_players(self) -> list[Player]:
        """Mark chip-less players as sitting out and report them."""
        eliminated = []
        for player in self.players:
            if player.chips <= 0 and player.status is not PlayerStatus.SITTING_OUT:
                player.status = PlayerStatus.SITTING_OUT
                eliminated.append(player)
                self.emit(PLAYER_ELIMINATED, seat=player.seat, name=player.name)
        return eliminated

    def is_game_over(self) -> bool:
        return len(self.seated) <= 1


def _can_post_blind(player: Player) -> bool:
    return player.status is not PlayerStatus.SITTING_OUT


def play_hand(table: Table, decide: Callable[[Table, int], Action]) -> None:
    """Play one hand to completion using ``decide`` as the decision source.

    ``decide`` receives the table and the acting seat and must return an
    :class:`Action`.  Any :class:`IllegalAction` it raises propagates: callers
    own the retry/fallback policy, which keeps the rules layer pure.
    """
    table.start_hand()
    guard = 0
    max_steps = 500
    while not table.is_hand_over:
        guard += 1
        if guard > max_steps:  # pragma: no cover - defensive
            raise RuntimeError("Hand did not terminate; engine bug")
        if table.actor is None:
            # Engine should always drive to a terminal state; this is a guard.
            break
        action = decide(table, table.actor)
        table.apply_action(table.actor, action)
