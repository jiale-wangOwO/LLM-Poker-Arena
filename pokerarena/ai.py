"""LLM decision layer.

Design: the engine owns legality, the model owns *character*.

The model is told the exact menu of legal actions and must reply with a
structured block::

    <action>raise 250</action>
    <say>Your turn to sweat.</say>
    <thought>He limped twice already, so his range is capped...</thought>

That reply is parsed into an :class:`~pokerarena.engine.Action`.  If the model
returns something illegal or unparseable, the layer retries with the exact
rejection reason appended to the conversation, and finally falls back to a
safe action derived from the legal menu.  A model can therefore never deadlock
the game or make a move the rules forbid.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .cards import Card, format_cards
from .engine import (
    Action,
    ActionType,
    IllegalAction,
    LegalActions,
    Player,
    PlayerStatus,
    Table,
)
from .personas import Persona
from .strength import strength

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_SAY_RE = re.compile(r"<say>\s*(.*?)\s*</say>", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"<thought>\s*(.*?)\s*</thought>", re.IGNORECASE | re.DOTALL)

# Loose fallbacks for models that ignore the requested format.
_LOOSE_ACTION_RE = re.compile(
    r"action\s*[:\-]\s*(.+)", re.IGNORECASE
)


class ParseError(ValueError):
    """The model's reply could not be turned into an action."""


@dataclass
class LLMReply:
    action_text: str
    say: str = ""
    thought: str = ""
    raw: str = ""


def parse_reply(text: str) -> LLMReply:
    """Extract the action / table talk / reasoning from a model reply."""
    if not text or not text.strip():
        raise ParseError("Empty response")

    action_match = _ACTION_RE.search(text)
    if action_match:
        action_text = action_match.group(1).strip()
    else:
        loose = _LOOSE_ACTION_RE.search(text)
        if not loose:
            # Last resort: the whole reply, stripped of markdown noise.
            candidate = text.strip().splitlines()[-1]
            action_text = candidate.strip()
        else:
            action_text = loose.group(1).strip()

    # Keep the first line, then strip markdown emphasis but *not* digits or
    # thousands separators, so "1,250" survives intact.
    first_line = action_text.splitlines()[0].strip() if action_text else ""
    cleaned = first_line.strip("`*_ ").strip()
    cleaned = cleaned.rstrip(".!").strip()
    if not cleaned:
        raise ParseError(f"Could not find an action in the reply: {text!r}")

    say = _SAY_RE.search(text)
    thought = _THOUGHT_RE.search(text)
    return LLMReply(
        action_text=cleaned,
        say=_clean(say.group(1)) if say else "",
        thought=_clean(thought.group(1)) if thought else "",
        raw=text,
    )


def _clean(text: str) -> str:
    return " ".join(text.split())[:400]


def parse_action_text(text: str) -> tuple[ActionType, int | None]:
    """Turn free text like ``"raise to 240"`` into an action type and amount."""
    lowered = text.strip().lower()

    # Pull the amount out *before* normalising whitespace, so thousands
    # separators ("raise to 1,250") survive.
    amount = _extract_amount(lowered)

    # Normalise whitespace, but never collapse digits together.
    lowered = re.sub(r"\s+", " ", lowered).strip()
    lowered = lowered.replace("all in", "all-in").replace("allin", "all-in")

    if lowered.startswith("fold") or lowered in {"f", "x"}:
        return ActionType.FOLD, None
    if lowered.startswith("check") or lowered in {"k", "pass"}:
        return ActionType.CHECK, None
    if "all-in" in lowered or "all in" in lowered or lowered in {"shove", "jam", "push"}:
        return ActionType.ALL_IN, None
    if lowered.startswith("call") or lowered in {"c"}:
        return ActionType.CALL, None
    if lowered.startswith(("raise", "bet", "r")) and amount is not None:
        return ActionType.RAISE, amount
    if lowered.startswith(("raise", "bet")):
        raise ParseError(f"A raise/bet needs an amount: {text!r}")
    if amount is not None:
        # A bare number means "make it this much".
        return ActionType.RAISE, amount
    raise ParseError(f"Unrecognised action: {text!r}")


_NUMBER_RE = re.compile(r"(\d[\d,_]*)\s*(k\b)?")
_TO_NUMBER_RE = re.compile(r"\bto\s+(\d[\d,_]*)\s*(k\b)?")


def _extract_amount(text: str) -> int | None:
    """Find the amount in a phrase, honouring 'raise to X' and the 'k' suffix."""
    # Prefer the number after 'to', which is the total commitment.
    match = _TO_NUMBER_RE.search(text) or _NUMBER_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(",", "").replace("_", "")
    try:
        value = int(raw)
    except ValueError:  # pragma: no cover - regex guards this
        return None
    if match.group(2):  # 'k' suffix
        value *= 1000
    return value


def reply_to_action(reply: LLMReply) -> Action:
    """Convert a parsed reply into an engine action (not yet validated).

    Table talk that gives the hand away is dropped here, so it never reaches the
    table: the other seats read ``<say>``, and a leaked holding would hand them a
    read nobody earned (including a false one, which is just as corrosive).
    """
    kind, amount = parse_action_text(reply.action_text)
    speech, _dropped = sanitize_speech(reply.say)
    return Action(
        kind,
        amount=amount or 0,
        thought=reply.thought,
        speech=speech,
        source="llm",
    )


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
SYSTEM_TEMPLATE = """You are {name}, a Texas Hold'em player at a live table.

WHO YOU ARE
{style}

YOUR VOICE
{voice}

HOW TO REPLY
Reply with EXACTLY these three tags and nothing else. No markdown, no code fences.

<action>YOUR ACTION HERE</action>
<say>one short line of table talk, or leave empty</say>
<thought>your private reasoning, 1-3 sentences</thought>

YOUR ACTION MUST BE ONE OF THE LEGAL ACTIONS LISTED IN THE PROMPT.
Available action formats:
  fold
  check
  call
  raise <TOTAL_AMOUNT>   (the TOTAL you are making your street commitment, not the increment)
  all-in

Do not invent action names. Do not raise to an amount outside the allowed range.

YOUR CARDS ARE SECRET
<say> and your actions are the only things the table can see. <thought> is private.
Never state, hint at, or act out what you are holding. That is the one rule of
table talk that even the loudest player at a real table keeps, and breaking it
ruins the hand for everyone.

Specifically, never say any of these in <say>:
  - the strength or rank of your hand ("two pair", "top pair", "I have aces",
    "a flush draw", "second nuts", "this is a monster")
  - the name of any card, whether or not you hold it ("the ace of spades",
    "that king"), or any card code such as "Ah" or "Kd"
  - a plain claim about where you stand, such as "I'm ahead", "I'm beat",
    "you have me", "I'm value betting", "I'm bluffing"
  - what you intend to do next, or what the other players are holding

Your cards belong in <thought>, where you should reason about them freely.

Do talk. Taunt, joke, needle, stall, compliment, complain about your luck,
comment on the pot, the board texture, the other players' *behaviour* -- all of
that is fair game and is what makes you a personality. Talk is how you build the
image you want; it does not have to be honest, but it must never be about your
actual holding."""


# --------------------------------------------------------------------------
# Card-secrecy filter
# --------------------------------------------------------------------------
# Even with the rule spelled out, a model occasionally narrates its hand.  That
# leaks private information into a public channel, so a reply that does it has
# its table talk dropped rather than shown: saying nothing is always safe, while
# reacting to revealed cards would hand the other seats a read nobody earned.
_CARD_RANK = (
    r"(?:ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|"
    r"deuce|trey|t\d{1,2}|[akqjt2-9])"
)
#: The same, allowing the spoken plural ("aces", "kings") but not a bare "s".
#: Written out rather than ``s?`` because ``\b`` after a singular fails on the
#: plural form, which is the common way players actually say it.
#: Spoken card ranks only.  A bare digit is deliberately excluded: "I have 3"
#: is far more likely to be pot odds ("I have 3 to 1") than a card, and treating
#: it as a rank produced a false positive that swallowed honest talk.
_CARD_RANK_SPOKEN = (
    r"(?:ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|"
    r"deuce|trey|10|[akqjt])"
)
_CARD_RANK_PLURAL = (
    r"(?:aces|kings|queens|jacks|tens|nines|eights|sevens|sixes|fives|fours|"
    r"threes|twos|deuces|treys|" + _CARD_RANK_SPOKEN + r")"
)
_HAND_RANK = (
    r"(?:royal flush|straight flush|quads|four of a kind|full house|flush|"
    r"straight|trips|set of|three of a kind|two pair|overpair|top pair|"
    r"middle pair|bottom pair|second pair|pocket pair|pair of|nut flush|"
    r"the nuts|second nuts|third nuts|ace high|king high|"
    r"flush draw|straight draw|gutshot|open[- ]ended|combo draw|backdoor)"
)
#: Ordinary words that double as rank names.  A pair-combo pattern containing
#: one of these is far more likely to be prose than a hand, so it is skipped.
_COMMON_WORD_RANK = r"(?:ten|tens|nine|nines|eight|eights|seven|sevens|six|sixes|five|fives|four|fours|three|threes|two|twos|deuce|deuces|trey|treys)"

#: One rank written compactly: "A", "K", "10", "7".
_RANK_SHORT = r"(?:10|[2-9akqjt])"
#: A compact rank pair: "Q5o", "AKs", "T9o".  ``\b`` cannot be used after a rank
#: because ``Q5`` is all word characters, so the boundary is expressed with
#: lookarounds instead.
_RANK_PAIR_SHORTHAND = (
    r"(?<![a-z0-9])" + _RANK_SHORT + _RANK_SHORT + r"[os]?(?![a-z0-9])"
)
#: "7-2", "10-4": two digits joined.  This cannot match odds, because those are
#: written "3 to 1" and never "3-1" in this context.
_DIGIT_PAIR = r"(?<!\d)(?:10|[2-9])\s*[-\u2013\u2014/]\s*(?:10|[2-9])(?!\d)"

#: Patterns that are unambiguous on their own, so they apply to even the
#: shortest line.  "Q5o" is three characters long and still names a hand.
_LEAK_PATTERNS_ALWAYS = [
    # "pocket queens", "pocket fives"
    re.compile(rf"\bpocket\s+{_CARD_RANK_PLURAL}\b", re.IGNORECASE),
    # Naming the two cards names the hand, whether or not the speaker still holds
    # it: "seven-deuce", "queen-nine offsuit", "7-2", "Q5o", "AK suited".
    #
    # A hyphen, dash or slash between two ranks is unambiguous, so it always
    # matches.  A space is riskier -- "two pair" and "three of a kind" are
    # ordinary prose -- so a spaced pair only matches when the *second* word is
    # not also a common English word.
    re.compile(
        rf"\b{_CARD_RANK_PLURAL}\s*[-\u2013\u2014/]\s*{_CARD_RANK_PLURAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_CARD_RANK_PLURAL}\s+(?!{_COMMON_WORD_RANK}\b){_CARD_RANK_PLURAL}\b",
        re.IGNORECASE,
    ),
    re.compile(_RANK_PAIR_SHORTHAND, re.IGNORECASE),
    # "7-2", "10-4" -- two digits joined are a hand, never pot odds.
    re.compile(_DIGIT_PAIR, re.IGNORECASE),
    re.compile(rf"\b{_CARD_RANK_PLURAL}\s+(?:suited|offsuit)\b", re.IGNORECASE),
    # "the ace of spades", "king of hearts"
    re.compile(
        rf"\b{_CARD_RANK_PLURAL}\s+of\s+(?:spades|hearts|diamonds|clubs)\b",
        re.IGNORECASE,
    ),
    # "Ah", "Kd", "10c" written as a card code
    re.compile(r"\b(?:10|[2-9akqjt])[shdc]\b", re.IGNORECASE),
]

#: Patterns that need a little context, because their keywords also occur in
#: ordinary speech ("a flush of anger", "I have 3 to 1 odds").
_LEAK_PATTERNS_CONTEXT = [
    # "two pair", "a flush draw", "the nuts", "quads"
    re.compile(rf"\b{_HAND_RANK}\b", re.IGNORECASE),
    # "my ace", "with my kings", "sitting on my queen" -- possessive phrasing
    # claims ownership of a card, which "the king on the board" does not.
    re.compile(
        r"\b(?:my|with\s+my|sitting\s+on\s+my|sittin'?\s+on\s+my)\s+"
        + _CARD_RANK_PLURAL
        + r"\b",
        re.IGNORECASE,
    ),
    # "I have aces", "I've got kings", "holding aces", "I got this king"
    re.compile(
        r"\b(?:i\s+have|i'?ve\s+got|i\s+got|i\s+hold|holding|i\s+flopped|"
        r"i\s+turned|i\s+rivered|i\s+made)\b"
        r"(?:\s+(?:a|an|the|this|that|my|two|pocket))?"
        r"\s+(?:" + _CARD_RANK_PLURAL + r"|pocket)\b",
        re.IGNORECASE,
    ),
    # "I'm ahead", "I'm beat", "I'm value betting", "I'm bluffing"
    re.compile(
        r"\bi(?:'m| am)\s+(?:ahead|behind|beat|good|dead|drawing|"
        r"value[- ]betting|bluffing|just calling|priced in)\b",
        re.IGNORECASE,
    ),
    # "you have me", "you've got me beat", "that's my card"
    re.compile(
        r"\b(?:you(?:'ve| have)?\s+(?:got me|have me|beat me)|"
        r"that(?:'s| is) my card)\b",
        re.IGNORECASE,
    ),
]

_LEAK_PATTERNS = _LEAK_PATTERNS_ALWAYS + _LEAK_PATTERNS_CONTEXT

MIN_SPEECH_FOR_FILTER = 10
"""Below this length only the unambiguous patterns are consulted, so a terse
"Q5o" is still caught while "Good luck" and "gg" are not second-guessed."""


def leaks_hand_information(speech: str) -> str | None:
    """Return the matched phrase when table talk gives a hand away, else None."""
    text = (speech or "").strip()
    if not text:
        return None
    patterns = _LEAK_PATTERNS_ALWAYS
    if len(text) >= MIN_SPEECH_FOR_FILTER:
        patterns = _LEAK_PATTERNS
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def sanitize_speech(speech: str) -> tuple[str, str]:
    """Drop table talk that reveals a hand.

    Returns ``(speech, dropped_reason)``; the reason is empty when nothing was
    removed.  Saying nothing is always safe, so the leaked line is discarded
    rather than rewritten -- a canned replacement could contradict the play.
    """
    leaked = leaks_hand_information(speech)
    if leaked is None:
        return speech, ""
    return "", leaked


def build_system_prompt(persona: Persona) -> str:
    return SYSTEM_TEMPLATE.format(
        name=persona.name, style=persona.style, voice=persona.voice
    )


def _hand_strength_note(player: Player, table: Table) -> str:
    """A calibrated strength anchor, including a percentile and equity.

    The model is free to disagree, but a percentile keeps its reasoning (and the
    offline bot) anchored to real hand values instead of vibes.

    Equity is reported separately from the percentile because they answer
    different questions, and confusing them causes real blunders.  The
    percentile ranks the *made* hand; equity is the chance of winning against a
    plausible range, which is what pot odds are compared against.  An ace-high
    gutshot is a weak made hand and still has well over half the equity, so
    folding it when the price is tiny would be a mistake.
    """
    if not player.hole_cards:
        return ""
    from .strength import describe_strength, equity

    note = f" -- {describe_strength(player.hole_cards, table.board)}"

    opponents = max(1, len([p for p in table.contenders if p.seat != player.seat]))
    if len(player.hole_cards) == 2:
        # One equity number is a fair comparison only heads-up.  Against a field
        # it overstates the hand badly, so say so rather than let the model
        # assume it applies.
        facing_raise = table.current_bet > table.big_blind
        share = equity(player.hole_cards, table.board, raising=facing_raise)
        against = "a raising range" if facing_raise else "a typical range"
        label = "equity heads-up" if opponents <= 1 else "equity vs one opponent"
        note += f", {share:.0%} {label} ({against})"
        if opponents > 1:
            note += f" -- {opponents} opponents, so win far less often than that"
    return note


def _position_note(table: Table, seat: int) -> str:
    """Where this seat sits relative to the button and the blinds.

    Position is not decoration: the same hand plays differently from the button
    than from the big blind, and a model that is not told where it is cannot
    reason about who acts behind it.
    """
    try:
        small_seat, big_seat = table.blind_seats()
    except Exception:  # pragma: no cover - defensive
        return ""
    order = [
        table.seated[(table.seated.index(table.players[seat]) + step) % len(table.seated)]
        for step in range(1, len(table.seated))
    ] if table.players[seat] in table.seated and table.seated else []
    behind = [p.name for p in order if p.in_hand]

    if seat == table.button:
        label = "the button (dealer)"
    elif seat == small_seat:
        label = "the small blind"
    elif seat == big_seat:
        label = "the big blind"
    else:
        label = f"{len(behind)} seat(s) off the button"
    if behind:
        return f"{label} -- {len(behind)} player(s) act after you: {', '.join(behind)}"
    return f"{label} -- you act LAST on every street"


def format_hand_log(table: Table, viewer: int) -> list[str]:
    """The current hand's betting, street by street, from the public log.

    This is the single biggest thing a seat needs and the old prompt lacked: a
    three-bet pot and a limped pot produce identical *state* but completely
    different reads.
    """
    if not table.hand_log:
        return ["  (no betting yet -- you are first to act)"]

    lines: list[str] = []
    current_street = None
    for entry in table.hand_log:
        if entry["street"] != current_street:
            current_street = entry["street"]
            lines.append(f"  -- {entry['street_label']} (pot {entry['pot']}) --")
        marker = " (you)" if entry["seat"] == viewer else ""
        if entry["action"] == "post_blind":
            detail = entry["described"]
        elif entry["action"] == "fold":
            detail = "folds"
        elif entry["action"] == "check":
            detail = "checks"
        elif entry["action"] == "call":
            detail = f"calls {entry['amount']}"
        elif entry["action"] == "bet":
            detail = f"bets {entry['amount']}"
        elif entry["action"] == "raise":
            raised = "raises" if entry["full_raise"] else "raises (short)"
            detail = f"{raised} to {entry['amount']}"
        elif entry["action"] == "all_in":
            detail = f"is ALL-IN for {entry['amount']}"
        else:  # pragma: no cover - future actions
            detail = entry["described"]
        lines.append(f"  - {entry['name']}{marker} {detail}")
    return lines


def _stack_notes(table: Table, seat: int) -> list[str]:
    """Effective stacks: how much is actually at risk against each opponent."""
    player = table.players[seat]
    others = [p for p in table.players if p.seat != seat and p.in_hand]
    if not others:
        return []
    lines = []
    for other in others:
        effective = min(player.chips, other.chips)
        lines.append(
            f"  - {other.name}: {other.chips} behind, effective stack vs you "
            f"{effective} ({effective / max(1, table.big_blind):.0f} BB)"
        )
    return lines


def _spr_note(table: Table, seat: int) -> str:
    """Stack-to-pot ratio, the standard way to judge commitment."""
    player = table.players[seat]
    others = [p for p in table.players if p.seat != seat and p.in_hand]
    if not others or table.pot_total <= 0:
        return ""
    effective = min([player.chips, *(o.chips for o in others)])
    return f"SPR {effective / table.pot_total:.1f} (effective stack / pot)"


def describe_table(
    table: Table,
    seat: int,
    *,
    recent_hands: list[dict] | None = None,
) -> str:
    """Build the user prompt: the whole public picture, plus the legal menu.

    A poker decision is not a function of "my cards and the pot".  It is a
    function of position, stack depths, what everybody has done so far on every
    street, and what has happened over the last few hands.  All of that is
    public information and all of it is included here.
    """
    player = table.players[seat]
    legal = table.legal_actions(seat)
    lines: list[str] = []

    lines.append(f"HAND #{table.hand_number}  STREET: {table.street.label}")
    lines.append(
        f"Blinds {table.small_blind}/{table.big_blind}"
        + (f", ante {table.ante}" if table.ante else "")
        + f"   (level {table.blind_level})"
    )
    lines.append(f"Your seat: {player.name}   Your stack: {player.chips}")
    lines.append(f"Your position: {_position_note(table, seat)}")
    lines.append(f"Pot: {table.pot_total}" + (f"   {_spr_note(table, seat)}" if _spr_note(table, seat) else ""))
    lines.append(
        f"Board: {format_cards(table.board) if table.board else '(none yet)'}"
    )
    lines.append(
        f"Your hole cards: {format_cards(player.hole_cards)}"
        f"{_hand_strength_note(player, table)}"
    )

    to_call = max(0, table.current_bet - player.street_bet)
    lines.append(
        f"Highest bet on this street: {table.current_bet}   "
        f"Already in from you: {player.street_bet}   "
        f"To call: {to_call}"
    )
    lines.append(f"Minimum full raise would be to: {table.last_full_raise_to + table.current_bet}")

    lines.append("")
    lines.append("BETTING SO FAR THIS HAND (in order):")
    lines.extend(format_hand_log(table, seat))

    lines.append("")
    lines.append("THE PLAYERS:")
    for other in table.players:
        if other.seat == seat:
            continue
        status = {
            PlayerStatus.ACTIVE: "still in the hand",
            PlayerStatus.ALL_IN: "ALL-IN",
            PlayerStatus.FOLDED: "folded this hand",
            PlayerStatus.SITTING_OUT: "busted out",
        }[other.status]
        persona = f" [{other.persona}]" if other.is_ai and other.persona else ""
        invested = table.hand_contributions.get(other.seat, 0)
        notes = [f"{other.chips} chips", f"{invested} in this hand"]
        if other.street_bet:
            notes.append(f"{other.street_bet} on this street")
        lines.append(f"  - {other.name}{persona}: {', '.join(notes)} -- {status}")

    stacks = _stack_notes(table, seat)
    if stacks:
        lines.append("")
        lines.append("EFFECTIVE STACKS (what is actually at risk):")
        lines.extend(stacks)

    if recent_hands:
        lines.append("")
        lines.append("RECENT HANDS AT THIS TABLE (newest first):")
        lines.extend(recent_hands)

    lines.append("")
    lines.append("YOUR LEGAL ACTIONS (choose exactly one):")
    for option in _legal_options(table, seat, legal):
        lines.append(f"  * {option}")

    lines.append("")
    lines.append(
        "Think about what the betting so far tells you about the other players' "
        "ranges, and about what your own line looks like to them -- not just "
        "about your own two cards."
    )
    lines.append(
        "Reply now with <action>, <say> and <thought>. Keep <say> under 20 words."
    )
    return "\n".join(lines)


def _legal_options(table: Table, seat: int, legal: LegalActions) -> list[str]:
    options: list[str] = []
    if legal.can_fold:
        options.append("fold")
    if legal.can_check:
        options.append("check")
    if legal.can_call:
        if legal.call_is_all_in:
            options.append(f"call  (this puts you all-in for {legal.call_cost})")
        else:
            options.append(f"call  (costs {legal.call_cost})")
    if legal.can_bet:
        options.append(
            f"raise <amount>  (bet total {legal.min_bet_to} to {legal.max_bet_to}, "
            "the TOTAL for this street)"
        )
    elif legal.can_raise:
        options.append(
            f"raise <amount>  (total {legal.min_raise_to} to {legal.max_raise_to}, "
            "the TOTAL for this street)"
        )
    if legal.all_in_to is not None:
        options.append(f"all-in  (total commitment {legal.all_in_to})")
    return options


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------
@dataclass
class ChatResult:
    """What a transport actually returned.

    ``reasoning`` holds a reasoning model's hidden chain-of-thought (e.g.
    DeepSeek's ``reasoning_content``) so it can be shown in the UI and reused as
    the ``<thought>`` when the model omits that tag.
    """

    content: str
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict | None = None

    @property
    def text(self) -> str:
        return self.content


class ChatTransport(Protocol):
    """Anything that can turn messages into a model reply."""

    def complete(
        self, system: str, messages: list[dict], *, temperature: float
    ) -> ChatResult:
        ...


@dataclass
class OpenAITransport:
    """OpenAI-compatible chat-completions transport.

    Handles both plain chat models and reasoning models.  For the latter the
    answer lands in ``content`` while the chain-of-thought lands in
    ``reasoning_content``; both are surfaced.  A reasoning model that runs out
    of token budget mid-thought returns empty ``content`` with
    ``finish_reason='length'``, which is reported as a clear error instead of
    being silently treated as an empty reply.
    """

    model: str
    base_url: str
    api_key: str
    max_tokens: int = 2500
    timeout: float = 120.0
    #: Called with the API's reported ``usage`` dict after each successful call,
    #: so a caller can total up token spend.  Defaults to doing nothing.
    on_usage: Callable[[dict], None] | None = None

    def complete(
        self, system: str, messages: list[dict], *, temperature: float
    ) -> ChatResult:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        # Reasoning models expose the chain-of-thought separately.
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        finish_reason = getattr(choice, "finish_reason", None)
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
            if self.on_usage is not None:
                self.on_usage(usage)

        if not content.strip():
            if finish_reason == "length":
                raise ParseError(
                    "Model used its whole token budget on reasoning and returned "
                    f"no answer (reasoning was {len(reasoning)} chars). "
                    "Increase max_tokens for this model."
                )
            if reasoning.strip():
                # No answer but we do have reasoning: let the parser try, since
                # some models put a usable action inside the reasoning text.
                return ChatResult(
                    content=reasoning,
                    reasoning=reasoning,
                    finish_reason=finish_reason,
                    usage=usage,
                )
            raise ParseError(
                f"Model returned no content (finish_reason={finish_reason!r})"
            )

        return ChatResult(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
        )


# --------------------------------------------------------------------------
# The decider
# --------------------------------------------------------------------------
@dataclass
class DecisionTrace:
    """Everything that happened on one AI decision -- for logs and the UI."""

    seat: int
    name: str
    persona: str
    model: str
    street: str
    prompt: str = ""
    attempts: list[dict] = field(default_factory=list)
    action: Action | None = None
    fell_back: bool = False
    error: str | None = None
    reasoning: str = ""
    """The model's hidden chain-of-thought, when the endpoint exposes one."""

    usage: dict | None = None

    @property
    def thought(self) -> str:
        return self.action.thought if self.action else ""

    @property
    def speech(self) -> str:
        return self.action.speech if self.action else ""


def safe_fallback(table: Table, seat: int, legal: LegalActions) -> Action:
    """A legal, non-committal action used when the model misbehaves.

    Checking is free, so it is always preferred.  Facing a bet we call only if
    it is cheap relative to the pot, otherwise we fold rather than bleed chips.
    """
    if legal.can_check:
        return Action(ActionType.CHECK, source="fallback", speech="")
    if legal.can_call:
        pot_odds_cost = legal.call_cost
        if pot_odds_cost <= max(table.big_blind, table.pot_total // 4):
            return Action(ActionType.CALL, amount=legal.call_cost, source="fallback")
    return Action(ActionType.FOLD, source="fallback")


def legal_options_text(table: Table, seat: int) -> str:
    """The bullet list of legal actions, shared by the prompt and retries."""
    return "\n".join(
        f"  - {option}" for option in _legal_options(table, seat, table.legal_actions(seat))
    )


def compact_state_line(
    *,
    hand: int,
    street: str,
    pot: int,
    board: list[Card],
    hole: list[Card],
) -> str:
    """One line describing the situation, for a seat's running context."""
    board_text = format_cards(board) if board else "(preflop)"
    return (
        f"[Hand {hand} / {street}]  pot {pot} | "
        f"board: {board_text} | your cards: {format_cards(hole)}"
    )


def compact_turn_record(
    player_name: str,
    action: Action,
    *,
    hand: int,
    street: str,
    pot: int,
    board: list[Card],
    hole: list[Card],
    intervening: list[dict] | None = None,
) -> str:
    """A record of one decision plus everything that happened around it.

    The record used to hold only the seat's own action, which meant a seat had
    no way to remember *what the others did* between its turns -- so it could
    never notice that the same opponent had raised three hands running.  The
    ``intervening`` actions (in order, since this seat last acted) restore that.
    """
    lines = [
        compact_state_line(
            hand=hand, street=street, pot=pot, board=board, hole=hole
        ),
    ]
    if intervening:
        lines.append("while you waited:")
        for entry in intervening:
            lines.append(f"  {_describe_log_entry(entry)}")
    lines.append(f"your action was: {action.describe()}")
    if action.speech:
        lines.append(f'you said: "{action.speech}"')
    thought = " ".join((action.thought or "").split())
    if thought:
        lines.append(f"you reasoned: {thought[:600]}")
    return "\n".join(lines)


def _describe_log_entry(entry: dict) -> str:
    """One human-readable line for a table :attr:`~Table.hand_log` entry."""
    name = entry.get("name", "?")
    action = entry.get("action")
    amount = entry.get("amount", 0)
    if action == "post_blind":
        return f"{name} {entry.get('described', 'posts a blind')}"
    if action == "fold":
        return f"{name} folds"
    if action == "check":
        return f"{name} checks"
    if action == "call":
        return f"{name} calls {amount}"
    if action == "bet":
        return f"{name} bets {amount}"
    if action == "raise":
        qualifier = "" if entry.get("full_raise", True) else " (short)"
        return f"{name} raises{qualifier} to {amount}"
    if action == "all_in":
        return f"{name} is ALL-IN for {amount}"
    return f"{name} {entry.get('described', action)}"


def recent_hand_summaries(hand_history: list[dict], limit: int = 4) -> list[str]:
    """The last few completed hands, as the table would remember them.

    Real players carry exactly this: who won, how big it was, and who showed
    what.  Without it every hand starts from zero and nobody can have a read.
    """
    lines: list[str] = []
    for record in reversed(hand_history[-limit:]):
        hand = record.get("hand_number") or record.get("hand")
        winners = record.get("winners") or []
        if not winners and record.get("winner"):
            winners = [record["winner"]]
        pot = record.get("pot") or 0
        who = ", ".join(str(w) for w in winners) if winners else "nobody"
        detail = f"  - Hand {hand}: {who} won {pot}" if winners else f"  - Hand {hand}: no flop"
        shown = record.get("showdown") or record.get("results") or []
        reveals = []
        for entry in shown:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            cards = entry.get("cards") or entry.get("hole_cards") or ""
            if isinstance(cards, list):
                cards = " ".join(str(c) for c in cards)
            name = entry.get("hand_name") or entry.get("hand") or ""
            reveals.append(f"{entry['name']} showed {cards} ({name})".replace("  ", " "))
        if reveals:
            detail += f"  ({'; '.join(reveals)})"
        lines.append(detail)
    return lines


class AIDecider:
    """Turns a persona + model + transport into legal engine actions.

    Each decider owns exactly one seat, so its :attr:`conversation` list is that
    player's private context.  The thread looks like::

        user:      <full table state + legal menu>
        assistant: <the model's own action / say / thought>
        user:      <compact record of the next situation>

    i.e. the first turn of the session carries the full picture, and everything
    after it is a running conversation.  The model can therefore refer back to
    what it said earlier, hold a grudge, or follow through on a plan.
    """

    def __init__(
        self,
        persona: Persona,
        transport: ChatTransport,
        *,
        model_label: str = "unknown",
        retries: int = 2,
        rng: random.Random | None = None,
        transcript: list[dict] | None = None,
        memory_turns: int = 4,
    ):
        self.persona = persona
        self.transport = transport
        self.model_label = model_label
        self.retries = max(0, retries)
        self.memory_turns = max(0, memory_turns)
        self.rng = rng or random.Random()
        self.system_prompt = build_system_prompt(persona)
        self.transcript = transcript if transcript is not None else []
        self.traces: list[DecisionTrace] = []
        self.hand_history: list[dict] = []
        """Completed hands, newest last -- what the table remembers.

        Shared with the arena (which appends to it) so a seat can see who has
        been winning and what has been shown down.
        """
        self._log_cursor = 0
        """How much of the current hand's log this seat has already been told."""

    # -- conversation context ---------------------------------------------
    def _build_messages(self, table: Table, seat: int) -> list[dict]:
        """Assemble the messages for the current decision.

        Every decision sends the **full** public picture -- position, stacks,
        the whole betting sequence of this hand, effective stacks, and the last
        few hands -- not a compact "pot / board / my cards" line.  A poker
        decision depends on the betting that produced the situation, and a seat
        that is only told the current state is reduced to a card-strength
        calculator.  Cost is deliberately not the constraint here.
        """
        player = table.players[seat]
        recent = recent_hand_summaries(self.hand_history)
        context = describe_table(table, seat, recent_hands=recent)
        if not player.conversation:
            player.opening_prompt = context
            return [{"role": "user", "content": context}]
        return [
            *self._sent_messages(player),
            {"role": "user", "content": context + "\n\nWhat do you do?"},
        ]

    def _remember(self, table: Table, seat: int, reply_text: str, action: Action) -> None:
        """Append this exchange to the seat's private context.

        The record is kept **in full** -- nothing is discarded -- so the whole
        session can be inspected later as a god's-eye view and scrubbed through.
        Token cost is bounded separately, in :meth:`_build_messages`, which only
        *sends* the most recent turns.
        """
        player = table.players[seat]
        player.conversation.append(
            {
                "role": "assistant",
                "content": reply_text,
                "action": action.describe(),
                "hand": table.hand_number,
                "street": table.street.label,
                "seq": len(player.history) + 1,
            }
        )
        player.conversation.append(
            {
                "role": "user",
                "content": compact_turn_record(
                    player.name,
                    action,
                    hand=table.hand_number,
                    street=table.street.label,
                    pot=table.pot_total,
                    board=table.board,
                    hole=player.hole_cards,
                    intervening=self._new_log_entries(table),
                ),
                "hand": table.hand_number,
                "street": table.street.label,
                "seq": len(player.history) + 1,
            }
        )

    def _new_log_entries(self, table: Table) -> list[dict]:
        """The table's actions since this seat was last consulted.

        Walking a cursor rather than reading the whole log keeps each record to
        "what happened while you waited", which is how a player actually
        experiences a hand.
        """
        log = getattr(table, "hand_log", [])
        start = min(self._log_cursor, len(log))
        entries = list(log[start:])
        self._log_cursor = len(log)
        return entries

    def _sent_messages(self, player: Player) -> list[dict]:
        """The slice of a seat's record actually sent to the model.

        The full record is always retained for display; this bounds what is
        charged to the prompt.  The cap is deliberately loose (50 turns by
        default): the compact records are a fraction of a full prompt, so a long
        memory is cheap relative to the value of continuity.
        """
        if not player.conversation:
            return []
        turn = max(0, self.memory_turns) * 2
        recent = player.conversation[-turn:] if turn else []
        messages = []
        if player.opening_prompt:
            messages.append({"role": "user", "content": player.opening_prompt})
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in recent
        )
        return messages

    # -- main entry point --------------------------------------------------
    def decide(self, table: Table, seat: int) -> Action:
        player = table.players[seat]
        legal = table.legal_actions(seat)
        trace = DecisionTrace(
            seat=seat,
            name=player.name,
            persona=self.persona.key,
            model=self.model_label,
            street=table.street.value,
        )
        self.traces.append(trace)

        base_prompt = describe_table(table, seat)
        trace.prompt = base_prompt
        # This seat's private thread: full prompt on the first turn of the
        # session, then a running conversation it remembers across hands.
        messages: list[dict] = self._build_messages(table, seat)

        last_error: str | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self.transport.complete(
                    self.system_prompt,
                    messages,
                    temperature=self.persona.temperature,
                )
            except Exception as exc:  # transport failure: retry, then fall back
                last_error = f"{type(exc).__name__}: {exc}"
                trace.attempts.append({"attempt": attempt, "error": last_error})
                # A transport error will not fix itself by rephrasing; back off.
                break

            raw = result.content
            hidden_reasoning = result.reasoning or ""
            if result.usage:
                trace.usage = result.usage

            try:
                reply = parse_reply(raw)
                action = reply_to_action(reply)
                applied = table.normalize_action(seat, action)
            except (ParseError, IllegalAction) as exc:
                last_error = str(exc)
                trace.attempts.append(
                    {"attempt": attempt, "raw": raw, "error": last_error}
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That was not legal: {last_error}\n"
                            f"Choose from exactly these:\n"
                            f"{legal_options_text(table, seat)}\n"
                            "Reply again with <action>, <say> and <thought>."
                        ),
                    }
                )
                continue

            # Keep the model's hidden reasoning even if it forgot the tag, so the
            # "why" is never lost for a reasoning model.
            if not applied.thought and hidden_reasoning:
                applied = Action(
                    applied.type,
                    amount=applied.amount,
                    thought=hidden_reasoning,
                    speech=applied.speech,
                    source=applied.source,
                )
            trace.reasoning = hidden_reasoning
            trace.attempts.append({"attempt": attempt, "raw": raw, "ok": True})
            trace.action = applied
            # Remember this for the player's table presence.
            if applied.speech:
                player.thoughts.append(applied.speech)
            # Per-seat history (for the UI's player panel and the hand history).
            player.history.append(
                {
                    "hand": table.hand_number,
                    "street": table.street.value,
                    "street_label": table.street.label,
                    "action": applied.describe(),
                    "action_type": applied.type.value,
                    "amount": applied.amount,
                    "speech": applied.speech,
                    "thought": applied.thought,
                    "reasoning": hidden_reasoning,
                    "source": applied.source,
                    "pot": table.pot_total,
                    "model": self.model_label,
                    "persona": self.persona.key,
                }
            )
            if applied.thought or applied.speech:
                self.transcript.append(
                    {
                        "hand": table.hand_number,
                        "street": table.street.value,
                        "street_label": table.street.label,
                        "seat": seat,
                        "name": player.name,
                        "persona": self.persona.key,
                        "thought": applied.thought,
                        "speech": applied.speech,
                        "action": applied.describe(),
                        "action_type": applied.type.value,
                        "amount": applied.amount,
                        "pot": table.pot_total,
                        "model": self.model_label,
                        "source": applied.source,
                    }
                )
            # Extend this seat's private context for its next decision.
            self._remember(table, seat, raw, applied)
            return applied

        # Every attempt failed: keep the game moving with a legal action.
        fallback = safe_fallback(table, seat, legal)
        fallback = Action(
            fallback.type,
            amount=fallback.amount,
            source="fallback",
            thought=f"(model fallback after error: {last_error})",
        )
        trace.fell_back = True
        trace.error = last_error
        trace.action = fallback
        player.history.append(
            {
                "hand": table.hand_number,
                "street": table.street.value,
                "street_label": table.street.label,
                "action": fallback.describe(),
                "action_type": fallback.type.value,
                "amount": fallback.amount,
                "speech": "",
                "thought": fallback.thought,
                "reasoning": "",
                "source": "fallback",
                "pot": table.pot_total,
                "model": self.model_label,
                "persona": self.persona.key,
                "error": last_error,
            }
        )
        return fallback


# --------------------------------------------------------------------------
# Scripted transport for tests and offline demos
# --------------------------------------------------------------------------
@dataclass
class ScriptedTransport:
    """Returns canned replies; used by the test suite and ``--demo``."""

    replies: list[str]
    index: int = 0
    on_exhausted: str = "<action>check</action><say></say><thought>nothing to do</thought>"
    reasoning: str = ""

    def complete(
        self, system: str, messages: list[dict], *, temperature: float
    ) -> ChatResult:
        if self.index < len(self.replies):
            reply = self.replies[self.index]
            self.index += 1
            return ChatResult(content=reply, reasoning=self.reasoning)
        return ChatResult(content=self.on_exhausted, reasoning=self.reasoning)


@dataclass
class HeuristicTransport:
    """A deterministic, persona-flavoured bot used for offline simulation.

    This is *not* the LLM path.  It exists so the engine can be exercised
    end-to-end without a network, so ``--demo`` always works, and so the test
    suite has a principled reference opponent.

    It plays percentile poker: :mod:`pokerarena.strength` maps the hand to a
    0..1 percentile, and the persona supplies the entry threshold, raise
    threshold and aggression.  That makes the personas genuinely different
    opponents rather than noise.
    """

    persona: Persona
    rng: random.Random = field(default_factory=random.Random)

    # -- ranges ------------------------------------------------------------
    # Landmarks measured from the real combo distribution of EQUITY_ORDER via
    # tools/calibrate_ranges.py, which maps a strength percentile to the share
    # of all 1326 starting combos that clear it:
    #     VPIP  8% -> 0.90 | 12% -> 0.86 | 18% -> 0.79 | 25% -> 0.72
    #     VPIP 35% -> 0.62 | 50% -> 0.46 | 65% -> 0.32 | 80% -> 0.17
    _VPIP_LANDMARKS: tuple[tuple[float, float], ...] = (
        (0.05, 0.94),
        (0.12, 0.86),
        (0.18, 0.79),
        (0.25, 0.72),
        (0.35, 0.62),
        (0.50, 0.46),
        (0.65, 0.32),
        (0.80, 0.17),
        (0.95, 0.02),
    )

    #: The table size the thresholds above were calibrated for.
    _FULL_RING = 6

    @property
    def target_vpip(self) -> float:
        """Share of hands this persona is willing to enter a pot with."""
        return max(0.05, min(0.95, 1.0 - self.persona.tightness))

    @property
    def _entry_threshold(self) -> float:
        """Minimum hand percentile this persona will put chips in with."""
        points = self._VPIP_LANDMARKS
        target = self.target_vpip
        if target <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if target <= x1:
                span = x1 - x0
                t = 0.0 if span == 0 else (target - x0) / span
                return y0 + (y1 - y0) * t
        return points[-1][1]

    @property
    def _raise_threshold(self) -> float:
        """Percentile above which the persona prefers raising to calling."""
        return min(0.99, self._entry_threshold + 0.10 + (1 - self.persona.aggression) * 0.10)

    def _shorthanded_entry(self, opponents: int) -> float:
        """Widen the entry range as the table shortens.

        A range calibrated for a full ring is far too tight three-handed: a nit
        waiting for the top 10% of hands simply folds every hand forever, which
        both plays badly and stalls the game.  Real short-handed play opens up
        roughly with the inverse of the field size.
        """
        base = self._entry_threshold
        if opponents >= self._FULL_RING:
            return base
        # At 3-handed the field is half the size, so the same *relative* hand
        # strength sits at a lower percentile.
        scale = max(0.45, opponents / self._FULL_RING)
        floor = 0.12  # never demand better than "any two decent cards"
        return max(floor, base * scale)

    def complete(
        self, system: str, messages: list[dict], *, temperature: float
    ) -> ChatResult:
        prompt = messages[-1]["content"] if messages else ""
        legal = _parse_legal_block(prompt)
        pct = _parse_percentile(prompt)
        share = _parse_equity(prompt)
        pot = _parse_int(prompt, r"Pot: (\d+)") or 0
        to_call = _parse_int(prompt, r"To call: (\d+)") or 0
        # The highest bet on this street: equals the big blind pre-flop, and is
        # raised as soon as anybody bets or raises.
        current_bet = _parse_int(prompt, r"Highest bet on this street: (\d+)") or 0
        blind = re.search(r"Blinds:?\s*(\d+)/(\d+)", prompt)
        big_blind = int(blind.group(2)) if blind else 0
        if big_blind <= 0:
            big_blind = current_bet or max(1, to_call)
        if current_bet <= 0:
            current_bet = big_blind

        # If no raise appears in the menu, strip raise data so we cannot attempt
        # one the engine would reject.
        if not legal.get("can_raise") and not legal.get("can_bet"):
            for key in ("min_raise_to", "max_raise_to", "min_bet_to", "max_bet_to"):
                legal.pop(key, None)

        style = self.persona
        # Ranges tighten as the field grows and widen as it shrinks.
        opponents = _parse_opponents(prompt)
        entry = self._shorthanded_entry(opponents)
        # Keep the raise threshold a consistent step above the (possibly
        # widened) entry threshold.
        raise_at = min(0.97, entry + 0.12 + (1 - style.aggression) * 0.10)
        can_raise = bool(legal.get("can_raise") or legal.get("can_bet"))
        # Facing a raise: someone has already put in more than the big blind.
        facing_raise = current_bet > big_blind
        # Aggression raises the frequency, not just the size.  Squaring makes
        # the archetypes separate cleanly: a passive station rarely raises, an
        # aggressive maniac raises almost every time it has a hand.
        raise_freq = 0.10 + (style.aggression**2) * 0.75
        # Bluffing: occasionally apply pressure with a hand that should not.
        bluffing = can_raise and self.rng.random() < style.bluff_frequency * 0.3

        text: str
        if legal.get("can_check"):
            # Nothing to call: either take a free card, bet for value, or bluff.
            if can_raise and (pct >= raise_at or bluffing):
                text = self._sized_raise(legal, pct, prompt)
            else:
                text = "check"
        elif legal.get("can_call"):
            call_cost = int(legal.get("call_cost") or to_call)
            all_in_to = int(legal.get("all_in_to") or 0)
            price = call_cost / max(1, pot + call_cost)
            tolerance = 0.6 - style.tightness * 0.2 + style.aggression * 0.1
            good_price = pct >= price * tolerance
            committing = all_in_to > 0 and call_cost >= 0.5 * all_in_to

            # Pot odds, done properly.  The percentile above ranks the *made*
            # hand, which is the wrong yardstick for a call: a gutshot with two
            # overcards is a weak made hand and still wins often enough to call
            # when the price is small.  Compare real equity against the
            # break-even share instead, with a margin so the bot is not calling
            # purely to break even.
            #
            # The reported equity is against a single opponent, which badly
            # overstates a hand in a multiway pot -- the same 54% that is a clear
            # call heads-up wins far less against four callers.  Raising it to
            # the power of the field size is the standard approximation for
            # beating several independent hands.
            effective_share = share
            if share is not None and opponents > 1:
                effective_share = share ** opponents
            equity_call = (
                effective_share is not None
                and call_cost > 0
                and effective_share >= price * 1.15
            )

            # Facing a raise, a hand should not have to clear the *opening*
            # range: the price on offer sets the bar for continuing.  Without
            # this every seat folds to any raise and almost no hand reaches a
            # flop (measured: 3% of full-ring hands saw a flop).
            defending_allowed = None
            if facing_raise and not can_raise:
                # Already has chips in (blinds) and is closing the action.
                defending_allowed = max(price, entry - 0.35)
            elif facing_raise:
                defending_allowed = max(price, entry - 0.22)
            elif not can_raise:
                # Nobody can raise behind, so a cheap call is always fine.
                defending_allowed = max(price, entry - 0.25)
            if defending_allowed is not None:
                defending_allowed = min(defending_allowed, 0.92)

            # Short-stack poker: fold equity disappears once you are committed.
            behind_after_call = max(0, all_in_to - call_cost)
            committed_short = (
                call_cost > 0
                and behind_after_call < max(call_cost, 4 * big_blind)
                and pct >= max(0.3, price)
            )

            entry_needed = (
                defending_allowed if defending_allowed is not None else entry
            )

            if pct < entry_needed and not committing and not committed_short:
                # Below what the price allows: only a cheap speculative call, a
                # genuinely +EV price, or a deliberate bluff raise keeps them in.
                if equity_call:
                    text = "call"
                elif bluffing and call_cost <= max(20, pot // 3):
                    text = "call"
                else:
                    text = "fold"
            elif (
                pct >= raise_at
                and can_raise
                and self.rng.random() < raise_freq
            ) or (committed_short and can_raise):
                text = self._sized_raise(legal, pct, prompt)
            elif good_price or committing or committed_short or equity_call or pct >= 0.85:
                text = "call"
            else:
                text = "fold"
        else:
            text = "fold"

        reply = (
            f"<action>{text}</action>"
            f"<say></say>"
            f"<thought>heuristic {style.key}: hand ~{int(pct * 100)}th pct "
            f"(entry {int(entry * 100)}, raise {int(raise_at * 100)}) -> {text}</thought>"
        )
        return ChatResult(content=reply)

    def _sized_raise(self, legal: dict, pct: float, prompt: str = "") -> str:
        """Choose a raise size relative to the *pot*, never a slice of the stack.

        Sizing between the legal minimum and the whole stack is how a bot ends up
        jamming 716 into a 70-chip pot, which destroys the game.  A raise here
        means "make it this much in total", built from the current bet and the
        pot, then clamped into the legal window.
        """
        lo = int(legal.get("min_raise_to") or legal.get("min_bet_to") or 0)
        cap = int(legal.get("max_raise_to") or 0)
        if not cap:
            return "call" if legal.get("can_call") else "check"
        if cap <= lo:
            return "all-in"

        current_bet = _parse_int(prompt, r"Highest bet on this street: (\d+)") or 0
        pot = _parse_int(prompt, r"Pot: (\d+)") or 0
        big_blind = current_bet or _parse_int(
            prompt, r"Minimum full raise would be to: (\d+)"
        ) or 0
        if big_blind <= 0:
            big_blind = max(lo, 20)

        # A standard line: open for ~2.5 big blinds; when facing a bet, raise to
        # about 2.5x that bet plus the pot.  Stronger hands and more aggressive
        # personas size up.  Deliberately *not* a slice of the stack, which is
        # how a bot ends up jamming hundreds into a small pot.
        if current_bet <= big_blind:
            to_amount = round(2.5 * big_blind)
        else:
            to_amount = round(current_bet * 2.5 + pot)
        to_amount = round(to_amount * (0.9 + pct * 0.3))
        if self.persona.aggression > 0.7 and pct > 0.75:
            to_amount = round(to_amount * 1.2)
        to_amount = round(to_amount * self.rng.uniform(0.92, 1.08))

        if to_amount >= cap:
            return "all-in"
        return f"raise {max(lo, min(cap, to_amount))}"


def _parse_legal_block(prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "check" in prompt.split("YOUR LEGAL ACTIONS")[-1]:
        out["can_check"] = True
    body = prompt.split("YOUR LEGAL ACTIONS (choose exactly one):")[-1]
    if "fold" in body:
        out["can_fold"] = True
    if "check" in body:
        out["can_check"] = True
    call = re.search(r"call\s+\(costs (\d+)\)", body)
    if call:
        out["can_call"] = True
        out["call_cost"] = int(call.group(1))
    elif "call" in body:
        out["can_call"] = True
        out["call_cost"] = _parse_int(prompt, r"To call: (\d+)") or 0
    bet = re.search(r"total (\d+) to (\d+)", body)
    if bet:
        lo, hi = int(bet.group(1)), int(bet.group(2))
        out["min_bet_to"] = lo
        out["max_bet_to"] = hi
        out["min_raise_to"] = lo
        out["max_raise_to"] = hi
        out["can_bet"] = True
        out["can_raise"] = True
    allin = re.search(r"all-in\s+\(total commitment (\d+)\)", body)
    if allin:
        out["all_in_to"] = int(allin.group(1))
        out.setdefault("max_raise_to", int(allin.group(1)))
    return out


def _parse_percentile(prompt: str) -> float:
    """Read the hand's percentile out of the prompt.

    ``describe_table`` writes the strength as ``~NNth percentile`` at the end of
    the hole-cards line, e.g.::

        Your hole cards: Jh 6h -- Three of a Kind (very strong, ~77th percentile)

    If that is missing (a hand-rolled prompt), fall back to re-deriving it from
    the cards so a caller never silently gets a default value.
    """
    match = re.search(r"~(\d+)(?:st|nd|rd|th) percentile", prompt)
    if match:
        return int(match.group(1)) / 100
    hole = re.search(r"Your hole cards:\s*(\S+)\s+(\S+)", prompt)
    if hole:
        try:
            cards = [Card.from_str(hole.group(1)), Card.from_str(hole.group(2))]
        except ValueError:
            return 0.25
        board_match = re.search(r"Board:\s*(.+)", prompt)
        board_cards: list[Card] = []
        if board_match and board_match.group(1).strip() not in {"(none yet)", ""}:
            for token in board_match.group(1).split():
                try:
                    board_cards.append(Card.from_str(token))
                except ValueError:
                    pass
        return strength(cards, board_cards)
    return 0.25


def _parse_int(prompt: str, pattern: str) -> int | None:
    match = re.search(pattern, prompt)
    return int(match.group(1)) if match else None


def _parse_equity(prompt: str) -> float | None:
    """Read the reported equity share (``51% equity ...``) out of the prompt."""
    match = re.search(r"(\d{1,3})%\s+equity", prompt)
    if not match:
        return None
    return min(1.0, int(match.group(1)) / 100)


def _parse_opponents(prompt: str) -> int:
    """How many *other* players are still in the hand.

    Counted from the player list rather than read from a dedicated field: an
    earlier version looked for a line that a later prompt rewrite had removed,
    so the count silently fell back to a full ring and every short-handed seat
    played as though it were nine-handed.

    Only the opponents' lines carry the "still in the hand" marker -- this
    seat's own line reads "7d Ah, 1000 chips" -- so the count is used as-is.
    """
    count = len(re.findall(r"--\s*still in the hand", prompt))
    if count:
        return count
    listed = _parse_int(prompt, r"Players still in this hand: (\d+)")
    if listed is not None:
        return max(0, listed - 1)
    return 1
