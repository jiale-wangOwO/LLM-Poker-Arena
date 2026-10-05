"""Table talk must never give a hand away.

The other seats read a player's ``<say>``; a leaked holding hands them a read
nobody earned.  Worse, a *false* claim ("two pair, baby!" while holding air) is
just as corrosive, because the model reading it has no way to tell truth from
taunt.  So the rule is absolute: table talk is never about the actual cards.

Two layers enforce it:

1. the system prompt states the rule explicitly, and
2. :func:`pokerarena.ai.sanitize_speech` drops any line that breaks it anyway.

This file tests layer 2 thoroughly, because layer 1 is a request rather than a
guarantee.
"""

from __future__ import annotations

import itertools

import pytest

from pokerarena.ai import (
    SYSTEM_TEMPLATE,
    leaks_hand_information,
    parse_reply,
    reply_to_action,
    sanitize_speech,
)

# Lines that describe the speaker's own holding and must be dropped.
LEAKY = [
    # the exact line a live run produced
    "Two pair, baby! Who wants to dance?",
    "I flopped a set, you're drawing dead.",
    "I've got the nuts here.",
    "This is top pair, good kicker.",
    "I have aces, sorry.",
    "Holding kings, what can you do?",
    "Pocket queens, obviously.",
    "I'm ahead, I know it.",
    "I'm bluffing, actually.",
    "I'm value betting this one.",
    "You've got me beat, I fold.",
    "The ace of spades is my card.",
    "Just a flush draw over here.",
    "Quads! Unbelievable.",
    "That's a full house for me.",
    "Second nuts is good enough.",
    "I'm priced in with a gutshot.",
    "You have me, nice hand.",
    "I rivered two pair on you.",
    "I got this pot locked with my ace.",
    # naming the two cards is naming the hand -- these came from live runs
    "Seven-deuce. Not even I'm that patient.",
    "Queen-nine offsuit. Still not a hand I raise.",
    "Q-five. No. Not worth the chips.",
    "I was dealt ace-king and I folded it.",
    "Q5o is a fold from there.",
    "AK suited, obviously.",
    "Holding 7-2 offsuit.",
    "10-4, nice and tidy.",
]

# Ordinary table talk: every one of these must survive untouched.
CLEAN = [
    "Wake up, table! I'm bringing the heat!",
    "All of it! You want the crown, come take it!",
    "I'll let you two fight over this one. For now.",
    "Suited connectors, huh? You'll never guess which two.",
    "Funny how these boards keep landing my way. Or do they?",
    "Nice hand, well played.",
    "Good luck, everyone.",
    "Come on, dealer, I've been good.",
    "You're taking your time. Scared?",
    "I came here to gamble, not to fold.",
    "Check it down?",
    "That's a scary board for both of us.",
    "I'll give you a free card. This time.",
    "You always raise there. Interesting.",
    "Show me something pretty.",
    "I'm not afraid of you.",
    "Big pot. Big decision.",
    "Let's see the turn.",
    "You bluff too much for your own good.",
    # pot odds are numbers, not cards
    "I have 3 to 1 odds.",
    "I have 2 to 1 on a call.",
    "10 to 1 is generous.",
    # talking about the *board* is fine; only your own holding is off limits
    "The king on the board scares me.",
    "That ace changed everything.",
    "Nice flop for the raiser.",
    "Board pairs the queen.",
    "Anyone else see that ace?",
]


@pytest.mark.parametrize("line", LEAKY)
def test_hand_descriptions_are_detected(line):
    assert leaks_hand_information(line) is not None, f"leaked through: {line!r}"


@pytest.mark.parametrize("line", CLEAN)
def test_ordinary_table_talk_is_left_alone(line):
    assert leaks_hand_information(line) is None, f"false positive on: {line!r}"


def test_every_single_card_code_is_caught():
    """A model may write its hand as a code; none of the 52 may slip through."""
    for rank, suit in itertools.product(list("23456789TJQKA") + ["10"], "shdc"):
        for phrase in (f"I have {rank}{suit}.", f"I'm holding {rank}{suit}"):
            assert leaks_hand_information(phrase) is not None, phrase


def test_the_sanitizer_drops_the_line_and_reports_why():
    speech, reason = sanitize_speech("Two pair, baby!")
    assert speech == "", "a leaked line must be dropped, not shown"
    assert reason, "the reason should be reported for logging"

    speech, reason = sanitize_speech("Nice hand, well played.")
    assert speech == "Nice hand, well played."
    assert reason == ""


def test_short_pleasantries_are_not_second_guessed():
    """"Good luck" is below the analysis threshold and must never be stripped."""
    for line in ("Good luck", "Nice hand", "gg", "Wow", "No way"):
        speech, reason = sanitize_speech(line)
        assert speech == line
        assert reason == ""


def test_a_real_reply_loses_only_its_speech():
    """The action and the private reasoning must be unaffected."""
    reply = parse_reply(
        "<action>raise 100</action>"
        "<say>Two pair, baby!</say>"
        "<thought>I have two pair, so I raise for value.</thought>"
    )
    action = reply_to_action(reply)
    assert action.speech == "", "the leaked table talk reached the table"
    assert action.type.value == "raise"
    assert action.amount == 100
    assert "two pair" in action.thought, "private reasoning must be preserved"


def test_a_clean_reply_keeps_its_speech():
    reply = parse_reply(
        "<action>bet 200</action>"
        "<say>Who wants to dance?</say>"
        "<thought>Board is dry, I'll take it down.</thought>"
    )
    action = reply_to_action(reply)
    assert action.speech == "Who wants to dance?"


def test_the_system_prompt_states_the_rule():
    """Layer 1: the model is told, in plain terms, before it can leak anything."""
    prompt = SYSTEM_TEMPLATE.lower()
    assert "secret" in prompt
    assert "<say>" in prompt
    assert "never say" in prompt
    # The forbidden categories are named, not just alluded to.
    for phrase in ("rank of your hand", "name of any card", "i'm ahead", "bluffing"):
        assert phrase in prompt, phrase
    # And the model is given something it *can* do instead of going silent.
    assert "taunt" in prompt
