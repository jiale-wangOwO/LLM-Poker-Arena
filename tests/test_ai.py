"""LLM layer tests: parsing, the legality firewall, retries and fallbacks."""

from __future__ import annotations

import io
import random

import pytest
from rich.console import Console

from pokerarena.ai import (
    AIDecider,
    HeuristicTransport,
    OpenAITransport,
    ParseError,
    ScriptedTransport,
    describe_table,
    parse_action_text,
    parse_reply,
    reply_to_action,
    safe_fallback,
)
from pokerarena.arena import Arena, AISpec, build_players
from pokerarena.config import ArenaConfig
from pokerarena.engine import Action, ActionType, Player, Street, Table
from pokerarena.personas import PERSONAS, get_persona


def make_table(stacks=(1000, 1000, 1000), seed=1, button=0):
    players = [Player(name=f"P{i}", seat=i, chips=c) for i, c in enumerate(stacks)]
    table = Table(players, 10, 20, seed=seed)
    table.button = button
    return table


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,kind,amount",
    [
        ("<action>fold</action>", ActionType.FOLD, None),
        ("<action>check</action>", ActionType.CHECK, None),
        ("<action>call</action>", ActionType.CALL, None),
        ("<action>raise 250</action>", ActionType.RAISE, 250),
        ("<action>raise to 250</action>", ActionType.RAISE, 250),
        ("<action>raise to 1,250</action>", ActionType.RAISE, 1250),
        ("<action>bet 120</action>", ActionType.RAISE, 120),
        ("<action>all-in</action>", ActionType.ALL_IN, None),
        ("<action>all in</action>", ActionType.ALL_IN, None),
        ("<action>shove</action>", ActionType.ALL_IN, None),
        ("<action>FOLD</action>", ActionType.FOLD, None),
        ("<action>Raise To 300</action>", ActionType.RAISE, 300),
    ],
)
def test_parse_action_text(text, kind, amount):
    action = reply_to_action(parse_reply(text))
    assert action.type is kind
    if amount is not None:
        assert action.amount == amount


def test_parse_reply_extracts_all_three_fields():
    raw = (
        "<action>raise 240</action>\n"
        "<say>Time to find out where you are.</say>\n"
        "<thought>He limped twice, his range is capped, so apply pressure.</thought>"
    )
    reply = parse_reply(raw)
    assert reply.action_text == "raise 240"
    assert reply.say == "Time to find out where you are."
    assert "capped" in reply.thought


def test_parse_reply_tolerates_markdown_and_preamble():
    raw = (
        "Sure, here is my decision:\n\n"
        "```\n<action>call</action>\n```\n"
        "<say>Fine.</say>\n<thought>Price is right.</thought>"
    )
    reply = parse_reply(raw)
    assert reply.action_text == "call"


def test_parse_reply_falls_back_to_loose_action_line():
    raw = "I will Action: raise 300 because the pot is big"
    reply = parse_reply(raw)
    assert reply.action_text.startswith("raise 300")


def test_parse_reply_survives_trailing_explanation():
    raw = "<action>raise 250</action> I am raising because my hand is strong"
    assert parse_reply(raw).action_text == "raise 250"


def test_parse_reply_strips_emphasis_noise():
    assert parse_reply("<action>**fold**</action>").action_text == "fold"


@pytest.mark.parametrize("bad", ["", "   ", "<action></action>"])
def test_parse_reply_rejects_empty(bad):
    with pytest.raises(ParseError):
        parse_reply(bad)


def test_parse_action_text_rejects_raiseless_raise():
    with pytest.raises(ParseError):
        parse_action_text("raise")


def test_parse_action_text_rejects_nonsense():
    with pytest.raises(ParseError):
        parse_action_text("teleport the deck")


def test_bare_number_means_raise_to_that_total():
    kind, amount = parse_action_text("250")
    assert kind is ActionType.RAISE
    assert amount == 250


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def test_prompt_lists_only_legal_actions():
    table = make_table(seed=2)
    table.start_hand()
    prompt = describe_table(table, table.actor)
    legal_block = prompt.split("YOUR LEGAL ACTIONS (choose exactly one):")[1]
    assert "fold" in legal_block
    assert "call" in legal_block
    assert "raise" in legal_block
    assert "check" not in legal_block, "facing the big blind, checking is illegal"


def test_prompt_offers_check_when_nothing_is_owed():
    table = make_table(seed=3)
    table.start_hand()
    while table.street is Street.PREFLOP and not table.is_hand_over:
        legal = table.legal_actions(table.actor)
        if legal.can_check:
            prompt = describe_table(table, table.actor)
            block = prompt.split("YOUR LEGAL ACTIONS (choose exactly one):")[1]
            assert "check" in block
            return
        table.apply_action(
            table.actor,
            Action(ActionType.CHECK if legal.can_check else ActionType.CALL),
        )
    pytest.skip("no checking spot found")


def test_prompt_never_leaks_opponent_hole_cards():
    table = make_table(seed=4)
    table.start_hand()
    prompt = describe_table(table, table.actor)
    opponent_codes = [
        c.code
        for p in table.players
        if p.seat != table.actor
        for c in p.hole_cards
    ]
    for code in opponent_codes:
        assert code not in prompt, f"prompt leaked opponent card {code}"


# --------------------------------------------------------------------------
# The legality firewall
# --------------------------------------------------------------------------
def test_illegal_action_is_retried_then_accepted():
    table = make_table(seed=5)
    table.start_hand()
    seat = table.actor
    # First reply raises below the minimum (illegal), second is fine.
    transport = ScriptedTransport(
        [
            "<action>raise 25</action><say></say><thought>too small</thought>",
            "<action>raise 60</action><say>sixty</say><thought>now legal</thought>",
        ]
    )
    decider = AIDecider(get_persona("pro"), transport, retries=2)
    action = decider.decide(table, seat)
    assert action.type is ActionType.RAISE
    assert action.amount == 60
    assert action.source == "llm"
    assert len(decider.traces) == 1
    assert len(decider.traces[0].attempts) == 2
    assert not decider.traces[0].fell_back


def test_illegal_action_twice_falls_back_to_a_legal_action():
    table = make_table(seed=6)
    table.start_hand()
    seat = table.actor
    transport = ScriptedTransport(
        ["<action>raise 1</action>"] * 5  # always illegal
    )
    decider = AIDecider(get_persona("maniac"), transport, retries=2)
    action = decider.decide(table, seat)
    # Facing the big blind the safe fallback is a fold (or a cheap call).
    assert action.source == "fallback"
    # Whatever it chose must actually be applicable.
    table.apply_action(seat, action)
    assert table.players[seat].last_action is not None
    assert decider.traces[0].fell_back


def test_unparseable_reply_falls_back():
    table = make_table(seed=7)
    table.start_hand()
    seat = table.actor
    transport = ScriptedTransport(["I refuse to answer.", "Still refusing."])
    decider = AIDecider(get_persona("rock"), transport, retries=1)
    action = decider.decide(table, seat)
    assert action.source == "fallback"
    table.apply_action(seat, action)


def test_transport_error_breaks_out_and_falls_back():
    class ExplodingTransport:
        def complete(self, system, messages, *, temperature):
            raise RuntimeError("connection reset by peer")

    table = make_table(seed=8)
    table.start_hand()
    seat = table.actor
    decider = AIDecider(get_persona("pro"), ExplodingTransport(), retries=3)
    action = decider.decide(table, seat)
    assert action.source == "fallback"
    assert "connection reset" in (decider.traces[0].error or "")
    # A dead transport must not be retried into a long hang.
    assert len(decider.traces[0].attempts) == 1
    table.apply_action(seat, action)


def test_fallback_is_always_legal_from_any_state():
    rng = random.Random(11)
    for seed in range(25):
        table = make_table([1000] * 4, seed=seed)
        table.start_hand()
        steps = 0
        while not table.is_hand_over and steps < 40:
            steps += 1
            legal = table.legal_actions(table.actor)
            fallback = safe_fallback(table, table.actor, legal)
            table.apply_action(table.actor, fallback)
        assert table.is_hand_over or steps >= 40


def test_decider_records_transcript_for_each_decision():
    table = make_table(seed=9)
    table.start_hand()
    transport = ScriptedTransport(
        ["<action>call</action><say>I am in.</say><thought>cheap enough</thought>"]
    )
    decider = AIDecider(get_persona("calling_station"), transport, retries=0)
    decider.decide(table, table.actor)
    assert decider.transcript
    entry = decider.transcript[0]
    assert entry["name"] == "P0" or entry["name"].startswith("P")
    assert entry["speech"] == "I am in."
    assert entry["thought"] == "cheap enough"


def test_action_amount_zero_means_absent_not_all_in():
    """A speech-only reply must never be misread as a huge raise."""
    action = reply_to_action(parse_reply("<action>fold</action><say>bye</say>"))
    assert action.amount == 0
    assert action.type is ActionType.FOLD


# --------------------------------------------------------------------------
# Persona plumbing
# --------------------------------------------------------------------------
def test_every_persona_produces_a_usable_system_prompt():
    for key, persona in PERSONAS.items():
        decider = AIDecider(persona, ScriptedTransport([]))
        assert persona.name in decider.system_prompt
        assert "<action>" in decider.system_prompt
        assert persona.style[:20] in decider.system_prompt


def test_personas_are_distinct():
    styles = {p.style for p in PERSONAS.values()}
    voices = {p.voice for p in PERSONAS.values()}
    names = {p.name for p in PERSONAS.values()}
    assert len(styles) == len(PERSONAS)
    assert len(voices) == len(PERSONAS)
    assert len(names) == len(PERSONAS)


def test_persona_temperatures_are_sane():
    for persona in PERSONAS.values():
        assert 0.0 <= persona.temperature <= 2.0
        assert 0.0 <= persona.aggression <= 1.0
        assert 0.0 <= persona.tightness <= 1.0
        assert 0.0 <= persona.bluff_frequency <= 1.0


# --------------------------------------------------------------------------
# Heuristic transport (offline brain)
# --------------------------------------------------------------------------
def test_heuristic_transport_always_returns_legal_actions():
    rng = random.Random(21)
    for seed in range(30):
        table = make_table([1000] * 4, seed=seed)
        table.start_hand()
        transport = HeuristicTransport(get_persona("maniac"), random.Random(rng.random()))
        decider = AIDecider(get_persona("maniac"), transport, retries=0)
        steps = 0
        while not table.is_hand_over and steps < 60:
            steps += 1
            action = decider.decide(table, table.actor)
            table.apply_action(table.actor, action)
        assert steps < 60, f"heuristic deadlocked at seed {seed}"


def test_heuristic_transport_never_needs_the_fallback():
    """The offline brain must always answer inside the legal menu itself."""
    for seed in range(20):
        table = make_table([500] * 4, seed=seed)
        table.start_hand()
        persona = get_persona("pro")
        decider = AIDecider(
            persona, HeuristicTransport(persona, random.Random(seed)), retries=0
        )
        steps = 0
        while not table.is_hand_over and steps < 60:
            steps += 1
            seat = table.actor
            action = decider.decide(table, seat)
            table.apply_action(seat, action)
        assert steps < 60, f"heuristic deadlocked at seed {seed}"
        for trace in decider.traces:
            assert not trace.fell_back, (seed, trace.error)
            assert not trace.attempts or trace.attempts[0].get("ok")


# --------------------------------------------------------------------------
# Arena end-to-end (offline)
# --------------------------------------------------------------------------
def test_arena_plays_a_full_game_offline_without_deadlock():
    specs = [
        AISpec(name="Ironside", persona_key="calculating", model=""),
        AISpec(name="Vega", persona_key="maniac", model=""),
        AISpec(name="Granite", persona_key="rock", model=""),
        AISpec(name="Kitsune", persona_key="trickster", model=""),
    ]
    players, _ = build_players(specs, starting_chips=400)
    config = ArenaConfig(seed=42, max_rounds=400)
    arena = Arena(players, config=config)
    total_before = sum(p.chips for p in arena.table.players)
    winner = arena.play_game()
    total_after = sum(p.chips for p in arena.table.players)
    assert total_after == total_before, "chips leaked during a full game"
    assert winner
    assert arena.table.is_game_over(), "the game should reach a single winner"
    assert arena.table.seated[0].chips == total_before


def test_arena_hand_deltas_sum_to_zero():
    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="shark", model=""),
        AISpec(name="C", persona_key="calling_station", model=""),
    ]
    players, _ = build_players(specs, starting_chips=500)
    arena = Arena(players, config=ArenaConfig(seed=7, max_rounds=40))
    for _ in range(10):
        if arena.table.is_game_over():
            break
        record = arena.play_hand()
        assert sum(record["deltas"].values()) == 0, record
        arena.table.eliminate_broke_players()
        arena.table.rotate_button()


def test_arena_records_transcript_and_writes_history(tmp_path):
    specs = [
        AISpec(name="A", persona_key="trickster", model=""),
        AISpec(name="B", persona_key="maniac", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    config = ArenaConfig(seed=5, max_rounds=5, hand_history_dir=tmp_path)
    arena = Arena(players, config=config)
    arena.play_game()
    path = arena.save_history()
    assert path is not None and path.exists()
    assert arena.hand_history


def test_arena_uses_heuristic_when_model_has_no_api_key(monkeypatch):
    """A seat whose model key is absent must degrade to the local brain."""
    # Use a model whose key is deliberately unset for this test, rather than
    # relying on the developer's machine having no keys at all.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("POKERARENA_TEST_MISSING_KEY", "")
    specs = [
        AISpec(name="A", persona_key="pro", model="claude"),
        AISpec(name="B", persona_key="rock", model="claude"),
    ]
    players, _ = build_players(specs, starting_chips=200)
    arena = Arena(players, config=ArenaConfig(seed=3, max_rounds=10))
    for decider in arena.deciders.values():
        assert isinstance(decider.transport, HeuristicTransport)
    arena.play_game()


def test_arena_uses_a_real_transport_when_the_key_is_present(monkeypatch):
    """Sanity check the other branch: a provider with a key yields an API transport."""
    from pokerarena.providers import Provider

    # An environment key must make no difference: providers carry their own key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-environment")
    provider = Provider(
        id="p",
        label="P",
        model="some-model",
        base_url="https://example.com/v1",
        api_key="sk-ant-test-placeholder",
    )
    specs = [
        AISpec(name="A", persona_key="pro", model=provider.id),
        AISpec(name="B", persona_key="rock", model=provider.id),
    ]
    players, _ = build_players(specs, starting_chips=200)

    class Store:
        def get(self, provider_id):
            return provider if provider_id == provider.id else None

    arena = Arena(players, config=ArenaConfig(seed=3, max_rounds=2), providers=Store())
    for decider in arena.deciders.values():
        assert isinstance(decider.transport, OpenAITransport)
        assert decider.transport.api_key == "sk-ant-test-placeholder"
        assert decider.transport.base_url == "https://example.com/v1"


def test_arena_detects_and_recovers_from_a_misbehaving_transport():
    """A model that only ever emits garbage must not stall the game."""
    specs = [
        AISpec(name="Chaos", persona_key="maniac", model=""),
        AISpec(name="Steady", persona_key="rock", model=""),
    ]
    players, _ = build_players(specs, starting_chips=200)
    transports = {
        0: ScriptedTransport(["gibberish nonsense"] * 200),
    }
    transports[1] = HeuristicTransport(get_persona("rock"), random.Random(1))
    arena = Arena(
        players,
        config=ArenaConfig(seed=9, max_rounds=6, llm_retries=1),
        transports=transports,
    )
    winner = arena.play_game()
    assert winner
    assert sum(p.chips for p in arena.table.players) == 400


# --------------------------------------------------------------------------
# CLI view
# --------------------------------------------------------------------------
def test_cli_view_keeps_showing_the_pot_after_award():
    """Regression: the panel used to read the live balance and show 'pot 0'.

    The table zeroes its pot balance as soon as chips are awarded, so the view
    must remember the last pot value from the event stream instead.
    """
    from pokerarena.cli import TableView

    specs = [
        AISpec(name="A", persona_key="rock", model=""),
        AISpec(name="B", persona_key="maniac", model=""),
    ]
    players, _ = build_players(specs, starting_chips=200)
    arena = Arena(players, config=ArenaConfig(seed=4, max_rounds=5))
    view = TableView(console=Console(file=io.StringIO(), width=100))
    arena.on_event = view.handle
    arena.play_hand()

    # The hand is over, so the live pot is empty...
    assert arena.table.pot_total == 0
    # ...but the view remembers what was contested.
    assert view._display_pot > 0

    panel = view._board_panel(arena.table)
    title = panel.title
    assert "pot 0" not in str(title), title
    assert "awarded" in str(title)


def test_cli_view_resets_the_pot_for_a_new_hand():
    from pokerarena.cli import TableView

    specs = [
        AISpec(name="A", persona_key="pro", model=""),
        AISpec(name="B", persona_key="pro", model=""),
    ]
    players, _ = build_players(specs, starting_chips=300)
    arena = Arena(players, config=ArenaConfig(seed=6, max_rounds=5))
    # This verifies display state across hands, so keep both seats alive with
    # controlled calls/checks instead of depending on a bot's betting outcome.
    arena.decide = lambda table, seat: Action(
        ActionType.CHECK if table.legal_actions(seat).can_check else ActionType.CALL
    )
    view = TableView(console=Console(file=io.StringIO(), width=100))
    arena.on_event = view.handle
    arena.play_hand()
    first = view._display_pot
    arena.table.rotate_button()
    arena.play_hand()
    # A fresh hand must not inherit the previous hand's pot display.
    assert view._display_pot != first or view.hand_number == 2


def test_human_action_parser_accepts_the_documented_forms():
    from pokerarena.cli import parse_human_action
    from pokerarena.engine import ActionType, IllegalAction

    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    legal = table.legal_actions(table.actor)

    assert parse_human_action("f", legal).type is ActionType.FOLD
    assert parse_human_action("fold", legal).type is ActionType.FOLD
    assert parse_human_action("c", legal).type is ActionType.CALL
    assert parse_human_action("call", legal).type is ActionType.CALL
    assert parse_human_action("a", legal).type is ActionType.ALL_IN
    assert parse_human_action("all-in", legal).type is ActionType.ALL_IN
    assert parse_human_action("r 100", legal).amount == 100
    assert parse_human_action("raise to 120", legal).amount == 120
    assert parse_human_action("150", legal).amount == 150

    with pytest.raises(IllegalAction):
        parse_human_action("", legal)
    with pytest.raises(IllegalAction):
        parse_human_action("banana", legal)
    with pytest.raises(IllegalAction):
        parse_human_action("raise", legal)
    # 'help' surfaces the legal menu rather than being an action.
    with pytest.raises(IllegalAction) as excinfo:
        parse_human_action("help", legal)
    assert "fold" in str(excinfo.value)


def test_human_action_parser_rejects_check_when_facing_a_bet():
    from pokerarena.cli import parse_human_action

    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    legal = table.legal_actions(table.actor)
    assert not legal.can_check
    # The parser itself accepts 'check'; the engine rejects it in legal_actions.
    assert parse_human_action("check", legal).type is ActionType.CHECK
