"""Real table context: correct opportunities, bounded memories and private information."""

from copy import deepcopy

import pytest

from pokerarena.ai import (
    AIDecider, ScriptedTransport, _position_label, _position_note,
    describe_table, format_hand_log, observed_table_reads, public_action_record,
)
from pokerarena.arena import Arena
from pokerarena.config import ArenaConfig
from pokerarena.cards import Card
from pokerarena.engine import Action, ActionType, Player, PlayerStatus, Street, Table
from pokerarena.personas import get_persona


def make_table(count=3, stacks=None):
    players = [Player(f"P{i}", i, (stacks or [1000] * count)[i], persona="pro") for i in range(count)]
    table = Table(players, 10, 20, seed=12)
    table.start_hand()
    return table


def test_six_player_positions_are_fixed_even_after_blinds_shove():
    table = make_table(6)
    assert [_position_label(table, i) for i in range(6)] == [
        "the button (dealer)", "the small blind", "the big blind",
        "under the gun (UTG)", "the hijack (HJ)", "the cutoff (CO)",
    ]
    table.players[1].chips = 0
    table.players[1].status = PlayerStatus.ALL_IN
    assert _position_label(table, 2) == "the big blind"
    table.players[4].status = PlayerStatus.FOLDED
    assert "P1" not in _position_note(table, table.actor)
    assert "P4" not in _position_note(table, table.actor)


def test_sparse_chairs_do_not_invent_busted_opponents_or_change_action_order():
    players = [Player(f"P{i}", i, 1000 if i in {1, 3, 5} else 0,
                      seated=i in {1, 3, 5}) for i in range(6)]
    table = Table(players, 10, 20, seed=12)
    table.start_hand()
    assert table.button == 1 and table.actor == 1
    prompt = describe_table(table, 1)
    assert "Pre-flop order of remaining active players: Seat 2 (P1) (you) -> Seat 4 (P3) -> Seat 6 (P5)" in prompt
    assert "Post-flop order of remaining active players: Seat 4 (P3) -> Seat 6 (P5) -> Seat 2 (P1) (you)" in prompt
    assert "P0" not in prompt and "P2" not in prompt and "P4" not in prompt
    assert _position_label(table, 3) == "the small blind"
    assert _position_label(table, 5) == "the big blind"


def test_heads_up_button_acts_first_preflop_and_last_postflop():
    table = make_table(2)
    assert table.actor == 0
    preflop = describe_table(table, 0)
    assert "the button / small blind" in preflop
    assert "Pre-flop order of remaining active players: Seat 1 (P0) (you) -> Seat 2 (P1)" in preflop
    assert "Post-flop order of remaining active players: Seat 2 (P1) -> Seat 1 (P0) (you)" in preflop
    assert "LAST on every street" not in preflop
    table.apply_action(0, Action(ActionType.CALL))
    table.apply_action(1, Action(ActionType.CHECK))
    assert table.street is Street.FLOP
    assert table.actor == 1
    assert "out of position to Seat 1 (P0)" in describe_table(table, 1)


def test_closing_preflop_call_does_not_claim_last_postflop_position():
    table = make_table()
    table.apply_action(0, Action(ActionType.CALL))
    table.apply_action(1, Action(ActionType.CALL))
    assert table.actor == 2
    prompt = describe_table(table, 2)
    assert "LAST in this betting round if you call/check" in prompt
    assert "out of position to Seat 1 (P0)" in prompt


def test_calling_price_uses_actual_cost_and_eligible_side_pots():
    table = make_table()
    table.players[0].chips = 100
    table.current_bet = 400
    table.players[1].street_bet = table.players[2].street_bet = 400
    table.hand_contributions = {0: 0, 1: 400, 2: 400}
    table.pot = 800
    prompt = describe_table(table, 0)
    assert "To call: 100" in prompt
    assert "Opponent's full price is 400" in prompt
    assert "eligible pot after calling: 300" in prompt
    assert "break-even equity: 33.3% (100 / 300)" in prompt
    assert "600 chips are above your contribution cap" in prompt


def test_calling_price_includes_dead_money_from_folded_seat():
    table = make_table()
    table.players[1].status = PlayerStatus.FOLDED
    assert "eligible pot after calling: 50" in describe_table(table, 0)
    assert "break-even equity: 40.0% (20 / 50)" in describe_table(table, 0)


def test_all_in_opponent_does_not_zero_the_active_opponents_spr():
    table = make_table()
    table.players[1].status = PlayerStatus.ALL_IN
    table.players[1].chips = 0
    prompt = describe_table(table, 0)
    assert "SPR 32.7" in prompt
    assert "ALL-IN: can win existing pots but cannot bet again" in prompt


def test_public_log_retains_safe_table_talk_but_no_private_fields():
    table = make_table()
    entry = {"seat": 1, "name": "P1", "street": "preflop", "street_label": "Pre-Flop",
             "action": "call", "amount": 10, "street_bet": 20, "pot": 40,
             "speech": "Who wants to dance?", "thought": "SECRET_PLAN",
             "hole_cards": ["SECRET_CARDS"], "reasoning": "SECRET_REASONING"}
    table.hand_log.append(entry)
    text = "\n".join(format_hand_log(table, 0))
    assert "Who wants to dance?" in text
    assert "SECRET_" not in text
    record = public_action_record(entry)
    assert set(record).isdisjoint({"thought", "reasoning", "hole_cards"})
    entry["speech"] = "My pocket aces are unbeatable."
    assert public_action_record(entry)["speech"] == ""


def blind(seat, total):
    return {"seat": seat, "street": "preflop", "action": "post_blind", "amount": total, "street_bet": total}


def test_public_reads_use_decision_opportunities_and_distinguish_shove_calls():
    table = make_table()
    history = [
        {"action_log": [blind(1, 10), blind(2, 20),
                        {"seat": 0, "street": "preflop", "action": "raise", "street_bet": 60},
                        {"seat": 1, "street": "preflop", "action": "all_in", "street_bet": 30},
                        {"seat": 2, "street": "preflop", "action": "fold", "street_bet": 20}],
         "showdown": []},
        # P1 went all-in posting a blind and had no preflop decision opportunity.
        {"action_log": [blind(1, 10), blind(2, 20)]},
    ]
    reads = "\n".join(observed_table_reads(table, history))
    assert "Seat 2 (P1): VPIP 1/1 (100%); pre-flop raises 0/1 (0%)" in reads
    assert "Seat 3 (P2): VPIP 0/1 (0%); pre-flop raises 0/1 (0%); folded facing a bet 1/1 decisions" in reads
    assert "small sample, weak read" in reads


def test_short_big_blind_does_not_create_a_false_preflop_raise_or_free_check():
    table = make_table()
    history = [
        {"big_blind": 20, "action_log": [blind(1, 10), blind(2, 5),
             {"seat": 1, "street": "preflop", "action": "all_in", "street_bet": 15}]},
        {"big_blind": 20, "action_log": [blind(1, 10), blind(2, 5),
             {"seat": 0, "street": "preflop", "action": "fold", "street_bet": 0},
             {"seat": 1, "street": "preflop", "action": "fold", "street_bet": 10}]},
    ]
    reads = "\n".join(observed_table_reads(table, history))
    assert "Seat 2 (P1): VPIP 1/2 (50%); pre-flop raises 0/2 (0%); folded facing a bet 1/2 decisions" in reads


def test_table_read_window_is_bounded_and_has_no_unearned_reads():
    table = make_table()
    history = [{"action_log": [blind(1, 10), blind(2, 20)],
                "private": "NEVER_SHARE", "results": [{"hole_cards": "NEVER_SHARE"}]}] * 100
    reads = "\n".join(observed_table_reads(table, history))
    assert "last 40 recorded hands" in reads
    assert "no observed pre-flop decisions yet" in reads
    assert "NEVER_SHARE" not in reads


def test_private_continuity_only_uses_this_seats_beliefs_and_net_results():
    table = make_table()
    player = table.players[0]
    player.conversation = [{"role": "assistant", "hand": 1, "street": "Flop", "action": "check",
                            "thought": "MY_PLAN: call small bets, reconsider on a paired turn.", "content": ""}]
    table.players[1].history = [{"thought": "OPPONENT_SECRET", "reasoning": "OPPONENT_REASONING"}]
    decider = AIDecider(get_persona("pro"), ScriptedTransport([]))
    decider.hand_history = [{"hand_number": 1, "deltas": {0: -40, 1: 40}, "winners": ["P1"], "pot": 80}]
    prompt = decider._build_messages(table, 0)[-1]["content"]
    assert "MY_PLAN" in prompt
    assert "your net result -40 chips" in prompt
    assert "OPPONENT_" not in prompt
    assert "no cards were publicly shown" in prompt


def test_trace_prompt_matches_the_current_context_sent_to_transport():
    class CapturingTransport:
        messages = None

        def complete(self, system, messages, *, temperature):
            self.messages = messages
            from pokerarena.ai import ChatResult
            return ChatResult("<action>call</action><say></say><thought>Price is reasonable.</thought>")

    table = make_table()
    transport = CapturingTransport()
    decider = AIDecider(get_persona("pro"), transport)
    decider.decide(table, 0)
    assert decider.traces[-1].prompt == transport.messages[-1]["content"]
    assert "OBSERVED TABLE READS" in decider.traces[-1].prompt


def test_opponents_hidden_cards_thoughts_and_spectator_mode_cannot_change_prompt():
    table = make_table()
    before = describe_table(table, 0)
    table.players[1].hole_cards = [Card.from_str("As"), Card.from_str("Ks")]
    table.players[1].thoughts = ["PRIVATE_TELL"]
    table.players[1].history = [{"thought": "PRIVATE_TELL", "reasoning": "PRIVATE_TELL"}]
    # Spectator flags are outside the player-information boundary.
    table.god_mode = True
    table.reveal_all = True
    assert describe_table(table, 0) == before


@pytest.mark.parametrize("kind,amount,public_action", [
    (ActionType.RAISE, 60, "raises to 60"),
    (ActionType.FOLD, 0, "folds"),
    (ActionType.ALL_IN, 0, "is ALL-IN for 1000"),
    (ActionType.CALL, 0, "calls 20"),
])
def test_each_llm_sees_human_presence_and_public_action_in_real_arena(
        tmp_path, kind, amount, public_action):
    """The real human callback reaches both other seats' model requests."""
    from pokerarena.ai import ChatResult

    class PassiveCapture:
        def complete(self, system, messages, *, temperature):
            legal = messages[-1]["content"].split("YOUR LEGAL ACTIONS")[-1]
            action = "check" if "  * check" in legal else "call"
            return ChatResult(f"<action>{action}</action><say></say><thought>Public price.</thought>")

    players = [Player("You", 0, 1000, is_ai=False, persona="human"),
               Player("P1", 1, 1000, persona="pro"),
               Player("P2", 2, 1000, persona="pro")]
    first_decision = True

    def human(table, seat):
        nonlocal first_decision
        if first_decision:
            first_decision = False
            return Action(kind, amount=amount, speech="Keep it moving.", thought="HUMAN_PRIVATE")
        legal = table.legal_actions(seat)
        return Action(ActionType.CHECK if legal.can_check else ActionType.CALL)

    arena = Arena(players, human_seat=0, human_decider=human,
                  config=ArenaConfig(seed=12, max_rounds=1, hand_history_dir=tmp_path),
                  transports={1: PassiveCapture(), 2: PassiveCapture()})
    record = arena.play_hand()
    for seat in (1, 2):
        traces = arena.deciders[seat].traces
        assert traces, f"Seat {seat + 1} never received its public context"
        prompt = traces[0].prompt
        assert f"Your seat: Seat {seat + 1} (P{seat})" in prompt
        assert f"Seat 1 (You) {public_action}" in prompt
        assert "Seat 1 (You):" in prompt
        assert "the button (dealer)" in prompt
        assert 'table talk: "Keep it moving."' in prompt
        assert "HUMAN_PRIVATE" not in prompt
        assert "[human]" not in prompt and "[pro]" not in prompt
    assert 0 in {entry["seat"] for entry in record["participants"]}
    assert any(entry["seat"] == 0 and entry["action"] == kind.value for entry in record["action_log"])
    assert "HUMAN_PRIVATE" not in str(record)
    assert all("source" not in entry for entry in record["action_log"])


def test_opponent_control_persona_model_and_action_source_cannot_change_llm_input():
    """Swapping only hidden control metadata leaves every sent byte unchanged."""
    table = make_table()
    table.players[0].name = "You"
    table.players[0].is_ai = False
    table.players[0].persona = "human"
    table.apply_action(0, Action(ActionType.RAISE, amount=60, source="human"))
    table.hand_log[-1]["name"] = "You"
    decider = AIDecider(get_persona("pro"), ScriptedTransport([]))
    decider.hand_history = [
        {"hand_number": 1, "big_blind": 20, "pot": 120, "winners": ["You"],
         "participants": [{"seat": 0, "name": "You"}],
         "action_log": [blind(1, 10), blind(2, 20),
                        {"seat": 0, "name": "You", "street": "preflop", "street_label": "Pre-Flop",
                         "action": "raise", "street_bet": 60, "amount": 60, "source": "human"}],
         "showdown": []},
        {"hand_number": 2, "big_blind": 20, "action_log": [blind(1, 10), blind(2, 20),
             {"seat": 2, "street": "preflop", "action": "raise", "street_bet": 60},
             {"seat": 0, "street": "preflop", "action": "fold", "street_bet": 0, "source": "human"}]},
        {"hand_number": 3, "big_blind": 20, "action_log": [blind(1, 10), blind(2, 20),
             {"seat": 0, "name": "You", "street": "preflop", "street_label": "Pre-Flop",
              "action": "all_in", "street_bet": 120, "amount": 120, "source": "human"}]},
    ]
    before = deepcopy(decider._build_messages(table, 1))
    reads = observed_table_reads(table, decider.hand_history)
    assert "Seat 1 (You): VPIP 2/3 (67%); pre-flop raises 2/3 (67%); folded facing a bet 1/3 decisions" in "\n".join(reads)
    assert "Seat 1 (You) won 120" in before[-1]["content"]

    for opponent in (table.players[0], table.players[2]):
        opponent.is_ai = not opponent.is_ai
        opponent.persona = "HIDDEN_PERSONA_CHANGED"
        opponent.model = "HIDDEN_MODEL_CHANGED"
        opponent.hole_cards = [Card.from_str("As"), Card.from_str("Ks")]
        opponent.thoughts = ["HIDDEN_TELL_CHANGED"]
        opponent.history = [{"thought": "HIDDEN_REASONING_CHANGED", "source": "llm"}]
        opponent.conversation = [{"role": "assistant", "content": "HIDDEN_PRIVATE_CHANGED"}]
    for entry in table.hand_log:
        entry.update(source="llm", is_ai=True, persona="HIDDEN_PERSONA_CHANGED", model="HIDDEN_MODEL_CHANGED")
    for record in decider.hand_history:
        for entry in record["action_log"]:
            entry.update(source="llm", is_ai=True, persona="HIDDEN_PERSONA_CHANGED", model="HIDDEN_MODEL_CHANGED")
    assert decider._build_messages(table, 1) == before
    assert observed_table_reads(table, decider.hand_history) == reads
    assert "HIDDEN_" not in str(before)


def test_failed_request_is_remembered_as_fallback_not_as_a_model_read():
    class FailingTransport:
        def complete(self, system, messages, *, temperature):
            raise RuntimeError("unavailable")

    table = make_table()
    decider = AIDecider(get_persona("pro"), FailingTransport())
    action = decider.decide(table, 0)
    assert action.source == "fallback"
    assert table.players[0].conversation[0]["source"] == "fallback"
    assert "(fallback)" in decider._build_messages(table, 0)[-1]["content"]


@pytest.mark.parametrize("raises", [False, True])
def test_stop_after_in_flight_invalid_or_error_reply_does_not_retry_or_record(raises):
    class CancelDuringRequest:
        calls = 0
        stopped = False

        def complete(self, system, messages, *, temperature):
            from pokerarena.ai import ChatResult
            self.calls += 1
            self.stopped = True
            if raises:
                raise RuntimeError("request failed after cancellation")
            return ChatResult("<action>raise 1</action><thought>Discard me.</thought>")

    table = make_table()
    transport = CancelDuringRequest()
    decider = AIDecider(get_persona("pro"), transport, retries=3)
    decider.should_stop = lambda: transport.stopped
    decider.decide(table, 0)
    assert transport.calls == 1
    assert table.players[0].history == []
    assert table.players[0].conversation == []
    assert decider.transcript == []
    assert decider.traces == []


def test_already_stopped_decider_does_not_start_a_model_call():
    table = make_table()
    transport = ScriptedTransport(["<action>call</action>"])
    decider = AIDecider(get_persona("pro"), transport)
    decider.should_stop = lambda: True
    decider.decide(table, 0)
    assert transport.index == 0
    assert table.players[0].conversation == []
    assert decider.traces == []


def test_arena_persists_public_action_history_without_opponents_private_decisions(tmp_path):
    players = [Player("P0", 0, 500, persona="pro"), Player("P1", 1, 500, persona="pro")]
    reply = "<action>call</action><say>Keep it moving.</say><thought>PRIVATE_PLAN</thought>"
    arena = Arena(players, config=ArenaConfig(seed=5, max_rounds=1, hand_history_dir=tmp_path),
                  transports={seat: ScriptedTransport([reply] * 10, on_exhausted="<action>check</action><thought>PRIVATE_PLAN</thought>") for seat in range(2)})
    record = arena.play_hand()
    assert record["action_log"]
    assert len(record["participants"]) == 2
    assert "PRIVATE_PLAN" not in str(record)
    assert any(entry.get("speech") == "Keep it moving." for entry in record["action_log"])
