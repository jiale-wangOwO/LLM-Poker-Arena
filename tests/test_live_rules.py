"""Real-table decisions: all-in responses, raise rights and public accounting."""

import pytest

from pokerarena.engine import Action, ActionType, IllegalAction, Player, Table


def table_for(stacks, *, button=0):
    table = Table([Player(f"P{i}", i, stack) for i, stack in enumerate(stacks)], 10, 20, seed=7)
    table.button = button
    return table


def test_only_live_stack_must_choose_whether_to_call_all_in_blind():
    table = table_for([1000, 20])
    table.start_hand()
    assert table.actor == 0
    assert not table.is_hand_over
    legal = table.legal_actions(0)
    assert legal.can_call and legal.call_cost == 10
    assert not legal.can_raise and legal.all_in_to is None
    table.apply_action(0, Action(ActionType.FOLD))
    assert table.is_hand_over
    assert table.board == []
    assert table.players[1].chips == 30
    assert table.pots_snapshot == [{"amount": 20, "eligible": [1], "side": False}]
    assert any(e.data.get("reason") == "uncalled_bet_returned" and e.data["amount"] == 10 for e in table.events)


def test_calling_all_in_runs_out_every_remaining_board_card():
    table = table_for([1000, 100])
    table.start_hand()
    table.apply_action(0, Action(ActionType.CALL))
    table.apply_action(1, Action(ActionType.ALL_IN))
    assert table.actor == 0
    legal = table.legal_actions(0)
    assert legal.call_cost == 80
    assert not legal.can_raise and legal.all_in_to is None
    table.apply_action(0, Action(ActionType.CALL))
    assert table.is_hand_over
    assert len(table.board) == 5
    assert len(table.deck) == 52 - 4 - 5 - 3
    assert sum(p.chips for p in table.players) == 1100
    assert [e.data["street"] for e in table.events if e.kind == "board_dealt"] == ["flop", "turn", "river"]


def test_short_all_in_does_not_grant_an_earlier_caller_raise_rights():
    table = table_for([30, 1000, 1000, 1000])
    table.start_hand()
    table.apply_action(3, Action(ActionType.CALL))
    table.apply_action(0, Action(ActionType.ALL_IN))
    table.apply_action(1, Action(ActionType.CALL))
    table.apply_action(2, Action(ActionType.CALL))
    assert table.actor == 3
    legal = table.legal_actions(3)
    assert legal.call_cost == 10
    assert not legal.can_raise and legal.all_in_to is None
    for action in (Action(ActionType.RAISE, 50), Action(ActionType.ALL_IN)):
        with pytest.raises(IllegalAction):
            table.apply_action(3, action)
    table.apply_action(3, Action(ActionType.CALL))
    assert table.street.value == "flop"


def test_cumulative_short_all_ins_reopen_after_a_full_raise_is_faced():
    table = table_for([1000, 30, 45, 1000, 1000])
    table.start_hand()
    for seat in (3, 4, 0):
        table.apply_action(seat, Action(ActionType.CALL))
    table.apply_action(1, Action(ActionType.ALL_IN))
    table.apply_action(2, Action(ActionType.ALL_IN))
    assert table.actor == 3
    legal = table.legal_actions(3)
    assert legal.can_raise and legal.min_raise_to == 65
    applied = table.apply_action(3, Action(ActionType.RAISE, 65))
    assert applied.amount == 65
    assert table.last_full_raise_to == 20


def test_short_big_blind_keeps_the_nominal_preflop_bring_in():
    table = table_for([1000, 1000, 5, 1000])
    table.start_hand()
    assert table.current_bet == 20
    legal = table.legal_actions(3)
    assert legal.call_cost == 20 and legal.min_raise_to == 40
    table.apply_action(3, Action(ActionType.CALL))
    assert table.players[3].street_bet == 20


def test_heads_up_only_pays_the_amount_the_short_blind_can_match():
    table = table_for([1000, 15])
    table.start_hand()
    assert table.actor == 0
    assert table.legal_actions(0).call_cost == 5
    table.apply_action(0, Action(ActionType.CALL))
    assert table.is_hand_over and len(table.board) == 5
    assert sum(p.chips for p in table.players) == 1015


def test_empty_initial_button_is_moved_to_a_live_seat():
    table = table_for([0, 1000, 0, 1000])
    table.players[0].seated = table.players[2].seated = False
    table.start_hand()
    assert table.button == 1
    assert table.blind_seats() == (1, 3)
    assert table.actor == 1


def test_blind_positions_survive_elimination_and_heads_up_avoids_repeat_bb():
    table = table_for([1000, 1000, 1000])
    table.start_hand()
    assert table.blind_seats() == (1, 2)
    table.players[0].chips = 0
    table.eliminate_broke_players()
    assert table.blind_seats() == (1, 2)
    table.rotate_button()
    assert table.button == 2
    table.start_hand()
    assert table.blind_seats() == (2, 1)


def test_public_betting_log_includes_spoken_table_talk_only():
    table = table_for([1000, 1000, 1000])
    table.start_hand()
    table.apply_action(0, Action(ActionType.CALL, thought="Private reasoning", speech="Let's see a flop."))
    assert table.hand_log[-1]["speech"] == "Let's see a flop."
    assert "thought" not in table.hand_log[-1]
