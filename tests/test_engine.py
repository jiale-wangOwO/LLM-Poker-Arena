"""Table engine tests: betting order, closure, all-ins, side pots, termination.

These tests are the contract the LLM layer relies on.  If the engine says an
action is legal, applying it must not corrupt chip accounting, and every hand
must terminate.
"""

from __future__ import annotations

import random

import pytest

from pokerarena.cards import Card
from pokerarena.engine import (
    Action,
    ActionType,
    IllegalAction,
    Player,
    PlayerStatus,
    Street,
    Table,
    play_hand,
)


def make_table(stacks, small=10, big=20, seed=1, button=0):
    players = [
        Player(name=f"P{i}", seat=i, chips=chips)
        for i, chips in enumerate(stacks)
    ]
    table = Table(players, small, big, seed=seed)
    table.button = button
    return table


def stack_total(table):
    return sum(p.chips for p in table.players) + table.pot_total


# --------------------------------------------------------------------------
# Seating, blinds and order
# --------------------------------------------------------------------------
def test_six_max_blind_and_order_positions():
    table = make_table([1000] * 6, button=0)
    table.start_hand()
    small, big = table.blind_seats()
    assert (small, big) == (1, 2)
    # Pre-flop action starts with UTG (seat 3); post-flop with SB (seat 1).
    assert table.actor == 3
    assert table.preflop_first_seat() == 3
    assert table.postflop_first_seat() == 1


def test_heads_up_button_posts_small_blind():
    table = make_table([1000, 1000], button=0)
    table.start_hand()
    small, big = table.blind_seats()
    assert small == 0, "heads-up button must post the small blind"
    assert big == 1
    # Heads-up pre-flop the button/SB acts first, post-flop the BB acts first.
    assert table.actor == 0
    assert table.postflop_first_seat() == 1


def test_blinds_are_deducted_once():
    table = make_table([1000] * 4, seed=5)
    table.start_hand()
    assert table.players[1].chips == 990
    assert table.players[2].chips == 980
    assert table.pot_total == 30
    assert table.current_bet == 20


def test_button_rotation_skips_eliminated_players():
    table = make_table([1000, 0, 1000, 1000], button=0)
    table.eliminate_broke_players()
    table.rotate_button()
    assert table.button == 2


# --------------------------------------------------------------------------
# Legal action menus
# --------------------------------------------------------------------------
def test_utg_facing_big_blind_can_fold_call_raise_but_not_check():
    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    legal = table.legal_actions(table.actor)
    assert legal.can_fold and legal.can_call and legal.can_raise
    assert not legal.can_check
    assert legal.call_cost == 20
    assert legal.min_raise_to == 40, "min raise is a full big blind over the current bet"
    assert legal.max_raise_to == 1000


def test_big_blind_gets_the_option_when_everyone_limps():
    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    order = []
    for _ in range(3):
        order.append(table.actor)
        table.apply_action(table.actor, Action(ActionType.CALL))
    # Seats 3, 0, 1 called; now the big blind (seat 2) must still get to act.
    assert order == [3, 0, 1]
    assert table.actor == 2
    legal = table.legal_actions(2)
    assert legal.can_check, "big blind may check its option"
    assert legal.can_raise, "big blind may raise its option"


def test_all_in_blinds_fast_forward_to_showdown_with_full_board():
    # 2 players, 20 each: the button/SB shoves its 10 remaining chips and the
    # big blind is already all-in, so no betting is possible.
    table = make_table([20, 20], small=10, big=20, button=0)
    table.start_hand()
    assert table.is_hand_over, "hand should fast-forward to showdown"
    assert len(table.board) == 5, "the whole board must be dealt"
    assert table.pot == 0, "the pot must be fully distributed"
    assert sum(p.chips for p in table.players) == 40


def test_action_out_of_turn_is_rejected():
    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    with pytest.raises(IllegalAction):
        table.apply_action((table.actor + 1) % 4, Action(ActionType.FOLD))


def test_check_when_facing_a_bet_is_rejected():
    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    with pytest.raises(IllegalAction):
        table.apply_action(table.actor, Action(ActionType.CHECK))


def test_raise_below_minimum_becomes_an_all_in_shove():
    """A sub-minimum raise is not illegal; it is the rest of the stack going in."""
    table = make_table([1000, 1000, 1000, 25], seed=2)
    table.start_hand()
    # Seat 3 is UTG with only 25 behind.
    assert table.actor == 3
    applied = table.apply_action(3, Action(ActionType.RAISE, 25))
    assert applied.type is ActionType.ALL_IN
    assert table.players[3].street_bet == 25
    assert table.current_bet == 25
    assert table.players[3].status is PlayerStatus.ALL_IN


def test_illegal_raise_target_that_is_not_a_shove_is_rejected():
    table = make_table([1000] * 4, seed=2)
    table.start_hand()
    assert table.actor == 3
    with pytest.raises(IllegalAction):
        table.apply_action(3, Action(ActionType.RAISE, 25))


def test_call_cannot_exceed_the_stack():
    table = make_table([15, 1000, 1000, 1000], seed=2)
    table.start_hand()
    # Blinds sit at seats 1 and 2, so seat 3 opens and seat 0 (15 behind) is next.
    assert table.actor == 3
    table.apply_action(3, Action(ActionType.CALL))
    assert table.actor == 0
    legal = table.legal_actions(0)
    assert legal.call_cost == 15
    assert legal.call_is_all_in
    table.apply_action(0, Action(ActionType.CALL))
    assert table.players[0].chips == 0
    assert table.players[0].status is PlayerStatus.ALL_IN


# --------------------------------------------------------------------------
# Betting closure
# --------------------------------------------------------------------------
def test_full_raise_reopens_action_for_players_who_already_called():
    table = make_table([1000] * 4, seed=3)
    table.start_hand()
    acting = table.actor  # seat 3
    table.apply_action(acting, Action(ActionType.CALL))  # seat 3 calls 20
    raiser = table.actor  # seat 0
    table.apply_action(raiser, Action(ActionType.RAISE, 60))
    # Seats 1, 2, 3 all owe 40 more; seat 3 already acted but must act again.
    actors = []
    while table.street is Street.PREFLOP and table.actor is not None:
        actors.append(table.actor)
        legal = table.legal_actions(table.actor)
        table.apply_action(table.actor, Action(ActionType.CALL))
    assert actors == [1, 2, 3], "the original caller must be given another turn"


def call_or_check(table):
    """Take the passive legal action for whoever is to act."""
    legal = table.legal_actions(table.actor)
    table.apply_action(
        table.actor,
        Action(ActionType.CHECK if legal.can_check else ActionType.CALL),
    )


def test_street_closes_when_everyone_matched():
    table = make_table([1000] * 4, seed=3)
    table.start_hand()
    steps = 0
    while table.street is Street.PREFLOP and not table.is_hand_over:
        call_or_check(table)
        steps += 1
        assert steps < 20
    # Everyone acted pre-flop, then the street reset for the flop.
    assert table.players[3].last_action is not None
    assert table.players[3].last_action.type is ActionType.CALL
    assert table.street is Street.FLOP
    assert len(table.board) == 3
    assert table.pot == 80
    # A new street clears the per-street action flags for everybody.
    assert not any(p.has_acted for p in table.players)


def test_fold_ends_the_hand_immediately():
    table = make_table([1000] * 4, seed=4)
    table.start_hand()
    for _ in range(3):
        if table.is_hand_over:
            break
        table.apply_action(table.actor, Action(ActionType.FOLD))
    assert table.is_hand_over
    winner = table.contenders[0]
    assert winner.name == "P2", "everyone folded to the big blind"
    assert winner.chips == 1010, "winner collects the blinds (10 + 20)"
    assert table.pot == 0, "the pot must be emptied on award"
    assert sum(p.chips for p in table.players) == 4000


def test_short_all_in_does_not_reopen_for_a_player_who_already_called():
    """A short all-in raise must not give an earlier caller a fresh action."""
    table = make_table([30, 1000, 1000, 1000], seed=11)
    table.start_hand()
    # Pre-flop order: seat 3 (UTG), seat 0, seat 1, seat 2.
    assert table.actor == 3
    table.apply_action(3, Action(ActionType.CALL))  # seat 3 limps for 20
    assert table.actor == 0
    # Seat 0 shoves its remaining 30: a raise of only 10 (less than the 20
    # minimum), so it is a short all-in.
    table.apply_action(0, Action(ActionType.ALL_IN))
    assert table.players[0].street_bet == 30
    assert table.current_bet == 30
    # Seat 1 is next in turn order (it still owes 20).
    assert table.actor == 1
    table.apply_action(1, Action(ActionType.CALL))
    table.apply_action(2, Action(ActionType.CALL))
    # Seat 3 limped for 20 and owes 10 more, so it does get to act -- exactly once.
    assert table.actor == 3
    table.apply_action(3, Action(ActionType.CALL))
    assert table.street is Street.FLOP, "street closes once everyone matched 30"
    assert table.pot == 30 * 4


def test_short_shove_does_not_repeat_turns_within_a_street():
    table = make_table([30, 1000, 1000, 1000], seed=12)
    table.start_hand()
    assert table.actor == 3
    table.apply_action(3, Action(ActionType.CALL))
    table.apply_action(0, Action(ActionType.ALL_IN))
    seen = []
    guard = 0
    while table.street is Street.PREFLOP and not table.is_hand_over:
        guard += 1
        assert guard < 10, "street is looping"
        seen.append(table.actor)
        call_or_check(table)
    # Seats 1 and 2 owe the full 30; seat 3 owes 10 more and acts last, once.
    assert seen == [1, 2, 3], "each player acts once, nobody is asked twice"
    assert table.players[0].status is PlayerStatus.ALL_IN
    assert table.pot == 120


# --------------------------------------------------------------------------
# All-ins, side pots and termination
# --------------------------------------------------------------------------
def test_multi_way_all_in_creates_side_pots_and_conserves_chips():
    stacks = [100, 300, 60, 500]
    table = make_table(stacks, seed=21)
    total_before = sum(stacks)
    table.start_hand()
    # Everyone shoves.
    while not table.is_hand_over:
        legal = table.legal_actions(table.actor)
        table.apply_action(table.actor, Action(ActionType.ALL_IN))
    assert table.is_hand_over
    assert sum(p.chips for p in table.players) == total_before
    assert len(table.pots_snapshot) >= 2, "unequal stacks must produce side pots"


def test_chip_conservation_over_many_random_hands():
    rng = random.Random(99)
    stacks = [1000] * 5
    players = [Player(name=f"P{i}", seat=i, chips=stacks[i]) for i in range(5)]
    table = Table(players, 10, 20, seed=99)
    for _ in range(300):
        if table.is_game_over():
            break
        before = sum(p.chips for p in table.players)
        table.eliminate_broke_players()
        if table.is_game_over():
            break

        def decide(tbl, seat, _rng=rng):
            legal = tbl.legal_actions(seat)
            roll = _rng.random()
            if legal.can_check and roll < 0.45:
                return Action(ActionType.CHECK)
            if roll < 0.75 and legal.can_call:
                return Action(ActionType.CALL)
            if roll < 0.82:
                return Action(ActionType.FOLD)
            if legal.can_raise and roll < 0.9:
                target = min(legal.max_raise_to, legal.min_raise_to + 20)
                return Action(ActionType.RAISE, target)
            if legal.all_in_to:
                return Action(ActionType.ALL_IN)
            return Action(ActionType.CHECK if legal.can_check else ActionType.CALL)

        play_hand(table, decide)
        after = sum(p.chips for p in table.players)
        assert after == before, f"chips leaked: {before} -> {after}"
        table.rotate_button()
    assert sum(p.chips for p in table.players) == sum(stacks)


def test_hand_always_terminates_with_aggressive_ai():
    stacks = [200, 200, 200, 200]
    table = make_table(stacks, seed=7)

    def decide(tbl, seat):
        legal = tbl.legal_actions(seat)
        if legal.can_raise:
            return Action(ActionType.RAISE, legal.max_raise_to)
        if legal.all_in_to:
            return Action(ActionType.ALL_IN)
        return Action(ActionType.CALL)

    for _ in range(30):
        if table.is_game_over():
            break
        play_hand(table, decide)
        table.eliminate_broke_players()
        table.rotate_button()
    assert sum(p.chips for p in table.players) == 800


def test_side_pot_awarded_correctly_end_to_end():
    """The short stack wins the main pot; a bigger stack wins the side pot."""
    table = make_table([300, 300, 300], seed=31)
    table.start_hand()
    # Rig the board so the result is deterministic: seat 0 (short) makes trips,
    # seat 1 makes two pair, seat 2 has nothing.
    table.board = [Card.from_str(c) for c in ("Kh", "8d", "2c", "5s", "9h")]
    table.players[0].hole_cards = [Card.from_str("Kd"), Card.from_str("Kc")]
    table.players[1].hole_cards = [Card.from_str("8h"), Card.from_str("2d")]
    table.players[2].hole_cards = [Card.from_str("3h"), Card.from_str("4d")]

    # Coherent all-in state: everyone started with 300.
    #   seat 0 was all-in for its whole 300
    #   seats 1 and 2 committed 300 each as well
    table.players[0].chips = 0
    table.players[1].chips = 0
    table.players[2].chips = 0
    table.hand_contributions = {0: 300, 1: 300, 2: 300}
    table.pot = 900
    table.players[0].status = PlayerStatus.ACTIVE
    table.players[1].status = PlayerStatus.ACTIVE
    table.players[2].status = PlayerStatus.ACTIVE
    table.street = Street.RIVER
    table.to_showdown()

    # Equal contributions means a single 900 main pot, taken by seat 0's trips.
    assert table.players[0].chips == 900, "short stack takes the main pot"
    assert table.players[1].chips == 0
    assert table.players[2].chips == 0
    assert sum(p.chips for p in table.players) == 900
    assert table.pot == 0


def test_short_all_in_wins_only_the_main_pot():
    """A short all-in cannot win the side pot it never paid into."""
    table = make_table([100, 500, 500], seed=32)
    table.start_hand()
    # Seat 0 is all-in for 100 and *does* have the best hand, but it may only
    # win the 300 main pot; the 800 side pot goes to the best of seats 1/2.
    table.board = [Card.from_str(c) for c in ("Kh", "8d", "2c", "5s", "9h")]
    table.players[0].hole_cards = [Card.from_str("Kd"), Card.from_str("Kc")]  # trips
    table.players[1].hole_cards = [Card.from_str("8h"), Card.from_str("2d")]  # two pair
    table.players[2].hole_cards = [Card.from_str("3h"), Card.from_str("4d")]  # nothing

    table.players[0].chips = 0
    table.players[1].chips = 100
    table.players[2].chips = 100
    table.hand_contributions = {0: 100, 1: 400, 2: 400}
    table.pot = 900
    for player in table.players:
        player.status = PlayerStatus.ACTIVE
    table.street = Street.RIVER
    table.to_showdown()

    # main pot 100*3 = 300 -> seat 0 (trips)
    # side pot 300*2 = 600 -> seat 1 (two pair)
    assert table.players[0].chips == 300
    assert table.players[1].chips == 100 + 600
    assert table.players[2].chips == 100
    assert sum(p.chips for p in table.players) == 1100
    assert len(table.pots_snapshot) == 2
    assert table.pot == 0


def test_tied_players_split_and_chips_are_conserved():
    table = make_table([200, 200], seed=41)
    table.start_hand()
    table.board = [Card.from_str(c) for c in ("Ah", "Kd", "Qc", "Js", "Th")]
    # Both play the board: a straight on the board is a split pot.
    table.players[0].hole_cards = [Card.from_str("2c"), Card.from_str("3d")]
    table.players[1].hole_cards = [Card.from_str("4c"), Card.from_str("5d")]
    table.hand_contributions = {0: 200, 1: 200}
    table.players[0].chips = 0
    table.players[1].chips = 0
    table.players[0].status = PlayerStatus.ALL_IN
    table.players[1].status = PlayerStatus.ALL_IN
    table.street = Street.RIVER
    table.to_showdown()
    assert table.players[0].chips == 200
    assert table.players[1].chips == 200


def test_uncalled_bet_is_returned():
    """An unmatched shove comes straight back; it is never counted as won."""
    table = make_table([1000, 1000], seed=51)
    table.start_hand()
    table.hand_contributions = {0: 1000, 1: 20}
    table.players[0].chips = 0
    table.players[1].chips = 0
    table.players[0].status = PlayerStatus.ACTIVE
    table.players[1].status = PlayerStatus.FOLDED
    table.street = Street.RIVER
    table._settle_pots({0: 5000}, [])

    # 980 of seat 0's shove was never matched and comes straight back.  Seat 1's
    # dead 20 forms a pot that only seat 0 can contest, so seat 0 collects it.
    # In real play this state is unreachable: the engine awards the pot via
    # _end_hand_early the moment everyone else folds.
    assert table.players[0].chips == 1020, "980 refunded + the dead 40 pot"
    assert table.players[1].chips == 0
    assert sum(p.chips for p in table.players) == 1020
    assert table.pot == 0


def test_uncalled_bet_in_a_real_hand_is_returned():
    """End-to-end: a shove nobody can call does not vanish from the stacks."""
    table = make_table([1000, 100], seed=52)
    table.start_hand()
    shover = table.actor
    # Seat 0 shoves 1000 pre-flop; everybody else folds.
    table.apply_action(shover, Action(ActionType.ALL_IN))
    while not table.is_hand_over:
        table.apply_action(table.actor, Action(ActionType.FOLD))
    assert sum(p.chips for p in table.players) == 1100
    assert table.pot == 0
    # Seat 1 only ever posted the big blind, so the shover must get the rest back.
    assert table.players[shover].chips == 1020


# --------------------------------------------------------------------------
# Board and street progression
# --------------------------------------------------------------------------
def test_board_grows_correctly_and_deck_accounts_for_burns():
    table = make_table([1000] * 4, seed=61)
    table.start_hand()
    while not table.is_hand_over:
        legal = table.legal_actions(table.actor)
        table.apply_action(table.actor, Action(ActionType.CHECK if legal.can_check else ActionType.CALL))
    assert len(table.board) == 5
    # 8 hole cards + 5 board + 3 burns = 16 cards consumed.
    assert len(table.deck) == 52 - 16


def test_showdown_identifies_category_for_every_contender():
    table = make_table([500] * 3, seed=71)
    table.start_hand()
    while not table.is_hand_over:
        legal = table.legal_actions(table.actor)
        table.apply_action(table.actor, Action(ActionType.CHECK if legal.can_check else ActionType.CALL))
    assert table.showdown_results
    for entry in table.showdown_results:
        assert entry["hand_name"]
        assert entry["score"] > 0


def test_elimination_marks_players_sitting_out():
    table = make_table([0, 500, 500], seed=81)
    eliminated = table.eliminate_broke_players()
    assert [p.name for p in eliminated] == ["P0"]
    assert table.players[0].status is PlayerStatus.SITTING_OUT
    assert not table.is_game_over()


def test_game_over_with_one_player_left():
    table = make_table([0, 500], seed=82)
    table.eliminate_broke_players()
    assert table.is_game_over()


# --------------------------------------------------------------------------
# Regression guards for bugs found in the original implementation
# --------------------------------------------------------------------------
def test_blinds_are_not_wiped_by_street_reset():
    """Original bug: the pre-flop street reset zeroed the freshly posted blinds."""
    table = make_table([1000] * 4, seed=91)
    table.start_hand()
    assert table.players[1].street_bet == 10
    assert table.players[2].street_bet == 20
    assert table.current_bet == 20


def test_raise_amount_is_a_total_not_a_delta():
    """Original bug: 'raise 100' added 100 to the pot without moving the price."""
    table = make_table([1000] * 4, seed=92)
    table.start_hand()
    # Seat 3 opens (UTG), seat 0 raises to a total of 100.
    table.apply_action(3, Action(ActionType.CALL))
    assert table.actor == 0
    table.apply_action(0, Action(ActionType.RAISE, 100))
    assert table.players[0].street_bet == 100
    assert table.current_bet == 100
    assert table.players[0].chips == 900
    # Seat 1 posted the small blind, so it owes 90 more.
    assert table.actor == 1
    assert table.legal_actions(1).call_cost == 90


def test_minimum_raise_tracks_the_previous_raise_size():
    table = make_table([1000] * 4, seed=94)
    table.start_hand()
    table.apply_action(3, Action(ActionType.CALL))
    table.apply_action(0, Action(ActionType.RAISE, 100))  # raise of 80 over 20
    legal = table.legal_actions(1)
    assert legal.min_raise_to == 180, "must raise by at least the previous raise (80)"


def test_pot_is_not_double_counted():
    table = make_table([1000] * 4, seed=93)
    table.start_hand()
    assert table.pot_total == 30
    table.apply_action(table.actor, Action(ActionType.CALL))
    assert table.pot_total == 50
    assert sum(p.chips for p in table.players) + table.pot_total == 4000
