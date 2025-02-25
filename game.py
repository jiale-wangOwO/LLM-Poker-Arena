# Author: [Your Name]
# File: game.py
# Description: This file contains the main game logic for a Texas Hold'em Poker game, 
# including the PokerGame class, the Player class, and other game-related methods like betting_round, post_blinds, etc.

# PokerGame handles the game flow (e.g., rotating dealers, removing broke players).
# Player holds information about the player (e.g., chips, cards, actions).

from card import Deck, Card
from hand import Hand
from utils import print_bold, logger

class Player:
    """
    Player class containing player name, chip count, private cards, and current betting status.
    """
    def __init__(self, name, chips):
        """
        Initialize the player object.

        :param name: Player's name.
        :param chips: Initial chip count.
        """
        self.name = name
        self.chips = chips
        self.hole_cards = []   # Private cards
        self.folded = False    # Has the player folded in this round?
        self.current_bet = 0   # Bet in this betting round
        self.total_bet = 0     # Total bet during the game round
        self.action_taken = False # Whether the player acted in the current betting round

    def reset_for_round(self):
        """
        Reset player status for a new game round (clear private cards, reset bets and fold status).
        """
        self.hole_cards = []
        self.folded = False
        self.current_bet = 0
        self.total_bet = 0     # Total bet during the game
        self.action_taken = False

    def place_bet(self, amount):
        """
        Player places a bet, deducting chips and updating the current bet amount.

        :param amount: Bet amount.
        :raises ValueError: If there are insufficient chips.
        """
        if amount > self.chips:
            raise ValueError("Insufficient chips")
        self.chips -= amount
        self.current_bet += amount
        self.total_bet += amount


class PokerGame:
    """
    Texas Hold'em Poker Game class to manage the flow of a game, including the deck, players, community cards, pot, and various stages.
    """
    def __init__(self, players, small_blind=50, big_blind=100):
        """
        Initialize the game.

        :param players: List of player objects.
        :param small_blind: Amount for the small blind.
        :param big_blind: Amount for the big blind.
        """
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_pos = 0      # Dealer position index
        self.pot = 0             # Current pot
        self.community_cards = []  # List of community cards
        self.deck = None
        self.current_bet = 0

    def rotate_dealer(self):
        """
        Rotate the dealer position for the next round.
        """
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)

    def remove_broke_players(self):
        """
        Remove players with zero chips.
        """
        for p in self.players:
            if p.chips == 0:
                print(f"{p.name} is broke and removed from the game.")
        self.players = [p for p in self.players if p.chips > 0]

    def setup_round(self):
        """
        Set up a new round: initialize the deck, reset the pot, community cards, and player states, and deal two cards to each player.
        """
        self.deck = Deck()
        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        for player in self.players:
            player.reset_for_round()
        for player in self.players:
            player.hole_cards = [self.deck.deal(), self.deck.deal()]

    def post_blinds(self):
        """
        Post blinds: the player after the dealer posts the small blind, and the next player posts the big blind.
        Update the pot and the current highest bet.
        """
        small_blind_player = self.players[(self.dealer_pos + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_pos + 2) % len(self.players)]
        try:
            small_blind_player.place_bet(self.small_blind)
            print(f"{small_blind_player.name} posts small blind: {self.small_blind}")
        except ValueError:
            print(f"{small_blind_player.name} cannot post small blind (insufficient chips).")
        try:
            big_blind_player.place_bet(self.big_blind)
            print(f"{big_blind_player.name} posts big blind: {self.big_blind}")
        except ValueError:
            print(f"{big_blind_player.name} cannot post big blind (insufficient chips).")
        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind

    def betting_round(self, round_name):
        """
        Conduct a betting round until all participants' bets are equal or only one player remains.

        :param round_name: Name of the betting round (e.g., "Pre-flop", "Flop", "Turn", "River").
        """
        print(f"=== {round_name} Betting Round ===")

        # For Pre-flop, don't reset the `current_bet` for players who have posted the blinds
        if round_name != "Pre-flop":
            for player in self.players:
                player.current_bet = 0  # Reset current bet for the current round
                player.action_taken = False
        active_players = [p for p in self.players if not p.folded and p.chips > 0]  # Remove folded players
        if len(active_players) <= 1:
            return

        # Pre-flop stage starts from the player after the big blind
        if round_name == "Pre-flop":
            current_player_index = (self.dealer_pos + 3) % len(self.players)  # Big blind's next player starts
        else:
            current_player_index = (self.dealer_pos + 1) % len(self.players)  # Other stages start from the next player after the dealer

        while True:
            # Skip folded players
            while self.players[current_player_index].folded:
                current_player_index = (current_player_index + 1) % len(self.players)  # Skip to next player

            player = self.players[current_player_index]
            print("-" * 40)
            print(player.name + "'s turn:")
            print(f"Pot: {self.pot}")
            print("Community Cards:", self.community_cards)
            print(f"Your hand: {player.hole_cards}")
            print(f"Your chips: {player.chips}")
            print(f"Your total bet: {player.total_bet}")
            print(f"Your current bet: {player.current_bet}")
            print(f"Table current bet: {self.current_bet}")
            print(f"Action taken this round: {player.action_taken}")
            print("-" * 40)

            # Allow the player to take action
            valid_action = False
            while not valid_action:
                action = input(f"Enter action for {player.name} (fold, call, raise [amount], check, all-in): ").strip().lower()

                if action == "fold":
                    player.folded = True
                    # print(f"{player.name} folds.")
                    print_bold(f"{player.name} folds.")
                    valid_action = True

                elif action == "call":
                    call_amount = self.current_bet - player.current_bet
                    if call_amount > player.chips:
                        call_amount = player.chips  # All-in if not enough chips
                        # print(f"{player.name} is all-in with {call_amount}.")
                        print_bold(f"{player.name} is all-in with {call_amount}.")
                    player.place_bet(call_amount)
                    self.pot += call_amount
                    # print(f"{player.name} calls {call_amount}.")
                    print_bold(f"{player.name} calls {call_amount}.")
                    valid_action = True

                elif action == "check":
                    if player.current_bet == self.current_bet:
                        # print(f"{player.name} checks.")
                        print_bold(f"{player.name} checks.")
                        valid_action = True
                    else:
                        print("Cannot check, you must call or raise.")

                elif action.startswith("raise"):
                    try:
                        raise_amount = int(action.split()[1])  # Get the raise amount
                        total_bet = raise_amount + player.current_bet
                    except (IndexError, ValueError):
                        print("Invalid raise amount. Please specify the amount to raise to.")
                        continue
                    player.place_bet(raise_amount)
                    self.pot += raise_amount
                    self.current_bet = total_bet
                    # print(f"{player.name} raises to {total_bet}.")
                    print_bold(f"{player.name} raises to {total_bet}.")
                    valid_action = True

                elif action == "all-in":
                    all_in_amount = player.chips  # All-in with remaining chips
                    player.place_bet(all_in_amount)
                    self.pot += all_in_amount
                    self.current_bet = max(player.current_bet, self.current_bet)
                    # print(f"{player.name} goes all-in with {all_in_amount}.")
                    print_bold(f"{player.name} goes all-in with {all_in_amount}.")
                    valid_action = True

                else:
                    print("Invalid action. Please enter a valid action (fold, call, raise [amount], check, all-in).")

            player.action_taken = True
            # Move to the next player in the round
            current_player_index = (current_player_index + 1) % len(self.players)  # Go to next player

            # Check if all players have called, raised, or folded
            bets = [p.current_bet for p in active_players if not p.folded]

            # If all bets are equal and all active players have taken action, end the round
            if len(set(bets)) == 1 and (all(p.action_taken for p in active_players if not p.folded)):
                break

    def deal_community_cards(self, number):
        """
        Deal community cards.

        :param number: Number of community cards to deal.
        """
        for _ in range(number):
            self.community_cards.append(self.deck.deal())
        print('-' * 40)
        print(' ')
        print("Community Cards:", self.community_cards)
        print(' ')
        print('-' * 40)

    def showdown(self):
        """
        Showdown stage: if multiple players remain, reveal hands,
        determine the best combination using Hand.best_hand, and compare hands using Hand.compare_hands.
        Award the pot to the winner (side pot logic not handled here).
        """
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            print(f"{winner.name} wins the pot of {self.pot} (all others folded).")
            winner.chips += self.pot
        else:
            print("Showdown!")
            for player in active_players:
                best = Hand.best_hand(player.hole_cards, self.community_cards)
                print(f"{player.name}'s best hand is {best.evaluate_hand()} with {player.hole_cards} + {self.community_cards}")
                player.best_hand = best  # Store the best hand for comparison
            best_player = active_players[0]
            for player in active_players[1:]:
                result = Hand.compare_hands(best_player.best_hand, player.best_hand)
                if result == "Player 2 wins":
                    best_player = player
            print(f"{best_player.name} wins the pot of {self.pot} with {best_player.best_hand.evaluate_hand()}!")
            best_player.chips += self.pot

    def play_round(self):
        """
        Execute a complete round of Texas Hold'em, including dealing cards, posting blinds, betting rounds, dealing community cards, and showdown.
        """
        self.setup_round()
        self.post_blinds()
        
        # Pre-flop betting
        self.betting_round("Pre-flop")
        
        if len([p for p in self.players if not p.folded]) > 1:
            # Flop
            self.deal_community_cards(3)
            self.current_bet = 0
            self.betting_round("Flop")
            
        if len([p for p in self.players if not p.folded]) > 1:
            # Turn
            self.deal_community_cards(1)
            self.current_bet = 0
            self.betting_round("Turn")
            
        if len([p for p in self.players if not p.folded]) > 1:
            # River
            self.deal_community_cards(1)
            self.current_bet = 0
            self.betting_round("River")
            
        self.showdown()

    def play_game(self):
        """
        Play the entire game loop until only one player remains (with chips greater than 0).
        After each round, update the dealer position and remove players with zero chips.
        """
        round_number = 1
        while len(self.players) > 1:
            print("=" * 40)
            print(f"Starting Round {round_number}")
            self.play_round()
            self.remove_broke_players()
            for player in self.players:
                print(f"{player.name}: {player.chips} chips")
            self.rotate_dealer()
            round_number += 1
        print(f"Game over! Winner is {self.players[0].name} with {self.players[0].chips} chips.")
