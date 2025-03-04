# Author: [Your Name]
# File: game.py
# Description: This file contains the main game logic for a Texas Hold'em Poker game, 
# including the PokerGame class, the Player class, and other game-related methods like betting_round, post_blinds, etc.

# PokerGame handles the game flow (e.g., rotating dealers, removing broke players).
# Player holds information about the player (e.g., chips, cards, actions).

from card import Deck, Card
from hand import Hand
from utils import print_bold, logger, log_sensitive
from ui_helper import GameUI
from datetime import datetime
from ai_helper import get_ai_action

class Player:
    """
    Player class containing player name, chip count, private cards, and current betting status.
    """
    def __init__(self, name, chips, is_ai=False):
        """
        Initialize the player object.

        :param name: Player's name.
        :param chips: Initial chip count.
        """
        self.name = name
        self.chips = chips
        self.hole_cards = []   # Private cards
        self.is_ai = is_ai
        self.folded = False    # Has the player folded in this round?
        self.current_bet = 0   # Bet in this betting round
        self.total_bet = 0     # Total bet during the game round
        self.action_taken = False # Whether the player acted in the current betting round
        self.action_history = []  # List of actions taken by the player in this round

    def reset_for_round(self):
        """
        Reset player status for a new game round (clear private cards, reset bets and fold status).
        """
        self.hole_cards = []
        self.folded = False
        self.current_bet = 0
        self.total_bet = 0     # Total bet during the game
        self.action_taken = False
        self.action_history = []  # Reset action history for new round

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

    def record_action(self, betting_round, action, amount=None):
        """Record an action taken by the player during a betting round"""
        action_record = {
            "round": betting_round,
            "action": action,
            "amount": amount,
            "timestamp": datetime.now()
        }
        self.action_history.append(action_record)


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
            # Log the cards dealt to this player
            log_sensitive(f"Dealt cards to {player.name}", {
                "player": player.name,
                "cards": player.hole_cards,
                "remaining_deck": len(self.deck.cards)
            })

    def post_blinds(self):
        """
        Post blinds: the player after the dealer posts the small blind, and the next player posts the big blind.
        Update the pot and the current highest bet.
        """
        GameUI.print_section("Posting Blinds")
        
        small_blind_player = self.players[(self.dealer_pos + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_pos + 2) % len(self.players)]
        
        try:
            small_blind_player.place_bet(self.small_blind)
            GameUI.print_action(small_blind_player.name, "posts small blind", self.small_blind)
            small_blind_player.record_action("Pre-flop", "post small blind", self.small_blind)
        except ValueError:
            GameUI.print_error(f"{small_blind_player.name} cannot post small blind (insufficient chips).")
            logger.warning(f"Player {small_blind_player.name} couldn't post small blind - insufficient chips")
        
        try:
            big_blind_player.place_bet(self.big_blind)
            GameUI.print_action(big_blind_player.name, "posts big blind", self.big_blind)
            big_blind_player.record_action("Pre-flop", "post big blind", self.big_blind)
        except ValueError:
            GameUI.print_error(f"{big_blind_player.name} cannot post big blind (insufficient chips).")
            logger.warning(f"Player {big_blind_player.name} couldn't post big blind - insufficient chips")
        
        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        
        GameUI.print_pot(self.pot)
        logger.info(f"Blinds posted: small={self.small_blind}, big={self.big_blind}, pot={self.pot}")

    def betting_round(self, round_name):
        """
        Conduct a betting round until all participants' bets are equal or only one player remains.

        :param round_name: Name of the betting round (e.g., "Pre-flop", "Flop", "Turn", "River").
        """
        GameUI.print_header(f"{round_name} Betting Round")
        logger.info(f"Starting {round_name} betting round")
        
        # For Pre-flop, don't reset the `current_bet` for players who have posted the blinds
        if round_name != "Pre-flop":
            for player in self.players:
                player.current_bet = 0  # Reset current bet for the current round
                player.action_taken = False
            
        active_players = [p for p in self.players if not p.folded and p.chips > 0]  # Remove folded players
        if len(active_players) <= 1:
            logger.info(f"Betting round {round_name} skipped - only one active player")
            return
        
        # Pre-flop stage starts from the player after the big blind
        if round_name == "Pre-flop":
            current_player_index = (self.dealer_pos + 3) % len(self.players)  # Big blind's next player starts
        else:
            current_player_index = (self.dealer_pos + 1) % len(self.players)  # Start with player after dealer
        
        while True:
            # Skip folded players
            while self.players[current_player_index].folded:
                current_player_index = (current_player_index + 1) % len(self.players)  # Skip to next player
            
            player = self.players[current_player_index]
            
            # Display comprehensive game state
            self.display_game_state(current_player_index, round_name)
            
            # Allow the player to take action
            valid_action = False
            while not valid_action:
                if player.is_ai:
                    # Construct game_state with all available information.
                    game_state = {
                        "community_cards": self.community_cards,
                        "pot": self.pot,
                        "current_bet": self.current_bet,
                        "round": round_name,
                        "your_position": self.get_player_position(current_player_index),
                        "opponents": [{
                            "name": opp.name,
                            "position": self.get_player_position(i),
                            "chips": opp.chips,
                            "current_bet": opp.current_bet,
                            "action_history": opp.action_history,
                            "folded": opp.folded
                        } for i, opp in enumerate(self.players) if opp != player]
                    }
                    # Get AI's decision
                    action = get_ai_action(player, game_state)
                else:
                    # For human players, prompt for input (assuming GameUI.prompt_action exists)
                    action = GameUI.prompt_action(player.name)
                
                
                if action == "fold":
                    player.folded = True
                    GameUI.print_action(player.name, "fold")
                    player.record_action(round_name, "fold")
                    valid_action = True
                    
                elif action == "call":
                    call_amount = self.current_bet - player.current_bet
                    if call_amount > player.chips:
                        call_amount = player.chips  # All-in if not enough chips
                        GameUI.print_action(player.name, "all-in", call_amount)
                        player.record_action(round_name, "all-in", call_amount)
                        logger.info(f"{player.name} forced all-in with {call_amount} when attempting to call")
                    else:
                        GameUI.print_action(player.name, "call", call_amount)
                        player.record_action(round_name, "call", call_amount)
                    player.place_bet(call_amount)
                    self.pot += call_amount
                    valid_action = True
                    
                elif action == "check":
                    if player.current_bet == self.current_bet:
                        GameUI.print_action(player.name, "check")
                        player.record_action(round_name, "check")
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
                    GameUI.print_action(player.name, "raise", raise_amount)
                    player.record_action(round_name, "raise", raise_amount)
                    valid_action = True

                elif action == "all-in":
                    all_in_amount = player.chips  # All-in with remaining chips
                    player.place_bet(all_in_amount)
                    self.pot += all_in_amount
                    self.current_bet = max(player.current_bet, self.current_bet)
                    GameUI.print_action(player.name, "all-in", all_in_amount)
                    player.record_action(round_name, "all-in", all_in_amount)
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

    def display_game_state(self, current_player_index, round_name):
        """Display comprehensive game state when it's a player's turn"""
        player = self.players[current_player_index]
        position = self.get_player_position(current_player_index)
        
        GameUI.print_section(f"{player.name}'s Turn - {position}")
        GameUI.print_info(f"Current Round: {round_name}")
        GameUI.print_pot(self.pot)
        GameUI.print_cards("Community Cards", self.community_cards)
        GameUI.print_player_info(player.name, player.hole_cards, player.chips, player.current_bet)
        print(f"Current Table Bet: {self.current_bet}")
        
        # Display opponent information
        GameUI.print_info("\nOpponent Information:\n")
        for i, opp in enumerate(self.players):
            if opp != player and not opp.folded:
                opp_position = self.get_player_position(i)
                print(f"{opp.name} ({opp_position}): Chips: {opp.chips}, Current Bet: {opp.current_bet}")
                
                # Show opponent action history
                if opp.action_history:
                    print(f"  Action History:")
                    for action in opp.action_history:
                        if action["amount"]:
                            print(f"    {action['round']}: {action['action']} {action['amount']}")
                        else:
                            print(f"    {action['round']}: {action['action']}")
                print()

    def deal_community_cards(self, number):
        """
        Deal community cards.

        :param number: Number of community cards to deal.
        """
        # Determine the current stage name
        stage_name = "Flop" if len(self.community_cards) == 0 else "Turn" if len(self.community_cards) == 3 else "River"
        GameUI.print_section(f"Dealing the {stage_name}")
        
        # Deal the cards
        new_cards = []
        for _ in range(number):
            card = self.deck.deal()
            self.community_cards.append(card)
            new_cards.append(card)
        
        GameUI.print_cards(f"New {stage_name} Card{'s' if number > 1 else ''}", new_cards)
        GameUI.print_cards("All Community Cards", self.community_cards)
        
        # Log the dealt cards
        logger.info(f"Dealt {stage_name}: {new_cards}")
        logger.info(f"Community cards now: {self.community_cards}")

    def showdown(self):
        """
        Showdown stage: if multiple players remain, reveal hands,
        determine the best combination using Hand.best_hand, and compare hands using Hand.compare_hands.
        Award the pot to the winner (side pot logic not handled here).
        """
        GameUI.print_header("Showdown")
        
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            GameUI.print_winner(winner.name, self.pot, "(all others folded)")
            winner.chips += self.pot
            logger.info(f"{winner.name} wins pot of {self.pot} by default (all others folded)")
        else:
            GameUI.print_info("Showdown! Players reveal their cards:")
            GameUI.print_cards("Community Cards", self.community_cards)
            # Display each player's hand and best combination
            for player in active_players:
                best = Hand.best_hand(player.hole_cards, self.community_cards)
                hand_desc = best.evaluate_hand()
                GameUI.print_player_info(player.name, player.hole_cards, player.chips, player.current_bet)
                GameUI.print_info(f"  Best hand: {hand_desc}")
                player.best_hand = best  # Store the best hand for comparison
                logger.info(f"{player.name}'s best hand is {hand_desc} with {player.hole_cards} + {self.community_cards}")
            
            # Determine the winner
            best_player = active_players[0]
            for player in active_players[1:]:
                result = Hand.compare_hands(best_player.best_hand, player.best_hand)
                if result == "Player 2 wins":
                    best_player = player
                
            # Award the pot
            best_player.chips += self.pot
            GameUI.print_winner(best_player.name, self.pot, best_player.best_hand.evaluate_hand())
            logger.info(f"{best_player.name} won showdown with {best_player.best_hand.evaluate_hand()}, winning {self.pot} chips")

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
        logger.info(f"Starting new game with {len(self.players)} players")
        
        while len(self.players) > 1:
            GameUI.print_header(f"Round {round_number}")
            logger.info(f"Starting Round {round_number}")
            
            # Log dealer position
            dealer_name = self.players[self.dealer_pos].name
            logger.info(f"Dealer: {dealer_name}")
            GameUI.print_info(f"Dealer: {dealer_name}")
            
            # Play the round
            self.play_round()
            
            # End of round cleanup
            self.remove_broke_players()
            GameUI.print_round_summary(self.players)
            self.rotate_dealer()
            round_number += 1
            
            # Pause between rounds if there are still multiple players
            if len(self.players) > 1:
                GameUI.prompt_next_round()
        
        # Game over
        if self.players:
            GameUI.print_game_over(self.players[0].name, self.players[0].chips)
        else:
            GameUI.print_error("Game ended with no players remaining!")

    def get_player_position(self, player_index):
        """Return the player's position name based on their index relative to dealer"""
        if player_index == self.dealer_pos:
            return "Dealer (BTN)"
        elif player_index == (self.dealer_pos + 1) % len(self.players):
            return "Small Blind (SB)"
        elif player_index == (self.dealer_pos + 2) % len(self.players):
            return "Big Blind (BB)"
        else:
            relative_pos = (player_index - self.dealer_pos) % len(self.players)
            positions = ["UTG", "UTG+1", "MP", "MP+1", "CO"]
            if relative_pos - 3 < len(positions):
                return positions[relative_pos - 3]
            return f"Position {relative_pos}"
