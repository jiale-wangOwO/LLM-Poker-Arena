import os
import sys
from datetime import datetime
from utils import logger

# ANSI color codes
COLORS = {
    'RESET': '\033[0m',
    'BOLD': '\033[1m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD_GREEN': '\033[1;32m',
    'BOLD_YELLOW': '\033[1;33m',
    'BOLD_BLUE': '\033[1;34m',
    'BOLD_RED': '\033[1;31m',
    'BOLD_CYAN': '\033[1;36m',
    'BOLD_MAGENTA': '\033[1;35m',
}

class GameUI:
    """Helper class for formatted terminal output and logging in the poker game."""
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen for better readability."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(text, width=80):
        """Print a header with centered text."""
        print(f"\n{COLORS['BOLD_BLUE']}{text.center(width, '=')}{COLORS['RESET']}\n")
        logger.info(f"HEADER: {text}")
    
    @staticmethod
    def print_section(text, width=80):
        """Print a section divider with text."""
        print(f"\n{COLORS['BOLD_CYAN']}{text.center(width, '-')}{COLORS['RESET']}\n")
        logger.debug(f"SECTION: {text}")
    
    @staticmethod
    def print_info(text):
        """Print general information text."""
        print(f"{COLORS['WHITE']}{text}{COLORS['RESET']}")
        logger.info(text)
    
    @staticmethod
    def print_player_info(name, hole_cards, chips, current_bet):
        """Print player information in a formatted way."""
        print(f"{COLORS['BOLD']}Player: {COLORS['BOLD_BLUE']}{name}{COLORS['RESET']}")
        print(f"  Hand: {COLORS['YELLOW']}{hole_cards}{COLORS['RESET']}")
        print(f"  Chips: {COLORS['GREEN']}{chips}{COLORS['RESET']}")
        print(f"  Current Bet: {COLORS['GREEN']}{current_bet}{COLORS['RESET']}")
        logger.debug(f"PLAYER INFO - {name}: Hand={hole_cards}, Chips={chips}, Bet={current_bet}")
    
    @staticmethod
    def print_action(player_name, action, amount=None):
        """Print player action with appropriate formatting."""
        if action == "fold":
            action_str = f"{player_name} folds"
            color = COLORS['RED']
        elif action == "check":
            action_str = f"{player_name} checks"
            color = COLORS['BLUE']
        elif action == "call":
            action_str = f"{player_name} calls {amount}" if amount else f"{player_name} calls"
            color = COLORS['GREEN']
        elif action == "raise":
            action_str = f"{player_name} raises to {amount}"
            color = COLORS['YELLOW']
        elif action == "all-in":
            action_str = f"{player_name} goes all-in with {amount}"
            color = COLORS['BOLD_RED']
        else:
            action_str = f"{player_name} {action} {amount if amount else ''}"
            color = COLORS['WHITE']
        
        print(f"{color}{action_str}{COLORS['RESET']}")
        logger.info(f"ACTION: {action_str}")
    
    @staticmethod
    def print_cards(label, cards):
        """Print cards with formatted output."""
        print(f"{COLORS['WHITE']}{label}: {COLORS['YELLOW']}{cards}{COLORS['RESET']}")
        logger.info(f"{label}: {cards}")
    
    @staticmethod
    def print_pot(amount):
        """Print current pot amount."""
        print(f"{COLORS['WHITE']}Pot: {COLORS['GREEN']}{amount}{COLORS['RESET']}")
        logger.info(f"Pot: {amount}")
    
    @staticmethod
    def print_winner(player_name, pot, hand_desc=None):
        """Print winner information with highlighting."""
        if hand_desc:
            message = f"{player_name} wins pot of {pot} with {hand_desc}!"
        else:
            message = f"{player_name} wins pot of {pot}!"
            
        print(f"\n{COLORS['BOLD_GREEN']}{message}{COLORS['RESET']}\n")
        logger.info(f"WINNER: {message}")
    
    @staticmethod
    def print_game_over(player_name, chips):
        """Print game over message."""
        message = f"Game over! Winner is {player_name} with {chips} chips."
        print(f"\n{COLORS['BOLD_GREEN']}{message}{COLORS['RESET']}\n")
        logger.info(f"GAME OVER: {message}")
    
    @staticmethod
    def prompt_action(player_name, options="fold, call, raise [amount], check, all-in"):
        """Prompt player for action with formatting."""
        prompt = f"{COLORS['BOLD']}Enter action for {COLORS['BOLD_BLUE']}{player_name}{COLORS['RESET']} ({options}): "
        action = input(prompt).strip().lower()
        logger.debug(f"Player {player_name} prompted for action. Response: {action}")
        return action
    
    @staticmethod
    def print_error(message):
        """Print error message with red highlighting."""
        print(f"{COLORS['BOLD_RED']}Error: {message}{COLORS['RESET']}")
        logger.error(f"ERROR: {message}")
    
    @staticmethod
    def print_round_summary(players):
        """Print a summary of all players' chip counts at the end of a round."""
        print(f"\n{COLORS['BOLD']}End of Round Summary:{COLORS['RESET']}")
        for player in players:
            print(f"  {COLORS['BOLD_BLUE']}{player.name}{COLORS['RESET']}: {COLORS['GREEN']}{player.chips}{COLORS['RESET']} chips")
        logger.info(f"ROUND SUMMARY: " + ", ".join([f"{p.name}={p.chips}" for p in players]))
    
    @staticmethod
    def prompt_next_round():
        """Prompt the user to continue to the next round."""
        print(f"\n{COLORS['BOLD_CYAN']}Press Enter to start the next round...{COLORS['RESET']}")
        input()
        GameUI.clear_screen() 