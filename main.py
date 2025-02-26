from game import PokerGame, Player
from ui_helper import GameUI
from utils import logger

if __name__ == "__main__":
    GameUI.clear_screen()
    GameUI.print_header("Texas Hold'em Poker Game", 80)
    
    try:
        num_players = int(input("Enter number of players (2-10): "))
        if num_players < 2 or num_players > 10:
            GameUI.print_error("Invalid number of players. Using default of 4 players.")
            num_players = 4
    except ValueError:
        GameUI.print_error("Invalid input. Using default of 4 players.")
        num_players = 4
    
    default_names = ["Alice", "Bob", "Carl", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
    starting_chips = 1000
    
    players = []
    for i in range(num_players):
        use_default = input(f"Use default name '{default_names[i]}' for Player {i+1}? (y/n): ").lower() == 'y'
        if use_default:
            name = default_names[i]
        else:
            name = input(f"Enter name for Player {i+1}: ")
        
        players.append(Player(name, starting_chips))
        logger.info(f"Added player: {name} with {starting_chips} chips")
    
    # Initialize and start the game
    small_blind = 10
    big_blind = 20
    game = PokerGame(players, small_blind, big_blind)
    logger.info(f"Game initialized with small blind={small_blind}, big blind={big_blind}")
    
    try:
        game.play_game()
    except KeyboardInterrupt:
        GameUI.print_error("Game interrupted by user")
        logger.warning("Game was interrupted by keyboard interrupt")
    except Exception as e:
        GameUI.print_error(f"Game error: {str(e)}")
        logger.error(f"Game error: {str(e)}", exc_info=True)
    
    logger.info("========== POKER GAME SESSION ENDED ==========")

