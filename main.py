from game import PokerGame, Player
from ui_helper import GameUI
from utils import logger

if __name__ == "__main__":
    GameUI.clear_screen()
    GameUI.print_header("Texas Hold'em Poker Game", 80)
    
    num_players = 4
    starting_chips = 1000

    players = []
    players.append(Player("User", starting_chips, is_ai=False))  # Human player
    for i in range(1, 4):
        players.append(Player(f"AI Player {i}", starting_chips, is_ai=True))

    
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

