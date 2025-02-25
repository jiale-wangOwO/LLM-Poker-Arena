from game import PokerGame, Player




if __name__ == "__main__":
    # try:
    #     num_players = int(input("Enter number of players: "))
    # except ValueError:
    #     print("Invalid input. Defaulting to 2 players.")
    #     num_players = 2
    num_players = 4
    default_names = ["Alice", "Bob", "Carl", "David"]
    
    players = []
    for i in range(num_players):
        # name = input(f"Enter name for Player {i+1}: ")
        name = default_names[i]
        players.append(Player(name, 1000))  # Each player starts with 1000 chips

    game = PokerGame(players)
    game.play_game()

