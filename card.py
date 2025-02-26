import random
from utils import log_sensitive

# Global constants for card suits, values, and hand rankings
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
VALUES_ORDER = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
HAND_RANKINGS = ["High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", 
                 "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]


class Card:
    """
    Represents a standard playing card in a deck.
    
    Attributes:
        value (str): The rank of the card ('2' to 'A').
        suit (str): The suit of the card ('Hearts', 'Diamonds', 'Clubs', 'Spades').
    """
    def __init__(self, value: str, suit: str):
        """
        Initializes a Card object with a rank and suit.

        Args:
            value (str): The rank of the card (must be in VALUES_ORDER, e.g., '2', 'A').
            suit (str): The suit of the card (must be in SUITS, e.g., 'Hearts', 'Spades').
        """
        self.value = value
        self.suit = suit

    def __repr__(self) -> str:
        """
        Returns a readable string representation of the card.

        Returns:
            str: A formatted string like "<value> of <suit>".
        """
        return f"{self.value} of {self.suit}"
    

class Deck:
    """
    Represents a deck of standard playing cards.

    This class provides methods to generate, shuffle, and deal cards from the deck.
    
    Attributes:
        cards (list[Card]): A list of Card objects representing the deck.
    """
    def __init__(self):
        """
        Initializes the deck with 52 cards and shuffles them.
        """
        self.cards = [Card(value, suit) for suit in SUITS for value in VALUES_ORDER]
        self.shuffle()

    def shuffle(self):
        """
        Shuffles the deck, randomizing the order of the cards.
        """
        # Log the deck state before shuffling
        log_sensitive("Deck before shuffling", {
            "cards": self.cards[:5] + ["..."] + self.cards[-5:] if len(self.cards) > 10 else self.cards
        })
        
        random.shuffle(self.cards)
        
        # Log partial deck state after shuffling (first few and last few cards)
        log_sensitive("Deck after shuffling", {
            "first_cards": self.cards[:5],
            "last_cards": self.cards[-5:]
        })

    def deal(self) -> Card:
        """
        Deals a card from the top of the deck.

        Returns:
            Card: The dealt Card object.
        """
        return self.cards.pop()
        # Example usage of the Deck class



