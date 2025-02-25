'''
Author: [Your Name]
Date: [Date]
File: hand.py
This file contains the logic for evaluating poker hands. It handles evaluating a player's hand, determining hand rankings, and comparing hands.
Classes:
    Hand: Represents a poker hand and provides methods for evaluation and comparison.
Methods:
    __init__(self, cards: list): Initializes a poker hand with a list of Card objects.
    evaluate_hand(self) -> str: Determines the hand ranking type of the poker hand.
    get_sorted_hand(self) -> list: Returns a list of card values sorted by rank.
    tiebreaker(self) -> tuple: Returns a tuple used for tie-breaking when comparing poker hands of the same type.
    best_hand(player_cards: list, community_cards: list): Determines the best possible 5-card hand from player's cards and community cards.
    compare_hands(hand1, hand2) -> str: Compares two poker hands and determines the winner.
    is_royal_flush(self) -> bool: Checks if the hand is a Royal Flush.
    is_straight_flush(self) -> bool: Checks if the hand is a Straight Flush.
    is_four_of_a_kind(self) -> bool: Checks if the hand is Four of a Kind.
    is_full_house(self) -> bool: Checks if the hand is a Full House.
    is_flush(self) -> bool: Checks if the hand is a Flush.
    is_straight(self) -> bool: Checks if the hand is a Straight.
    is_three_of_a_kind(self) -> bool: Checks if the hand is Three of a Kind.
    is_two_pair(self) -> bool: Checks if the hand is Two Pair.
    is_one_pair(self) -> bool: Checks if the hand is One Pair.
from collections import Counter
'''

from card import Card, SUITS, VALUES_ORDER
from itertools import combinations
from collections import Counter

HAND_RANKINGS = ["High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", 
                 "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]

class Hand:
    """
    Represents a poker hand and provides methods for evaluation and comparison.

    This class can calculate the hand ranking of a hand, 
    generate a tie-breaker tuple for comparing hands in case of a tie,
    compare two hands to determine the winner, 
    and also find the best possible hand combination from player's private cards and community cards.

    Attributes:
        cards (list[Card]): A list of Card objects in the hand.
    """
    def __init__(self, cards: list):
        """
        Initializes a poker hand with a list of Card objects.

        Args:
            cards (list[Card]): A list of 5 Card objects forming the hand.
        """
        self.cards = cards
        self.values = [card.value for card in self.cards]
        self.suits = [card.suit for card in self.cards]
        self.value_counts = Counter(self.values)
        self.suit_counts = Counter(self.suits)

    def evaluate_hand(self) -> str:
        """
        Determines the hand ranking type of the poker hand.

        Returns:
            str: Poker hand that can be made, e.g. "Royal Flush", "Two Pair".
        """
        if self.is_royal_flush():
            return "Royal Flush"
        elif self.is_straight_flush():
            return "Straight Flush"
        elif self.is_four_of_a_kind():
            return "Four of a Kind"
        elif self.is_full_house():
            return "Full House"
        elif self.is_flush():
            return "Flush"
        elif self.is_straight():
            return "Straight"
        elif self.is_three_of_a_kind():
            return "Three of a Kind"
        elif self.is_two_pair():
            return "Two Pair"
        elif self.is_one_pair():
            return "One Pair"
        return "High Card"

    def get_sorted_hand(self) -> list:
        """
        Returns a list of card values sorted by rank.

        returns:
            list: A list of card values sorted by rank and suit.
        """
        sorted_cards = sorted(
            self.cards,
            key=lambda card: (VALUES_ORDER.index(card.value), card.suit),
            reverse=True
        )
        return [VALUES_ORDER.index(card.value) for card in sorted_cards]

    def tiebreaker(self) -> tuple:
        """
        Returns a tuple used for tie-breaking when comparing poker hands of the same type.
        
        The tuple varies depending on the hand type and is used to break ties by ranking individual cards.
        For different hand types, the tuple format differs:
        - **Straight** or **Straight Flush**: Returns the highest card index (special handling for Ace-low case).
        - **Four of a Kind**: Returns a tuple of (four-of-a-kind card value index, kicker card value index).
        - **Full House**: Returns a tuple of (three-of-a-kind card value index, pair card value index).
        - **Flush** or **High Card**: Returns a tuple of all card values sorted in descending order.
        - **Three of a Kind**, **Two Pair**, **One Pair**: Returns a tuple of the three-of-a-kind or pair card value, followed by the kicker cards in descending order.
        
        The function ensures that even hands with the same ranking (e.g., both players having a "Two Pair") are properly compared by their card values.
        
        :return: A tuple representing the tie-breaker values for the hand, used to compare hands of the same rank.
        """
        # covert card values to indices in VALUES_ORDER, higher index means higher value
        values = [VALUES_ORDER.index(v) for v in self.values]
        counts = self.value_counts  # count of each card value
        hand_type = self.evaluate_hand()

        if hand_type in ["Straight", "Straight Flush"]:
            sorted_vals = sorted(values, reverse=True)
            if sorted_vals == [12, 3, 2, 1, 0]: # Ace-low straight (A-2-3-4-5), the kiker is '5'(index is 3)
                return (3,)
            return (sorted_vals[0],)
        elif hand_type == "Four of a Kind":
            quad = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 4][0]
            kicker = max([VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt != 4])
            return (quad, kicker)
        elif hand_type == "Full House":
            triple = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 3][0]
            pair = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 2][0]
            return (triple, pair)
        elif hand_type in ["Flush", "High Card"]:
            return tuple(sorted(values, reverse=True))
        elif hand_type == "Three of a Kind":
            triple = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 3][0]
            kickers = sorted(
                [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 1],
                reverse=True
            )
            return (triple,) + tuple(kickers)
        elif hand_type == "Two Pair":
            pairs = sorted(
                [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 2],
                reverse=True
            )
            kicker = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 1][0]
            return tuple(pairs) + (kicker,)
        elif hand_type == "One Pair":
            pair = [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 2][0]
            kickers = sorted(
                [VALUES_ORDER.index(v) for v, cnt in counts.items() if cnt == 1],
                reverse=True
            )
            return (pair,) + tuple(kickers)
        else:
            # default case: High Card, return all card indices.
            return tuple(sorted(values, reverse=True))
    
    @staticmethod
    def best_hand(player_cards: list, community_cards: list):
        """
        Determines the best possible 5-card hand from player's cards and community cards.

        The comparison is done using a custom sorting key:
        1. First, the hand type is evaluated by calling hand.evaluate_hand() (e.g., "Royal Flush", "Straight").
           We then look up the index of this hand type in the HAND_RANKINGS list, where hands like "Royal Flush" are ranked higher than "High Card".
        2. If two hands have the same hand type (e.g., both are "One Pair"), the tie-breaking logic is applied:
           hand.tiebreaker() returns a tuple of values that allow us to compare hands of the same type by considering individual card ranks.
           This tuple may include the value of the pair, kicker cards, etc., depending on the hand type.
        3. The max() function selects the hand with the highest value based on the tuple (first by hand type, then by tie-breaker values).
           This ensures that hands are ranked correctly, with stronger hands prioritized and ties resolved appropriately.        

        Args:
            player_cards (list[Card]): The player's two private cards.
            community_cards (list[Card]): The five community cards.

        Returns:
            Hand: The best possible 5-card Hand object.
        """
        all_cards = player_cards + community_cards # all 7 cards
        all_hands = [Hand(list(combo)) for combo in combinations(all_cards, 5)] # all possible 5-card hands        
        return max(all_hands, key=lambda hand: (HAND_RANKINGS.index(hand.evaluate_hand()), hand.tiebreaker()))


    @staticmethod
    def compare_hands(hand1, hand2) -> str:
        """
        Compares two poker hands and determines the winner.

        Args:
            hand1 (Hand): The first player's hand.
            hand2 (Hand): The second player's hand.

        Returns:
            str: "Player 1 wins", "Player 2 wins", or "It's a tie".
        """
        rank1, rank2 = HAND_RANKINGS.index(hand1.evaluate_hand()), HAND_RANKINGS.index(hand2.evaluate_hand())

        if rank1 > rank2:
            return "Player 1 wins"
        elif rank1 < rank2:
            return "Player 2 wins"
        else:
            tb1, tb2 = hand1.tiebreaker(), hand2.tiebreaker()
            if tb1 > tb2:
                return "Player 1 wins"
            elif tb1 < tb2:
                return "Player 2 wins"
            else:
                return "It's a tie"
        

    def is_royal_flush(self) -> bool:
        return self.is_flush() and set(self.values) == {'A', 'K', 'Q', 'J', 'T'}

    def is_straight_flush(self) -> bool:
        return self.is_flush() and self.is_straight()

    def is_four_of_a_kind(self) -> bool:
        return 4 in self.value_counts.values()

    def is_full_house(self) -> bool:
        return sorted(self.value_counts.values()) == [2, 3]

    def is_flush(self) -> bool:
        return len(set(self.suits)) == 1

    def is_straight(self) -> bool:
        sorted_values = sorted([VALUES_ORDER.index(v) for v in self.values if v in VALUES_ORDER])
        if len(set(sorted_values)) != 5:
            return False
        return sorted_values == [0, 1, 2, 3, 12] or sorted_values[-1] - sorted_values[0] == 4

    def is_three_of_a_kind(self) -> bool:
        return 3 in self.value_counts.values()

    def is_two_pair(self) -> bool:
        return list(self.value_counts.values()).count(2) == 2

    def is_one_pair(self) -> bool:
        return 2 in self.value_counts.values()
    

