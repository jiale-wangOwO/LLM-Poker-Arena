"""Legacy hand-evaluation checks, migrated to the v2 API.

The assertions are the originals; only the card construction changed.  The v2
``Card`` stores integer rank/suit for speed, so it is built from a code string.
"""

from card import Card, Hand  # noqa: F401  (Hand comes from hand.py via the shim)


def hand(*codes: str) -> Hand:
    return Hand([Card.from_str(code) for code in codes])


def test_evaluate_hand():
    cases = [
        (("As", "Ks", "Qs", "Js", "Ts"), "Royal Flush"),
        (("Th", "Kh", "Qh", "Jh", "Ah"), "Royal Flush"),
        (("Td", "Jd", "Qd", "Kd", "Ad"), "Royal Flush"),
        (("Js", "Ts", "As", "Ks", "Qs"), "Royal Flush"),
        (("9h", "9d", "9c", "9s", "5h"), "Four of a Kind"),
        (("2h", "2d", "2c", "2s", "Ah"), "Four of a Kind"),
        (("5h", "6h", "7h", "8h", "9h"), "Straight Flush"),
        (("2c", "3c", "4c", "5c", "6c"), "Straight Flush"),
        (("5d", "5c", "5h", "7s", "7c"), "Full House"),
        (("Ad", "Ac", "Ah", "2s", "2c"), "Full House"),
        (("2d", "4d", "6d", "8d", "Td"), "Flush"),
        (("Ah", "2d", "3c", "4s", "5h"), "Straight"),
        (("Th", "Jd", "Qc", "Ks", "Ah"), "Straight"),
        (("3h", "3d", "3c", "7s", "9h"), "Three of a Kind"),
        (("2h", "2d", "5c", "5s", "Ah"), "Two Pair"),
        (("Kh", "Kd", "2c", "7s", "9h"), "One Pair"),
        (("2h", "4d", "7c", "9s", "Kh"), "High Card"),
        (("As", "2s", "3s", "4s", "5s"), "Straight Flush"),
        (("8d", "9d", "Td", "Jd", "Qd"), "Straight Flush"),
        (("Qh", "Qd", "Qc", "Ts", "Th"), "Full House"),
        (("Kh", "Kd", "Qc", "Qs", "Jh"), "Two Pair"),
        (("Ah", "3h", "5h", "7h", "9h"), "Flush"),
        (("Qh", "Kh", "Ah", "2h", "3h"), "Flush"),
        (("5h", "6d", "7c", "8s", "8h"), "One Pair"),
        (("Th", "Td", "Tc", "Ts", "Kh"), "Four of a Kind"),
        (("Ah", "Kd", "Qc", "Js", "9h"), "High Card"),
        (("3h", "4d", "5c", "7s", "8h"), "High Card"),
    ]

    for codes, expected in cases:
        actual = hand(*codes).evaluate_hand()
        assert actual == expected, f"Expected {expected}, got {actual} for {codes}"

    print("test_evaluate_hand passed!")


def test_best_hand():
    cases = [
        (
            ("Ah", "Kh"),
            ("Qh", "Jh", "Th", "2d", "3c"),
            "Royal Flush",
        ),
        (
            ("Ah", "Ad"),
            ("Ac", "Kh", "Kd", "Qh", "Jc"),
            "Full House",
        ),
        (
            ("7c", "8c"),
            ("9c", "Tc", "Jc", "Ah", "Ad"),
            "Straight Flush",
        ),
        (
            ("6h", "7h"),
            ("8h", "9h", "Th", "Ad", "Ac"),
            "Straight Flush",
        ),
        (
            ("As", "2h"),
            ("3d", "4c", "5s", "Th", "Jd"),
            "Straight",
        ),
        (
            ("Qh", "Qd"),
            ("Ks", "Kh", "Kc", "2d", "3c"),
            "Full House",
        ),
        (
            ("8h", "9h"),
            ("Th", "Jh", "Qd", "2h", "3h"),
            "Flush",
        ),
        (
            ("Jc", "Jd"),
            ("Qh", "Qs", "Kc", "Kh", "2d"),
            "Two Pair",
        ),
        (
            ("5c", "9c"),
            ("6c", "7c", "8c", "Th", "Jd"),
            "Straight Flush",
        ),
        (
            ("As", "Kh"),
            ("Ad", "Qc", "Js", "Th", "9d"),
            "Straight",
        ),
        (
            ("7h", "7d"),
            ("8c", "8s", "9h", "9d", "Tc"),
            "Two Pair",
        ),
        (
            ("Qh", "Kh"),
            ("Ah", "2h", "3h", "Td", "Jc"),
            "Flush",
        ),
        (
            ("4h", "4d"),
            ("5h", "5d", "6c", "6s", "7h"),
            "Two Pair",
        ),
    ]

    for hole, board, expected in cases:
        best = Hand.best_hand(
            [Card.from_str(c) for c in hole], [Card.from_str(c) for c in board]
        )
        actual = best.evaluate_hand()
        assert actual == expected, (
            f"Expected {expected}, got {actual} for {hole} + {board}"
        )

    print("test_best_hand passed!")


def test_compare_hands():
    cases = [
        (
            ("Ah", "Kh", "Qh", "Jh", "Th"),
            ("Kc", "Qc", "Jc", "Tc", "9c"),
            "Player 1 wins",
        ),
        (
            ("5h", "5d", "5c", "5s", "Kh"),
            ("5h", "5d", "5c", "5s", "Qh"),
            "Player 1 wins",
        ),
        (
            ("Ah", "Kh", "Qh", "Jh", "Th"),
            ("Td", "Kd", "Qd", "Jd", "Ad"),
            "It's a tie",
        ),
        (
            ("8h", "9h", "Th", "Jh", "Qh"),
            ("Ah", "Ad", "Ac", "As", "Kh"),
            "Player 1 wins",
        ),
        (
            ("Kh", "Kd", "Qc", "Qs", "Ah"),
            ("Kc", "Ks", "Qh", "Qd", "Jc"),
            "Player 1 wins",
        ),
        (
            ("Ah", "Jd", "9c", "4s", "3h"),
            ("Kc", "Qh", "Js", "Td", "8c"),
            "Player 1 wins",
        ),
        (
            ("Ah", "2h", "3h", "4h", "5h"),
            ("2c", "3c", "4c", "5c", "6c"),
            "Player 2 wins",
        ),
        (
            ("Ah", "Kh", "Qh", "Jh", "9h"),
            ("Ad", "Kd", "Qd", "Td", "9d"),
            "Player 1 wins",
        ),
        (
            ("Kh", "Kd", "Kc", "Qd", "Qh"),
            ("Qc", "Qd", "Qh", "Ks", "Kd"),
            "Player 1 wins",
        ),
        (
            ("7h", "7d", "7c", "Ad", "Kc"),
            ("7d", "7h", "7s", "Ah", "Qc"),
            "Player 1 wins",
        ),
        (
            ("9h", "Th", "Jh", "Qh", "Kh"),
            ("8d", "9d", "Td", "Jd", "Qd"),
            "Player 1 wins",
        ),
        (
            ("Kh", "Kd", "Jc", "Js", "Qh"),
            ("Qc", "Qd", "Jh", "Jd", "As"),
            "Player 1 wins",
        ),
        (
            ("Ah", "Kh", "Qh", "Jh", "Th"),
            ("Tc", "Jc", "Qc", "Kc", "Ac"),
            "It's a tie",
        ),
        (
            ("Ah", "Kd", "Qc", "Js", "9h"),
            ("Ac", "Kh", "Qd", "Jc", "8s"),
            "Player 1 wins",
        ),
        (
            ("Th", "Jh", "Qh", "Kh", "Ah"),
            ("9d", "Td", "Jd", "Qd", "Kd"),
            "Player 1 wins",
        ),
    ]

    for first, second, expected in cases:
        result = Hand.compare_hands(
            Hand([Card.from_str(c) for c in first]),
            Hand([Card.from_str(c) for c in second]),
        )
        assert result == expected, (
            f"Expected {expected}, got {result} for {first} vs {second}"
        )

    print("test_compare_hands passed!")


if __name__ == "__main__":
    test_evaluate_hand()
    test_best_hand()
    test_compare_hands()
