"""
Feedback scoring for Word500.

Word500 reports how many letters are correct, not which ones.
This module is the source of truth for that rule. Both the game
(to answer a guess) and the solver (to test hypotheses) import it.
"""

from collections import Counter
from typing import NamedTuple

WORD_LENGTH = 5


class Feedback(NamedTuple):
    """
    The result of one guess: counts only, never letter positions.
    Reds are derived (the three counts always sum to WORD_LENGTH) and 
    storing the third would allow an inconsistent Feedback to exist.
    """

    greens: int
    yellows: int

    @property
    def reds(self) -> int:
        """Guess positions that earned nothing (Word500 shows these grey/red)."""
        return WORD_LENGTH - self.greens - self.yellows
    
    @property
    def is_win(self) -> bool:
        return self.greens == WORD_LENGTH


def score(guess: str, answer: str) -> Feedback:
    """
    Return how many letters of `guess` are right-place / right-letter.
    greens  = letters in the correct position
    yellows = letters present in the answer but in the wrong position,
    where it is also counted so no letter of the answer is credited twice
    """
    if len(guess) != WORD_LENGTH or len(answer) != WORD_LENGTH:
        raise ValueError(
            f"both words must be {WORD_LENGTH} letters, "
            f"got {guess!r} and {answer!r}"
        )

    greens = 0
    guess_rest: list[str] = []
    answer_rest: list[str] = []

    # Pass 1: take the greens out to keep the three answer types disjoint.
    for guess_ch, answer_ch in zip(guess, answer):
        if guess_ch == answer_ch:
            greens += 1
        else:
            guess_rest.append(guess_ch)
            answer_rest.append(answer_ch)

    # Pass 2: pair up the leftovers. `&` takes the min count per letter,
    # so a letter can't be credited more often than the answer has it.
    yellows = sum((Counter(guess_rest) & Counter(answer_rest)).values())

    return Feedback(greens, yellows)
