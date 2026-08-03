"""
Local Word500 test bench.

This is not the actual game people play but the one meant for the agents to train
against. It holds one secret word, answers guesses with counts, and
counts turns. It offers no hints and doesn't leak the secret word early.
"""

from word500.scoring import Feedback, score

MAX_GUESSES = 8


class GameOver(Exception):
    """Raised on an illegal action: guessing or revealing at the wrong time."""


class Game:
    """One round of Word500 against a fixed secret."""

    def __init__(self, secret: str, max_guesses: int = MAX_GUESSES) -> None:
        self._secret = secret.strip().upper()
        self.max_guesses = max_guesses
        self.history: list[tuple[str, Feedback]] = []
        self.won = False

    @property
    def guesses_made(self) -> int:
        return len(self.history)

    @property
    def guesses_left(self) -> int:
        return self.max_guesses - self.guesses_made

    @property
    def is_over(self) -> bool:
        return self.won or self.guesses_left == 0

    def guess(self, word: str) -> Feedback:
        """Judge one guess and record it."""
        if self.is_over:
            raise GameOver("the game has already ended")

        word = word.strip().upper()
        feedback = score(word, self._secret)
        self.history.append((word, feedback))
        if feedback.is_win:
            self.won = True
        return feedback

    def reveal(self) -> str:
        """The secret word. Legal only once the game is over."""
        if not self.is_over:
            raise GameOver("cannot reveal the secret mid-game")
        return self._secret
