"""
The interface Word500 agents have to implement.

A solver never receives the Game object and never sees the answer list.
It is handed a candidate pool at construction and (guess, feedback)
pairs thereafter. Restrction prevents cheating by accident. It also
means the same solver can play the real word500.com with a human
relaying counts.
"""

from abc import ABC, abstractmethod

from word500.scoring import Feedback, score


class Solver(ABC):
    """Base class for agents. Subclasses implement next_guess()."""

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None) -> None:
        # Words that could still be the secret. Shrinks as feedback arrives.
        self.candidates = list(candidates)
        # Words this solver may guess. Never shrinks -- a guess that cannot
        # be the answer can still be the most informative thing to try.
        self.guess_pool = list(guess_pool) if guess_pool is not None else list(candidates)
        self.history: list[tuple[str, Feedback]] = []

    @property
    def name(self) -> str:
        """The name of this solver (its class name)."""
        return type(self).__name__

    @abstractmethod
    def next_guess(self) -> str:
        """The word to guess now."""

    def update(self, word: str, feedback: Feedback) -> None:
        """
        Record a result and drop candidates it rules out.

        Only the newest constraint is applied, because the surviving
        candidates already satisfy every earlier one. Equivalent to
        re-filtering from scratch.
        """
        self.history.append((word, feedback))
        self.candidates = [w for w in self.candidates if score(word, w) == feedback]

    def consistent(self, words: list[str]) -> list[str]:
        """
        Which of `words` match every piece of feedback so far?

        The from-scratch version of update(). Needed when the history
        changes rather than grows (e.g. correcting a mistyped count
        during a real game).
        """
        return [w for w in words if all(score(g, w) == fb for g, fb in self.history)]
