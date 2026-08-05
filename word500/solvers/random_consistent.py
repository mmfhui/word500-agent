"""Baseline agent: guess a random word that is still possible."""

import random

from word500.solvers.base import Solver


class RandomConsistent(Solver):
    """
    Pick uniformly at random from the candidates still consistent.
    
    The simplest strategy that uses feedback at all. Every new
    solver has to beat this number to justify their complexity.
    """

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None,
                 rng: random.Random | None = None):
        super().__init__(candidates, guess_pool)
        self.rng = rng if rng is not None else random.Random()

    def next_guess(self) -> str:
        if not self.candidates:
            raise RuntimeError("no candidates left, feedback was inconsistent")
        return self.rng.choice(self.candidates)
