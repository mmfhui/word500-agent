"""
One-step-ahead solvers.

For each possible guess, partition the surviving candidates by the
feedback that guess would produce, then score the partition. Guess the
word whose partition scores best.

Every strategy in this family differs ONLY in how the partition is
scored, so they share one class and swap a scoring function:

    minimax        Knuth (1976-77)   minimize the largest partition
    expected size  Irving (1978-79)  minimize expected remaining count
    max entropy    Neuwirth (1982)   maximize Shannon entropy
    most parts     Kooi (2005)       maximize non-empty partition count

Which one wins is game-dependent -- the literature has different
winners for different Mastermind configurations -- so for Word500 it is
an empirical question, not a settled one.
"""

from collections import Counter
from math import log2
from typing import Callable

from word500.scoring import score
from word500.solvers.base import Solver


def max_entropy(sizes: list[int], n: int) -> float:
    """Shannon entropy of the partition, in bits."""
    return -sum(s / n * log2(s / n) for s in sizes)


def most_parts(sizes: list[int], _n: int) -> int:
    """How many distinct feedbacks the guess can produce."""
    return len(sizes)


def neg_expected_size(sizes: list[int], n: int) -> float:
    """Negated expected remaining candidates. Negated so bigger is better."""
    return -sum(s * s for s in sizes) / n


def neg_max_size(sizes: list[int], _n: int) -> int:
    """Negated worst-case remaining candidates (Knuth's minimax)."""
    return -max(sizes)


SCORERS = {
    "entropy": max_entropy,
    "parts": most_parts,
    "expected": neg_expected_size,
    "minimax": neg_max_size,
}


class OneStepAhead(Solver):
    """Pick the guess whose feedback partition scores best.

    full_pool_below: only consider the whole guess pool once the candidate
        set has shrunk to this size or less. Above it, guess from the
        candidates only. Purely a speed tradeoff -- None means always use
        the full pool, which is better play and far slower.
    opener: the fixed first guess. Turn one is identical in every game, so
        computing it repeatedly is pure waste.
    """

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None,
                 *, scorer: Callable[[list[int], int], float] = max_entropy,
                 full_pool_below: int | None = None,
                 opener: str | None = None) -> None:
        super().__init__(candidates, guess_pool)
        self.scorer = scorer
        self.full_pool_below = full_pool_below
        self.opener = opener

    @property
    def name(self) -> str:
        return f"OneStepAhead({self.scorer.__name__})"

    def next_guess(self) -> str:
        cands = self.candidates
        if not cands:
            raise RuntimeError("no candidates left -- feedback was inconsistent")
        # With two or fewer left, guessing one is optimal: it may win now,
        # and if not the next guess is forced.
        if len(cands) <= 2:
            return cands[0]
        if not self.history and self.opener:
            return self.opener

        pool = cands if (self.full_pool_below is not None
                         and len(cands) > self.full_pool_below) else self.guess_pool
        cand_set = set(cands)
        n = len(cands)

        best, best_key = None, None
        for guess in pool:
            sizes = list(Counter(score(guess, c) for c in cands).values())
            # Tie-break toward a guess that could itself be the answer.
            key = (self.scorer(sizes, n), guess in cand_set)
            if best_key is None or key > best_key:
                best, best_key = guess, key
        if best is None:
            raise RuntimeError("no guess selected")
        return best
