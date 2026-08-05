"""Baseline agent: guess the candidate with the most common letters."""

from collections import Counter

from word500.solvers.base import Solver


class GreedyLetters(Solver):
    """
    Pick the surviving candidate whose letters appear in the most others.

    A step up from random as a greedy letters approach (preffering guesses
    made of letters shared widely across the remaining candidates) on the theory 
    that testing common letters resolves more of the pool. It is still greedy and 
    only optimizes letter coverage, not information gain, and it only ever guesses 
    words that could be the answer.

    Deterministic: ties break alphabetically, so no rng and identical
    output on every run.
    """

    def next_guess(self) -> str:
        if not self.candidates:
            raise RuntimeError("no candidates left, feedback was inconsistent")
        if len(self.candidates) == 1:
            return self.candidates[0]

        # How many candidates contain each letter. set(w) so a word with a
        # doubled letter does not inflate that letter's count.
        freq = Counter(ch for w in self.candidates for ch in set(w))
        return max(self.candidates, key=lambda w: (sum(freq[c] for c in set(w)), w))
