"""Human-style agent: a few fixed opening probes, then guess by common letters.

Opens with a small, fixed set of guesses chosen once for broad letter
coverage, then switches to picking the surviving candidate whose letters
are most common among the others -- the same idea GreedyLetters uses,
just applied after the probes have already narrowed the field.

Deterministic: ties break alphabetically, so no rng and identical
output on every run.
"""

from collections import Counter

from word500.solvers.base import Solver


def pick_probes(guess_pool: list[str], num_probes: int) -> list[str]:
    """
    Greedily choose `num_probes` words from `guess_pool` that together
    cover as many distinct letters as possible, with minimal overlap.

    Mirrors how a person picks a couple of "starter words" that between
    them test as many different letters as they can, rather than
    searching for a jointly-optimal pair.
    """
    freq = Counter(ch for w in guess_pool for ch in set(w))
    covered: set[str] = set()
    probes: list[str] = []
    remaining = list(guess_pool)

    for _ in range(num_probes):
        if not remaining:
            break
        # Score by letters not yet covered, so the second probe is
        # pulled toward what the first one missed instead of repeating it.
        best = max(remaining, key=lambda w: (sum(freq[c] for c in set(w) - covered), w))
        probes.append(best)
        covered |= set(best)
        remaining.remove(best)

    return probes


class HumanProbe(Solver):
    """
    Open with a few fixed, disjoint-letter probes; close by picking the
    surviving candidate with the most letters in common with the rest.

    num_probes: how many opening guesses to use before switching to
        picking from the candidates. Defaults to 2.
    """

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None,
                 num_probes: int = 2) -> None:
        super().__init__(candidates, guess_pool)
        # Computed once per solver instance, over the full guess pool --
        # not cached across games, since the pool is the same every game
        # and recomputation cost has not been measured yet.
        self.probes = pick_probes(self.guess_pool, num_probes)

    def next_guess(self) -> str:
        if not self.candidates:
            raise RuntimeError("no candidates left, feedback was inconsistent")
        # With two or fewer left, guessing one is optimal regardless of
        # how many probes remain unused.
        if len(self.candidates) <= 2:
            return self.candidates[0]

        turn = len(self.history)
        if turn < len(self.probes):
            return self.probes[turn]

        # Closing phase: same "common letters win" intuition GreedyLetters
        # uses, just applied after the openers have already narrowed things.
        freq = Counter(ch for w in self.candidates for ch in set(w))
        return max(self.candidates, key=lambda w: (sum(freq[c] for c in set(w)), w))
