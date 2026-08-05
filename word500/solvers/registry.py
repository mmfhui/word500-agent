"""Which solvers the harness can run.

Each entry is a factory taking (candidates, seed) and returning a fresh
solver. Keeping the registry here rather than in harness.py means adding a
solver never touches the harness -- fewer merge conflicts when several
people are adding solvers and someone else is editing the harness.
"""

import random
from typing import Sequence, Callable
from word500.solvers.greedy_letters import GreedyLetters
from word500.solvers.one_step import SCORERS, OneStepAhead
from word500.solvers.random_consistent import RandomConsistent


# Entropy-best opening guess for standard mode. Turn 1 is identical in every
# game, so searching for it 1477 times is pure waste. Recompute with:
#   python -m word500.solvers.registry
OPENER = "TARES"

# Only search the whole guess pool once candidates have shrunk to this size.
# Above it, guess from the candidates only. Purely a speed tradeoff -- see the
# threshold experiment in the strategy-comparison issue.
FULL_POOL_BELOW = 150

def _one_step(scorer_key: str) -> Callable[[Sequence[str], int], OneStepAhead]:
    def build(candidates: Sequence[str], _seed: int) -> OneStepAhead:
        return OneStepAhead(list(candidates), scorer=SCORERS[scorer_key],
                            full_pool_below=FULL_POOL_BELOW, opener=OPENER)
    return build


SOLVERS = {
    "random": lambda candidates, seed: RandomConsistent(
        list(candidates), rng=random.Random(seed)),
    "greedy": lambda candidates, seed: GreedyLetters(list(candidates)),
    "entropy": _one_step("entropy"),
    "minimax": _one_step("minimax"),
    "expected": _one_step("expected"),
    "parts": _one_step("parts"),
}


if __name__ == "__main__":
    # Recompute the opener. Takes several minutes -- it searches the full
    # pool against the full pool.
    from word500.wordlist import Mode, load_allowed, possible_secrets

    pool = possible_secrets(load_allowed(), Mode.STANDARD)
    best = OneStepAhead(pool, scorer=SCORERS["entropy"]).next_guess()
    print(f"pool {len(pool)}\nOPENER = {best!r}")
