"""Which solvers the harness can run.

Each entry is a factory taking (candidates, seed) and returning a fresh
solver. Keeping the registry here rather than in harness.py means adding a
solver never touches the harness -- fewer merge conflicts when several
people are adding solvers and someone else is editing the harness.
"""

import random
from functools import lru_cache
from typing import Any, Callable

from word500.matrix import FeedbackTable
from word500.solvers.endgame_search import EndgameSolver
from word500.solvers.greedy_letters import GreedyLetters
from word500.solvers.human_probe import HumanProbe
from word500.solvers.matrix_solver import SCORERS as M_SCORERS
from word500.solvers.matrix_solver import MatrixOneStep, MatrixOneStepConfig
from word500.solvers.one_step import SCORERS, OneStepAhead
from word500.solvers.random_consistent import RandomConsistent
from word500.wordlist import load_allowed

# Entropy-best opening guess for standard mode. Turn 1 is identical in every
# game, so searching for it 1477 times is pure waste. Recompute with:
#   python -m word500.solvers.registry
OPENER = "TARES"

# Only search the whole guess pool once candidates have shrunk to this size.
# Above it, guess from the candidates only. Purely a speed tradeoff -- see the
# threshold experiment in the strategy-comparison issue.
FULL_POOL_BELOW = 150


@lru_cache(maxsize=1)
def table() -> FeedbackTable:
    """The feedback table, built once per process and cached on disk.

    Built over the FULL allowed list, not a mode-filtered pool, so one table
    serves every mode -- each mode's candidates are a subset of it.
    """
    return FeedbackTable(load_allowed())


def _matrix_one_step(scorer_key: str) -> Callable[[Any, int], MatrixOneStep]:
    def build(candidates: Any, _seed: int) -> MatrixOneStep:
        # Read the module globals here rather than when SOLVERS is populated,
        # so --full-pool-below can override the threshold at runtime. Building
        # a three-field frozen dataclass per game is free.
        config = MatrixOneStepConfig(scorer=M_SCORERS[scorer_key],
                                     full_pool_below=FULL_POOL_BELOW,
                                     opener=OPENER)
        # guess_pool is the FULL allowed list, not the mode-filtered candidates.
        # A word that cannot be the answer can still be the best probe -- and
        # restricting it would silently answer the repeated-letter question.
        return MatrixOneStep(candidates, guess_pool=load_allowed(),
                             table=table(), config=config)
    return build


def _one_step(scorer_key: str) -> Callable[[Any, int], OneStepAhead]:
    def build(candidates: Any, _seed: int) -> OneStepAhead:
        return OneStepAhead(candidates, guess_pool=load_allowed(),
                            scorer=SCORERS[scorer_key],
                            full_pool_below=FULL_POOL_BELOW, opener=OPENER)
    return build

# Pure-Python reference implementations. Kept so the matrix-backed versions
# can be cross-checked against them, but ~90x slower -- far too slow for a
# full sweep. --compare skips these unless asked for by name.
REFERENCE = {"py-entropy", "py-minimax"}
SOLVERS = {
    "random": lambda candidates, seed: RandomConsistent(
        candidates, rng=random.Random(seed)),
    "greedy": lambda candidates, seed: GreedyLetters(candidates),
    "human": lambda candidates, seed: HumanProbe(candidates),
    # Pure-Python reference implementations. Slow; kept for cross-checking.
    "py-entropy": _one_step("entropy"),
    "py-minimax": _one_step("minimax"),
    # Matrix-backed, and what you should actually run.
    "entropy": _matrix_one_step("entropy"),
    "minimax": _matrix_one_step("minimax"),
    "expected": _matrix_one_step("expected"),
    "parts": _matrix_one_step("parts"),
    # Endgame strategy: uses two-move lookahead once the candidate list is small.
    "endgame": lambda candidates, seed: EndgameSolver(
        candidates,
        guess_pool=load_allowed(),
        cutoff=10,
        use_two_step=True,
        first_metric="expected",
        second_metric="expected",
    ),
}


if __name__ == "__main__":
    # Recompute the opener. Takes several minutes -- it searches the full
    # pool against the full pool.
    from word500.wordlist import Mode, possible_secrets

    pool = possible_secrets(load_allowed(), Mode.STANDARD)
    best = OneStepAhead(pool, scorer=SCORERS["entropy"]).next_guess()
    print(f"pool {len(pool)}\nOPENER = {best!r}")
