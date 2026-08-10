"""Solvers backed by a precomputed feedback table.

Two things get faster here. Choosing a guess becomes a bincount instead of
thousands of score() calls -- but the bigger win is filtering, which runs on
every turn of every game for every solver and becomes one vectorised
comparison.

MatrixOneStep deliberately does not replace OneStepAhead. Keeping the pure
Python version as a reference implementation means the two can be
cross-checked against each other, and it leaves solvers/one_step.py alone
for whoever is working on strategies.
"""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from word500.matrix import CODE, N_CLASSES, FeedbackTable
from word500.scoring import Feedback
from word500.solvers.base import Solver


def entropy(counts: NDArray[np.int32], n: int) -> NDArray[np.float64]:
    """Shannon entropy of each guess's partition, in bits."""
    p = counts / n
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -(p * logp).sum(axis=1)


def most_parts(counts: NDArray[np.int32], _n: int) -> NDArray[np.float64]:
    """How many distinct feedbacks each guess can produce."""
    return (counts > 0).sum(axis=1).astype(np.float64)


def neg_expected_size(counts: NDArray[np.int32], n: int) -> NDArray[np.float64]:
    """Negated expected remaining candidates, so bigger is better."""
    return -(counts.astype(np.int64) ** 2).sum(axis=1) / n


def neg_max_size(counts: NDArray[np.int32], _n: int) -> NDArray[np.float64]:
    """Negated worst-case remaining candidates (Knuth's minimax)."""
    return -counts.max(axis=1).astype(np.float64)


SCORERS = {
    "entropy": entropy,
    "parts": most_parts,
    "expected": neg_expected_size,
    "minimax": neg_max_size,
}


@dataclass(frozen=True)
class MatrixOneStepConfig:
    """Configuration for MatrixOneStep."""
    scorer: Callable[[NDArray[np.int32], int], NDArray[np.float64]] = entropy
    full_pool_below: int | None = None
    opener: str | None = None


class MatrixSolver(Solver):
    """Keeps candidates as numpy indices into a FeedbackTable.

    `candidates` remains a list of words to anything looking from outside, so
    the harness and the base class are unaffected -- it is just materialised
    on access rather than stored.
    """

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None,
                 table: FeedbackTable | None = None) -> None:
        if table is None:
            raise ValueError("MatrixSolver requires a FeedbackTable")
        # Must precede super(), because the candidates setter needs it.
        self.table = table
        super().__init__(candidates, guess_pool)
        self.pool_idx = table.indices(self.guess_pool)

    @property
    def candidates(self) -> list[str]:
        """The remaining candidate words."""
        return [self.table.words[i] for i in self.cand_idx]

    @candidates.setter
    def candidates(self, words: list[str]) -> None:
        self.cand_idx = self.table.indices(words)

    def update(self, word: str, feedback: Feedback) -> None:
        self.history.append((word, feedback))
        code = CODE[(feedback.greens, feedback.yellows)]
        row = self.table.matrix[self.table.index[word]]
        self.cand_idx = self.cand_idx[row[self.cand_idx] == code]


def matrix_one_step_guess(
    table: FeedbackTable,
    pool_idx: NDArray[np.int32],
    idx: NDArray[np.int32],
    scorer: Callable[[NDArray[np.int32], int], NDArray[np.float64]],
) -> str:
    """Best one-step guess from `pool_idx` against candidates `idx`, via the table.

    Shared by MatrixOneStep and any other solver that wants matrix-backed
    one-step scoring for a large candidate pool.
    """
    n = len(idx)
    sub = table.matrix[np.ix_(pool_idx, idx)]
    counts = np.empty((len(pool_idx), N_CLASSES), dtype=np.int32)
    for code in range(N_CLASSES):
        counts[:, code] = (sub == code).sum(axis=1)

    scores = scorer(counts, n)
    tied = np.flatnonzero(scores == scores.max())
    # Prefer a guess that could itself be the answer; then lowest index,
    # matching the pure-Python implementation's tie-breaking.
    prefer = tied[np.isin(pool_idx[tied], idx)]
    best = int(pool_idx[prefer[0] if len(prefer) else tied[0]])
    return str(table.words[best])


class MatrixOneStep(MatrixSolver):
    """One-step-ahead search over the feedback table."""

    def __init__(self, candidates: list[str], guess_pool: list[str] | None = None,
                 table: FeedbackTable | None = None, *,
                 config: MatrixOneStepConfig | None = None) -> None:
        super().__init__(candidates, guess_pool, table=table)
        if config is None:
            config = MatrixOneStepConfig()
        self.scorer = config.scorer
        self.full_pool_below = config.full_pool_below
        # Fail here rather than mid-game. An opener outside the word list is a
        # config error: the real game would reject it as an invalid guess.
        if config.opener is not None and config.opener not in self.table.index:
            raise ValueError(f"opener {config.opener!r} is not in the word list")
        self.opener = config.opener

    @property
    def name(self) -> str:
        return f"MatrixOneStep({self.scorer.__name__})"

    def next_guess(self) -> str:
        idx = self.cand_idx
        n = len(idx)
        if n == 0:
            raise RuntimeError("no candidates left -- feedback was inconsistent")
        if n <= 2:
            return str(self.table.words[idx[0]])
        if not self.history and self.opener:
            return self.opener

        pool = (idx if (self.full_pool_below is not None
                        and n > self.full_pool_below) else self.pool_idx)

        # One (guesses x classes) count table, built with 20 vectorised passes,
        # then every guess scored in a single array operation. Scoring guesses
        # one at a time is dominated by numpy's per-call overhead on 20-element
        # arrays, which costs more than the arithmetic.
        return matrix_one_step_guess(self.table, pool, idx, self.scorer)
