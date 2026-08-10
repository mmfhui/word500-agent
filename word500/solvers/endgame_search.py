from __future__ import annotations

"""Endgame search utilities for Word500.

This module implements small-candidate endgame heuristics for the Word500
variant. It supports one-step and two-step lookahead scoring, candidate
partitioning by feedback, simulated solve runs, and a real Solver subclass
that can be registered with the harness.

Two-step search is restricted to small pools to keep the solver fast.
"""

from collections import Counter, defaultdict
from math import log2
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from word500.matrix import FeedbackTable
from word500.solvers.matrix_solver import SCORERS, MatrixSolver, matrix_one_step_guess

Feedback = Tuple[int, int, int]
FeedbackFn = Callable[[str, str], Feedback]

def word500_feedback(guess: str, secret: str) -> Feedback:
    """Compute Word500 feedback counts for a guess against a secret."""
    green = 0
    guess_remain: List[str] = []
    secret_remain: List[str] = []

    for g, s in zip(guess, secret):
        if g == s:
            green += 1
        else:
            guess_remain.append(g)
            secret_remain.append(s)

    common = Counter(secret_remain)
    yellow = 0
    for g in guess_remain:
        if common[g] > 0:
            yellow += 1
            common[g] -= 1

    red = len(guess) - green - yellow
    return green, yellow, red

def partition_candidates(
    candidates: Iterable[str],
    guess: str,
    feedback_fn: FeedbackFn,
) -> Dict[Feedback, List[str]]:
    """Partition candidate secrets by the feedback each would produce for a guess."""
    partitions: Dict[Feedback, List[str]] = defaultdict(list)
    for secret in candidates:
        partitions[feedback_fn(guess, secret)].append(secret)
    return partitions


def expected_partition_size(partitions: Dict[Feedback, List[str]]) -> float:
    total = sum(len(group) for group in partitions.values())
    if total == 0:
        return 0.0
    return sum(len(group) ** 2 for group in partitions.values()) / total

def worst_partition_size(partitions: Dict[Feedback, List[str]]) -> int:
    return max((len(group) for group in partitions.values()), default=0)

def negative_entropy(partitions: Dict[Feedback, List[str]]) -> float:
    """Negated Shannon entropy of the partition, in bits.

    Negated so that, like expected_partition_size and worst_partition_size,
    a lower score is a better guess -- callers minimise regardless of metric.
    """
    total = sum(len(group) for group in partitions.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for group in partitions.values():
        p = len(group) / total
        if p > 0:
            bits -= p * log2(p)
    return -bits

def one_step_score(
    candidates: Iterable[str],
    guess: str,
    feedback_fn: FeedbackFn,
    metric: str = "expected",
) -> float:
    partitions = partition_candidates(candidates, guess, feedback_fn)
    if metric == "expected":
        return expected_partition_size(partitions)
    if metric == "worst":
        return worst_partition_size(partitions)
    if metric == "entropy":
        return negative_entropy(partitions)
    raise ValueError(f"Unknown metric: {metric}")

def best_one_step_guess(
    candidates: List[str],
    guesses: List[str],
    feedback_fn: FeedbackFn,
    metric: str = "expected",
) -> Tuple[str, float]:
    best_guess = guesses[0]
    best_score = float("inf")
    for guess in guesses:
        score = one_step_score(candidates, guess, feedback_fn, metric)
        if score < best_score:
            best_score = score
            best_guess = guess
    return best_guess, best_score

def best_two_step_guess(
    candidates: List[str],
    guesses: List[str],
    feedback_fn: FeedbackFn,
    max_candidates: int = 10,
    first_metric: str = "expected",
    second_metric: str = "expected",
) -> Tuple[str, float]:
    """Choose the best first guess by looking ahead one additional move.

    The search is intentionally limited to small candidate pools by
    `max_candidates` because the two-step evaluation is expensive.
    """
    if len(candidates) > max_candidates:
        raise ValueError(
            "Two-step search is too slow above the cutoff; call only when len(candidates) <= max_candidates"
        )

    best_guess = guesses[0]
    best_score = float("inf")

    for first in guesses:
        partitions = partition_candidates(candidates, first, feedback_fn)

        if first_metric == "expected":
            total = len(candidates)
            score = 0.0
            for partition in partitions.values():
                if len(partition) <= 1:
                    second_score = 0.0
                else:
                    _, second_score = best_one_step_guess(
                        partition, guesses, feedback_fn, metric=second_metric
                    )
                score += (len(partition) / total) * second_score
        elif first_metric == "worst":
            score = 0.0
            for partition in partitions.values():
                if len(partition) <= 1:
                    second_score = 0.0
                else:
                    _, second_score = best_one_step_guess(
                        partition, guesses, feedback_fn, metric=second_metric
                    )
                score = max(score, second_score)
        else:
            raise ValueError(f"Unknown first_metric: {first_metric}")

        if score < best_score:
            best_score = score
            best_guess = first

    return best_guess, best_score

def choose_endgame_guess(
    candidates: List[str],
    guesses: List[str],
    feedback_fn: FeedbackFn,
    use_two_step: bool = True,
    cutoff: int = 10,
    first_metric: str = "expected",
    second_metric: str = "expected",
) -> Tuple[str, float]:
    """Pick either a one-step or two-step guess depending on the current pool size."""
    if use_two_step and len(candidates) <= cutoff:
        return best_two_step_guess(
            candidates,
            guesses,
            feedback_fn,
            max_candidates=cutoff,
            first_metric=first_metric,
            second_metric=second_metric,
        )
    return best_one_step_guess(candidates, guesses, feedback_fn, metric=second_metric)

def simulate_guessing(
    secret: str,
    candidates: List[str],
    guesses: List[str],
    feedback_fn: FeedbackFn,
    use_two_step: bool = False,
    cutoff: int = 10,
    max_turns: int = 6,
) -> Optional[int]:
    """Simulate a single game using the endgame strategy helpers."""
    pool = candidates[:]
    available_guesses = guesses[:]
    for turn in range(1, max_turns + 1):
        if not pool or not available_guesses:
            return None
        guess, _ = choose_endgame_guess(
            pool,
            available_guesses,
            feedback_fn,
            use_two_step=use_two_step,
            cutoff=cutoff,
            first_metric="expected",
            second_metric="expected",
        )
        if guess not in available_guesses:
            return None
        available_guesses.remove(guess)
        feedback = feedback_fn(guess, secret)
        if feedback == (5, 0, 0):
            return turn
        pool = [word for word in pool if feedback_fn(guess, word) == feedback]
    return None

def should_use_endgame(candidates: List[str], cutoff: int = 10) -> bool:
    return len(candidates) <= cutoff


class EndgameSolver(MatrixSolver):
    """A Solver implementation that uses endgame lookahead.

    Above `cutoff` candidates, guesses come from matrix-backed one-step
    scoring (the same FeedbackTable machinery as MatrixOneStep) -- pure
    Python one-step search over a multi-thousand-word pool takes tens of
    seconds per guess and makes a full sweep impractical. Once the pool
    shrinks to `cutoff` or below, it switches to the pure-Python two-step
    lookahead in this module, which is cheap precisely because it is only
    ever run over that small pool.
    """

    def __init__(
        self,
        candidates: list[str],
        guess_pool: list[str] | None = None,
        table: FeedbackTable | None = None,
        *,
        cutoff: int = 10,
        use_two_step: bool = True,
        first_metric: str = "expected",
        second_metric: str = "expected",
        scorer: Callable[[NDArray[np.int32], int], NDArray[np.float64]] | None = None,
        full_pool_below: int | None = None,
        opener: str | None = None,
    ) -> None:
        super().__init__(candidates, guess_pool, table=table)
        self.cutoff = cutoff
        self.use_two_step = use_two_step
        self.first_metric = first_metric
        self.second_metric = second_metric
        self.scorer = scorer if scorer is not None else SCORERS["expected"]
        self.full_pool_below = full_pool_below
        # Fail here rather than mid-game. An opener outside the word list is a
        # config error: the real game would reject it as an invalid guess.
        if opener is not None and opener not in self.table.index:
            raise ValueError(f"opener {opener!r} is not in the word list")
        self.opener = opener

    @property
    def name(self) -> str:
        return f"EndgameSolver(cutoff={self.cutoff}, two_step={self.use_two_step})"

    def next_guess(self) -> str:
        idx = self.cand_idx
        n = len(idx)
        if n == 0:
            raise RuntimeError("no candidates left -- feedback was inconsistent")
        if n == 1:
            return str(self.table.words[idx[0]])
        if not self.history and self.opener:
            return self.opener

        # Two-step search evaluates each candidate guess against every other
        # guess in the pool, so it must stay restricted to self.candidates --
        # running it over the full guess_pool is O(len(guess_pool)^2) pure
        # Python calls and never finishes in practice.
        if self.use_two_step and n <= self.cutoff:
            candidates = self.candidates
            guess, _ = choose_endgame_guess(
                candidates,
                candidates,
                word500_feedback,
                use_two_step=True,
                cutoff=self.cutoff,
                first_metric=self.first_metric,
                second_metric=self.second_metric,
            )
            return guess

        pool = (idx if (self.full_pool_below is not None
                        and n > self.full_pool_below) else self.pool_idx)
        return matrix_one_step_guess(self.table, pool, idx, self.scorer)


if __name__ == "__main__":
    ATER_GROUP = ["LATER", "WATER", "CATER", "BATER", "PATER"]
    IGHT_GROUP = ["MIGHT", "LIGHT", "RIGHT", "NIGHT", "SIGHT"]
    ITCH_GROUP = ["DITCH", "ITCHY", "RITCH", "VITCH"]  # placeholder words with same ending

    for name, group in [("_ATER", ATER_GROUP), ("_IGHT", IGHT_GROUP), ("_ITCH", ITCH_GROUP)]:
        one_guess, one_score = best_one_step_guess(group, group, word500_feedback, metric="expected")
        two_guess, two_score = best_two_step_guess(group, group, word500_feedback, max_candidates=10)
        print(f"{name}: one-step {one_guess} => {one_score:.4f}, two-step {two_guess} => {two_score:.4f}")

        for secret in group:
            turns_one = simulate_guessing(secret, group, group, word500_feedback, use_two_step=False)
            turns_two = simulate_guessing(secret, group, group, word500_feedback, use_two_step=True)
            print(
                f"  secret={secret}: one-step turns={turns_one}, two-step turns={turns_two}"
            )