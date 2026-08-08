from __future__ import annotations

"""Endgame search utilities for Word500.

This module implements small-candidate endgame heuristics for the Word500
variant. It supports one-step and two-step lookahead scoring, candidate
partitioning by feedback, simulated solve runs, and a real Solver subclass
that can be registered with the harness.

Two-step search is restricted to small pools to keep the solver fast.
"""

from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from word500.solvers.base import Solver  # New endgame strategy solver integration

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


class EndgameSolver(Solver):
    """A Solver implementation that uses endgame lookahead.

    This solver behaves like an ordinary agent in the harness but it
    switches from one-step scoring to a two-step endgame search once the
    remaining candidate pool is small.
    """

    def __init__(
        self,
        candidates: list[str],
        guess_pool: list[str] | None = None,
        *,
        cutoff: int = 10,
        use_two_step: bool = True,
        first_metric: str = "expected",
        second_metric: str = "expected",
    ) -> None:
        super().__init__(candidates, guess_pool)
        self.cutoff = cutoff
        self.use_two_step = use_two_step
        self.first_metric = first_metric
        self.second_metric = second_metric

    @property
    def name(self) -> str:
        return f"EndgameSolver(cutoff={self.cutoff}, two_step={self.use_two_step})"

    def next_guess(self) -> str:
        if not self.candidates:
            raise RuntimeError("no candidates left -- feedback was inconsistent")
        if len(self.candidates) == 1:
            return self.candidates[0]

        # Use candidate-only one-step before the endgame threshold to keep
        # the solver efficient. Switch to two-step lookahead in the endgame.
        if self.use_two_step and len(self.candidates) <= self.cutoff:
            guess, _ = choose_endgame_guess(
                self.candidates,
                self.guess_pool if self.guess_pool is not None else self.candidates,
                word500_feedback,
                use_two_step=True,
                cutoff=self.cutoff,
                first_metric=self.first_metric,
                second_metric=self.second_metric,
            )
        else:
            guess, _ = best_one_step_guess(
                self.candidates,
                self.candidates,
                word500_feedback,
                metric=self.second_metric,
            )
        return guess


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