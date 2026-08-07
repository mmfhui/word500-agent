import unittest
from math import log2

from word500.scoring import Feedback, score
from word500.solvers.one_step import (
    OneStepAhead,
    SCORERS,
    max_entropy,
    most_parts,
    neg_expected_size,
    neg_max_size,
)
from word500.wordlist import Mode, load_allowed, possible_secrets

class OneStepAheadTests(unittest.TestCase):
    def test_scorer_helpers_return_expected_values(self) -> None:
        expected_entropy = -((2 / 3) * log2(2 / 3) + (1 / 3) * log2(1 / 3))

        self.assertAlmostEqual(max_entropy([2, 1], 3), expected_entropy)
        self.assertEqual(most_parts([2, 1, 1], 4), 3)
        self.assertAlmostEqual(neg_expected_size([2, 1, 1], 4), -1.5)
        self.assertEqual(neg_max_size([2, 1, 1], 4), -2)

    def test_next_guess_returns_first_candidate_when_two_or_fewer_remain(self) -> None:
        solver = OneStepAhead(["apple", "apply"], guess_pool=["apple", "apply"])

        self.assertEqual(solver.next_guess(), "apple")

    def test_opener_is_used_on_the_first_turn(self) -> None:
        solver = OneStepAhead(
            ["apple", "apply", "apric"],
            guess_pool=["apple", "apply", "apric"],
            opener="zebra",
        )

        self.assertEqual(solver.next_guess(), "zebra")

    def test_parts_scorer_prefers_a_guess_that_creates_more_partitions(self) -> None:
        solver = OneStepAhead(
            ["abide", "baker", "cider"],
            guess_pool=["abide", "baker", "cider"],
            scorer=SCORERS["parts"],
        )

        self.assertEqual(solver.next_guess(), "baker")

    def test_real_game_mode_uses_the_wordlist_data(self) -> None:
        allowed = load_allowed()
        candidates = possible_secrets(allowed, Mode.STANDARD)[:20]
        solver = OneStepAhead(candidates, guess_pool=allowed, opener="TARES")

        self.assertEqual(solver.next_guess(), "TARES")

    def test_repeated_letter_feedback_filters_candidates(self) -> None:
        solver = OneStepAhead(
            ["aaabb", "ababa", "bbaaa", "ccccc"],
            guess_pool=["aaabb", "ababa", "bbaaa", "ccccc"],
        )

        solver.update("aaabb", Feedback(3, 2))

        self.assertEqual(solver.candidates, ["ababa"])

    def test_consistent_matches_all_history_invariants(self) -> None:
        candidates = ["aaabb", "ababa", "bbaaa", "ccccc"]
        solver = OneStepAhead(candidates, guess_pool=candidates)

        solver.update("aaabb", Feedback(3, 2))
        solver.update("ababa", Feedback(5, 0))

        self.assertEqual(solver.consistent(candidates), ["ababa"])
        self.assertEqual(solver.candidates, solver.consistent(candidates))
        self.assertTrue(all(score(g, w) == fb for g, fb in solver.history for w in solver.candidates))

    def test_history_does_not_change_consistent_results_when_recomputed(self) -> None:
        solver = OneStepAhead(["aaabb", "ababa", "bbaaa", "ccccc"], guess_pool=["aaabb", "ababa", "bbaaa", "ccccc"])
        solver.update("aaabb", Feedback(3, 2))

        self.assertEqual(solver.consistent(["aaabb", "ababa", "bbaaa", "ccccc"]), ["ababa"])
        self.assertEqual(solver.consistent(["aaabb", "ababa", "bbaaa", "ccccc"]), solver.consistent(["aaabb", "ababa", "bbaaa", "ccccc"]))

    def test_solver_name_and_scorer_reference_are_exposed(self) -> None:
        solver = OneStepAhead(["apple", "apply"], guess_pool=["apple", "apply"], scorer=SCORERS["entropy"])

        self.assertEqual(solver.name, "OneStepAhead(max_entropy)")
        self.assertIs(solver.scorer, SCORERS["entropy"])


if __name__ == "__main__":
    unittest.main()
