import unittest

import numpy as np

from word500.matrix import FeedbackTable
from word500.scoring import Feedback
from word500.solvers.matrix_solver import (
    SCORERS,
    MatrixOneStep,
    MatrixOneStepConfig,
    entropy,
    most_parts,
    neg_expected_size,
    neg_max_size,
)
from word500.solvers.one_step import SCORERS as PY_SCORERS
from word500.solvers.one_step import OneStepAhead
from word500.wordlist import Mode, load_allowed, possible_secrets


class ScorerTests(unittest.TestCase):
    def test_entropy_matches_the_pure_python_implementation(self) -> None:
        counts = np.array([[2, 1]])

        self.assertAlmostEqual(float(entropy(counts, 3)[0]), PY_SCORERS["entropy"]([2, 1], 3))

    def test_most_parts_matches_the_pure_python_implementation(self) -> None:
        counts = np.array([[2, 1, 1]])

        self.assertEqual(float(most_parts(counts, 4)[0]), PY_SCORERS["parts"]([2, 1, 1], 4))

    def test_neg_expected_size_matches_the_pure_python_implementation(self) -> None:
        counts = np.array([[2, 1, 1]])

        self.assertAlmostEqual(float(neg_expected_size(counts, 4)[0]),
                                PY_SCORERS["expected"]([2, 1, 1], 4))

    def test_neg_max_size_matches_the_pure_python_implementation(self) -> None:
        counts = np.array([[2, 1, 1]])

        self.assertEqual(float(neg_max_size(counts, 4)[0]), PY_SCORERS["minimax"]([2, 1, 1], 4))


class MatrixSolverTests(unittest.TestCase):
    def test_requires_a_feedback_table(self) -> None:
        # MatrixSolver itself is abstract (no next_guess), so the missing-table
        # check is exercised through the concrete MatrixOneStep instead.
        with self.assertRaises(ValueError):
            MatrixOneStep(["ABCDE", "FGHIJ"])

    def test_update_filters_candidates_the_same_way_as_other_solvers(self) -> None:
        words = ["AAABB", "ABABA", "BBAAA", "CCCCC"]
        table = FeedbackTable(words)
        solver = MatrixOneStep(words, guess_pool=words, table=table)

        solver.update("AAABB", Feedback(3, 2))

        self.assertEqual(solver.candidates, ["ABABA"])


class MatrixOneStepTests(unittest.TestCase):
    def test_next_guess_returns_first_candidate_when_two_or_fewer_remain(self) -> None:
        words = ["APPLE", "APPLY"]
        table = FeedbackTable(words)
        solver = MatrixOneStep(words, guess_pool=words, table=table)

        self.assertEqual(solver.next_guess(), "APPLE")

    def test_opener_is_used_on_the_first_turn(self) -> None:
        words = ["APPLE", "APPLY", "APRIC"]
        table = FeedbackTable(words)
        solver = MatrixOneStep(words, guess_pool=words, table=table,
                                config=MatrixOneStepConfig(opener="APRIC"))

        self.assertEqual(solver.next_guess(), "APRIC")

    def test_rejects_an_opener_outside_the_word_list(self) -> None:
        words = ["ABCDE", "FGHIJ", "KLMNO"]
        table = FeedbackTable(words)

        with self.assertRaises(ValueError):
            MatrixOneStep(words, guess_pool=words, table=table,
                          config=MatrixOneStepConfig(opener="ZZZZZ"))

    def test_next_guess_raises_when_no_candidates_remain(self) -> None:
        words = ["ABCDE", "FGHIJ", "KLMNO"]
        table = FeedbackTable(words)
        solver = MatrixOneStep(words, guess_pool=words, table=table)

        solver.update("ABCDE", Feedback(1, 1))

        with self.assertRaises(RuntimeError):
            solver.next_guess()

    def test_matches_the_pure_python_one_step_solver_for_every_scorer(self) -> None:
        words = ["ABCDE", "ABCFG", "HIJKL", "MNOPQ", "RSTUV"]
        table = FeedbackTable(words)

        for key in SCORERS:
            with self.subTest(scorer=key):
                py_solver = OneStepAhead(words, guess_pool=words, scorer=PY_SCORERS[key])
                matrix_solver = MatrixOneStep(
                    words, guess_pool=words, table=table,
                    config=MatrixOneStepConfig(scorer=SCORERS[key]))

                self.assertEqual(py_solver.next_guess(), matrix_solver.next_guess())

    def test_real_game_mode_uses_the_wordlist_data(self) -> None:
        allowed = load_allowed()
        subset = possible_secrets(allowed, Mode.STANDARD)[:15]
        table = FeedbackTable(subset)
        solver = MatrixOneStep(subset, guess_pool=subset, table=table)

        self.assertIn(solver.next_guess(), subset)


if __name__ == "__main__":
    unittest.main()
