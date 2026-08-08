"""Unit tests for the Word500 scoring model.

These tests ensure feedback values, repeated-letter handling, and scoring
invariants are correct for the core Word500 response model.
"""

import unittest

from word500.scoring import Feedback, score

class ScoringTests(unittest.TestCase):
    """Validation of the Feedback model and score() helper."""

    def test_exact_match_returns_full_green_feedback(self) -> None:
        feedback = score("apple", "apple")

        self.assertEqual(feedback, Feedback(5, 0))
        self.assertTrue(feedback.is_win)
        self.assertEqual(feedback.reds, 0)

    def test_no_overlap_returns_all_red_feedback(self) -> None:
        feedback = score("abcde", "fghij")

        self.assertEqual(feedback, Feedback(0, 0))
        self.assertFalse(feedback.is_win)
        self.assertEqual(feedback.reds, 5)

    def test_mixed_feedback_counts_greens_and_yellows_without_double_counting(self) -> None:
        feedback = score("aabbb", "bbbaa")

        self.assertEqual(feedback, Feedback(1, 4))
        self.assertEqual(feedback.reds, 0)

    def test_repeated_letters_are_not_overcounted(self) -> None:
        feedback = score("aaabb", "bbaaa")

        self.assertEqual(feedback, Feedback(1, 4))
        self.assertEqual(feedback.reds, 0)

    def test_words_must_be_five_letters_long(self) -> None:
        with self.assertRaises(ValueError):
            score("abc", "def")

        with self.assertRaises(ValueError):
            score("abcdef", "ghijkl")

    def test_feedback_reds_is_derived_from_green_and_yellow_counts(self) -> None:
        feedback = Feedback(2, 2)

        self.assertEqual(feedback.reds, 1)

    def test_all_greens_and_all_yellows_are_disjoint_counts(self) -> None:
        feedback = score("abcde", "abcde")
        self.assertEqual(feedback, Feedback(5, 0))

        feedback = score("abcde", "edcba")
        self.assertEqual(feedback, Feedback(1, 4))

    def test_repeated_letters_with_partial_overlap_keep_counts_consistent(self) -> None:
        feedback = score("aabca", "aacaa")

        self.assertEqual(feedback, Feedback(3, 1))
        self.assertEqual(feedback.reds, 1)

    def test_feedback_properties_remain_consistent_across_examples(self) -> None:
        for greens, yellows in [(0, 0), (2, 2), (5, 0)]:
            feedback = Feedback(greens, yellows)
            self.assertEqual(feedback.reds, 5 - greens - yellows)
            self.assertEqual(feedback.is_win, greens == 5)


if __name__ == "__main__":
    unittest.main()
