import random
import unittest

from word500.scoring import Feedback
from word500.solvers.random_consistent import RandomConsistent
from word500.wordlist import Mode, load_allowed, possible_secrets


class RandomConsistentTests(unittest.TestCase):
    def test_next_guess_returns_one_of_the_candidates(self) -> None:
        candidates = ["abcde", "fghij", "klmno"]
        solver = RandomConsistent(candidates, rng=random.Random(0))

        self.assertIn(solver.next_guess(), candidates)

    def test_next_guess_is_reproducible_with_a_seeded_rng(self) -> None:
        candidates = ["abcde", "fghij", "klmno", "pqrst"]
        first = RandomConsistent(candidates, rng=random.Random(42))
        second = RandomConsistent(candidates, rng=random.Random(42))

        self.assertEqual(first.next_guess(), second.next_guess())

    def test_a_default_rng_is_created_when_none_is_given(self) -> None:
        solver = RandomConsistent(["abcde", "fghij"])

        self.assertIsInstance(solver.rng, random.Random)

    def test_next_guess_raises_when_no_candidates_remain(self) -> None:
        solver = RandomConsistent([], rng=random.Random(0))

        with self.assertRaises(RuntimeError):
            solver.next_guess()

    def test_update_filters_candidates_the_same_way_as_other_solvers(self) -> None:
        candidates = ["aaabb", "ababa", "bbaaa", "ccccc"]
        solver = RandomConsistent(candidates, rng=random.Random(0))

        solver.update("aaabb", Feedback(3, 2))

        self.assertEqual(solver.candidates, ["ababa"])

    def test_real_game_mode_uses_the_wordlist_data(self) -> None:
        allowed = load_allowed()
        candidates = possible_secrets(allowed, Mode.STANDARD)[:20]
        solver = RandomConsistent(candidates, rng=random.Random(0))

        self.assertIn(solver.next_guess(), candidates)


if __name__ == "__main__":
    unittest.main()
