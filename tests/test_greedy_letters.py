import unittest

from word500.scoring import Feedback
from word500.solvers.greedy_letters import GreedyLetters
from word500.wordlist import Mode, load_allowed, possible_secrets


class GreedyLettersTests(unittest.TestCase):
    def test_next_guess_returns_the_only_candidate_when_one_remains(self) -> None:
        solver = GreedyLetters(["apple"])

        self.assertEqual(solver.next_guess(), "apple")

    def test_next_guess_prefers_the_candidate_with_the_most_shared_letters(self) -> None:
        candidates = ["abcde", "abcfg", "hijkl"]
        solver = GreedyLetters(candidates)

        self.assertEqual(solver.next_guess(), "abcfg")

    def test_ties_break_toward_the_alphabetically_last_word(self) -> None:
        candidates = ["abcde", "fghij", "klmno"]
        solver = GreedyLetters(candidates)

        self.assertEqual(solver.next_guess(), "klmno")

    def test_next_guess_raises_when_no_candidates_remain(self) -> None:
        solver = GreedyLetters([])

        with self.assertRaises(RuntimeError):
            solver.next_guess()

    def test_update_filters_candidates_the_same_way_as_other_solvers(self) -> None:
        candidates = ["aaabb", "ababa", "bbaaa", "ccccc"]
        solver = GreedyLetters(candidates)

        solver.update("aaabb", Feedback(3, 2))

        self.assertEqual(solver.candidates, ["ababa"])

    def test_real_game_mode_uses_the_wordlist_data(self) -> None:
        allowed = load_allowed()
        candidates = possible_secrets(allowed, Mode.STANDARD)[:20]
        solver = GreedyLetters(candidates)

        self.assertIn(solver.next_guess(), candidates)


if __name__ == "__main__":
    unittest.main()
