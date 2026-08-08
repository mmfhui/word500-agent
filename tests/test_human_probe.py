import unittest

from word500.scoring import Feedback
from word500.solvers.human_probe import HumanProbe, pick_probes
from word500.wordlist import Mode, load_allowed, possible_secrets


class PickProbesTests(unittest.TestCase):
    def test_prefers_the_word_that_covers_the_most_letters(self) -> None:
        pool = ["abcfg", "abcde", "hijkl"]

        self.assertEqual(pick_probes(pool, 1), ["abcfg"])

    def test_second_probe_avoids_letters_the_first_already_covered(self) -> None:
        pool = ["abcfg", "abcde", "hijkl"]

        self.assertEqual(pick_probes(pool, 2), ["abcfg", "hijkl"])

    def test_ties_break_toward_the_alphabetically_last_word(self) -> None:
        pool = ["abcde", "fghij", "klmno"]

        self.assertEqual(pick_probes(pool, 1), ["klmno"])

    def test_stops_early_when_the_pool_is_smaller_than_requested(self) -> None:
        pool = ["abcde", "fghij"]

        probes = pick_probes(pool, 5)

        self.assertEqual(sorted(probes), sorted(pool))

    def test_returns_empty_list_when_num_probes_is_zero(self) -> None:
        self.assertEqual(pick_probes(["abcde", "fghij"], 0), [])

    def test_returns_empty_list_when_pool_is_empty(self) -> None:
        self.assertEqual(pick_probes([], 2), [])


class HumanProbeTests(unittest.TestCase):
    def test_next_guess_returns_first_candidate_when_two_or_fewer_remain(self) -> None:
        solver = HumanProbe(["apple", "apply"], guess_pool=["apple", "apply"])

        self.assertEqual(solver.next_guess(), "apple")

    def test_next_guess_follows_probes_then_switches_to_closing_phase(self) -> None:
        pool = ["abcde", "fghij", "klmno", "pqrst", "uvwxy"]
        solver = HumanProbe(pool, guess_pool=pool)

        self.assertEqual(solver.probes, ["uvwxy", "pqrst"])

        self.assertEqual(solver.next_guess(), "uvwxy")
        solver.update("uvwxy", Feedback(0, 0))

        self.assertEqual(solver.next_guess(), "pqrst")
        solver.update("pqrst", Feedback(0, 0))

        # Both probes are used up but 3 candidates are still standing, so
        # this must come from the closing phase, not a third probe.
        self.assertEqual(solver.next_guess(), "klmno")

    def test_closing_phase_prefers_the_candidate_with_the_most_shared_letters(self) -> None:
        candidates = ["abcde", "abcfg", "hijkl"]
        solver = HumanProbe(candidates, guess_pool=candidates, num_probes=0)

        self.assertEqual(solver.next_guess(), "abcfg")

    def test_zero_probes_skips_straight_to_the_closing_phase(self) -> None:
        candidates = ["abcde", "fghij", "klmno"]
        solver = HumanProbe(candidates, guess_pool=candidates, num_probes=0)

        self.assertEqual(solver.probes, [])
        self.assertEqual(solver.next_guess(), "klmno")

    def test_update_filters_candidates_the_same_way_as_other_solvers(self) -> None:
        candidates = ["aaabb", "ababa", "bbaaa", "ccccc"]
        solver = HumanProbe(candidates, guess_pool=candidates)

        solver.update("aaabb", Feedback(3, 2))

        self.assertEqual(solver.candidates, ["ababa"])

    def test_next_guess_raises_when_no_candidates_remain(self) -> None:
        candidates = ["abcde", "fghij", "klmno"]
        solver = HumanProbe(candidates, guess_pool=candidates, num_probes=0)

        solver.update("abcde", Feedback(1, 1))

        with self.assertRaises(RuntimeError):
            solver.next_guess()

    def test_real_game_mode_uses_the_wordlist_data(self) -> None:
        allowed = load_allowed()
        candidates = possible_secrets(allowed, Mode.STANDARD)[:20]
        solver = HumanProbe(candidates, guess_pool=allowed)

        first_guess = solver.next_guess()

        self.assertIn(first_guess, allowed)
        self.assertEqual(len(solver.probes), 2)
        self.assertEqual(len(set(solver.probes)), 2)


if __name__ == "__main__":
    unittest.main()
