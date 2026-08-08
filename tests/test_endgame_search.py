"""Unit tests for the endgame search strategy helpers.

These tests compare one-step and two-step endgame behavior on the
problematic groups `_ATER`, `_IGHT`, and `_ITCH`.
"""

from word500.solvers.endgame_search import (
    best_one_step_guess,
    best_two_step_guess,
    choose_endgame_guess,
    simulate_guessing,
    word500_feedback,
)

# Small groups of candidate words that differ only in the first letter.
ATER_GROUP = ["LATER", "WATER", "CATER", "BATER", "PATER"]
IGHT_GROUP = ["MIGHT", "LIGHT", "RIGHT", "NIGHT", "SIGHT"]
ITCH_GROUP = ["DITCH", "ITCHY", "NITCH", "RITCH"]

def test_best_two_step_guess_improves_or_matches_one_step_for_ater():
    one_guess, one_score = best_one_step_guess(
        ATER_GROUP, ATER_GROUP, word500_feedback, metric="expected"
    )
    two_guess, two_score = best_two_step_guess(
        ATER_GROUP,
        ATER_GROUP,
        word500_feedback,
        max_candidates=10,
        first_metric="expected",
        second_metric="expected",
    )
    assert one_guess in ATER_GROUP
    assert two_guess in ATER_GROUP
    assert two_score <= one_score


def test_best_two_step_guess_improves_or_matches_one_step_for_ight():
    one_guess, one_score = best_one_step_guess(
        IGHT_GROUP, IGHT_GROUP, word500_feedback, metric="expected"
    )
    two_guess, two_score = best_two_step_guess(
        IGHT_GROUP,
        IGHT_GROUP,
        word500_feedback,
        max_candidates=10,
        first_metric="expected",
        second_metric="expected",
    )
    assert one_guess in IGHT_GROUP
    assert two_guess in IGHT_GROUP
    assert two_score <= one_score


def test_best_two_step_guess_improves_or_matches_one_step_for_itch():
    one_guess, one_score = best_one_step_guess(
        ITCH_GROUP, ITCH_GROUP, word500_feedback, metric="expected"
    )
    two_guess, two_score = best_two_step_guess(
        ITCH_GROUP,
        ITCH_GROUP,
        word500_feedback,
        max_candidates=10,
        first_metric="expected",
        second_metric="expected",
    )
    assert one_guess in ITCH_GROUP
    assert two_guess in ITCH_GROUP
    assert two_score <= one_score


def test_choose_endgame_guess_switches_at_cutoff():
    guess, score = choose_endgame_guess(
        ATER_GROUP,
        ATER_GROUP,
        word500_feedback,
        use_two_step=True,
        cutoff=10,
    )
    assert guess in ATER_GROUP
    assert score <= 5.0


def test_simulation_solves_small_groups():
    for group in [ATER_GROUP, IGHT_GROUP, ITCH_GROUP]:
        for secret in group:
            turns_one = simulate_guessing(
                secret,
                group,
                group,
                word500_feedback,
                use_two_step=False,
                cutoff=10,
                max_turns=6,
            )
            turns_two = simulate_guessing(
                secret,
                group,
                group,
                word500_feedback,
                use_two_step=True,
                cutoff=10,
                max_turns=6,
            )
            assert turns_one is not None
            assert turns_two is not None