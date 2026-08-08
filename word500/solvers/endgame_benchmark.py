"""Benchmark helper for comparing one-step vs two-step endgame behavior.

Evaluates the small problem groups `_ATER`, `_IGHT`, and `_ITCH`
using the endgame search helpers, and reports solve rates, average turns,
and any failures.
"""

from word500.solvers.endgame_search import (
    best_one_step_guess,
    best_two_step_guess,
    simulate_guessing,
    word500_feedback,
)

GROUPS = {
    "_ATER": ["LATER", "WATER", "CATER", "BATER", "PATER"],
    "_IGHT": ["MIGHT", "LIGHT", "RIGHT", "NIGHT", "SIGHT"],
    "_ITCH": ["DITCH", "ITCHY", "NITCH", "RITCH"],
}

def solve_group(group, use_two_step: bool):
    """Run the endgame simulation on a group of candidate secrets."""
    results = []
    for secret in group:
        turns = simulate_guessing(
            secret,
            group,
            group,
            word500_feedback,
            use_two_step=use_two_step,
            cutoff=10,
            max_turns=6,
        )
        results.append((secret, turns))

    solved = sum(1 for _, turns in results if turns is not None)
    avg_turns = (
        sum(turns for _, turns in results if turns is not None) / solved
        if solved
        else None
    )
    failed = [secret for secret, turns in results if turns is None]
    return {
        "results": results,
        "solved": solved,
        "avg_turns": avg_turns,
        "failed": failed,
    }

def run_group_comparison():
    """Compare one-step and two-step endgame choices and solve statistics."""
    for name, group in GROUPS.items():
        one_guess, one_score = best_one_step_guess(group, group, word500_feedback, metric="expected")
        two_guess, two_score = best_two_step_guess(
            group,
            group,
            word500_feedback,
            max_candidates=10,
            first_metric="expected",
            second_metric="expected",
        )
        print(f"{name}: one-step guess={one_guess}, score={one_score:.4f}; two-step guess={two_guess}, score={two_score:.4f}")

        one_stats = solve_group(group, use_two_step=False)
        two_stats = solve_group(group, use_two_step=True)

        print(
            f"  one-step: solved={one_stats['solved']}/{len(group)}, avg_turns={one_stats['avg_turns']}, failed={one_stats['failed']}"
        )
        print(
            f"  two-step: solved={two_stats['solved']}/{len(group)}, avg_turns={two_stats['avg_turns']}, failed={two_stats['failed']}"
        )
        print()


if __name__ == "__main__":
    run_group_comparison()
