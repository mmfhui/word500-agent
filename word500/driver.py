"""
Runs one game of word500 with an agent against the environment.

This acts as the game loop. Swapping solvers requires no
change here.
"""

from word500.game import Game
from word500.solvers.base import Solver


def play(game: Game, solver: Solver) -> int | None:
    """
    Play one full game.

    Returns the number of guesses used, or None if the solver ran out.
    """
    while not game.is_over:
        guess = solver.next_guess()
        feedback = game.guess(guess)
        if feedback.is_win:
            return game.guesses_made
        solver.update(guess, feedback)
    return None
