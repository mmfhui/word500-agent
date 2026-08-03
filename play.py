"""
Play Word500 yourself against the local test bench.
    python play.py                    # random Standard secret
    python play.py --mode advanced
    python play.py --secret crane     # fixed secret, for checking behaviour
    python play.py --guesses 5
At the prompt: a five-letter word, or 'q' to give up.
"""

import argparse
import random

from word500.game import Game
from word500.wordlist import (EXCLUDED_LETTERS, Mode, load_allowed, load_answers, possible_secrets)


MODE_HELP = {
    Mode.STANDARD:      "no repeated letters, and no J/Q/X/Z",
    Mode.STANDARD_PLUS: "no repeated letters (J/Q/X/Z allowed)",
    Mode.ADVANCED:      "anything goes, with repeats and rare letters allowed",
}


def choose_mode() -> Mode:
    """Prompts the user to choose a difficulty level. Used when --mode is not given."""
    print("\nDifficulty (this constrains the SECRET word, not your guesses):")
    for i, (m, desc) in enumerate(MODE_HELP.items(), start=1):
        print(f"  {i}. {m.value:14} {desc}")
    while True:
        choice = input("Pick 1-3 [1]: ").strip() or "1"
        if choice in {"1", "2", "3"}:
            return list(MODE_HELP)[int(choice) - 1]
        print("  please enter 1, 2, or 3")


def show_rules(mode: Mode, max_guesses: int) -> None:
    print(f"""
--- WORD500 ({mode.value}) ---

  Find the secret five-letter word in {max_guesses} guesses.
  After each guess you are told only HOW MANY letters are:

     #   right letter, right place
     +   right letter, wrong place
     .   not in the word at all

  The three counts always add up to 5. You are NOT told which
  letter is which, the additional constraint on top of Wordle.

  This round's secret: {MODE_HELP[mode]}.
  Type 'q' to give up.
""")


def show_board(game: Game) -> None:
    """Print every guess so far. Word500 is unplayable without this UI."""
    print()
    for turn, (word, fb) in enumerate(game.history, start=1):
        bar = "#" * fb.greens + "+" * fb.yellows + "." * fb.reds
        print(f"   {turn}  {' '.join(word)}   {bar}   {fb.greens}G {fb.yellows}Y {fb.reds}R")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Word500 locally.")
    parser.add_argument("--mode", default=None, choices=[m.value for m in Mode],
                        help="skip the interactive difficulty picker")
    parser.add_argument("--secret", help="fix the secret instead of drawing one")
    parser.add_argument("--guesses", type=int, default=8)
    args = parser.parse_args()

    mode = Mode(args.mode) if args.mode else choose_mode()
    allowed = set(load_allowed())

    # play.py sits on the harness side of the wall, so knowing the
    # secret here is fine. A solver must never see it.
    if args.secret:
        secret = args.secret.strip().upper()
    else:
        secret = random.choice(possible_secrets(load_answers(), mode))

    game = Game(secret, max_guesses=args.guesses)
    show_rules(mode, game.max_guesses)

    while not game.is_over:
        word = input(f"[{game.guesses_left} left] > ").strip().upper()

        if word in {"Q", "QUIT"}:
            print(f"Gave up, it was {secret}.")
            return
        if len(word) != 5:
            print("  five letters only please")
            continue
        if word not in allowed:
            print("  not in the word list")
            continue

        # Warn about letters the mode's rules already rule out, but let
        # the guess through -- whether such guesses are ever worth making
        # is a question for the solver to answer, not a rule of the game.
        notes = []
        if mode is not Mode.ADVANCED and len(set(word)) < 5:
            notes.append("a repeated letter")
        if mode is Mode.STANDARD:
            dead = sorted(set(word) & EXCLUDED_LETTERS)
            if dead:
                notes.append("/".join(dead))
        if notes:
            print(f"  note: {' and '.join(notes)} cannot be in a {mode.value} secret")
            if input("  use it anyway? [y/N] ").strip().lower() != "y":
                continue

        game.guess(word)
        show_board(game)

    print("Got it!" if game.won else f"Out of guesses. The secret word was {game.reveal()}.")


if __name__ == "__main__":
    main()
