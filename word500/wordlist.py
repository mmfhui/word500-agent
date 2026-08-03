"""
Loading the Word500 word lists (using Wordle as Word500 is not available).
`load_answers()` is the secret word the harness draws from and
`load_allowed()` is the pool the solver guesses from. The solver
imports only the second one, so it has no code path to the answers.
"""

from enum import Enum
from pathlib import Path

from word500.scoring import WORD_LENGTH

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXCLUDED_LETTERS = frozenset("JQXZ")

class Mode(Enum):
    """Word500 difficulty. Constrains the SECRET word, not your guesses."""

    STANDARD = "standard"        # no repeated letters, and no J/Q/X/Z
    STANDARD_PLUS = "standard+"  # no repeated letters
    ADVANCED = "advanced"        # no restrictions

def can_be_secret(word: str, mode: Mode) -> bool:
    """Filters if the word be the secret in the selected mode"""
    if mode is Mode.ADVANCED:
        return True
    if len(set(word)) != len(word):      # a letter appears twice
        return False
    if mode is Mode.STANDARD:
        return not (EXCLUDED_LETTERS & set(word))
    return True


def possible_secrets(words: list[str], mode: Mode) -> list[str]:
    """Narrow a word list to what the mode's rules permit as a secret."""
    return [w for w in words if can_be_secret(w, mode)]

def load_words(path: Path) -> list[str]:
    """Read one word per line (all uppercase) and rejects anything malformed."""
    words: list[str] = []
    seen: set[str] = set()

    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        word = line.strip().upper()
        if not word:
            continue
        if len(word) != WORD_LENGTH or not word.isalpha():
            raise ValueError(f"{path}:{lineno}: not a {WORD_LENGTH}-letter word: {word!r}")
        if word in seen:
            raise ValueError(f"{path}:{lineno}: duplicate word: {word!r}")
        seen.add(word)
        words.append(word)

    if not words:
        raise ValueError(f"{path} contained no words")
    return words


def load_answers() -> list[str]:
    """The secret word. Harness only, and the solver must never call this."""
    return load_words(DATA_DIR / "answers.txt")


def load_allowed() -> list[str]:
    """Every word the solver may guess. Must be a superset of the answers."""
    return load_words(DATA_DIR / "allowed.txt")


if __name__ == "__main__":
    answers, allowed = load_answers(), load_allowed()
    missing = set(answers) - set(allowed)
    print(f"answers: {len(answers)}")
    print(f"allowed: {len(allowed)}")
    if missing:
        raise SystemExit(
            f"BROKEN: {len(missing)} answers are not in allowed.txt, e.g. "
            f"{sorted(missing)[:3]}. Your allowed list is probably the "
            f"'additional guesses' file, concatenate the answers into it."
        )
    print("ok: every answer is a guessable word")
