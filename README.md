# Word500 Agent

An AI agent that plays [Word500](https://word500.com), a Wordle variant
where feedback tells you **how many** letters are correct but not **which**.

CS 4100 final project — Rishikesan, Sun, Michael

## Why this is harder than Wordle

Wordle's feedback is labelled: you see which letter went yellow. Word500's
is anonymous so you only get counts of red/yellow/green that always sum to 5.

## Setup

    ./download_data.sh      # fetch word lists, verify them
    python play.py          # play it yourself

## Difficulty modes (based on the official game)

Each constrains the **secret word**, not your guesses:

| mode | secret word rules |
|---|---|
| standard | no repeated letters, no J/Q/X/Z |
| standard+ | no repeated letters |
| advanced | no restrictions |

## Layout

| path | role |
|---|---|
| `word500/scoring.py` | the feedback rule — one function, everything depends on it |
| `word500/wordlist.py` | loading and validating word lists, mode filters |
| `word500/game.py` | the environment: holds one secret, judges guesses |
| `word500/solvers/` | the agents |
| `play.py` | play by hand against the local game |
| `harness.py` | run an agent over every secret, report results |

## Design constraint

The solver never sees the answer list and never receives the `Game`
object. It only sees `(guess, feedback)` pairs. This keeps it honest, and means
the same solver works against the real word500.com with a human relaying
the counts.

## Word lists

Wordle's lists stand in for Word500's, which isn't published. `answers.txt`
is the original 2,315 answers; `allowed.txt` is the current valid-guess
list (14,855 word). Not committed; `download_data.sh` fetches them.