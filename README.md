# Word500 Agent

An AI agent that plays [Word500](https://word500.com), a Wordle variant
where feedback tells you **how many** letters are correct but not **which**.

CS 4100 Final Project: 
**Team Members:** Rishikesan, Sun, Michael

## Why this is harder than Wordle?

Wordle's feedback is labelled: you see which letter went yellow. Word500's is
anonymous: you get three counts (green / yellow / red) that always sum to 5.

That leaves **20 distinct feedback outcomes** per guess against Wordle's 243 so
each guess carries at most log2(20) ≈ **4.3 bits** instead of 7.9. Same
dictionary, less than half the information per guess.

## Setup: 

    ./download_data.sh      # fetch and verify the word lists
    python play.py          # play it yourself

`download_data.sh` ends by running `python -m word500.wordlist`, which fails
loudly if a list is malformed or if any answer is missing from the guess pool.

## Difficulty modes (from the official game): 

Each constrains the **secret word**, not your guesses:

| mode | secret word rules |
|---|---|
| `standard` | no repeated letters, no J/Q/X/Z |
| `standard+` | no repeated letters |
| `advanced` | no restrictions |

Filtering the candidate pool by mode is legitimate deduction, the mode is shown
on screen before you guess, so it does not violate the design constraint below.

## Running the harness: 

    python harness.py                          # one solver, full sweep
    python harness.py --compare --oracle       # every solver, one table
    python harness.py --sample 300             # fast while iterating
    python harness.py --only FOUND -v          # replay one game, show guesses
    python harness.py --mode advanced
    python harness.py --guesses 5              # tighten until win rate matters
    python harness.py --verify                 # cross-check filtering (slow)
    python harness.py --csv results/run.csv    # per-game rows

Each game is seeded from `crc32(secret)` so any game replays identically in
isolation and `--sample` never changes how a game is played.

At the default 8-guess budget almost everything wins, making win rate a
near-useless metric: use `--guesses 5` when you need a number that
discriminates between strategies.

## Baselines: 

Standard mode, 8-guess budget, 1,477 secrets:

| solver | knows answers | avg guesses | solved | bits/guess |
|---|---|---|---|---|
| GreedyLetters | no | 5.379 | 99.5% | 2.43 |
| RandomConsistent | no | 5.609 | 99.3% | 2.33 |
| GreedyLetters | yes | 4.523 | 100% | 2.33 |
| RandomConsistent | yes | 4.784 | 100% | 2.20 |

The information floor is **3.02 guesses** so the best baseline leaves ~2.4
guesses of headroom. Withholding the answer list costs **~0.86 guesses** —
measured with `--oracle`, not estimated.

Notably, random-with-answers (4.784) beats greedy-without (5.379): the
information we deliberately withhold is worth more than the heuristic.

## Where does the baselines fail?

Failures cluster in families of words differing in a **single position**:
`_ATER`, `_AKER`, `_IGHT`, `_ITCH`, `_OUND`. Aggregated over ten seeds, seven of
the ten most frequently failed words end in `-ER`, against ~5.6% of the pool —
roughly 12x enrichment.

The mechanism is specific to anonymous feedback. Guessing `WATER` when the
answer is `LATER` returns `4G 0Y` and eliminates exactly one candidate, so the
solver degenerates to linear search through the family. In Wordle you would see
*which* letter matched; here you only get a count, and cannot tell that probing
initial consonants would be smarter.

No word is reliably unsolvable — even the worst is solved 60% of the time — so
these are systematically *risky*, not hard.

## Design constraint: 

The solver never sees the answer list and never receives the `Game` object. It
gets a candidate pool at construction and `(guess, feedback)` pairs thereafter.

This is enforced structurally: no file under `word500/solvers/` imports
`load_answers`. It also means the same solver can play the real word500.com
with a human relaying the counts.

## Correctness checks: 

Filtering fails silently: a bad filter produces no crash, just quietly worse
averages that look like a weak strategy. Two checks, both validated by
deliberately injecting broken filters:

- **Always on, free.** A lost game must still hold the secret among its
  candidates, since the secret is consistent with all feedback by definition.
  Catches a filter that eliminates too much.
- **`--verify`, costly.** Incremental filtering must exactly match re-filtering
  from scratch. Also catches a filter that eliminates too *little*, which the
  free check is structurally blind to.

Neither can catch an error in `score()` itself since both depend on it. That
requires cross-checking one guess against the live site.

## Layout: 

| path | role |
|---|---|
| `word500/scoring.py` | the feedback rule — one function, everything depends on it |
| `word500/wordlist.py` | loading and validating word lists, mode filters |
| `word500/game.py` | the environment: holds one secret, judges guesses |
| `word500/driver.py` | `play(game, solver)` — the game loop, 11 lines |
| `word500/solvers/base.py` | the `Solver` interface and shared filtering |
| `word500/solvers/` | the agents |
| `play.py` | play by hand against the local game |
| `harness.py` | sweep an agent over every secret and report |
| `download_data.sh` | fetch and verify the word lists |

## Word lists: 

Wordle's lists stand in for Word500's, which isn't published. `answers.txt` is
the original 2,315 answers; `allowed.txt` is the current valid-guess list
(14,855 words — larger than the commonly cited 12,972 since words have been
added since 2021). Neither is committed; `download_data.sh` fetches them.

Because Word500 uses its own unpublished dictionary, exact parity with
published Wordle solver benchmarks was never available.

## Adding a solver: 

Subclass `Solver`, implement `next_guess()`, and register it:

```python
from word500.solvers.base import Solver

class MySolver(Solver):
    def next_guess(self) -> str:
        return self.candidates[0]
```

`update()` and candidate filtering are inherited. `self.candidates` shrinks with
feedback; `self.guess_pool` does not, because a word that cannot be the answer
may still be the most informative thing to guess.

Register in `SOLVERS` and it picks up `--compare`, `--oracle`, `--verify`,
`--only`, and CSV output automatically.

## Useful identity: 

    greens + yellows = sum over letters of min(count in guess, count in answer)

So yellows never need computing separately, and the whole feedback table can be
built with 31 vectorised passes instead of one call per pair.
