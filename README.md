# Word500 Agent

An AI agent that plays [Word500](https://word500.com), a Wordle variant
where feedback tells you **how many** letters are correct but not **which**.

CS 4100 Final Project
**Team Members:** Rishikesan, Sun, Michael

## Why this is harder than Wordle?

Wordle's feedback is labelled: you see which letter went yellow. Word500's is
anonymous: you get three counts (green / yellow / red) that always sum to 5.

That leaves **20 distinct feedback outcomes** per guess against Wordle's 243, so
each guess carries at most log2(20) ≈ **4.3 bits** instead of 7.9. Same
dictionary, less than half the information per guess.

## Setup

    ./download_data.sh              # fetch and verify the word lists
    pip install -r requirements.txt
    python play.py                  # play it yourself

`download_data.sh` ends by running `python -m word500.wordlist`, which fails
loudly if a list is malformed or if any answer is missing from the guess pool.

## Difficulty modes (from the official game)

Each constrains the **secret word**, not your guesses:

| mode | secret word rules |
|---|---|
| `standard` | no repeated letters, no J/Q/X/Z |
| `standard+` | no repeated letters |
| `advanced` | no restrictions |

Filtering the candidate pool by mode is legitimate deduction — the mode is shown
on screen before you guess — so it does not violate the design constraint below.

## Running the harness

    python harness.py                          # one solver, full sweep
    python harness.py --compare --oracle       # every solver, one table
    python harness.py --solvers entropy,greedy # pick specific solvers
    python harness.py --sample 300             # fast while iterating
    python harness.py --only FOUND -v          # replay one game, show guesses
    python harness.py --mode advanced
    python harness.py --guesses 5              # tighten until win rate matters
    python harness.py --verify                 # cross-check filtering (slow)
    python harness.py --csv results/run.csv    # per-game rows

Each game is seeded from `crc32(secret)`, so any game replays identically in
isolation and `--sample` never changes how a game is played.

At the default 8-guess budget almost everything wins, making win rate a
near-useless metric: use `--guesses 5` when you need a number that
discriminates between strategies.

`--compare` skips the `py-` reference solvers, which are pure-Python versions
kept for cross-checking and are ~90x slower. Ask for them by name with
`--solvers`.

## Results

Standard mode, 8-guess budget, 400 secrets, seed 1.

| solver | knows answers | avg guesses | solved | worst | bits/guess | efficiency |
|---|---|---|---|---|---|---|
| entropy | no | **5.098** | 100% | 7 | 2.56 | 59.3% |
| expected | no | 5.103 | 100% | **6** | 2.56 | 59.2% |
| parts | no | 5.147 | 100% | 7 | 2.54 | 58.7% |
| minimax | no | 5.230 | 100% | 7 | 2.50 | 57.8% |
| greedy | no | 5.407 | 99.5% | 8 | 2.41 | 55.9% |
| random | no | 5.610 | 99.2% | 8 | 2.33 | 53.9% |
| entropy | yes | **4.367** | 100% | 6 | 2.41 | 55.8% |
| expected | yes | 4.388 | 100% | 6 | 2.40 | 55.5% |
| parts | yes | 4.412 | 100% | 6 | 2.39 | 55.2% |
| minimax | yes | 4.480 | 100% | 6 | 2.35 | 54.4% |
| greedy | yes | 4.575 | 100% | 7 | 2.30 | 53.2% |
| random | yes | 4.780 | 100% | 8 | 2.20 | 51.0% |

Full 1,477-secret sweeps confirm the baselines: greedy 5.379 (99.5%), random
5.609 (99.3%). `bits/guess` and `efficiency` are only comparable *within* a
knowledge setting, since they divide by the candidate pool.

Four findings worth stating:

**The four one-step strategies are barely distinguishable.** Entropy, expected
size, most parts, and minimax span 0.13 guesses. The Mastermind literature
reports the same pattern on other configurations — most parts and entropy were
statistically indistinguishable on MM(4,6) — so this reproduces a known result
on a novel game.

**Strategy matters less than expected.** Best baseline to best strategy is only
0.31 guesses. In Wordle, entropy solvers beat naive heuristics decisively. With
20 anonymous outcomes rather than 243 labelled ones, there is less room for a
clever guess to distinguish itself from a reasonable one.

**Greedy minimax does not minimise the worst case.** `expected` has a worst case
of 6; `minimax` has 7 — despite minimax existing to optimise the worst case. A
one-step-ahead minimax only minimises the worst *next* partition, not the worst
outcome of the game. Greedy is not optimal.

**Withholding the answer list costs more from weaker solvers:**

| solver | cost of being blind |
|---|---|
| expected | +0.715 |
| entropy | +0.730 |
| parts | +0.735 |
| minimax | +0.750 |
| greedy | +0.850 |
| random | +0.855 |

Better solvers lose less, because they extract more information per guess and
recover from the larger starting pool faster.

## On the information floor

The `floor` column reported by the harness is `log2(pool) / log2(20)` = **3.02
guesses**. That is a valid lower bound but a loose one: it assumes a guess
partitions the pool evenly across all 20 feedback classes, which no five-letter
word does.

Measured over all pairs in the pool, the feedback distribution carries **2.900
bits** against the 4.322-bit ceiling — over 60% of pairs land in just three of
the twenty classes. The best available opener (`TARES`) extracts **3.395 bits**
against a 2.818-bit average across all guesses, so choosing the opener well is
worth about 20% more information than an average word.

At 3.395 bits per guess the pool would need ~3.85 guesses. That is an estimate,
not a bound — later guesses face smaller candidate sets where more even
partitions are available — but it is closer to reality than 3.02.

## Where the baselines fail

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

**All four one-step strategies eliminate these failures entirely** (100%
solved, zero failures), while greedy and random both still reach the 8-guess
limit. Being able to probe with a word that cannot be the answer is what breaks
the linear search.

## Repeated-letter guesses

In standard mode the secret cannot contain a repeated letter, so a guess that
repeats one caps its own information: a doubled letter can match at most once,
leaving four distinct letters to test instead of five.

The solver picks one anyway, in a minority of games. 149 / 1477   (10.1%)
figure from `python harness.py --solver entropy --seed 1 | grep -A4 REPEATED` -->

These occur late, on small candidate sets, where breadth stops mattering and
positional precision starts to: a doubled letter probes one letter's position
across several slots at once, which anonymous aggregate feedback otherwise makes
expensive to obtain. The trade is real, and assuming distinct letters dominate
would have been wrong.

Measuring this at all requires giving the solver the **full** allowed word list
as its guess pool rather than the mode-filtered candidates — otherwise every
word it can reach is repeat-free by construction and the count is trivially
zero.

## Design constraint

The solver never sees the answer list and never receives the `Game` object. It
gets a candidate pool at construction and `(guess, feedback)` pairs thereafter.

This is enforced structurally: no file under `word500/solvers/` imports
`load_answers`. It also means the same solver can play the real word500.com
with a human relaying the counts.

## Correctness checks

Filtering fails silently: a bad filter produces no crash, just quietly worse
averages that look like a weak strategy. Two checks, both validated by
deliberately injecting broken filters:

- **Always on, free.** A lost game must still hold the secret among its
  candidates, since the secret is consistent with all feedback by definition.
  Catches a filter that eliminates too much.
- **`--verify`, costly.** Incremental filtering must exactly match re-filtering
  from scratch. Also catches a filter that eliminates too *little*, which the
  free check is structurally blind to.

The feedback table is checked the same way — against `score()` on random pairs,
with the validator itself verified by corrupting the table on purpose. And the
matrix-backed and pure-Python solvers produce **identical averages** on the same
secrets, which is the strongest evidence either is correct.

None of this can catch an error in `score()` itself, since everything depends on
it. That requires cross-checking one guess against the live site.

## Performance

A full 1,477-secret sweep of the entropy solver runs in **56 seconds**, down
from an estimated 7 hours. See `docs/benchmarks.txt` for the full table.

The key step is precomputing every pairwise feedback into a memory-mapped uint8
table, built with 31 vectorised numpy passes rather than 73 million Python
calls — 76x faster than a naive build, and it makes the opening-guess search
0.3 s instead of ~6 minutes.

Parallelism is deferred deliberately: 56 s per sweep means the full strategy
comparison is ~25 minutes of unattended compute, which did not justify the work.

## Layout

| path | role |
|---|---|
| `word500/scoring.py` | the feedback rule — one function, everything depends on it |
| `word500/wordlist.py` | loading and validating word lists, mode filters |
| `word500/game.py` | the environment: holds one secret, judges guesses |
| `word500/driver.py` | `play(game, solver)` — the game loop, 11 lines |
| `word500/matrix.py` | precomputed feedback table, cached and memory-mapped |
| `word500/solvers/base.py` | the `Solver` interface and shared filtering |
| `word500/solvers/registry.py` | which solvers the harness can run |
| `word500/solvers/` | the agents |
| `play.py` | play by hand against the local game |
| `harness.py` | sweep an agent over every secret and report |
| `download_data.sh` | fetch and verify the word lists |
| `docs/benchmarks.txt` | performance measurements |

## Word lists

Wordle's lists stand in for Word500's, which isn't published. `answers.txt` is
the original 2,315 answers; `allowed.txt` is the current valid-guess list
(14,855 words — larger than the commonly cited 12,972, since words have been
added since 2021). Neither is committed; `download_data.sh` fetches them.

Because Word500 uses its own unpublished dictionary, exact parity with published
Wordle solver benchmarks was never available.

## Adding a solver

Subclass `Solver` and implement `next_guess()`:

```python
from word500.solvers.base import Solver

class MySolver(Solver):
    def next_guess(self) -> str:
        return self.candidates[0]
```

`update()` and candidate filtering are inherited. `self.candidates` shrinks with
feedback; `self.guess_pool` does not, because a word that cannot be the answer
may still be the most informative thing to guess.

Then add a factory to `SOLVERS` in `word500/solvers/registry.py` — not in
`harness.py`, so that registering a solver never touches the harness. It picks
up `--compare`, `--oracle`, `--verify`, `--only`, and CSV output automatically.

For a solver that needs the feedback table, subclass `MatrixSolver` instead;
candidates are then numpy indices and filtering is one vectorised comparison.

## Useful identity

    greens + yellows = sum over letters of min(count in guess, count in answer)

So yellows never need computing separately, and the whole feedback table can be
built with 31 vectorised passes instead of one call per pair.