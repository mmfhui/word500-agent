"""Precomputed feedback lookup table.

score(guess, answer) is a pure function, so for a fixed word list every
answer it will ever give is already determined. This module computes all of
them once so solvers can read instead of recompute.

The build is vectorised via the identity

    greens + yellows = sum over letters of min(count in guess, count in answer)

which means yellows never need the per-pair multiset logic. The whole table
is 5 positional comparisons plus 26 letter-count minimums -- 31 array passes
instead of one Python call per pair.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from word500.scoring import WORD_LENGTH, Feedback, score

# Every valid (greens, yellows) pair. (4,1) is impossible: with four letters
# in place, one letter remains on each side, and if they matched it would be
# five greens -- so nothing is left to pair up.
FEEDBACK_CLASSES = [
    (g, y)
    for g in range(WORD_LENGTH + 1)
    for y in range(WORD_LENGTH + 1 - g)
    if (g, y) != (WORD_LENGTH - 1, 1)
]
CODE = {fb: i for i, fb in enumerate(FEEDBACK_CLASSES)}
N_CLASSES = len(FEEDBACK_CLASSES)
WIN_CODE = CODE[(WORD_LENGTH, 0)]

# (greens, yellows) -> code, as an array so encoding is one fancy-index op.
_LUT: NDArray[np.uint8] = np.full((WORD_LENGTH + 1, WORD_LENGTH + 1), 255, dtype=np.uint8)
for _fb, _c in CODE.items():
    _LUT[_fb] = _c


def letter_arrays(words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Words as an (n,5) array of letter codes and an (n,26) array of counts."""
    n = len(words)
    letters_flat = np.frombuffer("".join(words).encode(), dtype=np.uint8)
    letters = letters_flat.reshape(n, WORD_LENGTH) - ord("A")
    counts = np.zeros((n, 26), dtype=np.uint8)
    for lo in range(26):
        counts[:, lo] = (letters == lo).sum(axis=1)
    return letters, counts


def build(words: list[str]) -> np.ndarray:
    """The full (n,n) uint8 table. M[i,j] is the code for guess i vs answer j."""
    letters, counts = letter_arrays(words)
    n = len(words)

    greens = np.zeros((n, n), dtype=np.uint8)
    for p in range(WORD_LENGTH):
        greens += letters[:, p][:, None] == letters[None, :, p]

    total = np.zeros((n, n), dtype=np.uint8)
    for lo in range(26):
        total += np.minimum(counts[:, lo][:, None], counts[None, :, lo])

    return cast(NDArray[np.uint8], _LUT[greens, total - greens])


def decode(code: int) -> Feedback:
    """Code -> (greens, yellows) pair."""
    return Feedback(*FEEDBACK_CLASSES[code])


def validate(matrix: np.ndarray, words: list[str], n_pairs: int = 20000,
             seed: int = 0) -> int:
    """Check the table against score() on random pairs. Returns mismatches."""
    rng = np.random.default_rng(seed)
    n = len(words)
    i_idx = rng.integers(0, n, n_pairs)
    j_idx = rng.integers(0, n, n_pairs)
    bad = 0
    for i, j in zip(i_idx, j_idx):
        if decode(matrix[i, j]) != score(words[i], words[j]):
            bad += 1
    return bad


if __name__ == "__main__":
    # Build the table, prove the validator works, and print the numbers worth
    # recording. Run as: python -m word500.matrix
    import time
    from math import log2

    from word500.wordlist import Mode, load_allowed, possible_secrets

    pool = possible_secrets(load_allowed(), Mode.STANDARD)
    n = len(pool)
    print(f"pool {n}   valid feedback classes {N_CLASSES}\n")

    t = time.perf_counter()
    M = build(pool)
    vec = time.perf_counter() - t
    print(f"  vectorised build     {vec:6.2f}s   {M.nbytes / 1e6:.0f} MB uint8")

    t = time.perf_counter()
    for i in range(40):
        for j in range(n):
            score(pool[i], pool[j])
    naive = (time.perf_counter() - t) / 40 * n
    print(f"  naive score() loop   {naive:6.0f}s   {naive / vec:.0f}x slower")

    print(f"\n  validation           {validate(M, pool, 20_000)} mismatches in 20,000 pairs")

    # A validator that has never failed proves nothing. Break the table on
    # purpose and confirm it complains.
    shifted = np.minimum(M + 1, N_CLASSES - 1).astype(np.uint8)
    speckled = M.copy()
    flat = np.random.default_rng(2).choice(n * n, size=n * n // 1000, replace=False)
    speckled.flat[flat] = (speckled.flat[flat] + 3) % N_CLASSES
    print(f"  fault: all shifted   {validate(shifted, pool, 5000, seed=1)} / 5000 caught")
    print(f"  fault: 0.1% wrong    {validate(speckled, pool, 20000, seed=1)} caught "
          f"(expect ~20)")

    print("\n  structural checks")
    print(f"    diagonal is a win  {bool((M[np.arange(n), np.arange(n)] == WIN_CODE).all())}")
    print(f"    symmetric          {bool((M == M.T).all())}"
          "   (score is symmetric, so row/col order cannot be wrong)")
    print(f"    classes observed   {len(np.unique(M))} of {N_CLASSES}")

    freq = np.bincount(M.ravel(), minlength=N_CLASSES)
    p = freq[freq > 0] / freq.sum()
    print(f"\n  feedback entropy over all pairs   {-(p * np.log2(p)).sum():.3f} bits "
          f"of {log2(N_CLASSES):.3f} ceiling")
    print("  most common feedbacks:")
    for c in np.argsort(-freq)[:5]:
        g, y = FEEDBACK_CLASSES[c]
        print(f"    {g}G {y}Y {WORD_LENGTH - g - y}R   {freq[c] / freq.sum():6.2%}")

    # The 4.32-bit ceiling assumes the 20 outcomes are equally likely. They are
    # not, so the naive floor is optimistic. Find the genuinely best opener --
    # the same search that takes ~6 minutes in pure Python.
    t = time.perf_counter()
    entropies = np.empty(n)
    for i in range(n):
        counts = np.bincount(M[i], minlength=N_CLASSES)
        q = counts[counts > 0] / n
        entropies[i] = -(q * np.log2(q)).sum()
    best = int(entropies.argmax())
    print(f"\n  opener search        {time.perf_counter() - t:.1f}s  "
          "(~6 min in pure Python)")
    print(f"    best opener        {pool[best]}   {entropies[best]:.3f} bits")
    print(f"    mean over guesses  {entropies.mean():.3f} bits")
    print(f"\n  naive floor          {log2(n) / log2(N_CLASSES):.2f} guesses"
          "   (assumes 4.32 bits/guess)")
    print(f"  achievable floor     {log2(n) / entropies[best]:.2f} guesses"
          "   (no guess exceeds its own entropy)")