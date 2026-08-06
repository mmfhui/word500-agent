"""
Evaluate Word500 agents and report results.

    python harness.py                              # baseline, full sweep
    python harness.py --solver greedy
    python harness.py --compare                    # all solvers, one table
    python harness.py --sample 300                 # fast while iterating
    python harness.py --only FOUND MOUND -v        # replay specific games
    python harness.py --csv results/random.csv

Each game is seeded from its own secret word, so any single game can be
replayed in isolation and --sample never changes how a game is played.
"""

import argparse
import csv
import random
import statistics
import zlib
from collections import Counter
from typing import List, Optional, Callable, Dict, Any, Tuple
from math import log2
from pathlib import Path

from word500.driver import play
from word500.game import Game
from word500.solvers.registry import SOLVERS
import word500.solvers.registry as registry
from word500.wordlist import Mode, load_allowed, load_answers, possible_secrets


def cand_for_verify(candidates: List[str], verify: bool) -> Optional[List[str]]:
    """
    The pool --verify re-filters from scratch: whatever the solver started with.

    Return the original candidate list when verify is requested, else None.
    """
    return candidates if verify else None


BITS_PER_GUESS = log2(20)   # Word500 has 20 distinct feedbacks
BAR = "\u2587"


def game_seed(base: int, secret: str) -> int:
    """
    A per-game seed derived from the secret.

    crc32 rather than hash(): hash() is randomized per process, so it
    would give different results on every run.
    """
    return zlib.crc32(secret.encode()) ^ base


def evaluate(
    make_solver: Callable[[int], Any],
    secrets: List[str],
    options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Optional[int], List[Tuple[str, Any]]]]:
    """
    Play one game per secret. Returns [(secret, turns_or_None, history)].

    make_solver(seed) must build a fresh solver. It is given a seed, never
    the secret, the solver stays blind by construction.

    options may contain: max_guesses (int), pool (optional candidate pool),
    verify (bool), seed (int).
    """
    if options is None:
        options = {}
    max_guesses = options.get("max_guesses", 8)
    pool = options.get("pool", None)
    verify = options.get("verify", False)
    seed = options.get("seed", 0)
    results = []
    for secret in secrets:
        solver = make_solver(game_seed(seed, secret))
        game = Game(secret, max_guesses=max_guesses)
        turns = play(game, solver)

        # A lost game must still hold the secret among its candidates: the
        # secret is consistent with all feedback by definition. Catches a
        # filter that eliminates too much.
        if turns is None and secret not in solver.candidates:
            raise AssertionError(f"filter eliminated {secret}: {solver.history}")
        # Costly: also catches a filter that eliminates too little.
        if verify and pool is not None:
            expected = solver.consistent(pool)
            if sorted(solver.candidates) != sorted(expected):
                raise AssertionError(
                    f"filter disagrees with from-scratch on {secret}: "
                    f"kept {len(solver.candidates)}, expected {len(expected)}"
                )
        results.append((secret, turns, list(game.history)))
    return results


def summarise(
    results: List[Tuple[str, Optional[int], List[Tuple[str, Any]]]],
    max_guesses: int,
) -> Dict[str, Any]:
    """Reduce raw results to the numbers worth reporting."""
    wins: List[int] = [t for _, t, _ in results if t is not None]
    fails: List[str] = [s for s, t, _ in results if t is None]
    # Failures counted as max_guesses + 1 so a solver cannot look good by
    # failing on the words it would have found slowly.
    penalised: List[int] = wins + [max_guesses + 1] * len(fails)
    return {
        "n": len(results),
        "wins": len(wins),
        "fails": fails,
        "rate": len(wins) / len(results),
        "avg": statistics.mean(wins) if wins else float("nan"),
        "avg_pen": statistics.mean(penalised),
        "median": statistics.median(wins) if wins else float("nan"),
        "worst": max(wins) if wins else 0,
        "dist": Counter(wins),
    }


def report(
    title: str,
    summary: Dict[str, Any],
    max_guesses: int,
    report_info: Dict[str, Any],
) -> None:
    """Print a formatted summary report for one evaluation run."""
    cand_pool = report_info["cand_pool"]
    n_secrets = report_info["n_secrets"]
    knows = report_info["knows"]

    bits = log2(cand_pool)
    rule = "-" * 62
    print(f"\n{rule}\n  {title}\n{rule}\n")
    print(
        f"  SOLVED           {summary['wins']:>5} / {summary['n']:<6}"
        f"      {summary['rate']:>6.1%}"
    )
    print(f"  AVG GUESSES      {summary['avg']:>7.3f}             wins only")
    print(
        f"                   {summary['avg_pen']:>7.3f}"
        f"             failures counted as {max_guesses + 1}"
    )
    print(
        f"  MEDIAN           {summary['median']:>7.1f}"
        f"       WORST   {summary['worst']}"
    )

    if summary["dist"]:
        widest = max(summary["dist"].values())
        print("\n  GUESSES USED")
        for turn in range(1, max_guesses + 1):
            n = summary["dist"].get(turn, 0)
            print(
                f"    {turn}   {n:>5}  {n / summary['n']:>5.1%}"
                f"  {BAR * round(32 * n / widest)}"
            )

    print("\n  INFORMATION")
    print(
        f"    candidate pool     {cand_pool:>6} words"
        f"     {bits:>5.2f} bits to resolve"
    )
    print(
        f"    knows answers?     {knows:>6}             "
        f"{'' if knows == 'oracle' else f'(only {n_secrets} can occur)'}"
    )
    print(
        f"    ceiling            {BITS_PER_GUESS:>6.2f} bits/guess"
        f"  (20 possible feedbacks)"
    )
    print(f"    {'-' * 52}")
    print(f"    best possible      {bits / BITS_PER_GUESS:>6.2f} guesses")
    if summary["wins"]:
        got = bits / summary["avg"]
        print(
            f"    achieved           {got:>6.2f} bits/guess  ->  "
            f"{got / BITS_PER_GUESS:.1%} of the ceiling"
        )

    # Report repeated-letter statistics when present (useful for STANDARD mode
    # analysis).
    if 'repeat_guesses' in summary:
        print(f"\n  REPEATED-LETTER GUESSES")
        print(f"    total repeated guesses: {summary.get('repeat_guesses', 0)}")
        print(f"    games with >=1 repeated guess: {summary.get('repeat_games', 0)} / {n_secrets}")
        examples = summary.get('repeat_examples', [])
        if examples:
            for i in range(0, min(len(examples), 8), 8):
                print("    examples: " + ", ".join(examples[i:i+8]))

    if summary["fails"]:
        print(f"\n  FAILED ({len(summary['fails'])})")
        for i in range(0, min(len(summary["fails"]), 32), 8):
            print(
                (
                    "    "
                    + "  ".join(f"{word:<6}" for word in summary["fails"][i:i + 8])
                ).rstrip()
            )
        if len(summary["fails"]) > 32:
            print(f"    ... +{len(summary['fails']) - 32} more")
    print()


def compare(rows: List[Tuple[str, str, int, Dict[str, Any]]], max_guesses: int) -> None:
    """
    Compare the performance of different solvers.

    Three blocks: performance, information, distribution.

    Split rather than one wide table because the information columns push
    a combined table past 130 characters.
    """
    w = 19  # solver-name column width

    print("\n  PERFORMANCE")
    head = (f"  {'solver':<{w}}{'knows':>7}{'solved':>9}{'avg':>9}"
            f"{'pen':>9}{'med':>6}{'wst':>5}")
    print(head + "\n  " + "-" * (len(head) - 2))
    for name, knows, _, st in rows:
        print(f"  {name:<{w}}{knows:>7}{st['rate']:>8.1%}{st['avg']:>9.3f}"
              f"{st['avg_pen']:>9.3f}{st['median']:>6.1f}{st['worst']:>5}")

    print("\n  INFORMATION")
    head = (f"  {'solver':<{w}}{'knows':>7}{'pool':>7}{'bits':>7}{'floor':>7}"
            f"{'bits/g':>8}{'eff':>7}{'room':>7}")
    print(head + "\n  " + "-" * (len(head) - 2))
    for name, knows, pool_n, st in rows:
        bits = log2(pool_n)
        floor = bits / BITS_PER_GUESS
        got = bits / st["avg"] if st["wins"] else float("nan")
        print(f"  {name:<{w}}{knows:>7}{pool_n:>7}{bits:>7.2f}{floor:>7.2f}"
              f"{got:>8.2f}{got / BITS_PER_GUESS:>7.1%}{st['avg'] - floor:>7.2f}")
    print(f"  {'':<{w}}  bits/g and eff are only comparable WITHIN a knows setting")
    print(f"  {'':<{w}}  (they divide by the pool, which differs between them)")

    print("\n  DISTRIBUTION")
    turns = list(range(2, max_guesses + 1))
    head = (f"  {'solver':<{w}}{'knows':>7}   "
            + "".join(f"{t:>6}" for t in turns) + f"{'fail':>7}")
    print(head + "\n  " + "-" * (len(head) - 2))
    for name, knows, _, st in rows:
        print(f"  {name:<{w}}{knows:>7}   "
              + "".join(f"{st['dist'].get(t, 0):>6}" for t in turns)
              + f"{len(st['fails']):>7}")

    blind = {n: st for n, k, _, st in rows if k == "blind"}
    orac = {n: st for n, k, _, st in rows if k == "oracle"}
    if orac:
        print("\n  COST OF NOT KNOWING THE ANSWER LIST")
        for name in sorted(blind):
            if name in orac:
                d = blind[name]["avg_pen"] - orac[name]["avg_pen"]
                print(f"    {name:<{w}}{d:+7.3f} guesses")
    best = min(rows, key=lambda r: r[3]["avg_pen"])
    print(f"\n  best: {best[0]} ({best[1]}) at {best[3]['avg_pen']:.3f}\n")


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """
    Entry point: parse args and evaluate solvers.

    Kept intentionally concise; lint exemptions allow this function to
    orchestrate the evaluation flow.
    """
    ap = argparse.ArgumentParser(description="Evaluate Word500 agents.")
    ap.add_argument("--solver", default="random", choices=sorted(SOLVERS))
    ap.add_argument("--compare", action="store_true", help="run every solver")
    ap.add_argument("--mode", default=Mode.STANDARD.value,
                    choices=[m.value for m in Mode])
    ap.add_argument("--guesses", type=int, default=8)
    ap.add_argument("--sample", type=int, help="evaluate on N secrets, not all")
    ap.add_argument("--only", nargs="+", metavar="WORD",
                    help="evaluate just these secrets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check filtering from scratch (slow)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every guess of every game")
    ap.add_argument("--oracle", action="store_true",
                    help="also give the solver the answer list (upper bound)")
    ap.add_argument("--full-pool-below", type=int, default=None,
                    help="override registry FULL_POOL_BELOW threshold (None uses default)")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    mode = Mode(args.mode)

    # Allow overriding the registry threshold that controls when the full
    # guess pool is considered. Setting this here affects the factories in
    # word500.solvers.registry because they read the global at construction
    # time.
    if args.full_pool_below is not None:
        registry.FULL_POOL_BELOW = args.full_pool_below

    pool = possible_secrets(load_allowed(), mode)          # 'blind' candidates
    answer_pool = possible_secrets(load_answers(), mode)   # 'oracle' candidates
    secrets = list(answer_pool)
    n_secrets = len(secrets)

    if args.only:
        secrets = [w.strip().upper() for w in args.only]
        missing = [w for w in secrets if w not in set(pool)]
        if missing:
            raise SystemExit(
                f"not in the {mode.value} guess pool: {', '.join(missing)}\n"
                f"(a secret outside the pool is unwinnable -- wrong --mode?)")
    elif args.sample and args.sample < len(secrets):
        # Its own rng, so sampling cannot perturb how any game is played.
        secrets = sorted(random.Random(args.seed).sample(secrets, args.sample))

    chosen = sorted(SOLVERS.items()) if args.compare else [(args.solver, SOLVERS[args.solver])]

    # Each solver runs once per knowledge setting. 'blind' is the real
    # project constraint; 'oracle' is the upper bound it is measured against.
    settings = [("blind", pool)] + ([("oracle", answer_pool)] if args.oracle else [])

    rows = []
    for key, factory in chosen:
        for knows, candidates in settings:
            options = {
                "max_guesses": args.guesses,
                "pool": cand_for_verify(candidates, args.verify),
                "verify": args.verify,
                "seed": args.seed,
            }
            def make_solver(
                sd: int,
                f: Callable[[list[str], int], Any] = factory,
                candidates_: list[str] = candidates,
            ) -> object:
                return f(candidates_, sd)

            results = evaluate(make_solver, secrets, options)
            st = summarise(results, args.guesses)

            # Repeated-letter analysis: in STANDARD mode secrets cannot have
            # repeated letters; this checks whether the solver nevertheless
            # ever chooses guesses that repeat letters and how often.
            def _has_repeat(word: str) -> bool:
                return len(set(word)) != len(word)

            repeat_games = 0
            repeat_guesses = 0
            repeat_examples = []
            for secret, turns, history in results:
                g_repeated = False
                for guess, _ in history:
                    if _has_repeat(guess):
                        repeat_guesses += 1
                        g_repeated = True
                if g_repeated:
                    repeat_games += 1
                    if len(repeat_examples) < 8:
                        repeat_examples.append(secret)

            st['repeat_games'] = repeat_games
            st['repeat_guesses'] = repeat_guesses
            st['repeat_examples'] = repeat_examples

            rows.append((key, knows, len(candidates), st))

            if args.verbose:
                for secret, turns, history in results:
                    print(f"\n  {secret}  ({turns or 'FAILED'})  [{knows}]")
                    for i, (w, fb) in enumerate(history, 1):
                        print(f"    {i}  {w}  {fb.greens}G {fb.yellows}Y {fb.reds}R")

            if not args.compare:
                report(
                    f"{key}  |  {mode.value}  |  {knows}  |  "
                    f"{args.guesses}-guess budget  |  {st['n']} secrets",
                    st,
                    args.guesses,
                    {
                        "cand_pool": len(candidates),
                        "n_secrets": n_secrets,
                        "knows": knows,
                    },
                )

            if args.csv:
                # Always suffixed, so filenames are predictable and
                # self-documenting across runs.
                path = args.csv.with_name(
                    f"{args.csv.stem}_{key}_{knows}{args.csv.suffix}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["secret", "turns", "solver", "knows_answers",
                                     "mode", "max_guesses", "seed"])
                    for secret, turns, _ in results:
                        writer.writerow([secret, turns or "", key, knows,
                                         mode.value, args.guesses, args.seed])
                print(f"  wrote {path}")

    if args.compare:
        print(f"\n{mode.value} | {len(secrets)} secrets | {args.guesses}-guess budget")
        compare(rows, args.guesses)


if __name__ == "__main__":
    main()
