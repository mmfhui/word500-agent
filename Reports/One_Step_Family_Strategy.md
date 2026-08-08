# One-Step Family Strategy

## Standard-mode benchmark at an 8-guess budget:
On the standard-mode benchmark with 400 secrets, all four one-step scorers solved 100% of games but entropy was the strongest overall at 5.140 average guesses blind and 4.383 with the answer list available. Expected was very close behind, while parts and minimax were slightly weaker.

## Standard-mode benchmark at a 5-guess budget:
When the budget was tightened to 5 guesses, the win-rate differences became much clearer: entropy solved 72.5% of blind games and 98.8% with the answer list, while parts and minimax fell further behind. This shows that the stricter limit makes the quality of the partitioning strategy much more visible.

## Tuning full_pool_below:
The threshold sweep around 50, 100, 150, 200, and 300 suggested that 150 is a sensible default because performance was broadly similar across the middle range and no clear improvement came from pushing it higher. For now, I recommend keeping full_pool_below at 150 unless a much larger offline sweep is needed.

## Repeated-letter guesses in standard mode:
Yes! The best solver still sometimes chooses repeated-letter guesses even though standard-mode answers cannot contain repeated letters; the entropy run produced 41 repeated-letter guesses across 400 games (10.2%), including examples such as AAHED and ABBOT. Therefore, that confirms the solver occasionally values the extra positional information from a repeated-letter probe enough to use it.

## Overall Takeaway:
The one-step family is strong and surprisingly robust, with entropy emerging as the best overall scorer in both the relaxed and stricter benchmark settings. The main tradeoff is not between scorers alone but between the extra information gained by exploring the full guess pool and the computational cost of doing so.
