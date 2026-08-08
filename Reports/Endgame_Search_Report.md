# Endgame Search Report

## Completed work:

- Added a standalone endgame search module at `word500/solvers/endgame_search.py`.
  - Implemented Word500 feedback scoring and candidate partitioning.
  - Added one-step and two-step endgame scoring helpers.
  - Added a simulation helper for small candidate pools.
  - Added an `EndgameSolver` subclass so the endgame strategy can be used by the normal solver engine.
  - Added benchmark and test support.
  - Added `word500/solvers/endgame_benchmark.py` to compare one-step vs two-step solve performance.
  - Integrated the new solver into the solver registry.
  - Registered a new solver key: `endgame` in `word500/solvers/registry.py`.
  - This allows the harness and normal play flow to run the endgame strategy directly.

## Example result:

From the benchmark run on the small endgame groups:
- `_ATER`: one-step and two-step both solved `5/5`, with average turns `3.0`.

## Notes:
- It is not yet tuned in the broader harness beyond the small-group experiment but it can be run through the existing engine.
