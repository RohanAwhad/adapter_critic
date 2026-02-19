# Verifier Contract

## Verifier Chain (copy/paste)
```bash
uv run ruff format --check . && \
uv run ruff check . && \
uv run mypy . && \
uv run pytest -q tests/unit && \
uv run pytest -q tests/behavior && \
uv run pytest -q
```

## Pass/Fail Rules
- Pass:
  - all commands exit 0
  - no skipped critical behavior tests for direct/adapter/critic routing
- Fail:
  - any non-zero exit
  - missing telemetry assertions (`adapter_critic.intermediate` and `adapter_critic.tokens`)

## Milestone-Scoped Fast Checks
- M1/M2:
  - `uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -q tests/unit`
- M3+:
  - use full verifier chain above

## Anti-Thrash Rule
- If the same failure class persists for K=3 consecutive iterations:
  1. stop feature changes
  2. revert to prior checkpoint commit known green
  3. log rollback rationale and next hypothesis in `devlogs.md`

## Loop Check (Backburner)
- Before each verifier run completion:
  1. scan `plans/BACKBURNER.md` trigger conditions
  2. record result in `devlogs.md`
  3. either promote triggered item into `plans/WORK.md` or defer with explicit note

## Human Comments Check
- At verifier start and before final sign-off:
  1. read `plans/HUMAN_COMMENTS.md`
  2. confirm whether new actionable comments exist
  3. log action/defer decision in `devlogs.md`

## Stacked PR Verification
- Verify each active milestone branch is pushed to remote.
- Verify PR exists for branch when milestone is marked ready.
- Verify PR body follows required format from memory contract (Summary/Test plan/Test script/Test output/Unit tests).
- Verify non-first PRs include a parent-merge checkbox, e.g.:
  - `[ ] Parent PR merged: <url>`
  - first PR may instead use `Parent PR: N/A (first in stack)`

### Useful commands
```bash
git branch --show-current
git rev-parse --abbrev-ref --symbolic-full-name @{u}
gh pr view --json url,baseRefName,headRefName,body
```
