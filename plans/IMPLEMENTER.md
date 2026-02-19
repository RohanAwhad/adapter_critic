# Implementer Guide

## Guardrails
- Follow `plans/WORK.md` milestone order.
- Keep diffs minimal and local.
- Strong typing everywhere; no blanket `Any` drift.
- No `try/except` unless explicitly requested.
- No attribution/watermarks in code, docs, commits.
- Line length 120 via formatter/linter config.
- Check `plans/HUMAN_COMMENTS.md` every iteration and before verifier runs.

## Execution Rhythm (loop-friendly)
1. Check `plans/HUMAN_COMMENTS.md`; log new instructions in `devlogs.md`.
2. Pick one milestone slice.
3. Add/adjust tests first for that slice.
4. Implement smallest change to make tests pass.
5. Run milestone verifier subset.
6. Log results in `devlogs.md`.
7. Create checkpoint commit.

## Implementation Notes
- Prefer explicit contracts over implicit dict juggling.
- Separate state/config models from orchestration behavior.
- Keep workflow orchestrators isolated:
  - direct
  - adapter
  - critic
- Keep prompt text in dedicated module (`prompts.py`).

## Required Deliverables
- Working server endpoint `/v1/chat/completions`.
- Startup config model-id routing.
- Request overrides via `extra_body.x_adapter_critic`.
- Response extension with intermediate outputs + token usage breakdown.
- README usage for all three modes.

## Branching and PR Contract (stacked diffs)
- Branch model:
  - first branch: from base branch (e.g., `main`)
  - subsequent branch: from previous stack branch head
- Push every branch to remote once created and after updates.
- Open one PR per branch when branch slice is ready.
- PR body must follow memory contract format:
  - `## Summary`
  - `## Test plan`
  - `## Test script`
  - `## Test output`
  - `## Unit tests`
- For non-first PRs, include stack tracking checkbox:
  - `[ ] Parent PR merged: <url>`
- For first PR in stack:
  - `Parent PR: N/A (first in stack)`

## Stuck Policy
- If blocked by same issue for 3 iterations:
  - apply anti-thrash from `plans/VERIFIER.md`
  - rollback to last green checkpoint
  - record rationale in `devlogs.md`
