# Devlog Contract

## File
- Repo root: `devlogs.md`
- If missing when implementation starts: create it immediately.

## Update Frequency
- At least once per iteration.
- After every verifier run.
- After every rollback/promotion decision from BACKBURNER loop check.
- After every `plans/HUMAN_COMMENTS.md` review if any action item is added or deferred.

## Entry Template
```markdown
## YYYY-MM-DD HH:MM (local)
- Goal:
- Changes (paths):
- Commands + results:
- Decision:
- Next:
- Checkpoint commit:
```

## Checkpoint Policy
- Create meaningful checkpoint commits regularly (not giant delayed commits).
- Commit message should reflect milestone slice intent.
- Never add AI attribution.

## Required Cross-References
- Mention related milestone ID (e.g., `M4`).
- Mention failing/passing verifier command snippets.
- Mention BACKBURNER check result each loop.
- Mention `plans/HUMAN_COMMENTS.md` check result each loop.
- Mention branch name, remote push status, and PR URL/status for stacked diff tracking.
