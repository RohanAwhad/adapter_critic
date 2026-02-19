# Definition of Done Checklist

- [x] `POST /v1/chat/completions` works for direct mode.
- [x] Adapter mode implements API -> Adapter with search/replace (`lgtm` or edits).
- [x] Critic mode implements API -> Critic -> API sandwich.
- [x] Startup model-id routing config works without request overrides.
- [x] Request-level `extra_body.x_adapter_critic` overrides mode/model/base-url as planned.
- [x] Response includes intermediate stage outputs in `adapter_critic.intermediate`.
- [x] Response includes per-stage and total tokens in `adapter_critic.tokens`.
- [x] Unit tests cover contracts, edits, config resolution, and token aggregation.
- [x] Behavior tests cover direct/adapter/critic flow and routing/override behavior.
- [x] `ruff format` + `ruff` + `mypy` + `pytest` wired in pre-commit (line length 120).
- [x] README documents startup config and direct/adapter/critic usage examples.
- [x] `devlogs.md` maintained with per-iteration + verifier entries and checkpoint hashes.
- [x] `plans/HUMAN_COMMENTS.md` checked regularly and each check logged in `devlogs.md`.
- [x] Stacked-diff branch chain is maintained with one PR per branch and remote push done.
- [x] Non-first PRs include parent-merge checkbox and follow required PR format.
