# Definition of Done Checklist

- [ ] `POST /v1/chat/completions` works for direct mode.
- [ ] Adapter mode implements API -> Adapter with search/replace (`lgtm` or edits).
- [ ] Critic mode implements API -> Critic -> API sandwich.
- [ ] Startup model-id routing config works without request overrides.
- [ ] Request-level `extra_body.x_adapter_critic` overrides mode/model/base-url as planned.
- [ ] Response includes intermediate stage outputs in `adapter_critic.intermediate`.
- [ ] Response includes per-stage and total tokens in `adapter_critic.tokens`.
- [ ] Unit tests cover contracts, edits, config resolution, and token aggregation.
- [ ] Behavior tests cover direct/adapter/critic flow and routing/override behavior.
- [ ] `ruff format` + `ruff` + `mypy` + `pytest` wired in pre-commit (line length 120).
- [ ] README documents startup config and direct/adapter/critic usage examples.
- [ ] `devlogs.md` maintained with per-iteration + verifier entries and checkpoint hashes.
- [ ] `plans/HUMAN_COMMENTS.md` checked regularly and each check logged in `devlogs.md`.
- [ ] Stacked-diff branch chain is maintained with one PR per branch and remote push done.
- [ ] Non-first PRs include parent-merge checkbox and follow required PR format.
