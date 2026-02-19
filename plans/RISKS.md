# Risks and Rollback Triggers

## R-001 `extra_body` ingestion mismatch
- Risk: OpenAI SDK merges `extra_body` keys into top-level JSON; server may miss controls if it only expects nested `extra_body`.
- Mitigation: parse raw request payload and accept `x_adapter_critic` at top-level control namespace.
- Rollback trigger: repeated request parsing failures across direct/adapter/critic behavior tests.

## R-002 Response extension compatibility
- Risk: client wrappers may ignore or drop non-standard response fields.
- Mitigation: keep OpenAI-required fields untouched; place extras under a single `adapter_critic` namespace.
- Rollback trigger: SDK compatibility behavior test fails after schema changes.

## R-003 Token aggregation mismatch
- Risk: stage usage fields may differ by provider and break token totals.
- Mitigation: normalize usage fields in one module and test with fixture variants.
- Rollback trigger: deterministic unit token tests fail across workflows.

## R-004 Prompt regressions in adapter/critic modes
- Risk: prompts over-constrain or under-constrain model output.
- Mitigation: isolate prompts in module and test formatting contracts.
- Rollback trigger: behavior tests show invalid edit grammar or missing final response stages.

## R-005 Mode resolution ambiguity
- Risk: startup config and request overrides conflict unpredictably.
- Mitigation: codify precedence in contract tests.
- Rollback trigger: same resolution bug for 3 loops (anti-thrash threshold).

## R-006 No-try/except policy impact
- Risk: fail-fast exceptions may surface raw stack traces during early integration.
- Mitigation: keep validation strict at boundaries and test invalid inputs explicitly.
- Rollback trigger: repeated uncaught exceptions block milestone progress for 3 loops.

## R-007 Stacked diff chain drift
- Risk: branch stack diverges from intended PR chain, making review/merge order unclear.
- Mitigation: require parent-merge checkbox in non-first PRs and log PR chain in devlogs.
- Rollback trigger: verifier finds missing parent linkage for 2 consecutive iterations.

## R-008 Missed human comments
- Risk: loop proceeds without incorporating async instructions from `plans/HUMAN_COMMENTS.md`.
- Mitigation: mandatory comment-file checks at iteration start and verifier checkpoints.
- Rollback trigger: detected missed instruction with code divergence; pause and reconcile plan/work docs.
