# Backburner

## Queue

### BB-001 - Streaming mode support (SSE)
- Trigger condition: all non-streaming direct/adapter/critic behavior tests are stable and green for 2 iterations.
- Why it matters: many OpenAI-compatible clients expect `stream=true` support.
- Suggested next action: add SSE response adapter and stage-wise streaming policy.
- Estimate: L

### BB-002 - Persisted telemetry export
- Trigger condition: intermediate/token payload schema settles and no breaking changes for 3 iterations.
- Why it matters: useful for offline eval and token usage audits.
- Suggested next action: add optional sink (JSONL) for stage telemetry.
- Estimate: M

### BB-003 - Config schema as external YAML
- Trigger condition: startup config contract passes and team wants environment-specific deployments.
- Why it matters: easier ops management than env var-only config.
- Suggested next action: define YAML schema + loader + validation tests.
- Estimate: M

### BB-004 - Prompt tuning set for adapter/critic
- Trigger condition: baseline behavior tests pass but quality eval reveals systematic failure classes.
- Why it matters: improves final answer quality without architectural churn.
- Suggested next action: create prompt fixture set + golden behavior cases.
- Estimate: M

### BB-005 - Strict schema for `adapter_critic` extension
- Trigger condition: two client integrations consume extension payload.
- Why it matters: avoids accidental breaking changes in telemetry fields.
- Suggested next action: publish JSON schema and add contract tests.
- Estimate: S

## Loop Check
- Between iterations, verifier/loop must inspect all BB trigger conditions.
- For each check cycle:
  1. log check result in `devlogs.md`
  2. promote triggered item to `plans/WORK.md` or explicitly defer with reason
  3. include BB IDs reviewed in verifier notes
