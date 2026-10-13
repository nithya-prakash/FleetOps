# Runbook: quality gate rejected a deployment

`POST /api/v1/deployments` returned `422 quality_gate_failed`. Nothing was created and no device was contacted.

1. Read `error.details.failures` - each entry names the check:
   * `REGRESSION` - candidate is worse than a running model by more than allowed (values, limit, baseline version are included).
   * `BELOW_MINIMUM` / `ABOVE_MAXIMUM` - an absolute threshold was missed.
   * `MISSING_METRIC` - the candidate (or baseline) has no usable value for the metric.
   * `INCOMPARABLE` - the two models were evaluated on different benchmarks (suite, dataset fingerprint, metric definitions, threshold
     policy; for latency also the host class). Comparing them would be meaningless, so it is a failure. Retrain the older/newer model with the
     current pipeline so both carry the same evaluation metadata.
   * `MISSING_BASELINE` - an explicit baseline version is not registered, or `require_baseline` was set and nothing is running.
   * `GATE_REQUIRED` - this control plane requires a `quality_gate` on every deployment.
2. Decide: improve the candidate (usual), loosen the policy deliberately (e.g. accept a recall drop in exchange for precision, in the
   request, where it is reviewable), or - rarely - override.
3. **Override** is admin-only: add `"quality_override": {"reason": "..."}` (>= 10 characters). The deployment then runs under the normal canary
   and runtime health checks, is flagged `quality.overridden`, and `GET /audit/events?action=deployment.quality_gate_overridden` shows who,
   why and which checks were waived. Review these events periodically.
4. Latency checks are noisy (microsecond-scale, host dependent): use relative tolerances and compare only like with like.

Remember: the benchmark is synthetic. A passing gate means "not worse than the incumbent on this benchmark", not "good enough for the factory".
