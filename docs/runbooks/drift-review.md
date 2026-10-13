# Runbook: drift review

Alert `InputDriftDetected` / open items in `GET /api/v1/monitoring/recommendations?status=open`.
**Drift means input distributions changed. It does not mean the model is inaccurate, and nothing is deployed automatically.**
1. `GET /api/v1/monitoring/drift?status=alert`: flagged features with `psi`, `p_value`, reference vs current mean; `sample_count` and window times.
2. Check the quality block: if `measured: true`, precision/recall on labelled samples show whether performance actually moved; if `false`, quality is unmeasured - get labels before concluding.
3. Rule out data/sensor issues (calibration, a stuck or replaced sensor, a process change) before blaming the model. Compare per-device behaviour in Grafana.
4. Decide: acknowledge (`POST .../resolve {"action":"acknowledge"}`) and retrain/evaluate a candidate with `POST /models/train`, then roll it out through the normal canary + health gates; or dismiss with an audit trail.
Thresholds: `FLEETOPS_DRIFT_PSI_WARN` (0.10), `..._ALERT` (0.25), `..._P_VALUE` (0.01), `..._MIN_SAMPLES` (100), `..._WINDOW_SAMPLES` (500).
