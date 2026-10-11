# Forward predictor observations

The primary launcher defaults `JBR_CAPTURE_PRODUCTION_SCORES=true`. When its
ordinary freshness machinery actually invokes `ranker_predict`, the predictor
fingerprints the already-loaded model, scaler, ordered features, target, scored
column and posthoc calibrator before and after scoring. A changed bundle or a
failed required capture returns 2 before publishing predictions. This does not
train, load an additional model, change probabilities or promote a champion.
Explicit existing environment values are retained; ordinary predictor callers
without this environment flag retain their prior behavior.

Each invocation preserves only the current features snapshot's session, under
`data/production_score_history/<random-id>/`. It writes normalized/imputed input
features, matching symbol/timestamp probabilities and a manifest with exact saved
byte counts and SHA-256 values. Files use exclusive creation, fsync and complete
readback. Missing, duplicate, mismatched or invalid current-session rows fail
before publication. Historical rescoring rows are excluded. No outcomes or old
scores are retroactively bound. Reusing existing predictions creates no new
observation; only an actual predictor invocation can create a packet.

Prediction metadata references the immutable manifest. The bundle digest uses
protocol-4 pickle serialization of trusted in-memory objects; capture never
unpickles input and does not retain new model pickle files. Serialization can
vary by runtime. Equality identifies a serialized bundle, not mathematical model
equivalence or independent attestation of its deployment approval.

V2 observations also bind the predictor, capture and primary producer source
bytes before and after scoring. A source change prevents prediction publication.
Existing v1 captures remain unchanged. The [qualification consumer](production_evidence_qualification.md)
validates reviewed v2 evidence and joins it to later, mature exchange-session
outcomes; a saved monitor candidate cannot grant publication or acceptance.

Each observation has `outcomes_attached=false`, `research_qualified=false` and
`operating_acceptance=false`. It is evidence for later review, not an acceptance
token. Full acceptance still needs independently reviewed model identity and
evaluation populations, a reproducible join to subsequently observed mature
outcomes, sufficient comparable baseline/recent coverage, and a healthy monitor
followed by a new successful primary/postflight completion. Existing warnings,
seven-day limits and executor completion requirements remain in force. There is
no fixed promise that collecting a certain number of days clears a warning.

For canonical session preparation, both pipeline label-generation paths now
pass the signal session as `--run-date`. Label generation publishes under that
date instead of the wall-clock weekend date. Standalone callers without the
argument retain their New York clock default. Existing saved artifacts are not
renamed or deleted. This fixes identity of newly generated artifacts; it does
not manufacture newer outcome coverage.

Offline verification: `python -B -m unittest tests.test_acceptance_evidence
tests.test_ml_health_provenance tests.test_pipeline_postflight
tests.test_executor_preflight_health tests.test_ranker_monitor_contract`.
Tests use temporary synthetic inputs and forbid external execution. They do not
establish host integration, outcome causality, calibration or profitability.
