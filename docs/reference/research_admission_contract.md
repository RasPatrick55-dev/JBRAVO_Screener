# Research admission declaration boundary

Original work item: `JBRAVO_RESEARCH_ADMISSION_BOUNDARY_001`, source base
`c41016344f3a86925d590f0e43f6c8bc314f1d1d`. Narrow correction:
`JBRAVO_RESEARCH_ADMISSION_BOUNDARY_CORRECTION_001`, current source base
`3939fd951c7f346703fa054d814fbcf4bf565cd9`. Snapshot-retention continuation:
`JBRAVO_RESEARCH_ADMISSION_SNAPSHOT_RETENTION_001`, reconciled source base
`d67d36760e7a87a1ce7fa07a3d00a1ab86286125`. Historical strategy/foundation
provenance remains bound to its named original variant, not automatically
refreshed by current-base operational changes. Python API:
`scripts.research_admission.validate_specification(metadata)`.
This standard-library module checks caller-supplied metadata in memory. It has
no CLI, file/URI resolution, dataset-body hashing, credential/environment discovery,
logging setup, SDK, database, evaluator, model loading or operational imports.
It does not change PostgreSQL's authority or any collector/simulator/monitor.

## Decisions and immutable outputs

`AdmissionDecision` and `Reason` are frozen dataclasses. The detached snapshot
uses `FrozenMetadata.entries` tuples for objects and tuples for arrays. Scalar
values retain their supplied types and values. `decision.to_dict()` makes a fresh
mutable copy; modifying the submission or that copy cannot alter the decision.
There is no registry, timestamp generation, default input, repair or supersession.

`AdmissionDecision.specification` supports frozen objects, tuples for arrays,
and built-in str/int/float/bool/None scalars. Its frozen Boolean
`specification_retained` records presence independently of value or truthiness.
Direct construction of `AdmissionDecision` now requires that keyword-only flag;
the validator sets it explicitly and its public call signature is unchanged.
`to_dict()` keeps the same output keys. Safely inspected JSON null is reported as
`specification=null, specification_retained=true`; unsafe/oversized omission is
`specification=null, specification_retained=false`. False, zero, empty strings
and empty lists retain their exact types and values with presence true.

| Outcome | Meaning |
| --- | --- |
| `INVALID_SPEC` | Schema, type, value or traversal/size constraint failed. |
| `HOLD` | Parseable declarations have missing, unresolved, negative or conflicting evidence. |
| `STRUCTURALLY_READY_FOR_EVIDENCE_REVIEW` | Complete, internally consistent declarations only, awaiting independent evidence review. |

Every outcome has `research_qualified=false`, `execution_authorized=false` and
`acceptance_declarations_verified=false`. There is no replay-ready field. An
input `true`, hash-shaped string, named reviewer, acceptance reference or commit
does not prove a fact, reviewer independence, data acceptance, strategy parity,
deployed state or permission. A structurally complete `SYNTHETIC` specimen stays
synthetic; even a structurally complete `HISTORICAL` submission is unqualified.

Invalid reasons take precedence over HOLD reasons. All safely discoverable
schema/evidence reasons are retained, deduplicated only when all reason fields
are identical, and sorted by `(field, code, evidence_id, invalid)`. Every reason
contains its metadata path and, for evidence-specific blockers, its artifact ID.
Unsafe traversal stops immediately; these decisions have no snapshot and may
omit blockers below the stopped boundary. Finite malformed metadata is retained
without repair. Nonfinite floats are retained for diagnostics, with
`metadata_bytes=null`; that invalid snapshot cannot be serialized as strict JSON.

After safe traversal and aggregate-byte accounting, a non-object root is
`INVALID_SPEC` with root `TYPE` at `$`, specimen kind `UNKNOWN`, and all three
authority/qualification flags false. Its exact frozen scalar/list value is
retained without repair. Under-budget NaN, Infinity and -Infinity roots retain
diagnostic snapshots with both root `NONFINITE` and `TYPE`, and null byte counts.
Unsafe/oversized roots stop before freezing and omit snapshots; reasons below
the stopped boundary need not be discovered. All traversal and aggregate limits
remain unchanged. Snapshot retention supplies no schema or evidence acceptance.

## Exact v1 schema

Root keys are exactly `schema`, `specimen_kind`, `strategy`, `experiment`,
`inputs`, `acceptance_declarations`. Schema is `jbravo.research-admission.v1`;
specimen kind is `SYNTHETIC` or `HISTORICAL`. Objects below also reject unknown
or missing keys. This is a small declaration projection, **not** an arbitrary
upstream manifest reader. An upstream projection must explicitly retain the
recognized qualifiers; unknown manifest fields have not been validated.

Strings must be nonempty after whitespace inspection, but are never stripped
or rewritten. Reference/identity strings containing `latest` (case insensitive)
are rejected; immutable IDs and exact pins must replace dynamic selection.
Commits are exactly 40 hexadecimal characters; artifact SHA-256 values are
exactly 64. The module checks syntax without measuring bytes or consulting Git.
All schema dates are canonical `YYYY-MM-DD`, real Gregorian dates. Intervals
are `{start, end}`, **both inclusive**, with start <= end. Upstream security-status
intervals are start-inclusive/end-exclusive: a caller must explicitly translate
their evidence scope into this schema without claiming acceptance from that
translation. `knowledge_time` is exactly second-resolution UTC
`YYYY-MM-DDTHH:MM:SSZ`; it records a declaration, not proven causal availability.

### Strategy

Exactly these keys:

| Keys | Type and meaning |
| --- | --- |
| `id`, `version`, `repository`, `source_commit` | Nonempty identity/repository references and exact commit. |
| `variant` | `screener_v2`, `legacy_portfolio_simulator`, or `ml_forward_return_proxy`. No inferred equivalence. |
| `bindings` | Object with `strategy_source`, `strategy_config`, `parameters`: artifact IDs or explicit null blockers. |
| `parameters` | Nonempty array of `{name, value, unit}`; unique names, finite numeric values in [-1e12, 1e12], nonempty explicit units. |
| `feature_policy`, `warmup_policy` | Explicit nonempty feature construction and fresh/interruptible warm-up policies. |
| `entry_rules`, `exit_rules`, `sizing_rules`, `capital_rules` | Explicit nonempty rule declarations. |
| `signal_availability`, `execution_timing` | Separate completed-information and proposed execution declarations. |
| `fill_policy`, `stop_policy`, `expiry_policy`, `missing_session_policy`, `terminal_position_policy` | Explicit nonempty policies; no invented fallback fills or liquidation. |

Policy text is retained without interpreting or implementing strategy logic.
Whole-share rounding, gates/ranking, costs, exit precedence and timing must be
declared by the submitter in these pinned policies/configurations. This boundary
does not measure their correctness or enforce the simulator's causal contract.

### Experiment

Exactly `id`, `candidate`, `baseline`, `bindings`, `study`, `splits`,
`prior_exposure`, `warmup`, `benchmark`, `accounting`, `costs`, `cost_stress`,
`cost_basis`, `limits`, `decision_rules`.

- `id` is a nonempty immutable reference. Candidate and baseline each contain
  exactly `strategy_id`, `version`, `repository`, `source_commit`, `variant`.
  Candidate identity must match the strategy fields. Baseline remains separately
  declared and pinned through its artifact; baseline correctness is unverified.
- `bindings` has exactly `experiment_source`, `experiment_config`, `dependencies`,
  `ordering`, `baseline`, each an artifact ID or explicit null blocker.
- `study` and `warmup` are inclusive intervals; warm-up must finish before study
  begins. Input scope must cover the whole warm-up and study.
- `splits` has exactly `train`, `validation`, `test`. Each has `{interval,
  not_used_reason}`. An unused split requires null interval and a nonempty reason.
  A used split requires a valid interval and null reason; used splits are ordered,
  disjoint and within study. Coverage of every study date is not inferred.
- `prior_exposure` is `{status, references, note}`. Status is `KNOWN` or `UNKNOWN`,
  references is an array of immutable references, and note is nonempty. UNKNOWN
  or no references blocks. KNOWN is still an unverified declaration; it is not
  proof that a period was unseen. Duplicate exposure references are invalid.
- `benchmark` and `accounting` are explicit nonempty conventions, including units,
  dividends, gross/net/open costs and terminal marks where applicable. Their
  evidence is carried by pinned experiment configuration/source declarations.
- `costs` and `cost_stress` each contain `fee` and `slippage`; each cost contains
  exactly `{value, unit}`. Fee is finite in [0, 1e12] with unit
  `CURRENCY_PER_FILL` (currency in accounting). Slippage is finite in
  [0, 0.999999999999] with unit `FRACTION_PER_SIDE`. No Boolean-as-number,
  string/null, percentage conversion or implied units. Stress must have at least
  one nonzero cost. All-zero primary costs are a HOLD `ZERO_COST_REFERENCE`.
- `cost_basis` is `CALIBRATION_DECLARED` or `UNCALIBRATED_REFERENCE`; the latter
  blocks. The former does not verify calibration. Honest zero-cost legacy
  references cannot substitute for a calibrated net evaluation.
- `limits` is `{trials, runs, seconds, include_failures}`: integer trials/runs
  in [1, 10000], integer seconds in [1, 86400], include_failures exactly true.
  These are proposed experiment limits, **not** permission or this packet's
  test allowance. Boolean numeric values are invalid.
- `decision_rules` is a nonempty array of `{reference, disposition}` with an
  immutable reference and `PROPOSED` or `ADOPTED`. No performance thresholds are
  adopted or evaluated by this code, irrespective of the declaration's label.
  Duplicate rule references are invalid; inconsistent dispositions also block.

### Inputs, populations and artifact pins

Exactly `source_class`, `provider`, `feed`, `execution_adjustment`,
`feature_adjustment`, `execution_units`, `feature_units`, `knowledge_time`,
`security_ids`, `interval`, `universe_scope`, `selected_symbol_limitation`,
`artifacts`, `bindings`, `ml_lineage`, and the seven qualifier keys below.

Source class is one of `relational_daily_bars_snapshot`, `ml_artifact_set`,
`selected_symbol_capture`. Relational PostgreSQL `daily_bars` and ML artifact
populations are distinct; equal names/hashes do not merge them. ML proxy requires
an ML artifact population. Other variants remain named separately and do not
inherit behavior from a population name.

Provider/feed, execution versus feature adjustment semantics and units must be
nonempty explicit strings. Literal `UNKNOWN` is a blocker. Adjustment text is
not interpreted: no no-action, stitching, exclusion, dividend, ratio or share
conversion fallback is executed. Security IDs are a nonempty unique array of
immutable references; names/aliases/internal comparison keys are not verified
external identifiers.

`universe_scope` is `HISTORICAL_UNIVERSE` or `SELECTED_SYMBOLS`. Selected symbols
require a nonempty `selected_symbol_limitation`; historical universe requires
that field null. Selected capture requires SELECTED_SYMBOLS. The limitation is
retained and does not resolve independent action/identity/session/exposure gaps.
Historical membership is an asserted evidence role, not established by this enum.

`bindings` has exactly `bars`, `security`, `calendar`, `coverage`, `quality`,
`actions`, `eligibility`, `universe`; values are matching-role artifact IDs or
explicit null blockers. An empty artifact array is permitted but cannot resolve
these required bindings. No transport/pagination success, empty action inventory,
zero listed quality flags, match or alias can supply these missing declarations.

Each artifact has exactly:

```text
id, role, sha256, bytes, producer_id, request_id, run_id,
source_class, security_ids, interval, upstream_qualifiers
```

IDs and roles are unique across artifacts. Roles are the required bindings above,
the strategy/experiment binding keys, or ML `features`, `labels`, `predictions`.
Byte size is an integer in [0, 2^63-1], including explicit empty-body pins.
Producer/request/run IDs are nonempty immutable references; their claimed
immutability and artifact existence remain unverified. Every artifact must have
exactly the declared input security set and interval; this deliberately
conservative v1 does not join partial scopes or accept wider/overlapping evidence.
Data roles (`bars`, `features`, `labels`, `predictions`) must use the input source
class. ML feature/label/prediction roles require an ML population. Every other
non-metadata artifact must also use the input source class. Other evidence can
use `declaration_metadata` for a pinned declaration
projection; it never becomes a dataset population by being bound to this spec.

ML `ml_lineage` is exactly `{features, labels, predictions}` with three distinct
matching-role artifact IDs from one declared run. It explicitly links a single
feature/label/prediction set; other source classes require null lineage. The
module does not inspect feature vectors, label horizons, models or predictions,
or verify producer/run relationships.

The seven required qualifier keys are:

```text
qualified_for_research, research_qualified, action_coverage_complete,
executable, available_for_warmup, eligibility, status
```

The first five accept only true, false or null. False/null block and remain in
the snapshot. Eligibility is `DECLARED`, `UNKNOWN` or `INELIGIBLE`. Status is
`DECLARED`, `UNRESOLVED`, `HALTED` or `RESUMPTION_DOCUMENTED`; all except DECLARED
block. DECLARED means an unverified claim, never actual tradability. These
strings reuse the inspected security-status vocabulary without importing that
module. Every non-metadata artifact carries the exact qualifier object as
`upstream_qualifiers`; its blockers are independently retained and disagreements
with the input summary block. Metadata artifacts require null upstream qualifiers.
Missing keys or positive strings cannot hide false/null status. Current capture
and status outputs explicitly remain unqualified and will produce blockers.

### Acceptance declarations

Array (possibly empty) of objects with exactly `id`, `declarant`, `what`,
`evidence_id`, `decision_reference`, `security_ids`, `interval`. Identity,
declarant, evidence ID and decision references must be immutable nonempty strings;
what is a nonempty claim description. Every artifact needs a declaration, with
exactly its input security/interval scope. Declaration IDs must be unique;
multiple declarations for one artifact block rather than supersede one another.
Unbound evidence and different scope block. The module does not verify reference
contents, signatures, declarant authority/independence or approval truth. It does
not issue self-signed acceptance. No reference is opened or hashed.

## Input bounds and stable reasons

Only exact built-in dict/list/str/int/float/bool/None are accepted, not subclasses,
custom mappings, tuples, Decimal, datetime or dataset objects. Integer bit length
must be <=63; keys must have <=256 characters. Root depth is zero, keys and
values increase depth by one, with depth <=12. At most 3,000 metadata records:
each visited container/scalar **and each dictionary key** counts once. A shared
acyclic object is traversed separately at each reference; ancestor cycles stop.
Containers longer than 3,000 and individual strings longer than 512 KiB stop
before sorting/encoding their contents. No attacker-defined object methods run.

All metadata, including malformed/nonfinite submissions, must fit an aggregate
budget <=524,288 bytes. Finite metadata uses deterministic strict JSON:
`sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False`, UTF-8.
For budget accounting only, the encoder permits the literal ASCII tokens `NaN`,
`Infinity`, `-Infinity` (3, 8, 9 bytes). This is not a valid strict-JSON
representation, a repaired scalar, or factual acceptance. `metadata_bytes` stays
null when any nonfinite scalar was found; the internal aggregate budget still
applies. Finite inputs retain the exact same bytes and limit as strict JSON.
ASCII escaping makes encoded character count equal byte count. Bounded inspection
first discovers nonfinite reasons wherever traversal remains safe. Incremental
budget encoding then runs unconditionally, stopping on overflow **before**
semantic validation or freezing/copying the specification. Oversized nonfinite
submissions retain both NONFINITE and BYTE_LIMIT when inspection could safely
discover them, omit the full specification and retain deterministic partial
diagnostics. Earlier type/depth/record stops may prevent discovering later fields.
For finite overflow, `metadata_bytes` is the partial count at the stop, not a
measurement of a complete representation. Under-limit nonfinite inputs remain
invalid; their bounded snapshots retain the original invalid scalar for diagnostics.
No input values or references are changed, and nonfinite values never become
valid costs/budgets. Individual values remain bounded before aggregate encoding.

Stable invalid codes: `TYPE`, `VALUE`, `MISSING_FIELD`, `UNKNOWN_FIELD`,
`UNSUPPORTED_VALUE`, `EMPTY`, `NON_JSON_TYPE`, `NON_STRING_KEY`, `KEY_LIMIT`,
`INTEGER_LIMIT`, `NONFINITE`, `NUMBER`, `HASH`, `DATE`, `LATEST_BINDING`,
`DUPLICATE_ID`, `DUPLICATE_ROLE`, `REVERSED_INTERVAL`, `INCOMPATIBLE_INTERVAL`,
`ZERO_COST_STRESS`, `FAILURES_NOT_BUDGETED`, `RECORD_LIMIT`, `DEPTH_LIMIT`,
`BYTE_LIMIT`, `CYCLE`.

Stable HOLD codes: `MISSING_EVIDENCE`, `UNBOUND_EVIDENCE`, `ROLE_CONFLICT`,
`UNKNOWN_EVIDENCE`, `UNKNOWN_QUALIFIER`, `NEGATIVE_QUALIFIER`,
`UNRESOLVED_ELIGIBILITY`, `UNRESOLVED_STATUS`, `SCOPE_CONFLICT`,
`POPULATION_CONFLICT`, `IDENTITY_CONFLICT`, `QUALIFIER_CONFLICT`,
`LINEAGE_CONFLICT`, `CONFLICTING_DECLARATION`, `PRIOR_EXPOSURE_UNRESOLVED`,
`UNCALIBRATED_COSTS`, `ZERO_COST_REFERENCE`, `MISSING_ACCEPTANCE_DECLARATION`.

## Examples and verification boundary

The complete inline `specimen()` in `tests/test_research_admission.py` is an
executable schema example with fictional artifact pins, scopes, policies and
review declarations. It is not a production template or accepted historical data.
For an already supplied in-memory dictionary:

```python
from scripts.research_admission import validate_specification

decision = validate_specification(metadata)
assert decision.research_qualified is False
assert decision.execution_authorized is False
review_copy = decision.to_dict()  # No output file is created.
```

Setting `inputs.executable=None` yields HOLD (and conflicting artifact assertions
when present), retaining null. Setting an artifact upstream research qualifier
false yields HOLD even if the summary is true. Setting an action binding null,
leaving prior exposure UNKNOWN or using uncalibrated/zero primary costs blocks.
A malformed SHA-256 adds INVALID_SPEC while preserving discoverable HOLD reasons.
Fully consistent fictional declarations exercise only the structural outcome.

Run only the 52 scoped synthetic cases, with plugin autoload and bytecode disabled:

```text
python -B -m pytest --noconftest -p no:cacheprovider -o addopts= tests/test_research_admission.py -q
```

The original allocation stays exhausted at 2/2 invocations. The historical
aggregate-size correction remains 1/2 used, with 36/36 passed; its unused attempt
is not transferred. Its separate ceiling was two invocations, at most 36 cases
each, 60 seconds each and 120 cumulative controller seconds. Its four added
cases cover NaN/positive/negative infinity combined with two individually
under-limit but aggregate-oversized strings, bounded under-limit nonfinite
diagnostics, and finite exact-byte equality/one-byte overflow. The existing 32
cases and their effect guards remain, as do all 36 accepted aggregate-correction
cases. Snapshot retention adds 16 cases: eight finite root shapes, three
nonfinite roots, four oversized/unsafe root boundaries, and detached immutable
root snapshots. New validator/output-copy calls use the same effect guards.
This snapshot task has its own unchanged maximum of two pytest invocations,
52 collected cases each, externally capped at 60 seconds each and 120 cumulative
controller seconds. The retained old-base preflight stop used zero invocations,
cases and controller seconds; reconciliation continues this allocation without
a reset. Stop after the first complete success; attempt two needs an in-scope
fix or unresolved failure. No collection-only run or ad hoc probe is allowed.
Original, aggregate-correction and snapshot attempts remain separately retained.
The test fixture installs
socket/process/write/mkdir/logging/environment/import guards before normal
module import and every validator/output-copy call, then restores boundaries
between operations so pytest bookkeeping can run. Standard-library and source
reads are allowed. Deliberate prohibited calls only demonstrate guard interception;
they do not dispatch effects. No existing collector, status, simulator or bot
module is imported. Repository conftest, plugins and cache writes are disabled.

Synthetic tests establish bounded declaration behavior under mocked effect
boundaries; they do not prove live integration, independent evidence authenticity,
historical universe/prior exposure, cost calibration, evaluator correctness,
replay readiness, profitability or an operational department. Independently
accepted action/calendar/security/data evidence and a separately reviewed inert
evaluator adapter remain the next dependencies. Publication, runtime/host/paper
integration and trading require later authorization.
