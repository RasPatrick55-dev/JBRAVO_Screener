"""Read-only whole-signal research projection. No scanner or trading imports.

PostgreSQL identifies immutable daily archives. Only their hash-bound bytes from
the PythonAnywhere files API can supply outcomes. Candidate/exit-fill summaries
are never substituted for missing lifecycle evidence.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_GROUPS = 24
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_SPEC_BYTES = 2 * 1024 * 1024
MAX_LEDGER_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SIGNALS = 20000
MAX_FILLS = 100000
CACHE_SECONDS = 900
FAILURE_CACHE_SECONDS = 60
PROVIDER_INTERVAL_SECONDS = .5
REQUEST_SECONDS = 25
HEX = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
METRICS = ("closed_signals", "open_signals", "unfilled_signals", "wins", "losses",
           "breakeven", "win_rate_pct", "net_pnl", "expectancy_net_per_signal",
           "average_win", "average_loss", "payoff_ratio", "profit_factor")
LIMITATIONS = [
    "Research simulations, not actual paper-account performance or a success forecast.",
    "Each row covers its declared historical portfolio episode, not every qualifying setup.",
    "Distinct strategies and incompatible episodes are not combined into one portfolio.",
    "Closed-position P&L includes modeled fees/slippage and excludes cash dividends.",
    "Retrospective universe and historical availability assumptions remain unqualified.",
    "Week-cluster intervals describe this sample; they do not establish future performance.",
    "Few signal-week clusters give unstable intervals; grouping does not remove cross-week holding or regime dependence.",
    "Latest completed archives are historical evidence; newer scan attempts are not evaluated here.",
]

# Select only metadata first. No oversized JSON payload crosses the connection.
# Different episode/cost/universe contracts are intentionally separate rows.
ARCHIVES_SQL = """
WITH ranked AS (
 SELECT run_id, recorded_at, octet_length(result::text) AS payload_bytes,
 row_number() OVER (
  PARTITION BY result->>'strategy_id',
   result#>>'{summary,strategies,0,strategy,version}',
   result#>>'{summary,strategies,0,strategy,definition_sha256}',
   result#>>'{summary,strategies,0,episode_id}'
  ORDER BY result->>'signal_session' DESC, recorded_at DESC, run_id DESC
 ) AS ordinal
 FROM jbravo_research.research_runs
 WHERE result->>'schema'='jbravo.daily-strategy-scan.v1'
 AND result->>'status'='COMPLETED_PAPER_SCREEN'
)
SELECT run_id, payload_bytes FROM ranked WHERE ordinal=1
ORDER BY recorded_at DESC, run_id DESC LIMIT %s
"""
ARCHIVE_SQL = ("SELECT specification_sha256,result_sha256,result "
               "FROM jbravo_research.research_runs WHERE run_id=%s "
               "AND octet_length(result::text)<=%s")


class EvidenceError(ValueError):
    """Only fixed non-secret reason codes may reach the public API."""


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def invalid(_):
        raise EvidenceError("nonfinite_json")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "invalid_number")
    return float(value)


def day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            "invalid_session")
    return value


def reason(error):
    if isinstance(error, EvidenceError) and re.fullmatch(r"[a-z_]{1,70}", str(error)):
        return str(error)
    return "evidence_unavailable"


def archive_metadata(record, *, now):
    """Validate persisted identities without loading the live strategy catalog."""
    spec_sha, result_sha, value = record
    require(isinstance(value, dict) and digest(value) == result_sha.strip(),
            "archive_hash_mismatch")
    require(HEX.fullmatch(spec_sha.strip()) is not None, "specification_hash_invalid")
    require(value.get("schema") == "jbravo.daily-strategy-scan.v1" and
            value.get("status") == "COMPLETED_PAPER_SCREEN" and
            value.get("pipeline") == "pullback-reclaim-daily-paper-v1" and
            all(value.get(k) is False for k in
                ("research_qualified", "execution_authorized", "order_submission_authorized")),
            "archive_scope_invalid")
    session = day(value["signal_session"])
    require(session <= now.date().isoformat(), "future_archive")
    sid = value["strategy_id"]
    require(isinstance(sid, str) and IDENTIFIER.fullmatch(sid), "strategy_identity_invalid")
    require(value["run_id"] == digest(dict(schema="jbravo.daily-strategy-archive-id.v1",
            pipeline=value["pipeline"], strategy_id=sid, signal_session=session)),
            "archive_identity_mismatch")
    summary, handoff = value["summary"], value["handoff"]
    require(summary.get("schema") == "jbravo.strategy-scan.v1" and
            summary.get("status") == "COMPLETED_PAPER_SIMULATION" and
            summary.get("broker_operations") == 0 and summary.get("as_of") == session and
            summary.get("run_id") == value["scan_run_id"] and
            all(summary.get(k) is False for k in ("research_qualified", "order_submission_authorized")),
            "scan_scope_invalid")
    require(handoff.get("schema") == "jbravo.strategy-paper-proposals.v1" and
            handoff.get("scan_run_id") == value["scan_run_id"] and handoff.get("as_of") == session and
            handoff.get("paper_only") is True and handoff.get("order_submission_authorized") is False and
            handoff.get("proposals") == summary.get("top_candidates"), "handoff_mismatch")
    require(isinstance(summary["strategies"], list) and len(summary["strategies"]) == 1,
            "ambiguous_strategy_archive")
    item = summary["strategies"][0]
    strategy = item["strategy"]
    require(strategy["id"] == sid and item["as_of"] == session and
            isinstance(strategy["version"], str) and 0 < len(strategy["version"]) <= 32 and
            HEX.fullmatch(strategy["definition_sha256"]) and HEX.fullmatch(item["episode_id"]) and
            HEX.fullmatch(value["scan_manifest_sha256"]) and HEX.fullmatch(summary["dataset_sha256"]),
            "strategy_identity_invalid")
    # Historical definitions are evidence, not the current installed code's definition.
    definition = {k: v for k, v in strategy.items() if k not in
                  ("definition_sha256", "research_qualified")}
    definition["source_sha256"] = {k: v for k, v in definition["source_sha256"].items()
                                   if k != "strategy_library.py"}
    require(digest(definition) == strategy["definition_sha256"], "definition_hash_mismatch")
    return dict(strategy_id=sid, strategy_name=strategy.get("name", sid),
                strategy_version=strategy["version"], definition_sha256=strategy["definition_sha256"],
                episode_id=item["episode_id"], as_of=session)


def empty_row(identity):
    return dict(identity, **dict.fromkeys(METRICS),
        metric_basis="STRATEGY_RESEARCH_SIMULATION",
        scope="DECLARED_EPISODE_SIMULATED_PORTFOLIO_SIGNALS", episode=None,
        evidence_status="UNAVAILABLE", sample_status="UNKNOWN", profit_factor_status="UNKNOWN",
        costs=dict(basis="MODELED", fees=None, slippage=None, total=None, actual_costs_available=False),
        uncertainty=dict(method="SIGNAL_WEEK_CLUSTER_BOOTSTRAP", level=.95, low=None, high=None,
                         signal_weeks=None, samples=0, status="UNAVAILABLE", warning=None),
        evidence=dict(archive_count=1, distinct_signals=None, duplicate_observations=0),
        research_qualified=False, order_submission_authorized=False)


def trace_outcome(trace, identity):
    """Reconcile one whole position, including all partial exits and entry costs."""
    sid = trace["signal_id"]
    signal_day = day(trace["signal_session"])
    require(signal_day <= identity["as_of"] and
            trace["strategy_id"] == identity["strategy_id"] and
            trace["strategy_version"] == identity["strategy_version"] and
            re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,11}", trace["symbol"]) and
            sid == digest(dict(strategy_id=identity["strategy_id"], version=identity["strategy_version"],
                episode_id=identity["episode_id"], symbol=trace["symbol"], signal_session=signal_day)),
            "signal_identity_mismatch")
    fills = trace["fills"]
    require(isinstance(fills, list) and len(fills) <= 100, "fill_count_limit")
    bought = sold = 0
    previous = signal_day
    net = cash = fees = slippage = 0.0
    entry_price = None
    gross = 0.0
    for fill in fills:
        require(all(fill[k] == trace[k] for k in ("signal_id", "symbol", "strategy_id", "strategy_version")),
                "fill_lineage_mismatch")
        session = day(fill["session"])
        require(signal_day < session <= identity["as_of"] and previous <= session,
                "fill_session_invalid")
        previous = session
        quantity = fill["quantity"]
        require(type(quantity) is int and 0 < quantity <= 100000000, "fill_quantity_invalid")
        price, fee, slip = (number(fill[k]) for k in ("raw_price", "fee", "slippage"))
        # Research arithmetic can leave a tiny negative zero slippage residual.
        require(price > 0 and fee >= 0 and slip >= -1e-8, "fill_cost_invalid")
        delta = number(fill["cash_delta"])
        if fill["side"] == "BUY":
            require(not bought and not sold, "multiple_entry_unsupported")
            bought, entry_price = quantity, price
            expected_delta = -quantity * price - fee - slip
        else:
            require(fill["side"] == "SELL" and bought and sold + quantity <= bought,
                    "exit_quantity_invalid")
            sold += quantity
            net += number(fill["net_pnl"])
            gross += quantity * (price - entry_price)
            expected_delta = quantity * price - fee - slip
        require(math.isclose(delta, expected_delta, rel_tol=1e-10, abs_tol=1e-6),
                "fill_cash_mismatch")
        cash += delta
        fees += fee
        slippage += slip
    state = "CLOSED" if bought and bought == sold else "OPEN" if bought else "UNFILLED_PROPOSAL"
    require(type(trace["bought_quantity"]) is int and type(trace["sold_quantity"]) is int and
            trace["bought_quantity"] == bought and trace["sold_quantity"] == sold and
            trace["status"] == state, "signal_state_mismatch")
    if state == "CLOSED":
        require(math.isclose(net, cash, rel_tol=1e-10, abs_tol=1e-6) and
                math.isclose(net, gross-fees-slippage, rel_tol=1e-10, abs_tol=1e-6),
                "closed_signal_cost_mismatch")
    return dict(signal_id=sid, signal_session=signal_day, state=state,
                net_pnl=net if state == "CLOSED" else None, fees=fees, slippage=slippage)


def interval(trades, *, samples=400):
    weeks = defaultdict(list)
    for trade in trades:
        stamp = date.fromisoformat(trade["signal_session"])
        weeks[(stamp-timedelta(days=stamp.weekday())).isoformat()].append(trade["net_pnl"])
    result = dict(method="SIGNAL_WEEK_CLUSTER_BOOTSTRAP", level=.95, low=None, high=None,
                  signal_weeks=len(weeks), samples=0, status="INSUFFICIENT_SIGNAL_WEEKS",
                  warning="TOO_FEW_SIGNAL_WEEKS" if len(weeks) < 2 else
                          "FEW_SIGNAL_WEEK_CLUSTERS" if len(weeks) < 10 else None)
    if len(weeks) < 2:
        return result
    totals = [(math.fsum(weeks[k]), len(weeks[k])) for k in sorted(weeks)]
    rng = random.Random(271828)
    estimates = []
    for _ in range(samples):
        selected = rng.choices(totals, k=len(totals))
        estimates.append(math.fsum(t[0] for t in selected) / sum(t[1] for t in selected))
    estimates.sort()
    def percentile(p):
        index = (len(estimates)-1)*p
        left = int(index)
        return estimates[left] + (estimates[min(left+1, len(estimates)-1)]-estimates[left])*(index-left)
    return dict(result, low=percentile(.025), high=percentile(.975), samples=samples,
                status="DESCRIPTIVE_NOT_VALIDATION")


def aggregate(snapshots):
    """Deduplicate compatible full traces, not per-pick averages or exit fills.

    A later snapshot may append fills to an open position. Conflicting completed
    histories fail closed rather than selecting the more flattering outcome.
    Different strategy/definition/episode groups must be passed separately.
    """
    require(bool(snapshots), "missing_snapshots")
    snapshots = sorted(snapshots, key=lambda x: x["identity"]["as_of"])
    identity = snapshots[-1]["identity"]
    keys = ("strategy_id", "strategy_version", "definition_sha256", "episode_id")
    seen, duplicates, fill_count = {}, 0, 0
    for snapshot in snapshots:
        current = snapshot["identity"]
        require(all(current[k] == identity[k] for k in keys), "incompatible_episodes")
        rows = snapshot["lifecycle"]
        require(isinstance(rows, list) and len(rows) <= MAX_SIGNALS, "signal_count_limit")
        within = set()
        previously_filled = {sid for sid, (t, _) in seen.items() if t["fills"]}
        for trace in rows:
            outcome = trace_outcome(trace, current)
            sid = trace["signal_id"]
            require(sid not in within, "duplicate_signal_in_archive")
            within.add(sid)
            fill_count += len(trace["fills"])
            require(fill_count <= MAX_FILLS, "fill_count_limit")
            if sid in seen:
                duplicates += 1
                old, _ = seen[sid]
                require(trace["fills"][:len(old["fills"])] == old["fills"] and
                        (old["status"] != "CLOSED" or trace == old), "conflicting_signal_history")
            seen[sid] = (trace, outcome)
        require(previously_filled <= within, "conflicting_signal_history")
        require(len(seen) <= MAX_SIGNALS, "signal_count_limit")
    outcomes = [seen[s][1] for s in sorted(seen)]
    completed = [t for t in outcomes if t["state"] == "CLOSED"]
    nets = [t["net_pnl"] for t in completed]
    winners, losers = [v for v in nets if v > 0], [v for v in nets if v < 0]
    count = len(completed)
    gain, loss, total = math.fsum(winners), math.fsum(losers), math.fsum(nets)
    average_win = gain/len(winners) if winners else None
    average_loss = loss/len(losers) if losers else None
    fees, slippage = (math.fsum(t[k] for t in completed) for k in ("fees", "slippage"))
    return dict(closed_signals=count, open_signals=sum(t["state"] == "OPEN" for t in outcomes),
        unfilled_signals=sum(t["state"] == "UNFILLED_PROPOSAL" for t in outcomes),
        wins=len(winners), losses=len(losers), breakeven=count-len(winners)-len(losers),
        win_rate_pct=100*len(winners)/count if count else None, net_pnl=total,
        expectancy_net_per_signal=total/count if count else None,
        average_win=average_win, average_loss=average_loss,
        payoff_ratio=average_win/abs(average_loss) if winners and losers else None,
        profit_factor=gain/abs(loss) if losers else None,
        profit_factor_status="DEFINED" if losers else "UNDEFINED_NO_LOSSES" if count else "NO_CLOSED_SIGNALS",
        costs=dict(basis="MODELED", fees=fees, slippage=slippage, total=fees+slippage,
                   actual_costs_available=False), uncertainty=interval(completed),
        sample_status="OBSERVED_SAMPLE_NOT_QUALIFICATION" if count else "NO_CLOSED_SIGNALS",
        evidence_status="VERIFIED",
        evidence=dict(archive_count=len(snapshots), distinct_signals=len(seen),
                      duplicate_observations=duplicates))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EvidenceError("provider_redirect_refused")


class ProviderReader:
    """Existing PA credentials, fixed HTTPS files GET only; no retries/writes."""
    def __init__(self, *, env=None, deadline=None):
        env = os.environ if env is None else env
        self.token = env.get("PYTHONANYWHERE_API_TOKEN") or env.get("API_TOKEN")
        self.username = env.get("PYTHONANYWHERE_USERNAME") or env.get("PYTHONANYWHERE_USER")
        self.base = (env.get("PYTHONANYWHERE_API_BASE_URL") or
                     "https://www.pythonanywhere.com/api/v0/user").rstrip("/")
        require(self.base in {"https://www.pythonanywhere.com/api/v0/user",
                             "https://eu.pythonanywhere.com/api/v0/user"}, "provider_origin_refused")
        require(self.token and self.username and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.username),
                "provider_credentials_unavailable")
        self.deadline = deadline or time.monotonic()+REQUEST_SECONDS
        self.lock = threading.Lock()
        self.total = self.calls = 0
        self.next_call = 0.0

    def __call__(self, path, limit):
        require(isinstance(path, str) and path.startswith("/home/"+self.username+"/") and
                ".." not in PurePosixPath(path).parts and "\\" not in path,
                "provider_path_refused")
        with self.lock:
            self.calls += 1
            require(self.calls <= MAX_GROUPS*3 and time.monotonic() < self.deadline,
                    "provider_budget_exceeded")
            slot = max(time.monotonic(), self.next_call)
            self.next_call = slot+PROVIDER_INTERVAL_SECONDS
        delay = slot-time.monotonic()
        require(slot < self.deadline, "provider_budget_exceeded")
        if delay > 0:
            time.sleep(delay)
        require(time.monotonic() < self.deadline, "provider_budget_exceeded")
        url = self.base+"/"+self.username+"/files/path"+urllib.parse.quote(path, safe="/")
        req = urllib.request.Request(url, headers={"Authorization": "Token "+self.token,
                                                    "Accept-Encoding": "identity"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        timeout = max(.1, min(5, self.deadline-time.monotonic()))
        with opener.open(req, timeout=timeout) as response:
            require(response.status == 200 and response.headers.get("Content-Encoding", "identity")
                    in ("identity", ""), "provider_response_invalid")
            length = response.headers.get("Content-Length")
            require(length is None or int(length) <= limit, "artifact_size_limit")
            chunks, size = [], 0
            while True:
                require(time.monotonic() < self.deadline, "provider_budget_exceeded")
                # read1 performs at most one underlying read. A slow-drip body
                # cannot keep read(n) internally looping past the work deadline.
                part = response.read1(min(65536, limit-size+1))
                if not part:
                    break
                size += len(part)
                with self.lock:
                    self.total += len(part)
                    require(self.total <= MAX_TOTAL_BYTES, "provider_budget_exceeded")
                require(size <= limit, "artifact_size_limit")
                chunks.append(part)
            require(length is None or size == int(length), "artifact_size_mismatch")
            return b"".join(chunks)


def bound_lifecycle(record, identity, reader, producer_root, *, charge_expanded=None):
    """Only DB-manifest-bound provider bytes; no local artifact or CSV reads."""
    spec_sha, _, archive = record
    root = PurePosixPath(producer_root)
    require(root.is_absolute() and ".." not in root.parts and "\\" not in producer_root,
            "producer_root_invalid")
    primary = root/identity["as_of"]
    expected = primary if archive["strategy_id"] == "trend_pullback_reclaim_qualifying_close_v2" else (
        primary/"companions"/archive["strategy_id"])
    require(archive["output_root"] == str(expected), "archive_path_mismatch")
    folder = expected/"scan"
    raw = reader(str(folder/"manifest.json"), MAX_MANIFEST_BYTES)
    require(hashlib.sha256(raw).hexdigest() == archive["scan_manifest_sha256"], "manifest_hash_mismatch")
    manifest = strict_json(raw)
    bindings = manifest["bindings"]
    require(isinstance(bindings, list) and 1 <= len(bindings) <= 100, "manifest_count_invalid")
    pins = {}
    for pin in bindings:
        path = pin["path"]
        require(isinstance(path, str) and not PurePosixPath(path).is_absolute() and
                ".." not in PurePosixPath(path).parts and "\\" not in path and path not in pins and
                type(pin["bytes"]) is int and pin["bytes"] >= 0 and HEX.fullmatch(pin["sha256"]),
                "manifest_binding_invalid")
        pins[path] = pin
    require({"result.json", "paper_handoff.json", "specification.json"} <= pins.keys(),
            "manifest_incomplete")
    # save_json uses sorted two-space JSON plus a newline. Verify the DB's
    # summary/handoff match the manifest without spending two additional GETs.
    for name, value in (("result.json", archive["summary"]), ("paper_handoff.json", archive["handoff"])):
        expected_raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        require(pins[name]["bytes"] == len(expected_raw) and pins[name]["sha256"] == hashlib.sha256(expected_raw).hexdigest(),
                "manifest_archive_content_mismatch")
    def fetch(path, limit):
        pin = pins[path]
        require(pin["bytes"] <= limit, "artifact_size_limit")
        body = reader(str(folder/path), limit)
        require(len(body) == pin["bytes"] and hashlib.sha256(body).hexdigest() == pin["sha256"],
                "artifact_hash_mismatch")
        return body
    spec_raw = fetch("specification.json", MAX_SPEC_BYTES)
    require(hashlib.sha256(spec_raw).hexdigest() == spec_sha.strip(), "specification_hash_mismatch")
    spec = strict_json(spec_raw)
    summary = archive["summary"]
    described = summary["strategies"][0]["strategy"]
    folds = [f for f in spec["folds"] if day(f["test_start"]) <= identity["as_of"] <= day(f["test_end"])]
    require(len(folds) == 1 and len(spec["cost_cases"]) == 1 and
            spec["strategies"] == [described["research_method"]], "episode_specification_invalid")
    fold, cost = folds[0], spec["cost_cases"][0]
    # The canonical episode for market-wide scans intentionally binds universe
    # policy, not each day's changing membership. Exact coverage remains separate.
    coverage = summary.get("coverage")
    population = (dict(policy={k: coverage["policy"][k] for k in
        ("source", "status", "asset_class", "tradable")}, benchmark=spec["universe"]["benchmark"])
        if coverage else spec["universe"])
    episode = dict(strategy_id=identity["strategy_id"], version=identity["strategy_version"],
        fold_id=fold["id"], warmup_start=fold["warmup_start"], test_start=fold["test_start"],
        universe=population, portfolio=spec["portfolio"], cost_case=cost,
        definition_sha256=identity["definition_sha256"])
    require(digest(episode) == identity["episode_id"], "episode_hash_mismatch")
    require(digest(dict(spec=spec, as_of=identity["as_of"], sources=summary["source_sha256"],
        dataset_sha256=summary["dataset_sha256"], strategies=[identity["definition_sha256"]],
        cost_case=cost["id"])) == archive["scan_run_id"], "scan_identity_mismatch")
    names = [identity["strategy_id"]+"/lifecycle.json"+suffix for suffix in ("", ".gz")]
    names = [name for name in names if name in pins]
    require(len(names) == 1, "lifecycle_binding_unavailable")
    ledger = fetch(names[0], MAX_LEDGER_BYTES)
    if names[0].endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(ledger)) as stream:
            ledger = stream.read(MAX_EXPANDED_BYTES+1)
        require(len(ledger) <= MAX_EXPANDED_BYTES, "expanded_artifact_size_limit")
    if charge_expanded is not None:
        charge_expanded(len(ledger))
    traces = strict_json(ledger)
    require(isinstance(traces, list) and len(traces) <= MAX_SIGNALS, "signal_count_limit")
    symbols = set(spec["universe"]["symbols"])
    require(all(t["symbol"] in symbols and fold["test_start"] <= day(t["signal_session"]) <= identity["as_of"]
                for t in traces), "lifecycle_outside_declared_scope")
    return dict(identity=identity, lifecycle=traces), dict(test_start=fold["test_start"],
        as_of=identity["as_of"], universe_kind="DAILY_ACTIVE_TRADABLE_UNIVERSE" if coverage else "FIXED_COHORT",
        cost_case=cost, portfolio=spec["portfolio"], coverage=coverage,
        declared_symbols=len(spec["universe"]["symbols"]),
        historical_scope="SELECTED_SIMULATED_PORTFOLIO_SIGNALS_NOT_ALL_QUALIFYING_SETUPS")


def read_archives(connection, *, deadline=None):
    connection.set_session(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout='5000ms'")
        cursor.execute("SET LOCAL lock_timeout='1000ms'")
        cursor.execute(ARCHIVES_SQL, (MAX_GROUPS+1,))
        metadata = cursor.fetchall()
        records, rejected, total = [], [], 0
        for run_id, size in metadata[:MAX_GROUPS]:
            require(deadline is None or time.monotonic() < deadline, "database_budget_exceeded")
            if type(size) is not int or not 0 < size <= MAX_ARCHIVE_BYTES or total+size > MAX_TOTAL_BYTES:
                rejected.append("archive_size_limit")
                continue
            total += size
            cursor.execute(ARCHIVE_SQL, (run_id, MAX_ARCHIVE_BYTES))
            record = cursor.fetchone()
            if record is None or record[2].get("run_id") != run_id:
                rejected.append("archive_unavailable")
            else:
                records.append(record)
        return records, rejected, len(metadata) > MAX_GROUPS


def configuration(base_dir):
    """Read deployment configuration, never operational outcome artifacts."""
    def read(name):
        path = Path(base_dir)/"config"/name
        with path.open("rb") as stream:
            raw = stream.read(4097)
        require(len(raw) <= 4096, "configuration_size_limit")
        return strict_json(raw)
    cfg = read("strategy_pipeline.json")
    require(cfg.get("schema") == "jbravo.primary-strategy-route.v1" and cfg.get("enabled") is True and
            cfg.get("order_submission_authorized") is False, "configuration_invalid")
    enabled = {cfg["strategy_id"]}
    for name, schema in (("strategy_companions.json", "jbravo.strategy-companions.v1"),
                         ("chart_pattern_companions.json", "jbravo.additional-chart-patterns.v1")):
        if not (Path(base_dir)/"config"/name).exists():
            continue
        value = read(name)
        require(value.get("schema") == schema and type(value.get("enabled")) is bool,
                "configuration_invalid")
        if value["enabled"]:
            enabled.update(value.get("strategy_ids", [value.get("strategy_id")]))
    require(len(enabled) <= MAX_GROUPS and all(isinstance(sid, str) and IDENTIFIER.fullmatch(sid)
                                             for sid in enabled), "configuration_invalid")
    return cfg["producer_root"], enabled


def snapshot(*, base_dir, now=None, connection_factory=None, reader=None):
    now = now or datetime.now(timezone.utc)
    deadline = time.monotonic()+REQUEST_SECONDS
    result = dict(ok=False, status="UNAVAILABLE", source="postgres", rows=[], rows_final=0,
        artifact_source="pythonanywhere_api_hash_bound_to_postgres",
        metric_basis="STRATEGY_RESEARCH_SIMULATION", generated_at=now.isoformat(),
        cache_age_seconds=0, cache_ttl_seconds=CACHE_SECONDS, limitations=list(LIMITATIONS),
        coverage=dict(selection="LATEST_ARCHIVE_PER_STRATEGY_VERSION_DEFINITION_EPISODE",
                      max_groups=MAX_GROUPS, truncated=False, rejected_archives=0, reasons=[]),
        actual_paper=dict(status="UNAVAILABLE", net_metrics_available=False,
            reason="Complete actual paper cost accounting is not supplied by research archives."),
        research_qualified=False, order_submission_authorized=False)
    enabled, producer_root, config_error = set(), None, None
    try:
        producer_root, enabled = configuration(base_dir)
    except Exception as error:
        config_error = reason(error)
    connection, database_error = None, None
    try:
        if connection_factory is None:
            from scripts.research_store import connect
            connection_factory = connect
        connection = connection_factory()
        records, rejected, truncated = read_archives(connection, deadline=deadline)
    except Exception as error:
        records, rejected, truncated = [], [reason(error)], False
        database_error = reason(error)
    finally:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
    result["coverage"].update(truncated=truncated, rejected_archives=len(rejected), reasons=sorted(set(rejected)))
    accepted = []
    for record in records:
        try:
            accepted.append((record, archive_metadata(record, now=now)))
        except Exception as error:
            result["coverage"]["rejected_archives"] += 1
            result["coverage"]["reasons"].append(reason(error))
    provider_error = config_error
    try:
        if accepted and reader is None and provider_error is None:
            reader = ProviderReader(deadline=deadline)
    except Exception as error:
        provider_error = reason(error)
    expanded, budget_lock = [0], threading.Lock()
    def charge_expanded(size):
        with budget_lock:
            expanded[0] += size
            require(expanded[0] <= MAX_TOTAL_BYTES and time.monotonic() < deadline,
                    "expanded_artifact_budget_exceeded")
    def project(pair):
        record, identity = pair
        row = empty_row(identity)
        row["evidence"].update(archive_id=record[2]["run_id"], result_sha256=record[1].strip(),
            scan_run_id=record[2]["scan_run_id"], manifest_sha256=record[2]["scan_manifest_sha256"])
        try:
            require(provider_error is None, provider_error or "provider_unavailable")
            require(time.monotonic() < deadline, "provider_budget_exceeded")
            evidence, episode = bound_lifecycle(record, identity, reader, producer_root,
                                                charge_expanded=charge_expanded)
            metrics = aggregate([evidence])
            canonical(metrics)  # Never emit JSON NaN/Infinity after arithmetic overflow.
            metrics["evidence"] = dict(row["evidence"], **metrics["evidence"])
            row.update(metrics, episode=episode)
        except Exception as error:
            row["failure_code"] = reason(error)
            if row["failure_code"] == "conflicting_signal_history":
                row["evidence_status"] = "CONFLICT"
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(project, accepted))
    for sid in sorted(enabled-{r["strategy_id"] for r in rows}):
        row = empty_row(dict(strategy_id=sid, strategy_name=sid.replace("_", " "),
            strategy_version=None, definition_sha256=None, episode_id=None, as_of=None))
        row.update(failure_code=database_error or "completed_archive_unavailable")
        row["evidence"]["archive_count"] = 0
        rows.append(row)
    for row in rows:
        row["configured_enabled"] = row["strategy_id"] in enabled if config_error is None else None
        row["freshness"] = dict(basis="LATEST_COMPLETED_ARCHIVE_ONLY", evidence_as_of=row["as_of"],
            source_age_calendar_days=(now.date()-date.fromisoformat(row["as_of"])).days if row["as_of"] else None,
            latest_attempt_status="NOT_EVALUATED", latest_attempt_at=None, retained_historical_evidence=True)
    rows.sort(key=lambda r: tuple(r[k] or "" for k in
               ("strategy_id", "strategy_version", "definition_sha256", "episode_id")))
    verified = sum(r["evidence_status"] == "VERIFIED" for r in rows)
    incomplete = verified != len(rows) or result["coverage"]["rejected_archives"] or truncated or config_error
    status = ("PARTIAL" if incomplete else "COMPLETE") if verified else (
        "UNAVAILABLE" if rows or incomplete else "EMPTY")
    result.update(ok=status in {"COMPLETE", "PARTIAL", "EMPTY"}, status=status, rows=rows, rows_final=len(rows))
    result["cache_ttl_seconds"] = CACHE_SECONDS if verified else FAILURE_CACHE_SECONDS
    if database_error:
        result["failure_code"] = database_error
    result["coverage"]["configured_strategy_count"] = len(enabled) if config_error is None else None
    if config_error:
        result["coverage"]["reasons"].append(config_error)
    result["coverage"].update(verified_groups=verified, unavailable_groups=len(rows)-verified,
                              reasons=sorted(set(result["coverage"]["reasons"])))
    return result


_cache = {}
_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()


def api_payload(*, base_dir):
    """One bounded refresh per process; repeated public requests do not fan out."""
    key = str(base_dir)
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now-cached[0] < cached[1].get("cache_ttl_seconds", CACHE_SECONDS):
            return dict(deepcopy(cached[1]), cache_age_seconds=round(now-cached[0], 3))
    if not _refresh_lock.acquire(blocking=False):
        if cached:
            result = deepcopy(cached[1])
            result.update(status="PARTIAL", cache_age_seconds=round(now-cached[0], 3),
                          refresh_in_progress=True, stale=True)
            return result
        return dict(ok=False, status="UNAVAILABLE", rows=[], rows_final=0,
                    failure_code="refresh_in_progress", cache_age_seconds=None,
                    research_qualified=False, order_submission_authorized=False)
    try:
        result = snapshot(base_dir=base_dir)
        with _cache_lock:
            _cache.clear()  # Only the active deployment is retained; no unbounded key cache.
            _cache[key] = (time.monotonic(), deepcopy(result))
        return result
    finally:
        _refresh_lock.release()
