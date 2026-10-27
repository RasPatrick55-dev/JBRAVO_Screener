"""Bounded, reproducible strategy-testing CLI. Importing has no operational effects."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import html
import json
import math
from pathlib import Path
import platform
import sys
import time

import pandas as pd

from scripts.research_dataset import bounded_read, canonical_hash, digest_bytes, load_dataset, strict_json
from scripts.research_simulation import ResearchPortfolio
from scripts.pattern_strategy import PATTERNS
from scripts.research_strategies import BASELINE, BENCHMARK, CHALLENGER, METHODS, prepare_signals

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("research_testing.py", "research_dataset.py", "research_simulation.py",
                "research_strategies.py", "research_store.py", "backtest_core.py", "indicators.py")


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported result value: " + type(value).__name__)


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, default=json_default, allow_nan=False)
        handle.write("\n")


def validate_spec(spec):
    required = {"schema", "purpose", "capture_manifest", "actions_capture", "universe", "folds",
                "strategies", "cost_cases", "portfolio", "max_runs", "max_seconds", "research_qualified", "execution_authorized"}
    if set(spec) != required or spec["schema"] != "jbravo.strategy-study.v1":
        raise ValueError("Unexpected strategy study schema/fields")
    if spec["purpose"] != "exploratory_research" or spec["research_qualified"] is not False or spec["execution_authorized"] is not False:
        raise ValueError("Only explicit exploratory research is supported")
    if not 1 <= len(spec["strategies"]) <= 2 or len(set(spec["strategies"])) != len(spec["strategies"]) or not set(spec["strategies"]) <= set((*METHODS, *PATTERNS)):
        raise ValueError("Unknown/duplicate strategy; v1 has two frozen methods")
    folds = spec["folds"]
    if not 1 <= len(folds) <= 10:
        raise ValueError("Fold count limit exceeded")
    previous_end = None
    names = set()
    for fold in folds:
        if set(fold) != {"id", "warmup_start", "test_start", "test_end"} or fold["id"] in names:
            raise ValueError("Duplicate/malformed fold")
        names.add(fold["id"])
        for field in ("warmup_start", "test_start", "test_end"):
            if date.fromisoformat(fold[field]).isoformat() != fold[field]:
                raise ValueError("Noncanonical fold date")
        if not fold["warmup_start"] < fold["test_start"] <= fold["test_end"] or (previous_end and fold["test_start"] <= previous_end):
            raise ValueError("Warmup boundary or overlapping/unsorted evaluation folds")
        previous_end = fold["test_end"]
    costs = spec["cost_cases"]
    if not 1 <= len(costs) <= 5 or len({c["id"] for c in costs}) != len(costs):
        raise ValueError("Duplicate/invalid cost cases")
    for case in costs:
        if set(case) != {"id", "fee_per_fill", "slippage_bps"}:
            raise ValueError("Malformed cost case")
        for key, ceiling in (("fee_per_fill", 1000), ("slippage_bps", 1000)):
            value = case[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= ceiling:
                raise ValueError("Invalid explicit execution cost")
    portfolio = spec["portfolio"]
    allowed = {"initial_cash", "alloc_pct", "top_n", "max_positions", "trail_pct", "max_hold_days", "atr_multiple"}
    if set(portfolio) != allowed:
        raise ValueError("Malformed portfolio settings")
    for key, low, high in (("initial_cash", 100, 1e9), ("alloc_pct", 0.000001, 1),
                           ("trail_pct", 0.000001, 0.99), ("atr_multiple", 0.000001, 100)):
        value = portfolio[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid portfolio value: " + key)
    for key, ceiling in (("top_n", 100), ("max_positions", 100), ("max_hold_days", 365)):
        if type(portfolio[key]) is not int or not 1 <= portfolio[key] <= ceiling:
            raise ValueError("Invalid integer portfolio setting")
    if portfolio["top_n"] > portfolio["max_positions"]:
        raise ValueError("top_n exceeds capacity")
    expected_runs = len(folds) * len(costs) * (len(spec["strategies"]) + 1)
    if type(spec["max_runs"]) is not int or not expected_runs <= spec["max_runs"] <= 100:
        raise ValueError("Frozen trial budget exceeded")
    if type(spec["max_seconds"]) is not int or not 1 <= spec["max_seconds"] <= 3600:
        raise ValueError("Invalid wall-clock budget")
    return expected_runs


def starter_spec(manifest, actions):
    return dict(schema="jbravo.strategy-study.v1", purpose="exploratory_research",
                capture_manifest=str(Path(manifest).resolve()),
                actions_capture={"path": str(Path(actions).resolve()), "sha256": digest_bytes(bounded_read(actions))},
                universe={"symbols": ["ALK", "FN", "HMY", "LOGI", "MIRM", "NVST", "PLUG", "SKY", "SRRK", "STLD"],
                          "benchmark": "SPY", "excluded": {
                              "NBIS": "409 zero-volume/VWAP sessions and historical halt/identity questions; no silent filtering",
                              "VSCO": "Symbol continuity and missing post-June-1 sessions remain unresolved",
                              "VSXY": "Separate continuity supplement; no approved stitching to VSCO",
                              "HBAN": "Selected study excludes unresolved share-merger events involving this issuer",
                              "RGLD": "Selected study excludes unresolved share-merger events involving this issuer"}},
                folds=[{"id": str(year), "warmup_start": "2023-01-03", "test_start": f"{year}-01-02",
                        "test_end": f"{year}-12-31" if year < 2026 else "2026-10-23"}
                       for year in (2024, 2025, 2026)],
                strategies=[BASELINE, CHALLENGER],
                cost_cases=[dict(id="base_cost", fee_per_fill=1.0, slippage_bps=5.0),
                            dict(id="cost_stress", fee_per_fill=2.0, slippage_bps=10.0)],
                portfolio=dict(initial_cash=100000.0, alloc_pct=0.05, top_n=3, max_positions=4,
                               trail_pct=0.03, max_hold_days=7, atr_multiple=1.0),
                max_runs=18, max_seconds=600, research_qualified=False, execution_authorized=False)


def report_html(output, result, curves):
    rows = []
    for run in result["runs"]:
        rows.append({"method": run["method"], "fold": run["fold"], "cost": run["cost_case"],
                     **{k: run["metrics"][k] for k in ("total_return_pct", "max_drawdown_pct", "exit_fills", "dividend_income")}})
    table = pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)
    plots = ""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
        for fold in result["fold_ids"]:
            figure = go.Figure()
            for key, frame in curves.items():
                if key[1] == fold:
                    figure.add_trace(go.Scatter(x=frame.index, y=frame["equity"], name=key[0] + "/" + key[2]))
            figure.update_layout(title="Fold " + fold + " â€” modeled NAV including dividend receivables", yaxis_title="USD")
            plots += plot(figure, include_plotlyjs=True if not plots else False, output_type="div")
    except ImportError:
        plots = "<p>Plotly is unavailable; full equity CSV files are included.</p>"
    limitations = "".join("<li>" + html.escape(x) + "</li>" for x in result["limitations"])
    body = ("<!doctype html><html><head><meta charset='utf-8'><title>JBravo strategy study</title>"
            "<style>body{font:16px system-ui;max-width:1400px;margin:36px auto;padding:20px;color:#202b38}"
            "table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccd}th{text-align:left}"
            ".notice{padding:16px;background:#fff2d3}</style></head><body><h1>JBravo strategy testing</h1>"
            "<p class='notice'>Exploratory retrospective results. Data and strategies remain unqualified. "
            "No automatic winner selection, promotion or trading authorization.</p><p>Study "
            + html.escape(result["run_id"]) + "</p>" + table + plots + "<h2>Limits and assumptions</h2><ul>"
            + limitations + "</ul></body></html>")
    (output / "report.html").write_text(body, encoding="utf-8")


def run_study(spec, output, *, stop_file=None):
    expected_runs = validate_spec(spec)
    output = Path(output).resolve()
    if output == REPO or output.is_relative_to(REPO):
        raise ValueError("Research results must be outside the source/production tree")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "specification.json", spec)
    started = time.monotonic()
    deadline = started + spec["max_seconds"]
    stop_file = Path(stop_file) if stop_file else output / "STOP"
    source_hashes = {name: digest_bytes((REPO / "scripts" / name).read_bytes()) for name in SOURCE_FILES}
    source_dir = output / "source"
    source_dir.mkdir()
    for name in SOURCE_FILES:
        (source_dir / name).write_bytes((REPO / "scripts" / name).read_bytes())
    try:
        dataset = load_dataset(spec)
        save_json(output / "dataset.json", dataset.provenance)
        specification_hash = canonical_hash(spec)
        run_id = canonical_hash(dict(specification=specification_hash, dataset=dataset.provenance["dataset_sha256"], source=source_hashes))
        runs, curves = [], {}
        for fold in spec["folds"]:
            for boundary in ("warmup_start", "test_start", "test_end"):
                if fold[boundary] not in dataset.calendar.index:
                    raise ValueError("Fold boundary is not a captured calendar session")
            for cost in spec["cost_cases"]:
                for method in [*spec["strategies"], BENCHMARK]:
                    if time.monotonic() >= deadline or stop_file.exists():
                        raise RuntimeError("Research stopped before next trial")
                    symbols = spec["universe"]["symbols"] if method != BENCHMARK else [spec["universe"]["benchmark"]]
                    if method in PATTERNS:
                        from scripts.strategy_library import get_strategy
                        data = {symbol: get_strategy(method).prepare(dataset.raw[symbol], dataset.features[symbol], fold) for symbol in symbols}
                    else:
                        data = {symbol: prepare_signals(dataset.raw[symbol], dataset.features[symbol], method, fold) for symbol in symbols}
                    options = dict(spec["portfolio"], trade_cost=cost["fee_per_fill"], slippage=cost["slippage_bps"] / 10000)
                    if method == BENCHMARK:
                        options.update(alloc_pct=1.0, top_n=1, max_positions=1, trail_pct=None, atr_multiple=0,
                                       max_hold_days=100000, enable_macd_exit=False, enable_partial_exit=False,
                                       enable_rsi_divergence=False, enable_candlestick_exit=False,
                                       enable_ema_exit=False, enable_trailing_exit=False)
                    if method in PATTERNS:
                        from scripts.bull_flag_strategy import research_portfolio_class
                        if method != 'bull_flag_breakout_v1':
                            from scripts.pattern_strategy import research_portfolio_class
                        engine = research_portfolio_class()(data, dividends=dataset.dividends,
                                               deadline=deadline, stop_file=stop_file, **options)
                    else:
                        engine = ResearchPortfolio(data, method=method, dividends=dataset.dividends,
                                               deadline=deadline, stop_file=stop_file, **options)
                    engine.run()
                    metrics = engine.summary()
                    label = f"trial_{len(runs) + 1:03d}"
                    folder = output / label
                    folder.mkdir()
                    save_json(folder / "signals.json", engine.signals)
                    save_json(folder / "fills.json", engine.fills)
                    save_json(folder / "actions.json", engine.action_events)
                    save_json(folder / "ledger.json", engine.reconciliations)
                    engine.results().to_csv(folder / "trades.csv", index=False)
                    engine.equity().to_csv(folder / "equity.csv")
                    curve = engine.equity()
                    curves[(method, fold["id"], cost["id"])] = curve
                    record = dict(trial=label, method=method, fold=fold["id"], cost_case=cost["id"],
                                  metrics=metrics, outcome_sha256=canonical_hash(json.loads(json.dumps(
                                      dict(metrics=metrics, fills=engine.fills, ledger=engine.reconciliations), default=json_default))))
                    save_json(folder / "result.json", record)
                    runs.append(record)
                    print(json.dumps(dict(event="TRIAL_COMPLETE", trial=label, method=method,
                                          fold=fold["id"], cost_case=cost["id"], fills=len(engine.fills))), flush=True)
        if len(runs) != expected_runs:
            raise ValueError("Trial accounting mismatch")
        result = dict(schema="jbravo.strategy-study-result.v1", run_id=run_id,
                      specification_sha256=specification_hash, dataset_sha256=dataset.provenance["dataset_sha256"],
                      source_sha256=source_hashes, status="COMPLETED_EXPLORATORY", research_qualified=False,
                      execution_authorized=False, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                      environment=dict(python=platform.python_version(), pandas=pd.__version__),
                      planned_runs=expected_runs, completed_runs=len(runs), runs=runs,
                      fold_ids=[f["id"] for f in spec["folds"]],
                      limitations=dataset.provenance["limitations"] + [
                          "Historical folds reuse previously exposed data; chronological separation does not create an untouched holdout",
                          "Baseline is the legacy simulator core; this is not historical screener-v2 or broker-executor parity",
                          "Signals assume daily data available 20:15 Eastern; entries use next regular-session open, not the production premarket limit model",
                          "Daily OHLC cannot recover intraday order or actual liquidity; old protective stops precede completed-bar tightening",
                          "Flat per-fill fees and symmetric bps slippage are sensitivity assumptions, not calibrated execution costs",
                          "Win rate/profit factor count exit fills, including partials; zero-exit cases remain explicit",
                          "Terminal positions and dividend receivables remain marked; no invented liquidation or future payout",
                          "Cash interest, taxes, withholding, borrow, spread dynamics and market impact are not modeled",
                          "Two predefined strategies and two cost cases only; no optimization, statistical significance or winning-strategy claim"
                      ])
        save_json(output / "result.json", result)
        report_html(output, result, curves)
        bindings = [{"path": str(p.relative_to(output)), "sha256": digest_bytes(p.read_bytes()), "bytes": p.stat().st_size}
                    for p in sorted(output.rglob("*")) if p.is_file()]
        save_json(output / "manifest.json", dict(run_id=run_id, bindings=bindings, elapsed_seconds=time.monotonic() - started))
        return result
    except Exception as exc:
        save_json(output / "failure.json", dict(status="FAILED", error_type=type(exc).__name__,
                                                reason=str(exc), research_qualified=False, execution_authorized=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Load a local ignored environment file in-process; values are never printed")
    sub = parser.add_subparsers(dest="command", required=True)
    example = sub.add_parser("prepare-study")
    example.add_argument("--capture-manifest", required=True, type=Path)
    example.add_argument("--actions-capture", required=True, type=Path)
    example.add_argument("--output", required=True, type=Path)
    for command in ("inspect-data", "run"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True, type=Path)
        if command == "run":
            child.add_argument("--output", required=True, type=Path)
            child.add_argument("--stop-file", type=Path)
    sub.add_parser("db-status")
    snap = sub.add_parser("db-snapshot")
    snap.add_argument("--symbols", required=True)
    snap.add_argument("--start", required=True, type=date.fromisoformat)
    snap.add_argument("--end", required=True, type=date.fromisoformat)
    snap.add_argument("--output", required=True, type=Path)
    sub.add_parser("init-research-db")
    publish = sub.add_parser("publish-result")
    publish.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        if args.command == "prepare-study":
            spec = starter_spec(args.capture_manifest, args.actions_capture)
            validate_spec(spec)
            save_json(args.output, spec)
            print(json.dumps(dict(specification=str(args.output), planned_runs=spec["max_runs"])))
        elif args.command in ("inspect-data", "run"):
            spec = strict_json(bounded_read(args.spec, limit=1024 * 1024))
            validate_spec(spec)
            if args.command == "inspect-data":
                data = load_dataset(spec)
                print(json.dumps(data.provenance, indent=2))
            else:
                result = run_study(spec, args.output, stop_file=args.stop_file)
                print(json.dumps(dict(status=result["status"], run_id=result["run_id"], completed_runs=result["completed_runs"])))
        else:
            from scripts import research_store
            if args.command == "db-status":
                print(json.dumps(research_store.inspect_database(), indent=2))
            elif args.command == "db-snapshot":
                if args.start > args.end:
                    raise ValueError("Snapshot start follows end")
                snapshot = research_store.snapshot_daily_bars(args.symbols.split(","), args.start, args.end)
                save_json(args.output, snapshot)
                print(json.dumps(dict(saved=str(args.output), rows=len(snapshot["rows"]), research_qualified=False)))
            elif args.command == "init-research-db":
                research_store.initialize_database()
                print(json.dumps(dict(initialized="jbravo_research", production_tables_changed=False)))
            elif args.command == "publish-result":
                research_store.publish_result(args.result)
                print(json.dumps(dict(published=True, research_qualified=False)))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="FAILED", error_type=type(exc).__name__, reason=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
