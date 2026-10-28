"""Actual health AST, synthetic artifacts/clock; no app/config/provider imports."""
import ast
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import pytest

NOW = datetime(2026, 10, 27, 22, tzinfo=timezone.utc)


def health(metrics_age, connection_stamp, *, source='connection_health.json', trading_ok=True):
    path = Path(__file__).parents[1] / 'dashboards/data_io.py'
    names = {'_freshness', 'connection_check_snapshot', 'screener_health'}
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return NOW
    ns = dict(Any=Any, Dict=Dict, Optional=Optional, Path=Path, DATA_DIR=Path('synthetic-not-read'),
        dt=SimpleNamespace(datetime=Clock, timezone=timezone), os=SimpleNamespace(getenv=lambda *_: None),
        _mtime_iso=lambda _: None, _run_type_hint=lambda: 'nightly', _normalize_feed=lambda _: None,
        _update_coverage_history=lambda _: {},
        load_screener_metrics=lambda _: {'last_run_utc': (NOW-timedelta(seconds=metrics_age)).isoformat()},
        load_connection_health=lambda _: {'source': source, 'timestamp': connection_stamp, 'trading_ok': trading_ok, 'feed': 'sip'})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-health-clock-functions', 'exec'), ns)
    return ns['screener_health']()


def test_connection_freshness_independent_of_old_metrics():
    snapshot = health(72000, (NOW-timedelta(seconds=30)).isoformat())
    assert snapshot['freshness']['freshness_level'] == 'gray'
    check = snapshot['connection_check']
    assert check['status'] == 'fresh' and check['freshness']['age_seconds'] == 30
    assert check['source'] == 'connection_health.json' and check['trading_ok'] is True


def test_old_connection_never_becomes_fresh_from_new_metrics():
    snapshot = health(30, (NOW-timedelta(seconds=72000)).isoformat())
    assert snapshot['freshness']['freshness_level'] == 'green'
    assert snapshot['connection_check']['status'] == 'stale'
    assert snapshot['connection_check']['freshness']['freshness_level'] == 'gray'


@pytest.mark.parametrize('stamp', [None, '', 'invalid', '2026-02-30T00:00:00Z', '2026-10-27T22:00:00',
    '2026-10-28T00:00:00Z', '2026-10-27T22:00:00.000001+00:00'])
def test_missing_invalid_naive_or_future_timestamp_unknown(stamp):
    check = health(30, stamp)['connection_check']
    assert check['status'] == 'unknown'
    assert check['freshness']['freshness_level'] == 'gray'


def test_unverified_source_or_result_unknown():
    stamp = (NOW-timedelta(seconds=30)).isoformat()
    for source, result in [(None, True), ('unverified', True), ('connection_health.json', None)]:
        assert health(30, stamp, source=source, trading_ok=result)['connection_check']['status'] == 'unknown'


def test_equivalent_timezone_and_failed_connection_evidence_preserved():
    stamp = (NOW-timedelta(seconds=30)).astimezone(timezone(timedelta(hours=-4))).isoformat()
    check = health(30, stamp, source='connectivity.json', trading_ok=False)['connection_check']
    assert check['status'] == 'fresh' and check['trading_ok'] is False
    assert check['checked_at'] == (NOW-timedelta(seconds=30)).isoformat()


def test_connection_loader_labels_actual_selected_artifact():
    path = Path(__file__).parents[1] / 'dashboards/data_io.py'
    node = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'load_connection_health')
    for primary in (True, False):
        ns = dict(Any=Any, Dict=Dict, Optional=Optional, Path=Path, DATA_DIR=Path('synthetic'),
            _read_json_safe=lambda p: {'timestamp': '2026-10-27T21:59:30Z', 'trading_ok': True} if (p.name == 'connection_health.json') == primary else {},
            _coerce_bool=lambda v: v if type(v) is bool else None, _coerce_int=lambda _: None,
            _coerce_float=lambda _: None, _normalize_feed=lambda _: None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-connection-loader', 'exec'), ns)
        result = ns['load_connection_health']()
        assert result['source'] == ('connection_health.json' if primary else 'connectivity.json')
