from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, "non-object payload")
    except FileNotFoundError:
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, "missing")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, exc)
    return {}


def safe_read_metrics_csv(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if isinstance(row, dict):
            return row
    except FileNotFoundError:
        logger.warning("METRICS_READ_FAIL csv path=%s err=%s", path, "missing")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("METRICS_READ_FAIL csv path=%s err=%s", path, exc)
    return {}


def parse_pipeline_summary(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        logger.warning("PIPELINE_SUMMARY_PARSE_FAIL path=%s err=%s", path, "missing")
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("PIPELINE_SUMMARY_PARSE_FAIL path=%s err=%s", path, exc)
        return {}

    for line in reversed(text.splitlines()):
        if "PIPELINE_SUMMARY" not in line:
            continue
        match = re.search(
            r"symbols_in=(\d+)\s+with_bars=(\d+)\s+rows=(\d+)(?:[^\n\r]*?(?:\sbar_rows|\sbars_rows_total)=(\d+))?",
            line,
        )
        if match:
            payload = {
                "last_run_utc": None,
                "symbols_in": int(match.group(1)),
                "symbols_with_bars": int(match.group(2)),
                "rows": int(match.group(3)),
            }
            if match.group(4):
                payload["bars_rows_total"] = int(match.group(4))
            return payload
        break
    return {}


def coerce_kpi_types(metrics: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "rows"):
        value = metrics.get(key)
        if value in ("", None):
            result[key] = None
            continue
        try:
            result[key] = int(value)
        except (TypeError, ValueError):
            result[key] = None
    result["last_run_utc"] = (
        metrics.get("last_run_utc")
        or metrics.get("last_run")
        or None
    )
    return result
