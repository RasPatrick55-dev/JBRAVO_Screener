"""Evaluation-only auto-remediation runner for ML health-triggered autotune.

This script is paper-mode safe. It does not place trades and does not alter
execution semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.utils.champion_config import load_latest_champion  # noqa: E402
from scripts.utils.feature_schema import compute_feature_signature, load_features_meta_for_path  # noqa: E402
from scripts.utils.ml_health_guard import (  # noqa: E402
    decide_ml_enrichment,
    load_latest_ml_health,
    resolve_ml_health_max_age_days,
)
from scripts.ranker_predict import _find_latest as _ranker_predict_find_latest  # noqa: E402
from scripts.ranker_predict import _mtime_iso as _ranker_predict_mtime_iso  # noqa: E402
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_autoremediate")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_TARGET = "label_5d_pos_300bp"
DEFAULT_TRIALS = 10
DEFAULT_REPREDICT_STRICT_MAX_MISSING_FRACTION = 0.2


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = datetime.strptime(value[:10], "%Y-%m-%d")
    return parsed.date()


def _parse_boolish(value: str | bool | None) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _merge_split_args(raw: str | None, split_tokens: list[str] | None) -> list[str]:
    merged: list[str] = []
    if raw:
        merged.extend(shlex.split(str(raw)))
    if split_tokens:
        merged.extend([str(token) for token in split_tokens if str(token).strip()])
    return merged


def _parse_passthrough_args(raw: str | None, split_mode: bool) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if split_mode:
        return [token for token in text.split() if token.strip()]
    return shlex.split(text)


def _sanitize_feature_columns(columns: Any) -> list[str]:
    if not isinstance(columns, (list, tuple)):
        return []
    seen: set[str] = set()
    resolved: list[str] = []
    for value in columns:
        column = str(value).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        resolved.append(column)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summary_path_for_model(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None
    match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
    if not match:
        return None
    return model_path.parent / f"ranker_summary_{match.group(1)}.json"


def _load_autotune_summary(run_date: date | None) -> dict[str, Any]:
    if db.db_enabled():
        payload = db.load_ml_artifact_payload("ranker_autotune", run_date=run_date)
        if isinstance(payload, dict) and payload:
            return payload
    return _read_json(BASE_DIR / "data" / "ranker_autotune" / "latest.json")


def _champion_pointer(run_date: date | None) -> dict[str, Any]:
    champion = load_latest_champion(BASE_DIR, run_date=run_date)
    fs_path = BASE_DIR / "data" / "ranker_autotune" / "champion.json"
    if not champion:
        return {
            "present": False,
            "path": str(fs_path),
            "source": "missing",
            "run_date": None,
            "champion_status": None,
            "champion_params": {},
        }
    return {
        "present": True,
        "path": str(fs_path),
        "source": champion.get("_champion_source") or "unknown",
        "run_date": champion.get("_champion_run_date") or champion.get("run_date"),
        "champion_status": champion.get("champion_status"),
        "champion_params": champion.get("champion_params") or {},
        "holdout_metrics": champion.get("holdout_metrics") or {},
        "tune_metrics": champion.get("tune_metrics") or {},
    }


def _run_autotune_subprocess(
    *,
    target: str,
    trials: int,
    run_date: date | None,
    passthrough_args: list[str],
) -> tuple[int, float, list[str]]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ranker_autotune",
        "--target",
        str(target),
        "--trials",
        str(int(trials)),
    ]
    if run_date is not None:
        cmd.extend(["--run-date", run_date.isoformat()])
    if passthrough_args:
        cmd.extend(passthrough_args)

    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
        check=False,
    )
    elapsed = float(time.time() - started)
    return int(completed.returncode or 0), elapsed, cmd


def _run_recalibrate_subprocess(
    *,
    target: str,
    run_date: date | None,
) -> tuple[int, float, list[str]]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ranker_recalibrate",
        "--target",
        str(target),
    ]
    if run_date is not None:
        cmd.extend(["--run-date", run_date.isoformat()])
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
        check=False,
    )
    elapsed = float(time.time() - started)
    return int(completed.returncode or 0), elapsed, cmd


def _latest_model_identity() -> dict[str, Any]:
    model_path = _ranker_predict_find_latest(BASE_DIR / "data" / "models", "ranker_*.pkl")
    if model_path is None:
        return {
            "model_path": None,
            "model_mtime_utc": None,
            "feature_set": None,
            "feature_signature": None,
            "feature_count": 0,
        }
    model_mtime_utc = _ranker_predict_mtime_iso(model_path)
    payload_feature_set: str | None = None
    payload_feature_signature: str | None = None
    payload_feature_columns: list[str] = []
    try:
        import joblib  # type: ignore

        payload = joblib.load(model_path)
        if isinstance(payload, dict):
            payload_feature_set = str(payload.get("feature_set") or "").strip().lower() or None
            payload_feature_signature = str(payload.get("feature_signature") or "").strip() or None
            payload_feature_columns = _sanitize_feature_columns(payload.get("feature_columns"))
    except Exception:
        payload = None
    summary_feature_set: str | None = None
    summary_feature_signature: str | None = None
    summary_feature_columns: list[str] = []
    summary_path = _summary_path_for_model(model_path)
    if summary_path is not None and summary_path.exists():
        summary_payload = _read_json(summary_path)
        summary_feature_set = str(summary_payload.get("feature_set") or "").strip().lower() or None
        summary_feature_signature = str(summary_payload.get("feature_signature") or "").strip() or None
        summary_feature_columns = _sanitize_feature_columns(summary_payload.get("feature_columns"))
    feature_columns = payload_feature_columns or summary_feature_columns
    computed_signature = compute_feature_signature(feature_columns) if feature_columns else None
    if payload_feature_signature and computed_signature and payload_feature_signature != computed_signature:
        LOG.warning(
            "[WARN] AUTOREMEDIATE_MODEL_SIGNATURE_MISMATCH stored=%s computed=%s model_path=%s",
            payload_feature_signature,
            computed_signature,
            model_path,
        )
    resolved_feature_set = payload_feature_set or summary_feature_set
    resolved_feature_signature = computed_signature or payload_feature_signature or summary_feature_signature
    return {
        "model_path": str(model_path),
        "model_mtime_utc": model_mtime_utc,
        "feature_set": resolved_feature_set,
        "feature_signature": resolved_feature_signature,
        "feature_count": int(len(feature_columns)),
    }


def _model_identity_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_path = str(before.get("model_path") or "")
    after_path = str(after.get("model_path") or "")
    before_mtime = str(before.get("model_mtime_utc") or "")
    after_mtime = str(after.get("model_mtime_utc") or "")
    return (before_path != after_path) or (before_mtime != after_mtime)


def _predictions_source_state() -> str:
    if db.db_enabled():
        try:
            present = bool(db.fetch_latest_ml_artifact("predictions"))
        except Exception:
            present = False
        return f"db:{'present' if present else 'missing'}"
    try:
        predictions_path = _ranker_predict_find_latest(
            BASE_DIR / "data" / "predictions",
            "predictions_*.csv",
        )
        present = predictions_path is not None
    except Exception:
        present = False
    return f"fs:{'present' if present else 'missing'}"


def _features_freshness_for_model(model_meta: dict[str, Any]) -> dict[str, Any]:
    features_path = None if db.db_enabled() else _ranker_predict_find_latest(
        BASE_DIR / "data" / "features",
        "features_*.csv",
    )
    features_meta, features_meta_source = load_features_meta_for_path(
        features_path,
        base_dir=BASE_DIR,
        prefer_db=bool(db.db_enabled()),
    )
    model_feature_set = str(model_meta.get("feature_set") or "").strip().lower() or None
    model_feature_signature = str(model_meta.get("feature_signature") or "").strip() or None
    features_feature_set = str(features_meta.get("feature_set") or "").strip().lower() or None
    meta_feature_signature = str(features_meta.get("feature_signature") or "").strip() or None
    features_columns = _sanitize_feature_columns(features_meta.get("feature_columns"))
    computed_features_signature = (
        compute_feature_signature(features_columns) if features_columns else None
    )
    if (
        meta_feature_signature
        and computed_features_signature
        and meta_feature_signature != computed_features_signature
    ):
        LOG.warning(
            "[WARN] AUTOREMEDIATE_FEATURES_META_SIGNATURE_MISMATCH stored=%s computed=%s source=%s",
            meta_feature_signature,
            computed_features_signature,
            features_meta_source,
        )
    features_feature_signature = computed_features_signature or meta_feature_signature or None
    stale = False
    reason = "fresh"
    if not features_meta:
        stale = True
        reason = "features_meta_missing"
    elif not model_feature_signature:
        stale = True
        reason = "model_signature_missing"
    elif not features_feature_signature:
        stale = True
        reason = "features_signature_missing"
    elif model_feature_signature != features_feature_signature:
        stale = True
        reason = "feature_signature_mismatch"
    elif (
        model_feature_set
        and features_feature_set
        and model_feature_set != features_feature_set
    ):
        stale = True
        reason = "feature_set_mismatch"
    payload = {
        "stale": bool(stale),
        "reason": reason,
        "model_feature_set": model_feature_set,
        "features_feature_set": features_feature_set,
        "model_feature_signature": model_feature_signature,
        "features_feature_signature": features_feature_signature,
        "features_meta_source": features_meta_source,
        "features_path": str(features_path) if features_path is not None else None,
    }
    LOG.info(
        "[INFO] AUTOREMEDIATE_FEATURES_FRESHNESS stale=%s reason=%s model_feature_set=%s features_feature_set=%s model_feature_signature=%s features_feature_signature=%s",
        str(bool(stale)).lower(),
        reason,
        model_feature_set or None,
        features_feature_set or None,
        model_feature_signature or None,
        features_feature_signature or None,
    )
    return payload


def _run_feature_generator_subprocess(
    *,
    feature_set: str | None,
    passthrough_args: list[str],
) -> tuple[int, float, list[str], str | None]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.feature_generator",
    ]
    if passthrough_args:
        cmd.extend(passthrough_args)
    child_env = os.environ.copy()
    feature_set_used = None
    if str(feature_set or "").strip().lower() in {"v1", "v2"}:
        feature_set_used = str(feature_set).strip().lower()
        child_env["JBR_ML_FEATURE_SET"] = feature_set_used
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=child_env,
        check=False,
    )
    elapsed = float(time.time() - started)
    return int(completed.returncode or 0), elapsed, cmd, feature_set_used


def _extract_flag_value(args_list: list[str], flag_name: str) -> str | None:
    for idx, token in enumerate(args_list):
        text = str(token or "").strip()
        if not text:
            continue
        if text == flag_name:
            if idx + 1 < len(args_list):
                nxt = str(args_list[idx + 1] or "").strip()
                if nxt and not nxt.startswith("--"):
                    return nxt
            return "true"
        if text.startswith(f"{flag_name}="):
            return text.split("=", 1)[1]
    return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _prepare_predict_args_safe(user_args: list[str]) -> tuple[list[str], dict[str, Any]]:
    args_out = list(user_args)
    strict_injected = False
    max_missing_injected = False

    strict_value = _extract_flag_value(args_out, "--strict-feature-match")
    if strict_value is None:
        args_out.extend(["--strict-feature-match", "true"])
        strict_injected = True
        strict_effective = True
    else:
        strict_effective = _parse_boolish(strict_value)

    max_missing_value = _extract_flag_value(args_out, "--max-missing-feature-fraction")
    if max_missing_value is None:
        args_out.extend(
            [
                "--max-missing-feature-fraction",
                str(DEFAULT_REPREDICT_STRICT_MAX_MISSING_FRACTION),
            ]
        )
        max_missing_injected = True
        max_missing_effective = float(DEFAULT_REPREDICT_STRICT_MAX_MISSING_FRACTION)
    else:
        max_missing_effective = _coerce_float(
            max_missing_value,
            DEFAULT_REPREDICT_STRICT_MAX_MISSING_FRACTION,
        )

    return args_out, {
        "strict": bool(strict_effective),
        "max_missing_feature_fraction": float(max_missing_effective),
        "strict_injected": bool(strict_injected),
        "max_missing_injected": bool(max_missing_injected),
    }


def _run_predict_subprocess(
    *,
    passthrough_args: list[str],
) -> tuple[int, float, list[str]]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ranker_predict",
    ]
    if passthrough_args:
        cmd.extend(passthrough_args)
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
        check=False,
    )
    elapsed = float(time.time() - started)
    return int(completed.returncode or 0), elapsed, cmd


def run_autoremediate(args: argparse.Namespace) -> dict[str, Any]:
    run_date_value = args.run_date or datetime.now(timezone.utc).date()
    dry_run = bool(args.dry_run)
    mode = str(args.mode)

    LOG.info(
        "[INFO] AUTOREMEDIATE_START target=%s trials=%d mode=%s dry_run=%s",
        args.target,
        int(args.trials),
        mode,
        str(dry_run).lower(),
    )

    health_status = load_latest_ml_health(
        base_dir=BASE_DIR,
        logger=LOG,
        source=args.ml_health_source,
        path=args.ml_health_path,
    )
    max_age_days = resolve_ml_health_max_age_days(args.ml_health_max_age_days)
    decision = decide_ml_enrichment(
        health_status,
        mode=mode,
        max_age_days=max_age_days,
        pipeline_run_date=run_date_value,
    )
    reasons = list(decision.get("reasons") or [])
    reason_text = ",".join(reasons) if reasons else "none"
    LOG.info(
        "[INFO] AUTOREMEDIATE_DECISION decision=%s reason=%s action=%s monitor_run_date=%s",
        decision.get("decision"),
        reason_text,
        decision.get("recommended_action"),
        decision.get("monitor_run_date"),
    )

    executed = False
    model_before = _latest_model_identity()
    autotune_rc: int | None = None
    autotune_elapsed_secs: float | None = None
    autotune_cmd: list[str] = []
    autotune_summary: dict[str, Any] = {}
    recalibrate_rc: int | None = None
    recalibrate_elapsed_secs: float | None = None
    recalibrate_cmd: list[str] = []
    remediation_kind = "none"
    repredict_enabled = bool(args.refresh_predictions)
    repredict_executed = False
    repredict_rc: int | None = None
    repredict_cmd: list[str] = []
    repredict_skipped_reason: str | None = None
    repredict_elapsed_secs: float | None = None
    repredict_reason: str | None = None
    repredict_strict: dict[str, Any] = {
        "strict": None,
        "max_missing_feature_fraction": None,
        "strict_injected": False,
        "max_missing_injected": False,
    }
    features_freshness_before: dict[str, Any] | None = None
    features_freshness_after: dict[str, Any] | None = None
    features_refresh_attempted = False
    features_refresh_rc: int | None = None
    features_refresh_elapsed_secs: float | None = None
    features_refresh_cmd: list[str] = []
    feature_set_used_for_refresh: str | None = None

    action = str(decision.get("recommended_action") or "none").strip().lower()
    wants_recalibrate = "action_recalibrate" in reasons or action == "recalibrate"

    if decision.get("decision") != "allow" and not dry_run:
        if wants_recalibrate:
            remediation_kind = "recalibrate"
            LOG.info("[INFO] AUTOREMEDIATE_RECALIBRATE_START target=%s", args.target)
            recalibrate_rc, recalibrate_elapsed_secs, recalibrate_cmd = _run_recalibrate_subprocess(
                target=args.target,
                run_date=args.run_date,
            )
            LOG.info(
                "[INFO] AUTOREMEDIATE_RECALIBRATE_END rc=%s elapsed_secs=%.2f",
                recalibrate_rc,
                float(recalibrate_elapsed_secs),
            )
            executed = recalibrate_rc == 0
            if recalibrate_rc != 0:
                raise RuntimeError(f"ranker_recalibrate failed rc={recalibrate_rc}")
        else:
            remediation_kind = "retrain"
            LOG.info(
                "[INFO] AUTOREMEDIATE_AUTOTUNE_START target=%s trials=%d",
                args.target,
                int(args.trials),
            )
            autotune_rc, autotune_elapsed_secs, autotune_cmd = _run_autotune_subprocess(
                target=args.target,
                trials=args.trials,
                run_date=args.run_date,
                passthrough_args=args.autotune_args,
            )
            LOG.info(
                "[INFO] AUTOREMEDIATE_AUTOTUNE_END rc=%s elapsed_secs=%.2f",
                autotune_rc,
                float(autotune_elapsed_secs),
            )
            executed = autotune_rc == 0
            if autotune_rc != 0:
                raise RuntimeError(f"ranker_autotune failed rc={autotune_rc}")
            autotune_summary = _load_autotune_summary(args.run_date)

    model_after = _latest_model_identity()
    model_changed = _model_identity_changed(model_before, model_after)
    predictions_source = _predictions_source_state()
    if not repredict_enabled:
        repredict_skipped_reason = "refresh_disabled"
        LOG.info(
            "[INFO] AUTOREMEDIATE_REPREDICT_SKIPPED reason=%s",
            repredict_skipped_reason,
        )
    elif not executed:
        repredict_skipped_reason = "prior_step_failed"
        LOG.info(
            "[INFO] AUTOREMEDIATE_REPREDICT_SKIPPED reason=%s",
            repredict_skipped_reason,
        )
    elif bool(args.refresh_only_if_model_changed) and not model_changed:
        repredict_skipped_reason = "model_unchanged"
        LOG.info(
            "[INFO] AUTOREMEDIATE_REPREDICT_SKIPPED reason=%s",
            repredict_skipped_reason,
        )
    else:
        repredict_reason = "model_changed" if model_changed else "forced"
        features_freshness_before = _features_freshness_for_model(model_after)
        stale_before = bool(features_freshness_before.get("stale"))
        should_refresh_features = bool(args.refresh_features) and (
            (not bool(args.refresh_features_only_if_stale)) or stale_before
        )
        if should_refresh_features:
            features_refresh_attempted = True
            LOG.info(
                "[INFO] AUTOREMEDIATE_REFRESH_FEATURES enabled=true stale=%s -> running feature_generator feature_set=%s",
                str(stale_before).lower(),
                str(model_after.get("feature_set") or "env/default"),
            )
            (
                features_refresh_rc,
                features_refresh_elapsed_secs,
                features_refresh_cmd,
                feature_set_used_for_refresh,
            ) = _run_feature_generator_subprocess(
                feature_set=str(model_after.get("feature_set") or "").strip().lower() or None,
                passthrough_args=args.feature_generator_args,
            )
            LOG.info(
                "[INFO] AUTOREMEDIATE_REFRESH_FEATURES_DONE rc=%s",
                features_refresh_rc,
            )
            if int(features_refresh_rc or 0) == 0:
                features_freshness_after = _features_freshness_for_model(_latest_model_identity())
            else:
                repredict_skipped_reason = "features_refresh_failed"
                LOG.warning(
                    "[WARN] AUTOREMEDIATE_REPREDICT_SKIPPED reason=%s",
                    repredict_skipped_reason,
                )

        if repredict_skipped_reason == "features_refresh_failed":
            pass
        else:
            safe_predict_args, repredict_strict = _prepare_predict_args_safe(
                list(args.ranker_predict_args)
            )
            if features_freshness_after is None:
                features_freshness_after = (
                    _features_freshness_for_model(_latest_model_identity())
                    if should_refresh_features
                    else features_freshness_before
                )
            if features_freshness_after is not None and bool(features_freshness_after.get("stale")):
                repredict_reason = (
                    f"{repredict_reason},stale_features"
                    if repredict_reason
                    else "stale_features"
                )
            repredict_cmd_preview = [
                sys.executable,
                "-m",
                "scripts.ranker_predict",
                *list(safe_predict_args),
            ]
            LOG.info(
                "[INFO] AUTOREMEDIATE_REPREDICT_START reason=%s args=%s",
                repredict_reason,
                shlex.join(repredict_cmd_preview),
            )
            repredict_rc, repredict_elapsed_secs, repredict_cmd = _run_predict_subprocess(
                passthrough_args=safe_predict_args
            )
            repredict_executed = True
            predictions_source = _predictions_source_state()
            if int(repredict_rc or 0) == 0:
                LOG.info(
                    "[INFO] AUTOREMEDIATE_REPREDICT_END rc=%s predictions_source=%s",
                    repredict_rc,
                    predictions_source,
                )
            elif int(repredict_rc or 0) == 2:
                LOG.warning(
                    "[WARN] AUTOREMEDIATE_REPREDICT_END rc=%s predictions_source=%s reason=feature_mismatch_fatal",
                    repredict_rc,
                    predictions_source,
                )
            else:
                LOG.warning(
                    "[WARN] AUTOREMEDIATE_REPREDICT_END rc=%s predictions_source=%s reason=ranker_predict_failed",
                    repredict_rc,
                    predictions_source,
                )

    champion = _champion_pointer(args.run_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest.json"

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date_value.isoformat(),
        "target": args.target,
        "mode": mode,
        "dry_run": bool(dry_run),
        "trials": int(args.trials),
        "ml_health_source": args.ml_health_source or "auto",
        "ml_health_path": str(args.ml_health_path) if args.ml_health_path else None,
        "ml_health_max_age_days": int(max_age_days),
        "decision": {
            "decision": decision.get("decision"),
            "mode": decision.get("mode"),
            "reasons": reasons,
            "monitor_run_date": decision.get("monitor_run_date"),
            "recommended_action": decision.get("recommended_action"),
            "psi_score": decision.get("psi_score"),
            "recent_sharpe": decision.get("recent_sharpe"),
            "source": decision.get("source"),
        },
        "executed": bool(executed),
        "remediation_kind": remediation_kind,
        "recalibrate": {
            "executed": bool(remediation_kind == "recalibrate" and executed),
            "requested": bool(decision.get("decision") != "allow" and wants_recalibrate),
            "rc": recalibrate_rc,
            "elapsed_secs": recalibrate_elapsed_secs,
            "cmd": recalibrate_cmd,
        },
        "autotune": {
            "executed": bool(remediation_kind == "retrain" and executed),
            "requested": bool(decision.get("decision") != "allow" and not wants_recalibrate),
            "rc": autotune_rc,
            "elapsed_secs": autotune_elapsed_secs,
            "cmd": autotune_cmd,
            "summary": autotune_summary,
        },
        "repredict": {
            "enabled": repredict_enabled,
            "executed": repredict_executed,
            "rc": repredict_rc,
            "elapsed_secs": repredict_elapsed_secs,
            "cmd": repredict_cmd,
            "skipped_reason": repredict_skipped_reason,
            "reason": repredict_reason,
            "model_before": model_before,
            "model_after": model_after,
            "predictions_source": predictions_source,
            "refresh_only_if_model_changed": bool(args.refresh_only_if_model_changed),
            "features_freshness_before": features_freshness_before,
            "features_freshness_after": features_freshness_after,
            "features_refresh": {
                "enabled": bool(args.refresh_features),
                "attempted": bool(features_refresh_attempted),
                "rc": features_refresh_rc,
                "elapsed_secs": features_refresh_elapsed_secs,
                "cmd": features_refresh_cmd,
                "feature_set_used": feature_set_used_for_refresh,
                "refresh_only_if_stale": bool(args.refresh_features_only_if_stale),
            },
            "repredict_strict": repredict_strict,
        },
        "champion": champion,
        "output_files": {
            "latest_json": str(output_path),
            "champion_json": str(BASE_DIR / "data" / "ranker_autotune" / "champion.json"),
            "autotune_latest_json": str(BASE_DIR / "data" / "ranker_autotune" / "latest.json"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_autoremediate",
            run_date_value,
            payload=payload,
            rows_count=1,
            source="ranker_autoremediate",
            file_name=output_path.name,
        ):
            LOG.info(
                "[INFO] AUTOREMEDIATE_DB_WRITTEN artifact_type=ranker_autoremediate run_date=%s",
                run_date_value,
            )
    else:
        LOG.warning("[WARN] DB_DISABLED ranker_autoremediate_using_fs_fallback=true")

    LOG.info(
        "[INFO] AUTOREMEDIATE_END executed=%s output=%s",
        str(bool(executed)).lower(),
        output_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    env_mode = str(os.getenv("JBR_ML_HEALTH_GUARD_MODE") or "").strip().lower()
    if env_mode not in {"warn", "block"}:
        env_mode = "warn"
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Label target forwarded to ranker_autotune (default: label_5d_pos_300bp).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(float(os.getenv("JBR_RANKER_AUTOREMEDIATE_TRIALS", DEFAULT_TRIALS))),
        help="Bounded number of autotune trials when remediation executes (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const="true",
        default="false",
        help="When true, compute decision only and skip autotune execution.",
    )
    parser.add_argument(
        "--mode",
        choices=("warn", "block"),
        default=env_mode,
        help="Decision policy for unhealthy monitor states (default: warn).",
    )
    parser.add_argument(
        "--ml-health-source",
        choices=("auto", "db", "fs"),
        default=None,
        help="Optional health artifact source override (default from env or auto).",
    )
    parser.add_argument(
        "--ml-health-path",
        type=Path,
        default=None,
        help="Optional health artifact path override (used when source=fs/auto).",
    )
    parser.add_argument(
        "--ml-health-max-age-days",
        type=int,
        default=None,
        help="Optional staleness override for health payload age in days.",
    )
    parser.add_argument(
        "--autotune-args",
        type=str,
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_AUTOTUNE_ARGS", ""),
        help="Extra CLI args forwarded to scripts.ranker_autotune.",
    )
    parser.add_argument(
        "--autotune-args-split",
        nargs="*",
        default=None,
        help="Extra ranker_autotune CLI args as split tokens.",
    )
    parser.add_argument(
        "--refresh-predictions",
        nargs="?",
        const="true",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_REFRESH_PREDICTIONS", "false"),
        help="When true, run scripts.ranker_predict after successful remediation.",
    )
    parser.add_argument(
        "--refresh-only-if-model-changed",
        nargs="?",
        const="true",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_REFRESH_ONLY_IF_MODEL_CHANGED", "true"),
        help="When true, skip post-remediation repredict unless model identity changed.",
    )
    parser.add_argument(
        "--refresh-features",
        nargs="?",
        const="true",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_REFRESH_FEATURES", "false"),
        help="When true, optionally refresh features before repredict.",
    )
    parser.add_argument(
        "--refresh-features-only-if-stale",
        nargs="?",
        const="true",
        default=os.getenv(
            "JBR_RANKER_AUTOREMEDIATE_REFRESH_FEATURES_ONLY_IF_STALE",
            "true",
        ),
        help="When true, refresh features only when stale/mismatched.",
    )
    parser.add_argument(
        "--feature-generator-args",
        type=str,
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_FEATURE_GENERATOR_ARGS", ""),
        help="Extra CLI args forwarded to scripts.feature_generator during refresh.",
    )
    parser.add_argument(
        "--feature-generator-args-split",
        nargs="?",
        const="true",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_FEATURE_GENERATOR_ARGS_SPLIT", "false"),
        help="Interpret --feature-generator-args as split tokens (whitespace-separated).",
    )
    parser.add_argument(
        "--ranker-predict-args",
        type=str,
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_RANKER_PREDICT_ARGS", ""),
        help="Extra CLI args forwarded to scripts.ranker_predict during refresh.",
    )
    parser.add_argument(
        "--ranker-predict-args-split",
        nargs="?",
        const="true",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_RANKER_PREDICT_ARGS_SPLIT", "false"),
        help="Interpret --ranker-predict-args as split tokens (whitespace-separated).",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional run date (YYYY-MM-DD) for decision timing and artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_autoremediate",
        help="Output directory for summary payload (default: data/ranker_autoremediate).",
    )
    parsed = parser.parse_args(argv)

    if int(parsed.trials) < 1:
        parser.error("--trials must be >= 1")
    if parsed.ml_health_max_age_days is not None and int(parsed.ml_health_max_age_days) < 0:
        parser.error("--ml-health-max-age-days must be >= 0")
    mode = str(parsed.mode or "warn").strip().lower()
    if mode not in {"warn", "block"}:
        parser.error("--mode must be one of: warn, block")
    parsed.mode = mode
    parsed.dry_run = _parse_boolish(parsed.dry_run)
    parsed.refresh_predictions = _parse_boolish(parsed.refresh_predictions)
    parsed.refresh_only_if_model_changed = _parse_boolish(parsed.refresh_only_if_model_changed)
    parsed.refresh_features = _parse_boolish(parsed.refresh_features)
    parsed.refresh_features_only_if_stale = _parse_boolish(parsed.refresh_features_only_if_stale)
    parsed.ranker_predict_args_split = _parse_boolish(parsed.ranker_predict_args_split)
    parsed.feature_generator_args_split = _parse_boolish(parsed.feature_generator_args_split)
    try:
        parsed.run_date = _parse_run_date(parsed.run_date)
    except Exception:
        parser.error(f"Invalid --run-date: {parsed.run_date}")
    parsed.autotune_args = _merge_split_args(parsed.autotune_args, parsed.autotune_args_split)
    parsed.ranker_predict_args = _parse_passthrough_args(
        parsed.ranker_predict_args,
        parsed.ranker_predict_args_split,
    )
    parsed.feature_generator_args = _parse_passthrough_args(
        parsed.feature_generator_args,
        parsed.feature_generator_args_split,
    )
    return parsed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_autoremediate(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("AUTOREMEDIATE_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
