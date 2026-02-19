"""Normalization helpers for bar payloads."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .frame_guards import ensure_symbol_column

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

_TZ_STR_PATCHED = False


def _ensure_utc_dtype_str_patch() -> None:
    """Normalize pandas' UTC dtype string to satisfy downstream expectations."""

    global _TZ_STR_PATCHED
    if _TZ_STR_PATCHED:
        return
    try:
        from pandas.api.types import DatetimeTZDtype
    except Exception:  # pragma: no cover - pandas internals unavailable
        _TZ_STR_PATCHED = True
        return

    original_str = DatetimeTZDtype.__str__

    def _patched(self: "DatetimeTZDtype") -> str:  # type: ignore[name-defined]
        base = original_str(self)
        tz = getattr(self.tz, "zone", None) or str(self.tz)
        if isinstance(tz, str) and tz.upper() == "UTC":
            return "datetime64[UTC]"
        return base

    DatetimeTZDtype.__str__ = _patched  # type: ignore[assignment]
    _TZ_STR_PATCHED = True


def _object_to_dict(obj: Any) -> dict[str, Any]:
    """Coerce an arbitrary bar-like object into a dictionary."""

    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "_asdict"):
        try:
            return dict(obj._asdict())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "__dict__"):
        data = {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        if data:
            return data
    return {"value": obj}


def _maybe_iter(obj: Any) -> Iterable[Any]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.items()
    if isinstance(obj, (list, tuple, set)):
        return obj
    if hasattr(obj, "values") and callable(getattr(obj, "values")):
        values = obj.values()
        if isinstance(values, dict):
            return values.items()
        return values
    return [obj]


def to_bars_df(obj: Any) -> pd.DataFrame:
    """Normalize a bars payload into a canonical DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]

    records: list[dict[str, Any]] | None = None

    if isinstance(obj, dict):
        records = []
        for key, value in obj.items():
            values_iter = (
                value
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))
                else [value]
            )
            for bar in values_iter:
                record = _object_to_dict(bar)
                if "S" not in record and "symbol" not in record:
                    record["symbol"] = key
                records.append(record)
    elif isinstance(obj, (list, tuple, set)):
        records = [_object_to_dict(item) for item in obj]
    elif hasattr(obj, "data"):
        data_attr = getattr(obj, "data")
        if isinstance(data_attr, dict):
            records = []
            for key, value in data_attr.items():
                values_iter = (
                    value
                    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))
                    else [value]
                )
                for bar in values_iter:
                    record = _object_to_dict(bar)
                    if "S" not in record and "symbol" not in record:
                        record["symbol"] = key
                    records.append(record)
        else:
            try:
                records = [_object_to_dict(item) for item in _maybe_iter(data_attr)]
            except Exception:  # pragma: no cover - defensive
                records = None

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif hasattr(obj, "df") and isinstance(getattr(obj, "df"), pd.DataFrame):
        df = getattr(obj, "df").copy()
    elif records is not None:
        df = pd.DataFrame(records)
    else:
        if obj is None:
            obj = []
        df = pd.DataFrame(obj)

    original_index_names = list(df.index.names) if hasattr(df.index, "names") else []
    if not df.index.equals(pd.RangeIndex(start=0, stop=len(df))):
        df = df.reset_index()
        rename_map: dict[str, str] = {}
        for name in original_index_names:
            if not name:
                continue
            name_str = str(name)
            if name_str.lower() == "symbol" and name_str in df.columns:
                rename_map[name_str] = "symbol"
            if name_str.lower() == "timestamp" and name_str in df.columns:
                rename_map[name_str] = "timestamp"
        if "symbol" not in rename_map and "level_0" in df.columns and "symbol" not in df.columns:
            rename_map["level_0"] = "symbol"
        if (
            "timestamp" not in rename_map
            and "level_1" in df.columns
            and "timestamp" not in df.columns
        ):
            rename_map["level_1"] = "timestamp"
        if rename_map:
            df = df.rename(columns=rename_map)
    df = df.rename(
        columns={
            "S": "symbol",
            "Symbol": "symbol",
            "t": "timestamp",
            "T": "timestamp",
            "time": "timestamp",
            "Time": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "symbol" not in df.columns:
        df = ensure_symbol_column(df)

    if "timestamp" not in df.columns:
        for column in df.columns:
            if column == "symbol":
                continue
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                df.rename(columns={column: "timestamp"}, inplace=True)
                break

    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    try:
        import pytz

        df["timestamp"] = df["timestamp"].dt.tz_convert(pytz.UTC)
        _ensure_utc_dtype_str_patch()
    except Exception:  # pragma: no cover - fallback
        pass
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    volume_series = pd.to_numeric(df["volume"], errors="coerce")
    if not volume_series.isna().all():
        df["volume"] = volume_series.round().astype("Int64")
    else:
        df["volume"] = volume_series.astype("Int64")

    return df[CANON].copy()


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
