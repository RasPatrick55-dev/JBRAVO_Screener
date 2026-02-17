import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

try:
    from scripts import db
except Exception:  # pragma: no cover - optional dependency
    db = None  # type: ignore
DEFAULT_OUTPUT = BASE_DIR / "data" / "pythonanywhere_usage.json"
DEFAULT_API_BASE = "https://www.pythonanywhere.com/api/v0/user"


def _pythonanywhere_token() -> Optional[str]:
    return os.getenv("PYTHONANYWHERE_API_TOKEN") or os.getenv("API_TOKEN")


def _pythonanywhere_username() -> Optional[str]:
    return os.getenv("PYTHONANYWHERE_USERNAME") or os.getenv("PYTHONANYWHERE_USER")


def _pythonanywhere_api_base() -> str:
    return (os.getenv("PYTHONANYWHERE_API_BASE_URL") or DEFAULT_API_BASE).rstrip("/")


def _pythonanywhere_api_get_json(path: str) -> Optional[dict]:
    token = _pythonanywhere_token()
    username = _pythonanywhere_username()
    if not token or not username:
        return None
    cleaned = path.lstrip("/")
    url = f"{_pythonanywhere_api_base()}/{username}/{cleaned}"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None


def pythonanywhere_cpu_usage() -> Optional[dict]:
    payload = _pythonanywhere_api_get_json("cpu/")
    if not isinstance(payload, dict):
        return None
    used = payload.get("daily_cpu_total_usage_seconds")
    limit = payload.get("daily_cpu_limit_seconds")
    try:
        used_val = float(used) if used is not None else None
        limit_val = float(limit) if limit is not None else None
    except (TypeError, ValueError):
        used_val = None
        limit_val = None
    percent = percent_used(used_val, limit_val)
    return {
        "used_seconds": used_val,
        "limit_seconds": limit_val,
        "percent": round(percent, 2) if percent is not None else None,
        "next_reset_time": payload.get("next_reset_time"),
        "source": "pythonanywhere",
    }


def parse_size_to_bytes(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    number = ""
    unit = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        else:
            unit += char
    if not number:
        return None
    try:
        numeric = float(number)
    except ValueError:
        return None
    unit = unit.strip() or "b"
    factors = {
        "b": 1,
        "kb": 1024,
        "k": 1024,
        "mb": 1024**2,
        "m": 1024**2,
        "gb": 1024**3,
        "g": 1024**3,
        "tb": 1024**4,
        "t": 1024**4,
    }
    factor = factors.get(unit, 1)
    return int(numeric * factor)


def percent_used(used: Optional[float], limit: Optional[float]) -> Optional[float]:
    if used is None or limit is None or limit <= 0:
        return None
    return max(0.0, min(100.0, (used / limit) * 100))


def directory_size_bytes(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        output = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(output.split()[0])
    except Exception:
        total = 0
        try:
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
        except Exception:
            return None
        return total


def get_used_bytes_via_du() -> int:
    try:
        output = subprocess.check_output(
            [
                "bash",
                "-lc",
                "du -s -B 1 /tmp ~/.[!.]* ~/* /var/www/ 2>/dev/null | awk '{s+=$1}END{print s+0}'",
            ],
            text=True,
        )
        return int(output.strip() or "0")
    except Exception:
        return 0


def pythonanywhere_file_quota_bytes() -> int:
    try:
        quota_gb = float(os.getenv("PYTHONANYWHERE_DISK_QUOTA_GB", "10"))
    except (TypeError, ValueError):
        quota_gb = 10.0
    if quota_gb <= 0:
        quota_gb = 10.0
    return int(quota_gb * (1024**3))


def pythonanywhere_postgres_quota_gb() -> float:
    try:
        quota_gb = float(os.getenv("PYTHONANYWHERE_POSTGRES_QUOTA_GB", "10"))
    except (TypeError, ValueError):
        quota_gb = 10.0
    if quota_gb <= 0:
        quota_gb = 10.0
    return quota_gb


def pythonanywhere_postgres_quota_bytes() -> int:
    override = os.getenv("PYTHONANYWHERE_POSTGRES_QUOTA_BYTES")
    if override is not None:
        try:
            quota_bytes = int(override)
            if quota_bytes > 0:
                return quota_bytes
        except (TypeError, ValueError):
            pass
    return int(pythonanywhere_postgres_quota_gb() * (1024**3))


def pythonanywhere_postgres_storage_percent(used_bytes: int, quota_bytes: int) -> int:
    if quota_bytes <= 0:
        return 0
    percent = math.floor(100 * used_bytes / quota_bytes)
    return max(0, min(100, percent))


def pythonanywhere_file_storage_percent(used_bytes: int, quota_bytes: int) -> int:
    if quota_bytes <= 0:
        return 0
    percent = round(100 * used_bytes / quota_bytes)
    return max(0, min(100, percent))


def storage_used_bytes(storage_path: Path, mode: str) -> Optional[int]:
    if mode == "pythonanywhere":
        return get_used_bytes_via_du()
    return directory_size_bytes(storage_path)


def _postgres_wal_bytes(conn) -> Optional[int]:
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COALESCE(SUM(size), 0) FROM pg_ls_waldir()")
            row = cursor.fetchone()
            if not row:
                return None
            return int(row[0]) if row[0] is not None else None
    except Exception:
        return None


def postgres_size_bytes(mode: str = "database") -> Optional[int]:
    if db is None:
        return None
    conn = db.get_db_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            if mode in {"total", "total_wal"}:
                cursor.execute("SELECT SUM(pg_database_size(datname)) AS size_bytes FROM pg_database")
            else:
                cursor.execute("SELECT pg_database_size(current_database()) AS size_bytes")
            row = cursor.fetchone()
            if not row:
                return None
            base_size = int(row[0]) if row[0] is not None else None
            if base_size is None:
                return None
            if mode in {"database_wal", "total_wal"}:
                wal_bytes = _postgres_wal_bytes(conn) or 0
                return base_size + wal_bytes
            return base_size
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_pg_used_bytes_and_pretty(database_url: Optional[str]) -> tuple[int, str]:
    if db is None:
        raise RuntimeError("DB module unavailable")
    if database_url:
        os.environ["DATABASE_URL"] = database_url
    conn = db.get_db_conn()
    if conn is None:
        raise RuntimeError("Could not connect to Postgres")
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_database_size(current_database()) AS used_bytes,
                       pg_size_pretty(pg_database_size(current_database())) AS used_pretty
                """
            )
            row = cursor.fetchone()
            if not row or row[0] is None:
                raise RuntimeError("Postgres size query returned no data")
            return int(row[0]), str(row[1] or "")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_payload(
    storage_path: Path,
    storage_limit: Optional[str],
    postgres_limit: Optional[str],
    storage_mode: str,
    postgres_mode: str,
) -> dict:
    cpu_payload = pythonanywhere_cpu_usage()
    used_bytes = storage_used_bytes(storage_path, storage_mode)
    if storage_mode == "pythonanywhere":
        quota_bytes = pythonanywhere_file_quota_bytes()
        used = int(used_bytes or 0)
        storage_percent = pythonanywhere_file_storage_percent(used, quota_bytes)
        limit_bytes = quota_bytes
    else:
        limit_bytes = parse_size_to_bytes(storage_limit)
        storage_percent = percent_used(
            float(used_bytes) if used_bytes is not None else None,
            float(limit_bytes) if limit_bytes is not None else None,
        )

    pg_used_bytes: Optional[int] = None
    pg_used_pretty: Optional[str] = None
    pg_error: Optional[str] = None
    try:
        pg_used_bytes, pg_used_pretty = get_pg_used_bytes_and_pretty(os.getenv("DATABASE_URL"))
    except Exception as exc:
        pg_error = str(exc)

    pg_quota_bytes = pythonanywhere_postgres_quota_bytes()
    pg_quota_gb = pythonanywhere_postgres_quota_gb()
    pg_percent = (
        pythonanywhere_postgres_storage_percent(pg_used_bytes, pg_quota_bytes)
        if pg_used_bytes is not None
        else None
    )

    return {
        "cpu": cpu_payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_storage": {
            "used_bytes": used_bytes,
            "limit_bytes": limit_bytes,
            "percent": round(storage_percent, 2) if isinstance(storage_percent, float) else storage_percent,
            "file_used_bytes": int(used_bytes or 0),
            "file_quota_bytes": int(limit_bytes or 0),
            "file_storage_percent": int(storage_percent or 0),
            "file_used_gib": round((int(used_bytes or 0) / (1024**3)), 3),
            "path": str(storage_path),
            "mode": storage_mode,
        },
        "postgres_storage": {
            "used_bytes": pg_used_bytes,
            "limit_bytes": pg_quota_bytes,
            "percent": pg_percent,
            "pg_used_bytes": pg_used_bytes,
            "pg_quota_bytes": pg_quota_bytes,
            "pg_storage_percent": pg_percent,
            "pg_used_pretty": pg_used_pretty,
            "pg_used_gib": round((pg_used_bytes / (1024**3)), 3) if pg_used_bytes is not None else None,
            "pg_quota_gb": pg_quota_gb,
            "pg_error": pg_error,
            "mode": postgres_mode,
        },
        "version": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture PythonAnywhere resource usage for the dashboard."
    )
    parser.add_argument(
        "--output",
        default=os.getenv("PYTHONANYWHERE_USAGE_OUTPUT") or str(DEFAULT_OUTPUT),
        help="Path to write the JSON usage snapshot.",
    )
    parser.add_argument(
        "--storage-path",
        default=os.getenv("PYTHONANYWHERE_STORAGE_PATH") or os.getenv("HOME") or str(BASE_DIR),
        help="Path to measure for file storage usage.",
    )
    parser.add_argument(
        "--storage-mode",
        default=os.getenv("PYTHONANYWHERE_STORAGE_MODE")
        or ("pythonanywhere" if os.getenv("PYTHONANYWHERE_DOMAIN") or os.getenv("PYTHONANYWHERE_API_TOKEN") else "path"),
        choices=["path", "pythonanywhere"],
        help="Storage usage mode: 'path' (default) or 'pythonanywhere' quota calculation.",
    )
    parser.add_argument(
        "--storage-limit",
        default=os.getenv("PYTHONANYWHERE_STORAGE_LIMIT_BYTES")
        or os.getenv("PYTHONANYWHERE_STORAGE_LIMIT"),
        help="Optional file storage limit (bytes or 1g/512m).",
    )
    parser.add_argument(
        "--postgres-limit",
        default=os.getenv("PYTHONANYWHERE_POSTGRES_LIMIT_BYTES")
        or os.getenv("PYTHONANYWHERE_POSTGRES_LIMIT"),
        help="Optional Postgres storage limit (bytes or 1g/512m).",
    )
    parser.add_argument(
        "--postgres-mode",
        default=os.getenv("PYTHONANYWHERE_POSTGRES_MODE") or "database",
        choices=["database", "total", "database_wal", "total_wal"],
        help=(
            "Postgres usage mode: 'database' (current DB), 'total' (all DBs), "
            "'database_wal' (current DB + WAL), or 'total_wal' (all DBs + WAL)."
        ),
    )

    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        storage_path=Path(args.storage_path).expanduser(),
        storage_limit=args.storage_limit,
        postgres_limit=args.postgres_limit,
        storage_mode=args.storage_mode,
        postgres_mode=args.postgres_mode,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
