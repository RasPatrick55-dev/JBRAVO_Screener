import argparse
import json
import os
import subprocess
import sys
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


def postgres_size_bytes() -> Optional[int]:
    if db is None:
        return None
    conn = db.get_db_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT pg_database_size(current_database()) AS size_bytes")
            row = cursor.fetchone()
            if not row:
                return None
            return int(row[0]) if row[0] is not None else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_payload(storage_path: Path, storage_limit: Optional[str], postgres_limit: Optional[str]) -> dict:
    used_bytes = directory_size_bytes(storage_path)
    limit_bytes = parse_size_to_bytes(storage_limit)
    storage_percent = percent_used(
        float(used_bytes) if used_bytes is not None else None,
        float(limit_bytes) if limit_bytes is not None else None,
    )

    pg_used_bytes = postgres_size_bytes()
    pg_limit_bytes = parse_size_to_bytes(postgres_limit)
    pg_percent = percent_used(
        float(pg_used_bytes) if pg_used_bytes is not None else None,
        float(pg_limit_bytes) if pg_limit_bytes is not None else None,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_storage": {
            "used_bytes": used_bytes,
            "limit_bytes": limit_bytes,
            "percent": round(storage_percent, 2) if storage_percent is not None else None,
            "path": str(storage_path),
        },
        "postgres_storage": {
            "used_bytes": pg_used_bytes,
            "limit_bytes": pg_limit_bytes,
            "percent": round(pg_percent, 2) if pg_percent is not None else None,
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

    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        storage_path=Path(args.storage_path).expanduser(),
        storage_limit=args.storage_limit,
        postgres_limit=args.postgres_limit,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
