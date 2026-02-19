import argparse
import json
from datetime import datetime, timedelta, timezone

from scripts import db


def _emit_summary(payload: dict) -> None:
    print(f"DB_VERIFY_SUMMARY {json.dumps(payload, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify recent monitor DB events")
    parser.add_argument("--since-minutes", type=int, default=60)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    since_minutes = max(1, int(args.since_minutes))
    limit = max(1, int(args.limit))
    since_ts = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)

    conn = db.get_db_conn()
    if conn is None:
        print("DB_VERIFY rc=2 reason=disabled_or_unreachable")
        return 2

    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM order_events
                    WHERE event_time >= %(since_ts)s
                    """,
                    {"since_ts": since_ts},
                )
                count_row = cursor.fetchone()
                event_count = int(count_row[0] if count_row else 0)

                cursor.execute(
                    """
                    SELECT event_type, symbol, qty, order_id, status, event_time
                    FROM order_events
                    WHERE event_time >= %(since_ts)s
                    ORDER BY event_time DESC
                    LIMIT %(limit)s
                    """,
                    {"since_ts": since_ts, "limit": limit},
                )
                rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT MAX(event_time)
                    FROM order_events
                    WHERE event_time >= now() - interval '24 hours'
                    """
                )
                max_row = cursor.fetchone()
                max_event_time = max_row[0] if max_row else None

                cursor.execute(
                    """
                    SELECT event_type, COUNT(*)
                    FROM order_events
                    WHERE event_time >= now() - interval '24 hours'
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                    """
                )
                counts_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT event_time, symbol, order_id, status, raw
                    FROM order_events
                    WHERE event_type = 'SELL_SUBMIT'
                    ORDER BY event_time DESC
                    LIMIT 3
                    """
                )
                recent_submit_rows = cursor.fetchall()

        events = []
        for row in rows:
            events.append(
                {
                    "event_type": row[0],
                    "symbol": row[1],
                    "qty": row[2],
                    "order_id": row[3],
                    "status": row[4],
                    "event_time": row[5].isoformat() if row[5] else None,
                }
            )

        counts_24h: dict[str, int] = {}
        for row in counts_rows:
            counts_24h[str(row[0])] = int(row[1])

        recent_submits = []
        for row in recent_submit_rows:
            raw_value = row[4]
            if isinstance(raw_value, str):
                try:
                    raw_value = json.loads(raw_value)
                except Exception:
                    raw_value = {}
            if not isinstance(raw_value, dict):
                raw_value = {}
            raw_keys = sorted(raw_value.keys())
            recent_submits.append(
                {
                    "event_time": row[0].isoformat() if row[0] else None,
                    "symbol": row[1],
                    "order_id": row[2],
                    "status": row[3],
                    "exit_reason_code_present": bool(raw_value.get("exit_reason_code")),
                    "raw_keys": raw_keys,
                }
            )

        summary = {
            "since_minutes": since_minutes,
            "order_events_count": event_count,
            "events": events,
            "max_event_time_24h": max_event_time.isoformat() if max_event_time else None,
            "counts_24h": counts_24h,
            "recent_sell_submit": recent_submits,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _emit_summary(summary)
        return 0
    except Exception as exc:
        print(f"DB_VERIFY rc=1 err={exc}")
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
