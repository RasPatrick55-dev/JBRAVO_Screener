"""Pure position metadata and CSV schema shared by all position writers.

No clients, environment loading, orders or persistence at import time.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import math
import os
import time
import types
import pandas as pd


@contextmanager
def position_csv_lock(path):
    """Serialize each writer's metadata read/build/write; kernel releases on exit."""
    with open(str(path) + '.lock', 'a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'\0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        deadline = time.monotonic() + 10
        while True:
            try:
                acquire(); break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('position_csv_writer_busy') from None
                time.sleep(.05)
        try:
            yield
        finally:
            release()


def _position_side(position):
    value = str(getattr(getattr(position, 'side', ''), 'value', getattr(position, 'side', ''))).lower()
    if value in {'long', 'short'}:
        return value
    try:
        return 'short' if float(getattr(position, 'qty', 0)) < 0 else 'long'
    except (TypeError, ValueError):
        return 'long'


def _positive_accounting_number(value, *, absolute=False):
    if isinstance(value, bool):
        return None
    try:
        number = abs(float(value)) if absolute else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _observed_entry_time(value):
    try:
        stamp = pd.to_datetime(value, errors='coerce')
        if pd.isna(stamp) or getattr(stamp, 'tzinfo', None) is None:
            return None
        return stamp.tz_convert('UTC').isoformat()
    except (TypeError, ValueError, AttributeError):
        return None


def _accounted_exit_ids(record):
    value = record.get('accounted_exit_order_ids')
    if value is None or (not isinstance(value, (str, list)) and pd.isna(value)):
        return set()
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, list) or len(value) > 128
            or any(not isinstance(item, str) or not item or len(item) > 128 for item in value)):
        raise ValueError('position_exit_identity_unverified')
    return set(value)


def _position_closure_record(position, retained):
    """Use current basis; retain old time only for a continuing basis and side."""
    basis = _positive_accounting_number(getattr(position, 'avg_entry_price', None))
    side = _position_side(position)
    entry_time = (_observed_entry_time(getattr(position, 'entry_time', None))
                  or _observed_entry_time(getattr(position, 'created_at', None)))
    prior = retained[retained['symbol'] == position.symbol] if 'symbol' in retained else pd.DataFrame()
    accounted_ids = set()
    if basis is not None and len(prior) == 1:
        previous = prior.iloc[0]
        old_basis = _positive_accounting_number(previous.get('entry_price'))
        old_side = _position_side(types.SimpleNamespace(**previous.to_dict()))
        if old_basis is not None and math.isclose(basis, old_basis) and side == old_side:
            previous_time = _observed_entry_time(previous.get('entry_time'))
            if entry_time is None:
                entry_time = previous_time
            if previous_time is None or entry_time == previous_time:
                accounted_ids = _accounted_exit_ids(previous)
    return {'symbol': position.symbol, 'qty': position.qty, 'side': side,
            'entry_price': basis, 'entry_time': entry_time,
            'accounted_exit_order_ids': json.dumps(sorted(accounted_ids))}


POSITION_COLUMNS = (
    'symbol', 'qty', 'avg_entry_price', 'entry_price', 'current_price',
    'unrealized_pl', 'net_pnl', 'entry_time', 'days_in_trade', 'side',
    'order_status', 'pnl', 'order_type', 'accounted_exit_order_ids',
)


def position_frame(positions, retained):
    """Build the shared monitor/dashboard schema using current broker metadata."""
    rows = []
    for position in positions or []:
        row = dict(vars(position))
        row.update(_position_closure_record(position, retained))
        row.update(net_pnl=getattr(position, 'unrealized_pl', None),
                   pnl=getattr(position, 'unrealized_pl', None),
                   order_status='open', order_type=getattr(position, 'order_type', 'limit'))
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in POSITION_COLUMNS:
        if column not in frame:
            frame[column] = None
    # Parse each aware timestamp independently; mixed fractional precision is valid.
    frame['entry_time'] = frame['entry_time'].map(_observed_entry_time)
    stamps = pd.to_datetime(frame['entry_time'], utc=True, errors='coerce', format='mixed')
    frame['days_in_trade'] = (datetime.now(timezone.utc) - stamps).dt.days
    for column in ('qty', 'avg_entry_price', 'entry_price', 'current_price', 'unrealized_pl', 'net_pnl', 'pnl'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    # Broker objects can contain nested metadata/UUIDs that SQLite cannot store.
    return frame[list(POSITION_COLUMNS)]
