import os
import csv
from atomicwrites import atomic_write


def write_csv_atomic(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Atomically writes rows (list of dicts) to a CSV file.

    Parameters:
    - path (str): Full path to the CSV file.
    - rows (list[dict]): List of rows, each row being a dictionary.
    - fieldnames (list[str], optional): List of CSV headers. If None, inferred from rows[0].
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if not rows:
        if fieldnames is None:
            raise ValueError("fieldnames must be provided when rows is empty")
        with atomic_write(path, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with atomic_write(path, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
