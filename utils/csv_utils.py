import os
import csv
from typing import Union, Optional

import pandas as pd
from atomicwrites import atomic_write


def write_csv_atomic(
    path: str,
    rows: Union[pd.DataFrame, list[dict]],
    fieldnames: Optional[list[str]] = None,
) -> None:
    """Atomically writes rows (list of dicts) to a CSV file.

    Parameters:
    - path (str): Full path to the CSV file.
    - rows (list[dict] | pandas.DataFrame): Data to write.
    - fieldnames (list[str], optional): List of CSV headers. If None, inferred from rows[0].
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if isinstance(rows, pd.DataFrame):
        with atomic_write(path, overwrite=True, newline="", encoding="utf-8") as f:
            rows.to_csv(f, index=False)
        return

    if not rows:
        if fieldnames is None:
            raise ValueError("fieldnames must be provided when rows is empty")
        with atomic_write(path, overwrite=True, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with atomic_write(path, overwrite=True, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
