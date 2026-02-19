"""Deprecated wrapper around :mod:`scripts.execute_trades`.

The "enhanced" entry point previously contained an alternate executor
implementation.  The modern flow now lives in ``scripts.execute_trades``.
This shim remains to preserve backwards compatibility for downstream
orchestration while clearly signalling the deprecation.
"""

from __future__ import annotations

import sys
import warnings
from typing import Iterable, Optional

from . import execute_trades


def main(argv: Optional[Iterable[str]] = None) -> int:
    warnings.warn(
        "scripts.enhanced_execute_trades is deprecated; use scripts.execute_trades instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return execute_trades.main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
