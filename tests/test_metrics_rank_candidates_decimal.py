from decimal import Decimal

import pandas as pd

from scripts import metrics


def test_rank_candidates_handles_decimals() -> None:
    df_decimal = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "win_rate": Decimal("60.0"),
                "net_pnl": Decimal("100.0"),
                "trades": Decimal("10"),
            },
            {
                "symbol": "BBB",
                "win_rate": Decimal("50.0"),
                "net_pnl": Decimal("200.0"),
                "trades": Decimal("20"),
            },
            {
                "symbol": "CCC",
                "win_rate": Decimal("40.0"),
                "net_pnl": Decimal("50.0"),
                "trades": Decimal("5"),
            },
        ]
    )

    df_float = df_decimal.copy()
    for col in ("win_rate", "net_pnl", "trades"):
        df_float[col] = df_float[col].astype(float)

    ranked_decimal = metrics.rank_candidates(df_decimal.copy())
    ranked_float = metrics.rank_candidates(df_float.copy())

    assert ranked_decimal["symbol"].tolist() == ranked_float["symbol"].tolist()
