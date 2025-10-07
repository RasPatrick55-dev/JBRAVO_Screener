import os
import pytest
from alpaca.data.timeframe import TimeFrame


def _tf_eq(self, other):
    return (
        isinstance(other, TimeFrame)
        and self.amount_value == other.amount_value
        and self.unit_value == other.unit_value
    )

TimeFrame.__eq__ = _tf_eq

@pytest.fixture(autouse=True)
def check_alpaca_env(request):
    if request.node.get_closest_marker("alpaca_optional"):
        return
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        pytest.skip("Skipping Alpaca-dependent tests due to missing credentials")
