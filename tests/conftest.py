import pandas as pd
import pytest

from src.data import load_raw

KEY = ["TransactionID", "TransactionDT", "isFraud"]


def _fingerprint(df):
    return df.shape, list(df.columns), pd.util.hash_pandas_object(df[KEY], index=True).sum()


@pytest.fixture(scope="session")
def df():
    """Loaded once per run and shared, so READ-ONLY: a test that changes data must use df.copy()."""
    data = load_raw()
    before = _fingerprint(data)
    yield data
    assert _fingerprint(data) == before, "a test changed the shared df; use df.copy()"
