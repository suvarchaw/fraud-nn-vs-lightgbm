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


@pytest.fixture
def one_thread(monkeypatch):
    """LightGBM and PyTorch each bring their own OpenMP (thread) library. After LightGBM has trained in this
    process, PyTorch on 4 threads deadlocks. One thread avoids it and is still a fixed count, so results still
    repeat. Real runs (python -m src.nn, python -m src.compare) are separate processes with 4 threads.
    Test files that train a network opt in with pytestmark."""
    import torch

    import src.nn as nnet
    monkeypatch.setattr(nnet, "THREADS", 1)
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def prepared(df):
    """The network's real pipeline, run once per test run: (prep, train data, validation data)."""
    import src.nn as nnet
    return nnet.prepare(df)


@pytest.fixture(scope="session")
def scrambled(df):
    """Copy of df with every test-block AND gap row scrambled: features, labels and 5 category columns get a new label.
    A Phase 4 trial or Phase 5 choice must come out exactly the same on it, proving nothing reads those rows."""
    from src.lgbm import feature_names
    from src.split import split_by_time
    tr, val, _ = split_by_time(df)
    out = df.drop(tr.index.union(val.index))  # test block plus both gaps
    d = df.copy()
    feats = feature_names(df)
    cat = [c for c in feats if str(d[c].dtype) == "category"]
    for c in feats:
        if c in cat[:5]:
            d[c] = d[c].cat.add_categories(["never-seen.example"])
            d.loc[out.index, c] = "never-seen.example"
        elif c not in cat:
            d.loc[out.index, c] = d.loc[out.index, c] * -7 + 3
    d.loc[out.index, "isFraud"] = 1 - d.loc[out.index, "isFraud"]
    return d
