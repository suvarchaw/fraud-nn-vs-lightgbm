from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
KEEP_INT64 = ["TransactionID", "TransactionDT"]  # never turned into floats


def _mb(df):
    return df.memory_usage(deep=True).sum() / 1e6


def load_raw(verbose=True):
    """Load transactions, left-join identity, shrink memory. No splitting/scaling/encoding."""
    tx = pd.read_csv(RAW / "train_transaction.csv")
    idn = pd.read_csv(RAW / "train_identity.csv")

    assert idn["TransactionID"].is_unique, "TransactionID not unique in identity file"
    n_tx = len(tx)

    df = tx.merge(idn, on="TransactionID", how="left")
    assert len(df) == n_tx, f"join changed row count: {n_tx} -> {len(df)}"
    del tx, idn

    before = _mb(df)
    for c in df.columns:
        if c in KEEP_INT64:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int32")
        elif df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("category")
    if verbose:
        print(f"memory: {before:.0f} MB -> {_mb(df):.0f} MB")
    return df
