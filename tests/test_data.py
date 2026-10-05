import pandas as pd

from src.data import RAW


def test_identity_transaction_id_unique():
    ids = pd.read_csv(RAW / "train_identity.csv", usecols=["TransactionID"])
    assert ids["TransactionID"].is_unique


def test_row_count_matches_transaction_file(df):
    expected = len(pd.read_csv(RAW / "train_transaction.csv", usecols=["TransactionID"]))
    assert len(df) == expected


def test_key_columns_are_integers(df):
    for c in ["TransactionID", "TransactionDT", "isFraud"]:
        assert pd.api.types.is_integer_dtype(df[c]), c


def test_no_object_columns_left(df):
    assert not [c for c in df.columns if df[c].dtype == object]
    assert str(df["ProductCD"].dtype) == "category"
