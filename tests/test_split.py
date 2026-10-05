from src import split
from src.split import DAY, GAP_DAYS, split_by_time


def ids(s):
    return set(s["TransactionID"])


def test_no_row_in_two_sets(df):
    train, val, test = split_by_time(df)
    assert not ids(train) & ids(val)
    assert not ids(train) & ids(test)
    assert not ids(val) & ids(test)


def test_time_order_with_gap(df):
    assert df["TransactionDT"].min() == split.DT_START
    assert df["TransactionDT"].max() == split.DT_END
    train, val, test = (s["TransactionDT"] for s in split_by_time(df))
    assert val.min() - train.max() >= GAP_DAYS * DAY
    assert test.min() - val.max() >= GAP_DAYS * DAY


def test_only_gap_rows_lost(df):
    b = split.BOUNDS
    t = df["TransactionDT"]
    in_gap = ((t >= b["train_end"]) & (t < b["val_start"])) | ((t >= b["val_end"]) & (t < b["test_start"]))
    sets = split_by_time(df)
    assert sum(len(s) for s in sets) + in_gap.sum() == len(df)
    assert in_gap.sum() > 0
    for s in sets:  # kept rows are untouched originals: nothing scaled or encoded
        assert s.equals(df.loc[s.index])


def test_boundaries_live_in_bounds(df, monkeypatch):
    shifted = {k: v + 3 * DAY for k, v in split.BOUNDS.items()}
    monkeypatch.setattr(split, "BOUNDS", shifted)
    t = df["TransactionDT"]
    expected = [
        df[t < shifted["train_end"]],
        df[(t >= shifted["val_start"]) & (t < shifted["val_end"])],
        df[t >= shifted["test_start"]],
    ]
    for got, want in zip(split_by_time(df), expected):
        assert ids(got) == ids(want)
