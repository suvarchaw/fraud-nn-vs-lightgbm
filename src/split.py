"""Time-based train / validation / test split. Selects rows only: no fitting, scaling or encoding.

The test set is never used for tuning. Every choice in later phases is made on validation.
"""
import pandas as pd

DAY = 86_400  # seconds

# The only place split boundaries live. Other code imports these and never repeats the numbers.
DT_START = 86_400       # first TransactionDT in the data (checked by a test)
DT_END = 15_811_131     # last TransactionDT in the data (checked by a test)
TRAIN_END_FRAC = 0.60   # train ends at 60% of the time range
VAL_END_FRAC = 0.80     # validation ends at 80% of the time range
GAP_DAYS = 7            # cut from the start of validation and of test

_span = DT_END - DT_START
_train_end = DT_START + round(TRAIN_END_FRAC * _span)
_val_end = DT_START + round(VAL_END_FRAC * _span)
BOUNDS = {
    "train_end": _train_end,
    "val_start": _train_end + GAP_DAYS * DAY,
    "val_end": _val_end,
    "test_start": _val_end + GAP_DAYS * DAY,
}


def split_by_time(df):
    """Return (train, val, test) as new tables. Half-open blocks; rows in the gaps are dropped."""
    b = BOUNDS  # read at call time, so a changed BOUNDS is always used
    t = df["TransactionDT"]
    train = df[t < b["train_end"]]
    val = df[(t >= b["val_start"]) & (t < b["val_end"])]
    test = df[t >= b["test_start"]]
    return train, val, test


def summary(df):
    """Aggregates only: rows, shares, day range and fraud rate per set, plus the gaps."""
    sets = dict(zip(["train", "val", "test"], split_by_time(df)))
    rows = []
    for name, s in sets.items():
        days = (s["TransactionDT"] - DT_START) / DAY
        rows.append({
            "set": name,
            "rows": len(s),
            "row_share_%": 100 * len(s) / len(df),
            "first_day": days.min(),
            "last_day": days.max(),
            "span_days": days.max() - days.min(),
            "time_share_%": 100 * (days.max() - days.min()) * DAY / _span,
            "fraud_rate_%": 100 * s["isFraud"].mean(),
        })
    lost = len(df) - sum(len(s) for s in sets.values())
    rows.append({
        "set": "gaps",
        "rows": lost,
        "row_share_%": 100 * lost / len(df),
        "span_days": 2 * GAP_DAYS,
        "time_share_%": 100 * 2 * GAP_DAYS * DAY / _span,
    })
    return pd.DataFrame(rows).set_index("set").round(2)


if __name__ == "__main__":
    from src.data import load_raw

    print(summary(load_raw()).to_string())
