from src.lgbm import feature_names, scores, train
from src.split import split_by_time

SMALL = 50  # tree cap so the training tests stay fast


def test_features_exclude_answer_id_and_time(df):
    feats = feature_names(df)
    assert feats
    for bad in ["isFraud", "TransactionID", "TransactionDT"]:
        assert bad not in feats


def test_fitted_only_on_train_rows(df):
    tr, val, test = split_by_time(df)
    model, feats, idx, _, _ = train(df, max_trees=SMALL)
    # the model's OWN record of how many rows it saw, not what our function reports
    seen = int(model.booster_.trees_to_dataframe().query("tree_index == 0").iloc[0]["count"])
    assert seen == len(tr)
    assert seen != len(tr) + len(val)
    # the table handed to fit is exactly the train block, disjoint from validation and test
    assert idx.equals(tr.index)
    assert not set(idx) & set(val.index)
    assert not set(idx) & set(test.index)
    # category labels the model stored are the train table's labels
    assert model.booster_.pandas_categorical == [
        list(tr[c].cat.categories) for c in feats if str(tr[c].dtype) == "category"
    ]


def test_same_seed_same_score(df):
    a = scores(*(lambda r: (r[0], r[3], r[4]))(train(df, seed=7, max_trees=SMALL)))
    b = scores(*(lambda r: (r[0], r[3], r[4]))(train(df, seed=7, max_trees=SMALL)))
    assert a == b
