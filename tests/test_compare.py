"""Phase 4 checks that train nothing. Training checks live in each model's own test file (test_lgbm.py,
test_nn.py): once PyTorch has started its thread pool, LightGBM training in the same process crashes."""
import src.compare as cmp


def test_equal_trials_same_method_values_in_range():
    assert len({len(s) for s in cmp.SPACE.values()}) == 1  # same number of knobs
    for m, space in cmp.SPACE.items():
        cfgs = cmp.sample_configs(m)
        assert len(cfgs) == cmp.N_TRIALS
        assert cfgs[0] == cmp.BASELINE[m]
        assert cfgs == cmp.sample_configs(m)  # fixed search seed: repeatable
        for cfg in cfgs[1:]:
            assert cfg.keys() == space.keys()
            for k, spec in space.items():
                assert cfg[k] in spec if isinstance(spec, list) else spec[1] <= cfg[k] <= spec[2], (m, k)


def test_final_seeds_distinct():
    assert len(cmp.SEEDS) >= 5
    assert len(set(cmp.SEEDS)) == len(cmp.SEEDS)
    assert cmp.TUNE_SEED not in cmp.SEEDS


def test_verdict_rule():
    assert cmp.verdict([0.91, 0.92], [0.93, 0.94]) == "flipped"
    assert cmp.verdict([0.90, 0.92], [0.91, 0.93]) == "gap closed"     # ranges overlap
    assert cmp.verdict([0.920, 0.921], [0.916, 0.917]) == "gap closed"  # no overlap, mean gap 0.004
    assert cmp.verdict([0.920, 0.921], [0.88, 0.89]) == "gap stayed"
