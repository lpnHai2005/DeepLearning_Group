from src.run_experiment import expand


def test_expand_vary_is_cartesian():
    cfg = {"base": {"lr": 0.1}, "vary": {"n_train": [200, 400], "activation": ["tanh", "relu"]}}
    names = [n for n, _ in expand(cfg)]
    assert names == ["200_tanh", "200_relu", "400_tanh", "400_relu"]


def test_expand_cases_change_keys_together():
    cfg = {"base": {"lr": 0.1, "patience": 200},
           "cases": [{"n_train": 200, "max_epochs": 9143, "patience": 914},
                     {"n_train": 1000, "max_epochs": 2000, "patience": 200}]}
    out = list(expand(cfg))
    assert [n for n, _ in out] == ["200", "1000"]
    assert out[0][1] == {"lr": 0.1, "n_train": 200, "max_epochs": 9143, "patience": 914}
