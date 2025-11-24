import numpy as np
import pandas as pd

from src.models import NeuralNetworkWithEmbedding


def make_multiindex_df(n_dates=6, n_permnos=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    permnos = np.arange(1000, 1000 + n_permnos)

    rows = []
    for d in dates:
        for p in permnos:
            rows.append((d, p))

    index = pd.MultiIndex.from_tuples(rows, names=["date", "permno"])
    N = len(index)

    f1 = rng.normal(size=N)
    f2 = rng.normal(size=N)

    # permno effect so embedding can learn something
    perm_effect = np.repeat(np.linspace(-1.0, 1.0, n_permnos), n_dates)

    target = 0.6 * f1 - 0.3 * f2 + perm_effect + rng.normal(scale=0.1, size=N)

    df = pd.DataFrame({"f1": f1, "f2": f2, "retadj_next": target}, index=index)
    return df


def split_df(df):
    N = len(df)
    train_end = int(N * 0.5)
    val_end = int(N * 0.75)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def test_neural_network_with_embedding_basic():
    df = make_multiindex_df(n_dates=6, n_permnos=4, seed=42)
    train, val, test = split_df(df)

    hyperparameters = {
        "embedding_dim_list": [2],
        "hidden_units_list": [[8]],
        "learning_rate_list": [1e-3],
        "epochs": 5,
        "batch_size": 8,
    }

    model = NeuralNetworkWithEmbedding(train_data=train, val_data=val, test_data=test, hyperparameters=hyperparameters)

    # auto_tune should run quickly with the tiny grid and produce tuned_params
    res = model.auto_tune()
    assert isinstance(res, dict)
    assert model.tuned_params is not None

    # train (retrain on train+val)
    fitted = model.train()
    assert isinstance(fitted, dict)
    assert "model" in fitted and "scaler" in fitted

    # predict on test
    preds = model.predict("test")
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(test)

    # evaluate returns metrics
    metrics = model.evaluate("test")
    assert set(metrics.keys()) >= {"MSE", "MAE", "R2"}
