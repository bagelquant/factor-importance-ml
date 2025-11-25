import numpy as np
import pandas as pd
from ml_backend.models import NeuralNetwork


def make_panel_data(n_rows=100, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_rows, n_features))
    # simple linear target with noise
    coefs = rng.normal(size=(n_features,))
    y = X.dot(coefs) + rng.normal(scale=0.1, size=(n_rows,))
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["retadj_next"] = y
    # create a simple panel index (date, permno)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    permnos = np.arange(n_rows) % 10
    df.index = pd.MultiIndex.from_arrays([dates, permnos], names=("date", "permno"))
    return df


def test_neuralnetwork_basic_flow():
    # small dataset
    train = make_panel_data(n_rows=60, n_features=4, seed=1)
    val = make_panel_data(n_rows=20, n_features=4, seed=2)
    test = make_panel_data(n_rows=20, n_features=4, seed=3)

    # Minimal hyperparameter grid for quick testing
    hyperparams = {
        "hidden_layers": [1],
        "neurons_per_layer": [8],
        "activation_functions": ["relu"],
        "learning_rates": [0.01],
        "batch_sizes": [8],
        "epochs": [2],
    }

    nn = NeuralNetwork(train_df=train, val_df=val, test_df=test, hyperparams=hyperparams)

    # Auto-tune should run without error and set tuned_params/model
    nn.auto_tune()
    assert isinstance(nn.tuned_params, dict) and len(nn.tuned_params) > 0
    assert nn.model is not None

    # Train final should run and produce a model
    nn.train_final()
    assert nn.model is not None

    # Predict should return a pd.Series of the same length as test
    preds = nn.predict()
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(test)
    # predictions should have the same index as test
    assert preds.index.equals(test.index)

