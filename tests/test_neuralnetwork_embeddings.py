import numpy as np
import pandas as pd
from ml_backend.models import NeuralNetworkWithEmbeddings


def make_panel_data(n_rows=100, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_rows, n_features))
    # simple linear target with noise
    coefs = rng.normal(size=(n_features,))
    y = X.dot(coefs) + rng.normal(scale=0.1, size=(n_rows,))
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["retadj_next"] = y
    # create a panel index with multiple firms per date
    num_permnos = 10
    num_dates = max(1, n_rows // num_permnos)
    dates = np.repeat(pd.date_range("2020-01-01", periods=num_dates, freq="D"), num_permnos)
    permnos = np.tile(np.arange(num_permnos), num_dates)
    # trim to requested length
    dates = dates[:n_rows]
    permnos = permnos[:n_rows]
    df.index = pd.MultiIndex.from_arrays([dates, permnos], names=("date", "permno"))
    return df


def test_neuralnetwork_embeddings_basic_flow():
    # small dataset
    train = make_panel_data(n_rows=60, n_features=4, seed=1)
    val = make_panel_data(n_rows=20, n_features=4, seed=2)
    test = make_panel_data(n_rows=20, n_features=4, seed=3)

    # Minimal hyperparameter grid for quick testing
    hyperparams = {
        "embedding_dims": [4],
        "hidden_layers": [1],
        "neurons_per_layer": [8],
        "activation_functions": ["relu"],
        "learning_rates": [0.01],
        "batch_sizes": [8],
    }

    nn = NeuralNetworkWithEmbeddings(train_df=train, val_df=val, test_df=test, hyperparams=hyperparams)

    # Auto-tune should run without error and set tuned_params/model
    nn.auto_tune()
    assert isinstance(nn.tuned_params, dict) and len(nn.tuned_params) > 0
    assert nn.model is not None

    # Ensure permno mapping was created during tuning
    assert hasattr(nn, "_permno_mapping") and isinstance(nn._permno_mapping, dict)
    assert len(nn._permno_mapping) > 0

    # Make final training quick by injecting a small epoch value then train
    nn.tuned_params["epochs"] = 2
    nn.train_final()
    assert nn.model is not None

    # Predict should return a pd.Series of the same length as test
    preds = nn.predict()
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(test)
    # predictions should have the same index as test
    assert preds.index.equals(test.index)

    # peer-weighted features using embedding distances (new API: no args)
    peer_df = nn.distance_weighted_feature()
    assert isinstance(peer_df, pd.DataFrame)
    # output index should correspond to train + val concatenation (we compute on train+val)
    combined = pd.concat([train, val], axis=0)
    assert peer_df.index.equals(combined.index)
    # expect the target column to be present and have at least one non-NaN value
    assert "retadj_next" in peer_df.columns
    assert peer_df["retadj_next"].notna().any()
