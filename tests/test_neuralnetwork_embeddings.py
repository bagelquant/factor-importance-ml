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


def test_extract_embedding_df_after_auto_tune_and_train_final():
    """_extract_embedding_df should return a DataFrame indexed by permno.

    We validate that after auto_tune and train_final the helper has
    produced non-empty DataFrames whose indices match the permnos seen
    during training/combined train+val.
    """
    train = make_panel_data(n_rows=40, n_features=3, seed=10)
    val = make_panel_data(n_rows=20, n_features=3, seed=11)
    test = make_panel_data(n_rows=10, n_features=3, seed=12)

    hyperparams = {
        "embedding_dims": [4],
        "hidden_layers": [1],
        "neurons_per_layer": [8],
        "activation_functions": ["relu"],
        "learning_rates": [0.01],
        "batch_sizes": [8],
    }

    nn = NeuralNetworkWithEmbeddings(train_df=train, val_df=val, test_df=test, hyperparams=hyperparams)

    nn.auto_tune()
    # best_train_embedding_vectors should be set and indexed by permno
    assert nn.best_train_embedding_vectors is not None
    emb_df = nn.best_train_embedding_vectors
    assert isinstance(emb_df, pd.DataFrame)
    assert emb_df.index.name == "permno"
    # All permnos from training should be present in the index
    train_permnos = pd.Index(train.index.get_level_values("permno")).unique()
    assert set(train_permnos).issubset(set(emb_df.index))

    # Now train_final and check combined_train_val_embedding_vectors
    nn.tuned_params["epochs"] = 2
    nn.train_final()
    assert nn.combined_train_val_embedding_vectors is not None
    comb_emb_df = nn.combined_train_val_embedding_vectors
    assert isinstance(comb_emb_df, pd.DataFrame)
    assert comb_emb_df.index.name == "permno"
    combined = pd.concat([train, val], axis=0)
    combined_permnos = pd.Index(combined.index.get_level_values("permno")).unique()
    assert set(combined_permnos).issubset(set(comb_emb_df.index))
