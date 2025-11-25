import numpy as np
import pandas as pd

from ml_backend.models import ElasticNet


def make_panel(n_samples=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # create simple linear target with noise
    coef = np.arange(1, n_features + 1)
    y = X.dot(coef) + rng.normal(scale=0.1, size=n_samples)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["retadj_next"] = y
    # simple time/index to simulate panel index
    df.index = pd.RangeIndex(start=0, stop=n_samples)
    return df


def test_elasticnet_end_to_end():
    # Create train/val/test splits
    df = make_panel(n_samples=150, n_features=4, seed=42)
    train_df = df.iloc[:100].copy()
    val_df = df.iloc[100:125].copy()
    test_df = df.iloc[125:].copy()

    # Define a tiny hyperparam grid so the unit test is fast
    hyperparams = {"alpha": [0.1, 1.0], "l1_ratio": [0.0, 0.5, 1.0]}

    en = ElasticNet(train_df=train_df, val_df=val_df, test_df=test_df, hyperparams=hyperparams)

    # Auto-tune should populate tuned_params and model
    en.auto_tune()
    assert isinstance(en.tuned_params, dict) and len(en.tuned_params) > 0
    assert en.model is not None

    # Train final model and check feature_columns recorded
    en.train_final()
    assert getattr(en, "feature_columns", None) is not None
    assert isinstance(en.feature_columns, list) and len(en.feature_columns) > 0

    # Predict and validate output
    preds = en.predict()
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(test_df)
    assert not preds.isna().any()

    # Predictions should be finite numbers
    assert np.isfinite(preds.values).all()
