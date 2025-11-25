import numpy as np
import pandas as pd

from ml_backend.models import GradientBoostingTree


def make_panel(n_samples=200, n_features=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = np.arange(1, n_features + 1)
    y = X.dot(coef) + rng.normal(scale=0.1, size=n_samples)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["retadj_next"] = y
    df.index = pd.RangeIndex(start=0, stop=n_samples)
    return df


def test_gradient_boosting_end_to_end():
    df = make_panel(n_samples=150, n_features=4, seed=42)
    train_df = df.iloc[:100].copy()
    val_df = df.iloc[100:125].copy()
    test_df = df.iloc[125:].copy()

    # Minimal hyperparameter grid for a quick unit test
    hyperparams = {"n_leaves": [3, 7]}

    gbt = GradientBoostingTree(train_df=train_df, val_df=val_df, test_df=test_df, hyperparams=hyperparams)

    # Auto-tune should run and set tuned_params and model
    gbt.auto_tune()
    assert isinstance(gbt.tuned_params, dict) and len(gbt.tuned_params) > 0
    assert gbt.model is not None

    # Train final and ensure feature columns recorded
    gbt.train_final()
    assert getattr(gbt, "feature_columns", None) is not None
    assert isinstance(gbt.feature_columns, list) and len(gbt.feature_columns) > 0

    # Predict and validate output
    preds = gbt.predict()
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(test_df)
    assert not preds.isna().any()
    assert np.isfinite(preds.values).all()
