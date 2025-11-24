import numpy as np
import pandas as pd

from src.models import GradientBoostingTree, NeuralNetwork


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    y = 2.0 * x1 - 1.0 * x2 + noise
    df = pd.DataFrame({"x1": x1, "x2": x2, "retadj_next": y})
    return df


def test_gradient_boosting_tree_workflow():
    df = make_data(300, seed=1)
    train = df.iloc[:200].reset_index(drop=True)
    val = df.iloc[200:250].reset_index(drop=True)
    test = df.iloc[250:].reset_index(drop=True)

    model = GradientBoostingTree(
        train_data=train,
        val_data=val,
        test_data=test,
        target_column="retadj_next",
        hyperparameters={
            "num_leaves_list": [7],
            "learning_rate_list": [0.1],
            "n_estimators_list": [10],
            "max_depth_list": [3],
        },
    )

    res = model.auto_tune()
    assert isinstance(res, dict)
    assert model.tuned_params is not None
    assert model.fitted_model is not None

    model.train()
    assert model.fitted_model is not None

    preds = model.predict()
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == test.shape[0]

    metrics = model.evaluate("test")
    assert set(metrics.keys()) == {"MSE", "MAE", "R2"}
    assert metrics["MSE"] < 5.0


def test_neural_network_workflow():
    df = make_data(300, seed=2)
    train = df.iloc[:200].reset_index(drop=True)
    val = df.iloc[200:250].reset_index(drop=True)
    test = df.iloc[250:].reset_index(drop=True)

    model = NeuralNetwork(
        train_data=train,
        val_data=val,
        test_data=test,
        target_column="retadj_next",
        hyperparameters={
            "hidden_units_list": [[8], [16, 8]],
            "learning_rate_list": [1e-2],
            "epochs": 5,
            "batch_size": 16,
        },
    )

    res = model.auto_tune()
    assert isinstance(res, dict)
    assert model.tuned_params is not None
    assert model.fitted_model is not None

    model.train()
    assert model.fitted_model is not None

    preds = model.predict()
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == test.shape[0]

    metrics = model.evaluate("test")
    assert set(metrics.keys()) == {"MSE", "MAE", "R2"}
    assert metrics["MSE"] < 5.0
