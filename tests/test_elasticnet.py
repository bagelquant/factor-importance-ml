import numpy as np
import pandas as pd

from models import ElasticNet


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    y = 2.0 * x1 - 1.0 * x2 + noise
    df = pd.DataFrame({"x1": x1, "x2": x2, "retadj_next": y})
    return df


def test_elasticnet_workflow():
    df = make_data(300, seed=42)
    train = df.iloc[:200].reset_index(drop=True)
    val = df.iloc[200:250].reset_index(drop=True)
    test = df.iloc[250:].reset_index(drop=True)

    model = ElasticNet(
        train_data=train,
        val_data=val,
        test_data=test,
        target_column="retadj_next",
        hyperparameters={
            "alpha_list": [0.01, 0.1],
            "l1_ratio_list": [0.1, 0.5],
            "max_iter": 10000,
        },
    )

    # auto_tune should return a dict and set tuned_params
    res = model.auto_tune()
    assert isinstance(res, dict)
    assert model.tuned_params is not None

    # fitted model from tuning should be set
    assert model.fitted_model is not None

    # retrain on train+val
    model.train()
    assert model.fitted_model is not None

    preds = model.predict()
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == test.shape[0]

    metrics = model.evaluate("test")
    assert set(metrics.keys()) == {"MSE", "MAE", "R2"}
    # with small noise synthetic data MSE should be reasonably small
    assert metrics["MSE"] < 1.0
