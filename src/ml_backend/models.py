"""
This module will provide a unified interface for different machine learning models.

Models included:

- ElasticNet
- GradientBoostingTree (lightgbm)
- NeuralNetwork (tensorflow + keras)
- NeuralNetwork + embeddings layer using permno (tensorflow + keras) 
"""

import json
import pandas as pd
from pathlib import Path

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# sklearn for ElasticNet tuning
from itertools import product
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from typing import Any, cast

import lightgbm as lgb  # type: ignore


__all__ = [
    "ElasticNet",
    "GradientBoostingTree",
    "NeuralNetwork",
    "NeuralNetworkWithEmbeddings",
]

# Load config for model hyperparameters.
# Search upward from the package directory for a `configs.json` file
# so the repo-root config (e.g. `/path/to/ml_project/configs.json`) is found
# even when this module lives under `src/`.
CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = None
for p in (CURRENT_DIR, *CURRENT_DIR.parents):
    candidate = p / "configs.json"
    if candidate.exists():
        CONFIG_PATH = candidate
        break

if CONFIG_PATH is None:
    raise FileNotFoundError(
        f"configs.json not found when searching from {CURRENT_DIR}.\n"
        "Please place a `configs.json` in the repository root or set the "
        "environment to locate it."
    )

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)
    # be defensive: allow missing 'hyperparameters' key to produce empty dict
    hyper_params = cfg.get("hyperparameters", {})
RANDOM_SEED = hyper_params.get("random_seed", 42)
ELASTICNET_HYPERPARAMS = hyper_params.get("ElasticNet", {})
GBT_HYPERPARAMS = hyper_params.get("GradientTreeBoosting", {})
NN_HYPERPARAMS = hyper_params.get("NeuralNetwork", {})
NN_EMBEDDING_HYPERPARAMS = NN_HYPERPARAMS.copy()
NN_EMBEDDING_HYPERPARAMS.update(hyper_params.get("NeuralNetworkWithEmbeddings", {}))


@dataclass(slots=True)
class BaseModel(ABC):
    """
    A unified interface for different machine learning models 
    from various libraries.

    Example: GBT from lightgbm, Neural Networks from TensorFlow/Keras, etc.
    
    The BaseModel class dealing with pannel data structure.

    It will calculate:

        1. Use val_df for auto tuning hyperparameters and early stopping.
        2. Use train_df + val_df for final training.
        3. Use test_df to predict and evaluate the model.

    Attributes(Required)
    --------------------
    train_df: pd.DataFrame
        Training dataset, panel data structure required.
    val_df: pd.DataFrame
        Validation dataset, panel data structure required.
    test_df: pd.DataFrame
        Testing dataset, panel data structure required.
    hyperparams: dict[str, Iterable]
        Hyperparameters for model tuning, will use grid search and early stopping.

    Attributes(Optional)
    --------------------
    target_col: str = "retadj_next"

    Methods
    -------
    auto_tune(self) -> None:
        Auto-tune hyperparameters using val_df.
    train_final(self) -> None:
        Train the final model using train_df + val_df.
    predict(self) -> pd.DataFrame:
        Predict on test_df and return predictions.
    """
    train_df: pd.DataFrame = field(repr=False)
    val_df: pd.DataFrame = field(repr=False)
    test_df: pd.DataFrame = field(repr=False)
    hyperparams: dict[str, list] = field(default_factory=dict)
    tuned_params: dict[str, object] = field(init=False, default_factory=dict)
    target_col: str = "retadj_next"

    model: object = field(init=False, repr=False)

    def __post_init__(self):
        # Verify target column exists
        for df, name in zip(
            [self.train_df, self.val_df, self.test_df],
            ["train_df", "val_df", "test_df"]
        ):
            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in {name} columns.")

        # Verify hyperparams is not empty
        if not self.hyperparams:
            raise ValueError("Hyperparameters dictionary is empty. Please provide hyperparameters for tuning.")

    @abstractmethod
    def auto_tune(self) -> None:
        ...

    @abstractmethod
    def train_final(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> pd.Series:
        ...


@dataclass(slots=True)
class ElasticNet(BaseModel):

    hyperparams: dict[str, list] = field(default_factory=lambda: ELASTICNET_HYPERPARAMS)
    feature_columns: list = field(init=False, repr=False, default_factory=list)

    
    def auto_tune(self) -> None:
        """
        Grid-search over hyperparameters provided in `self.hyperparams`.

        Behavior:
        - Uses `self.train_df` to fit models and `self.val_df` to evaluate.
        - Standardizes numeric features with `StandardScaler` inside a pipeline.
        - Evaluates models by mean squared error on the validation set.
        - Stores best parameters in `self.tuned_params` and the fitted pipeline
          in `self.model`.
        """
        # Prepare feature matrices and targets. Drop the target column.
        X_train = self.train_df.drop(columns=[self.target_col])
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col])
        y_val = self.val_df[self.target_col]

        # The user supplies panel data and already ensures numeric features.
        # Use the data as-is (except the target column removed above).
        X_train = X_train.copy()
        X_val = X_val.copy()

        param_keys = list(self.hyperparams.keys())
        param_values = [self.hyperparams[k] for k in param_keys]

        best_mse = float("inf")
        best_params = {}
        best_pipeline = None

        for vals in product(*param_values):
            params = dict(zip(param_keys, vals))

            # Build pipeline: scaling -> ElasticNet (sklearn alias used)
            enet_kwargs = {}
            if "alpha" in params:
                 enet_kwargs["alpha"] = cast(float, params["alpha"])
            if "l1_ratio" in params:
                 enet_kwargs["l1_ratio"] = cast(float, params["l1_ratio"])
            if "max_iter" in params:
                 enet_kwargs["max_iter"] = cast(int, params["max_iter"])
            if "tol" in params:
                 enet_kwargs["tol"] = cast(float, params["tol"])
            enet = SkElasticNet(random_state=RANDOM_SEED, **enet_kwargs)
            pipeline = make_pipeline(StandardScaler(), enet)

            try:
                pipeline.fit(X_train.values, y_train.values)
            except Exception as exc:
                # If a particular parameter combination fails, skip it
                print(f"Skipping params {params} due to error: {exc}")
                continue

            preds = pipeline.predict(X_val.values)
            mse = mean_squared_error(y_val.values, preds)  # type: ignore

            if mse < best_mse:
                best_mse = mse
                best_params = params
                best_pipeline = pipeline

        if best_pipeline is None:
            raise RuntimeError("Failed to fit any ElasticNet models during tuning.")

        # Save results
        self.tuned_params = best_params
        self.model = best_pipeline
        print(f"ElasticNet tuning complete. Best MSE={best_mse:.6f}, params={best_params}")
            
    def train_final(self) -> None:
        """
        Train the final ElasticNet model on `train_df` + `val_df`.

        Behavior:
        - Concatenates `train_df` and `val_df` and fits a pipeline
          (`StandardScaler` -> `SkElasticNet`) using `self.tuned_params` if
          available, otherwise uses the first value from each hyperparameter list.
        - Stores the fitted pipeline in `self.model` and records feature
          column order in `self.feature_columns` for prediction alignment.
        """
        # Concatenate training and validation data for final training
        combined = pd.concat([self.train_df, self.val_df], axis=0)
        X = combined.drop(columns=[self.target_col]).copy()
        y = combined[self.target_col]

        # Remember feature columns to align test data during prediction
        self.feature_columns = X.columns.tolist()

        # Choose parameters: prefer tuned_params, otherwise pick first value from lists
        if getattr(self, "tuned_params", None):
            params = self.tuned_params
        else:
            params = {k: (v if not isinstance(v, list) else v[0]) for k, v in self.hyperparams.items()}

        # Build typed kwargs for sklearn's ElasticNet constructor
        enet_kwargs = {}
        if "alpha" in params:
            enet_kwargs["alpha"] = float(params["alpha"])  # type: ignore
        if "l1_ratio" in params:
            enet_kwargs["l1_ratio"] = float(params["l1_ratio"])  # type: ignore
        if "max_iter" in params:
            enet_kwargs["max_iter"] = int(params["max_iter"])  # type: ignore
        if "tol" in params:
            enet_kwargs["tol"] = float(params["tol"])  # type: ignore

        enet = SkElasticNet(random_state=RANDOM_SEED, **enet_kwargs)
        pipeline = make_pipeline(StandardScaler(), enet)
        pipeline.fit(X.values, y.values)

        # Save fitted model and params
        self.model = pipeline
        self.tuned_params = params
        print(f"ElasticNet final training complete. Params={params}")

    def predict(self) -> pd.Series:
        """
        Predict using the trained ElasticNet pipeline on `test_df`.

        Returns a `pd.Series` of predictions indexed the same as `test_df`.
        """
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")

        X_test = self.test_df.drop(columns=[self.target_col]).copy()

        # If we stored feature columns during training, reindex test accordingly
        # Reindex test features to training feature order
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        # Call predict dynamically to avoid static-type issues with `model: object`
        predict_fn = getattr(self.model, "predict", None)
        if predict_fn is None:
            raise RuntimeError("Trained model does not expose a 'predict' method.")
        preds = predict_fn(X_test.values)
        return pd.Series(preds, index=self.test_df.index, name=f"{self.target_col}_pred")


@dataclass(slots=True)
class GradientBoostingTree(BaseModel):

    hyperparams: dict[str, list] = field(default_factory=lambda: GBT_HYPERPARAMS)
    feature_columns: list = field(init=False, repr=False, default_factory=list)

    def _build_gbt_model(self, params: dict) -> object:
        """Construct a GBT model (LightGBM if available, otherwise sklearn) with typed params."""
        # Map config names to LightGBM kwargs
        kwargs: dict[str, Any] = {}
        if "n_leaves" in params:
            kwargs["num_leaves"] = int(params["n_leaves"])
        if "depth" in params:
            kwargs["max_depth"] = int(params["depth"])
        if "learning_rate_lambda" in params:
            kwargs["learning_rate"] = float(params["learning_rate_lambda"])
        if "n_estimators" in params:
            kwargs["n_estimators"] = int(params["n_estimators"])
        if "column_sample_rate" in params:
            kwargs["colsample_bytree"] = float(params["column_sample_rate"])

        # Ensure reproducibility
        kwargs.setdefault("random_state", RANDOM_SEED)
        return getattr(lgb, "LGBMRegressor")(**kwargs)


    def auto_tune(self) -> None:
        """
        Grid-search over hyperparameters provided in `self.hyperparams`.

        Uses `train_df` to fit and `val_df` to evaluate, selecting the
        model with lowest validation MSE.
        """
        X_train = self.train_df.drop(columns=[self.target_col]).copy()
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col]).copy()
        y_val = self.val_df[self.target_col]

        param_keys = list(self.hyperparams.keys())
        param_values = [self.hyperparams[k] for k in param_keys]

        best_mse = float("inf")
        best_params: dict[str, Any] = {}
        best_model = None

        for vals in product(*param_values):
            params = dict(zip(param_keys, vals))

            try:
                model = self._build_gbt_model(params)
                # Use LightGBM early stopping against the validation set to speedup tuning
                # Simple fit without early stopping (removed per project preference)
                cast(Any, model).fit(X_train.values, y_train.values)
            except Exception as exc:
                print(f"Skipping params {params} due to error: {exc}")
                continue

            preds = cast(Any, model).predict(X_val.values)
            mse = mean_squared_error(cast(Any, y_val.values), cast(Any, preds))

            if mse < best_mse:
                best_mse = mse
                best_params = params
                best_model = model

        if best_model is None:
            raise RuntimeError("Failed to fit any GradientBoostingTree models during tuning.")

        self.tuned_params = best_params
        self.model = best_model
        print(f"GradientBoostingTree tuning complete. Best MSE={best_mse:.6f}, params={best_params}")

    def train_final(self) -> None:
        # Train final model on train + val
        combined = pd.concat([self.train_df, self.val_df], axis=0)
        X = combined.drop(columns=[self.target_col]).copy()
        y = combined[self.target_col]

        # Record feature order
        self.feature_columns = X.columns.tolist()

        if getattr(self, "tuned_params", None):
            params = self.tuned_params
        else:
            params = {k: (v if not isinstance(v, list) else v[0]) for k, v in self.hyperparams.items()}

        model = self._build_gbt_model(params)
        cast(Any, model).fit(X.values, y.values)

        self.model = model
        self.tuned_params = params
        print(f"GradientBoostingTree final training complete. Params={params}")

    def predict(self) -> pd.Series:
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")

        X_test = self.test_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        predict_fn = getattr(self.model, "predict", None)
        if predict_fn is None:
            raise RuntimeError("Trained model does not expose a 'predict' method.")
        preds = predict_fn(X_test.values)
        return pd.Series(preds, index=self.test_df.index, name=f"{self.target_col}_pred")


@dataclass(slots=True)
class NeuralNetwork(BaseModel):

    hyperparams: dict[str, list] = field(default_factory=lambda: NN_HYPERPARAMS)

    def auto_tune(self) -> None:
        # Implement auto-tuning logic for Neural Network
        raise NotImplementedError("Auto-tuning not implemented for NeuralNetwork yet.")

    def train_final(self) -> None:
        # Implement final training logic for Neural Network
        raise NotImplementedError("Final training not implemented for NeuralNetwork yet.")

    def predict(self) -> pd.Series:
        # Implement prediction logic for Neural Network
        raise NotImplementedError("Prediction not implemented for NeuralNetwork yet.")


@dataclass(slots=True)
class NeuralNetworkWithEmbeddings(BaseModel):

    hyperparams: dict[str, list] = field(default_factory=lambda: NN_EMBEDDING_HYPERPARAMS)

    def auto_tune(self) -> None:
        # Implement auto-tuning logic for Neural Network with Embeddings
        raise NotImplementedError("Auto-tuning not implemented for NeuralNetworkWithEmbeddings yet.")

    def train_final(self) -> None:
        # Implement final training logic for Neural Network with Embeddings
        raise NotImplementedError("Final training not implemented for NeuralNetworkWithEmbeddings yet.")

    def predict(self) -> pd.Series:
        # Implement prediction logic for Neural Network with Embeddings
        raise NotImplementedError("Prediction not implemented for NeuralNetworkWithEmbeddings yet.")



























