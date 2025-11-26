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
import numpy as np
from pathlib import Path

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# sklearn for ElasticNet tuning
from itertools import product
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from typing import Any, cast, Optional
import keras  # standalone keras package
from keras import layers

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
NN_TRAINING = hyper_params.get("NeuralNetworkTraining", {})


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
                pipeline.fit(X_train.to_numpy(), y_train.to_numpy())
            except Exception as exc:
                # If a particular parameter combination fails, skip it
                print(f"Skipping params {params} due to error: {exc}")
                continue

            preds = pipeline.predict(X_val.to_numpy())
            mse = mean_squared_error(y_val.to_numpy(), preds)

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
        pipeline.fit(X.to_numpy(), y.to_numpy())

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
        preds = predict_fn(X_test.to_numpy())
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
                # Simple fit without early stopping (removed per project preference)
                cast(Any, model).fit(X_train.to_numpy(), y_train.to_numpy())
            except Exception as exc:
                print(f"Skipping params {params} due to error: {exc}")
                continue

            preds = cast(Any, model).predict(X_val.to_numpy())
            mse = mean_squared_error(y_val.to_numpy(), preds)

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
        cast(Any, model).fit(X.to_numpy(), y.to_numpy())

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
        preds = predict_fn(X_test.to_numpy())
        return pd.Series(preds, index=self.test_df.index, name=f"{self.target_col}_pred")


@dataclass(slots=True)
class NeuralNetwork(BaseModel):
    hyperparams: dict[str, list] = field(default_factory=lambda: NN_HYPERPARAMS)
    feature_columns: list = field(init=False, repr=False, default_factory=list)

    def _build_keras_model(self, input_dim: int, params: dict) -> "keras.Model":
        # Build a simple feed-forward network according to params
        hidden_layers = int(params.get("hidden_layers", 1))
        neurons = int(params.get("neurons_per_layer", 32))
        activation = params.get("activation_functions", "relu")
        lr = float(params.get("learning_rates", 0.001))

        model = keras.Sequential()
        model.add(keras.Input(shape=(input_dim,)))
        for _ in range(hidden_layers):
            model.add(layers.Dense(neurons, activation=activation))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer=cast(Any, keras.optimizers.Adam(learning_rate=lr)), loss="mse")
        return model

    # sklearn fallback removed; we assume standalone Keras is installed and used.

    def auto_tune(self) -> None:
        """
        Simple grid-search over provided hyperparameters. Uses a small number
        of epochs for Keras tuning to keep tuning quick.
        """
        X_train = self.train_df.drop(columns=[self.target_col]).copy()
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col]).copy()
        y_val = self.val_df[self.target_col]

        self.feature_columns = X_train.columns.tolist()

        param_keys = list(self.hyperparams.keys())
        param_values = [self.hyperparams[k] for k in param_keys]

        best_mse = float("inf")
        best_params: dict[str, Any] = {}
        best_model = None

        # tuning epochs: if epochs present in hyperparams and is an int/list, pick small value for tuning
        tuning_epochs = 5
        if "epochs" in self.hyperparams:
            ep = self.hyperparams.get("epochs")
            if isinstance(ep, list) and len(ep) > 0:
                tuning_epochs = min(5, int(ep[0]))
            elif isinstance(ep, int):
                tuning_epochs = min(5, ep)

        for vals in product(*param_values):
            params = dict(zip(param_keys, vals))
            print(f"Trying NeuralNetwork params: {params}")

            try:
                model = self._build_keras_model(X_train.shape[1], params)
                batch_size = int(cast(Any, params.get("batch_sizes", 32)))
                model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=tuning_epochs, batch_size=batch_size, verbose=cast(Any, 0))
                preds = model.predict(X_val.to_numpy()).ravel()
            except Exception as exc:
                print(f"Skipping params {params} due to error: {exc}")
                continue

            mse = mean_squared_error(y_val.to_numpy(), preds)
            if mse < best_mse:
                best_mse = mse
                best_params = params
                best_model = model

        if best_model is None:
            raise RuntimeError("Failed to fit any NeuralNetwork models during tuning.")

        self.tuned_params = best_params
        self.model = best_model
        print(f"NeuralNetwork tuning complete. Best MSE={best_mse:.6f}, params={best_params}")

    def train_final(self) -> None:
        # Train final NN on train + val using tuned params (or defaults)
        combined = pd.concat([self.train_df, self.val_df], axis=0)
        X = combined.drop(columns=[self.target_col]).copy()
        y = combined[self.target_col]

        self.feature_columns = X.columns.tolist()

        if getattr(self, "tuned_params", None):
            params = self.tuned_params
        else:
            params = {k: (v if not isinstance(v, list) else v[0]) for k, v in self.hyperparams.items()}

        # Determine epochs: prefer explicit param if present, otherwise use NN_TRAINING final_epochs
        if "epochs" in params:
            try:
                epochs = int(params["epochs"])  # type: ignore
            except Exception:
                epochs = int(cast(Any, NN_TRAINING.get("final_epochs", 100)))
        else:
            epochs = int(cast(Any, NN_TRAINING.get("final_epochs", 100)))

        batch_size = int(cast(Any, params.get("batch_sizes", 32)))
        model = self._build_keras_model(X.shape[1], params)

        # Optionally use early stopping with validation on the provided val_df
        use_es = bool(NN_TRAINING.get("use_early_stopping", False))
        if use_es:
            patience = int(cast(Any, NN_TRAINING.get("early_stopping_patience", 10)))
            callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
            # Use val_df as validation_data (it's okay if val_df is also part of combined)
            X_val = self.val_df.drop(columns=[self.target_col]).copy()
            y_val = self.val_df[self.target_col]
            model.fit(X.to_numpy(), y.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=cast(Any, 0), validation_data=(X_val.to_numpy(), y_val.to_numpy()), callbacks=callbacks)
        else:
            model.fit(X.to_numpy(), y.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=cast(Any, 0))

        self.model = model
        self.tuned_params = params
        print(f"NeuralNetwork final training complete. Params={params}")

    def predict(self) -> pd.Series:
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")

        X_test = self.test_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        preds = cast(Any, self.model).predict(X_test.to_numpy()).ravel()
        return pd.Series(preds, index=self.test_df.index, name=f"{self.target_col}_pred")


@dataclass(slots=True)
class NeuralNetworkWithEmbeddings(BaseModel):
    hyperparams: dict[str, list] = field(default_factory=lambda: NN_EMBEDDING_HYPERPARAMS)
    feature_columns: list = field(init=False, repr=False, default_factory=list)
    _permno_mapping: dict = field(init=False, repr=False, default_factory=dict)

    def _get_permnos(self, df: pd.DataFrame) -> pd.Series:
        # Try to extract `permno` from index first, then from columns.
        if isinstance(df.index, pd.MultiIndex) and "permno" in df.index.names:
            return pd.Series(df.index.get_level_values("permno"), index=df.index)
        if "permno" in df.columns:
            return df["permno"].astype(int)
        raise ValueError("permno not found in index levels or columns; required for embeddings.")

    def _build_embedding_model(self, input_dim: int, n_permnos: int, embedding_dim: int, params: dict) -> "keras.Model":
        hidden_layers = int(params.get("hidden_layers", 1))
        neurons = int(params.get("neurons_per_layer", 32))
        activation = params.get("activation_functions", "relu")
        lr = float(params.get("learning_rates", 0.001))

        numeric_input = keras.Input(shape=(input_dim,), name="numeric_input")
        permno_input = keras.Input(shape=(1,), dtype="int32", name="permno_input")
        embed = layers.Embedding(input_dim=n_permnos, output_dim=embedding_dim, input_length=1)(permno_input)
        embed = layers.Flatten()(embed)

        x = layers.Concatenate()([numeric_input, embed])
        for _ in range(hidden_layers):
            x = layers.Dense(neurons, activation=activation)(x)
        out = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=[numeric_input, permno_input], outputs=out)
        model.compile(optimizer=cast(Any, keras.optimizers.Adam(learning_rate=lr)), loss="mse")
        return model

    def auto_tune(self) -> None:
        """
        Grid-search over embedding dimensions and standard NN hyperparameters.
        Uses `train_df` to fit and `val_df` to evaluate; keeps tuning epochs small.
        """
        X_train = self.train_df.drop(columns=[self.target_col]).copy()
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col]).copy()
        y_val = self.val_df[self.target_col]

        # extract permnos and factorize to 0..n-1 for embedding indices
        permnos_train = self._get_permnos(self.train_df)
        permnos_val = self._get_permnos(self.val_df)

        # create mapping from permno -> id using train set
        unique_permnos = pd.Index(permnos_train.unique())
        self._permno_mapping = {int(p): i for i, p in enumerate(unique_permnos)}

        def map_permno_array(s: pd.Series) -> np.ndarray:
            return np.array([self._permno_mapping.get(int(x), 0) for x in s])

        X_train_num = X_train.to_numpy()
        X_val_num = X_val.to_numpy()
        p_train = map_permno_array(permnos_train)
        p_val = map_permno_array(permnos_val)

        param_keys = list(self.hyperparams.keys())
        param_values = [self.hyperparams[k] for k in param_keys]

        best_mse = float("inf")
        best_params: dict[str, Any] = {}
        best_model = None

        tuning_epochs = 3

        for vals in product(*param_values):
            params = dict(zip(param_keys, vals))
            try:
                embedding_dim = int(cast(Any, params.get("embedding_dims", 8)))
                model = self._build_embedding_model(X_train_num.shape[1], len(unique_permnos), embedding_dim, params)
                batch_size = int(cast(Any, params.get("batch_sizes", 32)))
                model.fit([X_train_num, p_train], y_train.to_numpy(), epochs=tuning_epochs, batch_size=batch_size, verbose=cast(Any, 0))
                preds = model.predict([X_val_num, p_val]).ravel()
            except Exception as exc:
                print(f"Skipping params {params} due to error: {exc}")
                continue
            mse = mean_squared_error(y_val.to_numpy(), preds)
            if mse < best_mse:
                best_mse = mse
                best_params = params
                best_model = model

        if best_model is None:
            raise RuntimeError("Failed to fit any NeuralNetworkWithEmbeddings models during tuning.")

        self.tuned_params = best_params
        self.model = best_model
        self.feature_columns = X_train.columns.tolist()
        print(f"NeuralNetworkWithEmbeddings tuning complete. Best MSE={best_mse:.6f}, params={best_params}")

    def train_final(self) -> None:
        combined = pd.concat([self.train_df, self.val_df], axis=0)
        X = combined.drop(columns=[self.target_col]).copy()
        y = combined[self.target_col]

        # permno mapping from combined dataset ensures we have IDs for all seen permnos
        permnos_combined = self._get_permnos(pd.concat([self.train_df, self.val_df], axis=0))
        unique_permnos = pd.Index(permnos_combined.unique())
        self._permno_mapping = {int(p): i for i, p in enumerate(unique_permnos)}

        def map_permno_array_series(s: pd.Series) -> np.ndarray:
            return np.array([self._permno_mapping.get(int(x), 0) for x in s])

        self.feature_columns = X.columns.tolist()

        if getattr(self, "tuned_params", None):
            params = self.tuned_params
        else:
            params = {k: (v if not isinstance(v, list) else v[0]) for k, v in self.hyperparams.items()}

        # determine embedding dim and epochs
        embedding_dim = int(cast(Any, params.get("embedding_dims", 8)))
        epochs = int(cast(Any, NN_TRAINING.get("final_epochs", 100)))
        batch_size = int(cast(Any, params.get("batch_sizes", 32)))

        model = self._build_embedding_model(X.shape[1], len(unique_permnos), embedding_dim, params)

        use_es = bool(NN_TRAINING.get("use_early_stopping", False))
        if use_es:
            patience = int(cast(Any, NN_TRAINING.get("early_stopping_patience", 10)))
            callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
            X_val = self.val_df.drop(columns=[self.target_col]).copy()
            y_val = self.val_df[self.target_col]
            p_train = map_permno_array_series(self._get_permnos(self.train_df))
            p_val = map_permno_array_series(self._get_permnos(self.val_df))
            X_train_num = self.train_df.drop(columns=[self.target_col]).to_numpy()
            y_train_num = self.train_df[self.target_col].to_numpy()
            model.fit([X_train_num, p_train], y_train_num, epochs=epochs, batch_size=batch_size, validation_data=( [X_val.to_numpy(), p_val], y_val.to_numpy()), callbacks=callbacks, verbose=cast(Any, 0))
        else:
            X_comb_num = X.to_numpy()
            p_comb = map_permno_array_series(self._get_permnos(combined))
            model.fit([X_comb_num, p_comb], y.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=cast(Any, 0))

        self.model = model
        self.tuned_params = params
        print(f"NeuralNetworkWithEmbeddings final training complete. Params={params}")

    def predict(self) -> pd.Series:
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")

        X_test = self.test_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        permnos_test = self._get_permnos(self.test_df)
        p_test = np.array([self._permno_mapping.get(int(x), 0) for x in permnos_test])

        preds = cast(Any, self.model).predict([X_test.to_numpy(), p_test]).ravel()
        return pd.Series(preds, index=self.test_df.index, name=f"{self.target_col}_pred")

    def distance_weighted_feature(self, col_name: str, df: Optional[pd.DataFrame] = None, bandwidth: Optional[float] = None) -> pd.Series:
        """Compute cross-sectional peer-weighted feature per date using embedding similarity.

        For each date t and firm i present in `df` (or `self.test_df` by default):
          - compute distances between i's embedding and all other firms j at date t
          - form Gaussian weights w_ij = exp(-d_ij^2 / (2*bandwidth^2)), set w_ii = 0
          - normalize weights so sum_j w_ij = 1
          - compute peer feature for i as sum_j w_ij * q_{j,t}

        Parameters
        ----------
        col_name: str
            Column name present in `df` whose cross-sectional peer-weighted
            average will be computed (q_{j,t}).
        df: Optional[pd.DataFrame]
            DataFrame to compute peer features on. If None, uses `self.test_df`.
        bandwidth: Optional[float]
            If provided, uses this value as the Gaussian kernel bandwidth; otherwise
            uses the median of non-zero pairwise distances per date (fallback 1.0).

        Returns
        -------
        pd.Series
            Series indexed like `df.index` containing the peer-weighted feature,
            named `peer_feat_{col_name}`. If a firm has no peers on a date, value is NaN.
        """
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")

        if df is None:
            df = self.test_df

        # find embedding layer
        emb_layer = None
        m = cast(Any, self.model)
        for layer in m.layers:
            if layer.__class__.__name__ == "Embedding":
                emb_layer = layer
                break
        if emb_layer is None:
            raise RuntimeError("Embedding layer not found in the trained model.")

        emb_weights = emb_layer.get_weights()[0]  # (n_permnos, emb_dim)

        # ensure column exists
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame.")

        # extract date index for grouping (support MultiIndex with 'date' level or a 'date' column)
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            dates = df.index.get_level_values("date")
        elif "date" in df.columns:
            dates = df["date"]
        else:
            raise ValueError("DataFrame must contain a 'date' level in the index or a 'date' column for cross-sectional grouping.")

        permnos = self._get_permnos(df)
        mapped = np.array([self._permno_mapping.get(int(x), 0) for x in permnos])

        q_vals_all = df[col_name].to_numpy(dtype=float)
        n = len(df)
        result = np.full(n, np.nan, dtype=float)

        # operate date-by-date
        unique_dates = pd.Index(dates).unique()
        for d in unique_dates:
            mask = (dates == d)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                continue
            if idx.size == 1:
                result[idx[0]] = np.nan
                continue

            emb_sub = emb_weights[mapped[idx]]  # shape (k, emb_dim)
            # pairwise squared distances
            norms = np.sum(emb_sub * emb_sub, axis=1)
            D2 = norms[:, None] + norms[None, :] - 2.0 * (emb_sub @ emb_sub.T)
            D2 = np.maximum(D2, 0.0)
            D = np.sqrt(D2)

            # choose bandwidth per date if not provided
            if bandwidth is None:
                nonzero = D[D > 0]
                bw = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
                if bw == 0.0:
                    bw = 1.0
            else:
                bw = float(bandwidth)

            # Gaussian kernel
            K = np.exp(- (D ** 2) / (2.0 * (bw ** 2)))
            np.fill_diagonal(K, 0.0)

            row_sums = K.sum(axis=1)
            q_sub = q_vals_all[idx]

            # compute weighted average for each row
            for i_pos, global_i in enumerate(idx):
                s = row_sums[i_pos]
                if s > 0:
                    result[global_i] = float(K[i_pos].dot(q_sub) / s)
                else:
                    result[global_i] = np.nan

        return pd.Series(result, index=df.index, name=f"peer_feat_{col_name}")



























