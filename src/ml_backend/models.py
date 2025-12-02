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
import os

_TF_AVAILABLE = False
try:
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except Exception:
    tf = None  # type: ignore

import lightgbm as lgb  # type: ignore
import optuna


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

# If TensorFlow is available, optionally enable mixed precision and tune threading
if _TF_AVAILABLE:
    try:
        # Mixed precision can significantly speed up training on Apple Silicon
        if bool(NN_TRAINING.get("use_mixed_precision", False)):
            try:
                from tensorflow.keras import mixed_precision  # type: ignore
                mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass
        # Optionally enable XLA (JIT) to improve kernel fusion/throughput on some workloads
        if bool(NN_TRAINING.get("enable_xla", False)):
            try:
                cast(Any, tf).config.optimizer.set_jit(True)
            except Exception:
                pass

        # Tune threading to match available cores (user can override in config)
        cpu_count = os.cpu_count() or 1
        intra = int(NN_TRAINING.get("intra_op_threads", max(1, cpu_count - 2)))
        inter = int(NN_TRAINING.get("inter_op_threads", max(1, cpu_count // 2)))
        try:
            cast(Any, tf).config.threading.set_intra_op_parallelism_threads(intra)
            cast(Any, tf).config.threading.set_inter_op_parallelism_threads(inter)
        except Exception:
            # not critical if we can't set threading options
            pass
    except Exception:
        pass


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
    
    # Optional datasets for final training (e.g. with embeddings trained on train+val)
    train_df_final: Optional[pd.DataFrame] = field(default=None, repr=False)
    val_df_final: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Optional dataset for final prediction
    test_df_final: Optional[pd.DataFrame] = field(default=None, repr=False)

    model: object = field(init=False, repr=False)

    def __post_init__(self):
        # Verify target column exists
        dfs_to_check = [self.train_df, self.val_df, self.test_df]
        names_to_check = ["train_df", "val_df", "test_df"]
        
        if self.train_df_final is not None:
            dfs_to_check.append(self.train_df_final)
            names_to_check.append("train_df_final")
        if self.val_df_final is not None:
            dfs_to_check.append(self.val_df_final)
            names_to_check.append("val_df_final")
        if self.test_df_final is not None:
            dfs_to_check.append(self.test_df_final)
            names_to_check.append("test_df_final")

        for df, name in zip(dfs_to_check, names_to_check):
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
        - Concatenates `train_df` and `val_df` (or `train_df_final` and `val_df_final` if provided)
          and fits a pipeline (`StandardScaler` -> `SkElasticNet`) using `self.tuned_params` if
          available, otherwise uses the first value from each hyperparameter list.
        - Stores the fitted pipeline in `self.model` and records feature
          column order in `self.feature_columns` for prediction alignment.
        """
        # Use final datasets if provided, else standard
        t_df = self.train_df_final if self.train_df_final is not None else self.train_df
        v_df = self.val_df_final if self.val_df_final is not None else self.val_df

        # Concatenate training and validation data for final training
        combined = pd.concat([t_df, v_df], axis=0)
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
        Predict using the trained ElasticNet pipeline on `test_df` (or `test_df_final`).

        Returns a `pd.Series` of predictions indexed the same as the test dataframe.
        """
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")
        
        target_df = self.test_df_final if self.test_df_final is not None else self.test_df

        X_test = target_df.drop(columns=[self.target_col]).copy()

        # If we stored feature columns during training, reindex test accordingly
        # Reindex test features to training feature order
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        # Call predict dynamically to avoid static-type issues with `model: object`
        predict_fn = getattr(self.model, "predict", None)
        if predict_fn is None:
            raise RuntimeError("Trained model does not expose a 'predict' method.")
        preds = predict_fn(X_test.to_numpy())
        return pd.Series(preds, index=target_df.index, name=f"{self.target_col}_pred")


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
        # Use final datasets if provided, else standard
        t_df = self.train_df_final if self.train_df_final is not None else self.train_df
        v_df = self.val_df_final if self.val_df_final is not None else self.val_df

        # Train final model on train + val
        combined = pd.concat([t_df, v_df], axis=0)
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
        
        target_df = self.test_df_final if self.test_df_final is not None else self.test_df

        X_test = target_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        predict_fn = getattr(self.model, "predict", None)
        if predict_fn is None:
            raise RuntimeError("Trained model does not expose a 'predict' method.")
        preds = predict_fn(X_test.to_numpy())
        return pd.Series(preds, index=target_df.index, name=f"{self.target_col}_pred")


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
        # Prepare feature matrices and targets. Drop the target column.
        X_train = self.train_df.drop(columns=[self.target_col]).to_numpy()
        y_train = self.train_df[self.target_col].to_numpy()
        X_val = self.val_df.drop(columns=[self.target_col]).to_numpy()
        y_val = self.val_df[self.target_col].to_numpy()

        self.feature_columns = self.train_df.drop(columns=[self.target_col]).columns.tolist()

        # tuning controls from config
        tuning_epochs = int(cast(Any, NN_TRAINING.get("tuning_epochs", 5)))
        n_trials = int(cast(Any, NN_TRAINING.get("tuning_trials", 20)))
        pruner_name = NN_TRAINING.get("pruner", "median")

        best_loss = float("inf")
        best_params: dict[str, Any] = {}
        best_model = None

        def objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_loss, best_params, best_model

            # sample params (categorical sampling from supplied lists)
            params: dict[str, Any] = {}
            for k, opts in self.hyperparams.items():
                if isinstance(opts, list):
                    params[k] = trial.suggest_categorical(k, opts)
                else:
                    params[k] = trial.suggest_categorical(k, [opts])

            try:
                model = self._build_keras_model(X_train.shape[1], params)
                batch_size = int(cast(Any, params.get("batch_sizes", 32)))

                callbacks = []
                # Local pruning callback: report val metric each epoch and prune if requested
                class _OptunaKerasPruningCallback(keras.callbacks.Callback):
                    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss") -> None:
                        super().__init__()
                        self._trial = trial
                        self._monitor = monitor

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        if self._monitor not in logs:
                            return
                        val = logs.get(self._monitor)
                        if val is None:
                            return
                        try:
                            self._trial.report(float(val), step=epoch)
                        except Exception:
                            pass
                        if self._trial.should_prune():
                            raise optuna.TrialPruned()

                callbacks.append(_OptunaKerasPruningCallback(trial, monitor="val_loss"))

                # prefer tf.data pipelines on TF-enabled envs for better throughput
                if _TF_AVAILABLE and getattr(tf, "data", None) is not None:
                    try:
                        train_ds = cast(Any, tf).data.Dataset.from_tensor_slices((X_train, y_train))
                        train_ds = train_ds.cache().shuffle(1024).batch(batch_size).prefetch(cast(Any, tf).data.AUTOTUNE)

                        # allow non-deterministic execution and other dataset-level optimizations
                        try:
                            options = cast(Any, tf).data.Options()
                            # allow parallelized map/processing and non-deterministic order for speed
                            options.experimental_deterministic = False
                            train_ds = train_ds.with_options(options)
                        except Exception:
                            pass

                        val_ds = cast(Any, tf).data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(cast(Any, tf).data.AUTOTUNE)
                        try:
                            options = cast(Any, tf).data.Options()
                            options.experimental_deterministic = False
                            val_ds = val_ds.with_options(options)
                        except Exception:
                            pass

                        history = model.fit(train_ds, validation_data=val_ds, epochs=tuning_epochs, callbacks=callbacks, verbose=cast(Any, 0))
                    except Exception:
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=tuning_epochs, batch_size=batch_size, callbacks=callbacks, verbose=cast(Any, 0))
                else:
                    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=tuning_epochs, batch_size=batch_size, callbacks=callbacks, verbose=cast(Any, 0))

                val_loss = float(history.history.get("val_loss", [np.inf])[-1])
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                # treat fit errors as pruned/failed trial
                raise optuna.TrialPruned(f"fit-failed: {exc}")

            trial.report(val_loss, step=len(history.history.get("loss", [])))
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                best_model = model

            return val_loss

        pruner = None
        if isinstance(pruner_name, str) and pruner_name.lower() == "median":
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(NN_TRAINING.get("pruner_warmup", 1)))

        # Decide sampler: use GridSampler when hyperparameter space is a small discrete grid
        # Build a categorical grid dict from self.hyperparams (ensure lists)
        grid_space: dict[str, list] = {}
        for k, v in self.hyperparams.items():
            grid_space[k] = v if isinstance(v, list) else [v]

        total_combinations = 1
        for opts in grid_space.values():
            total_combinations *= max(1, len(opts))

        max_grid = int(NN_TRAINING.get("max_grid_combinations", 2000))
        if total_combinations <= max_grid:
            sampler = optuna.samplers.GridSampler(grid_space)
        else:
            sampler = optuna.samplers.RandomSampler(seed=int(RANDOM_SEED))

        study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        if best_model is None:
            raise RuntimeError("Failed to fit any NeuralNetwork models during tuning.")

        self.tuned_params = best_params
        self.model = best_model
        print(f"NeuralNetwork tuning complete. Best val_loss={best_loss:.6f}, params={best_params}")

    def train_final(self) -> None:
        # Use final datasets if provided, else standard
        t_df = self.train_df_final if self.train_df_final is not None else self.train_df
        v_df = self.val_df_final if self.val_df_final is not None else self.val_df

        # Train final NN on train + val using tuned params (or defaults)
        combined = pd.concat([t_df, v_df], axis=0)
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
            # Use the selected v_df as validation_data
            X_val = v_df.drop(columns=[self.target_col]).copy()
            y_val = v_df[self.target_col]
            model.fit(X.to_numpy(), y.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=cast(Any, 0), validation_data=(X_val.to_numpy(), y_val.to_numpy()), callbacks=callbacks)
        else:
            model.fit(X.to_numpy(), y.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=cast(Any, 0))

        self.model = model
        self.tuned_params = params
        print(f"NeuralNetwork final training complete. Params={params}")

    def predict(self) -> pd.Series:
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")
            
        target_df = self.test_df_final if self.test_df_final is not None else self.test_df

        X_test = target_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        preds = cast(Any, self.model).predict(X_test.to_numpy()).ravel()
        return pd.Series(preds, index=target_df.index, name=f"{self.target_col}_pred")


@dataclass(slots=True)
class NeuralNetworkWithEmbeddings(BaseModel):
    hyperparams: dict[str, list] = field(default_factory=lambda: NN_EMBEDDING_HYPERPARAMS)
    feature_columns: list = field(init=False, repr=False, default_factory=list)
    _permno_mapping: dict = field(init=False, repr=False, default_factory=dict)

    # Stores learned embedding matrices for inspection/analysis
    # after tuning and after final training respectively.
    best_train_embedding_vectors: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)
    combined_train_val_embedding_vectors: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)

    def _get_permnos(self, df: pd.DataFrame) -> pd.Series:
        # Try to extract `permno` from index first, then from columns.
        if isinstance(df.index, pd.MultiIndex) and "permno" in df.index.names:
            return pd.Series(df.index.get_level_values("permno"), index=df.index)
        if "permno" in df.columns:
            return df["permno"].astype(int)  # type: ignore
        raise ValueError("permno not found in index levels or columns; required for embeddings.")

    def _build_embedding_model(self, input_dim: int, n_permnos: int, embedding_dim: int, params: dict) -> "keras.Model":
        hidden_layers = int(params.get("hidden_layers", 1))
        neurons = int(params.get("neurons_per_layer", 32))
        activation = params.get("activation_functions", "relu")
        lr = float(params.get("learning_rates", 0.001))

        numeric_input = keras.Input(shape=(input_dim,), name="numeric_input")
        permno_input = keras.Input(shape=(1,), dtype="int32", name="permno_input")
        embed = layers.Embedding(input_dim=n_permnos, output_dim=embedding_dim, input_length=1, name="permno_embedding")(permno_input)
        embed = layers.Flatten()(embed)
        x = layers.Concatenate()([numeric_input, embed])
        for _ in range(hidden_layers):
            x = layers.Dense(neurons, activation=activation)(x)
        out = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=[numeric_input, permno_input], outputs=out)
        model.compile(optimizer=cast(Any, keras.optimizers.Adam(learning_rate=lr)), loss="mse")
        return model

    def _extract_embedding_df(self, model: "keras.Model") -> pd.DataFrame:
        """Return embedding matrix as DataFrame indexed by permno.

        Assumes the embedding layer is named "permno_embedding" and that
        `self._permno_mapping` maps permno -> embedding index.
        """
        try:
            emb_layer = model.get_layer("permno_embedding")
        except Exception:
            raise RuntimeError("Embedding layer 'permno_embedding' not found in model.")

        weights = emb_layer.get_weights()
        if not weights:
            raise RuntimeError("Embedding layer has no weights.")

        emb_matrix = weights[0]
        # build reverse mapping from index to permno
        index_to_permno = {idx: perm for perm, idx in self._permno_mapping.items()}
        permnos = []
        rows = []
        for idx in range(emb_matrix.shape[0]):
            permno = index_to_permno.get(idx)
            if permno is None:
                # index without explicit permno (e.g., unknown bucket); skip
                continue
            permnos.append(permno)
            rows.append(emb_matrix[idx])

        return pd.DataFrame(rows, index=pd.Index(permnos, name="permno"))

    def auto_tune(self) -> None:
        """Use Optuna to tune embedding NN with pruning support.

        Samples from `self.hyperparams` and uses a pruner similarly to
        `NeuralNetwork.auto_tune`. The permno mapping is derived from the
        training set and kept for final modeling.
        """
        X_train = self.train_df.drop(columns=[self.target_col]).copy()
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col]).copy()
        y_val = self.val_df[self.target_col]

        # extract permnos and factorize to 0..n-1 for embedding indices
        permnos_train = self._get_permnos(self.train_df)
        permnos_val = self._get_permnos(self.val_df)

        unique_permnos = pd.Index(permnos_train.unique())
        # Reserve index 0 for OOV; start mapping at 1
        self._permno_mapping = {int(p): i + 1 for i, p in enumerate(unique_permnos)}

        def map_permno_array(s: pd.Series) -> np.ndarray:
            return np.array([self._permno_mapping.get(int(x), 0) for x in s])

        X_train_num = X_train.to_numpy()
        X_val_num = X_val.to_numpy()
        p_train = map_permno_array(permnos_train)
        p_val = map_permno_array(permnos_val)

        # tuning controls
        tuning_epochs = int(cast(Any, NN_TRAINING.get("tuning_epochs", 3)))
        n_trials = int(cast(Any, NN_TRAINING.get("tuning_trials", 20)))
        pruner_name = NN_TRAINING.get("pruner", "median")

        best_loss = float("inf")
        best_params: dict[str, Any] = {}
        best_model = None

        def objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_loss, best_params, best_model

            params: dict[str, Any] = {}
            for k, opts in self.hyperparams.items():
                if isinstance(opts, list):
                    params[k] = trial.suggest_categorical(k, opts)
                else:
                    params[k] = trial.suggest_categorical(k, [opts])

            try:
                embedding_dim = int(cast(Any, params.get("embedding_dims", 8)))
                # Vocabulary size = len(unique) + 1 (for OOV at index 0)
                model = self._build_embedding_model(X_train_num.shape[1], len(unique_permnos) + 1, embedding_dim, params)
                batch_size = int(cast(Any, params.get("batch_sizes", 32)))

                callbacks = []
                # Local pruning callback for embedding model as well
                class _OptunaKerasPruningCallback(keras.callbacks.Callback):
                    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss") -> None:
                        super().__init__()
                        self._trial = trial
                        self._monitor = monitor

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        if self._monitor not in logs:
                            return
                        val = logs.get(self._monitor)
                        if val is None:
                            return
                        try:
                            self._trial.report(float(val), step=epoch)
                        except Exception:
                            pass
                        if self._trial.should_prune():
                            raise optuna.TrialPruned()

                callbacks.append(_OptunaKerasPruningCallback(trial, monitor="val_loss"))

                # use tf.data pipeline if TF is available for better throughput
                if _TF_AVAILABLE and getattr(tf, "data", None) is not None:
                    try:
                        train_ds = cast(Any, tf).data.Dataset.from_tensor_slices(((X_train_num, p_train), y_train.to_numpy()))
                        train_ds = train_ds.cache().shuffle(1024).batch(batch_size).prefetch(cast(Any, tf).data.AUTOTUNE)

                        # allow non-deterministic execution and other dataset-level optimizations
                        try:
                            options = cast(Any, tf).data.Options()
                            options.experimental_deterministic = False
                            train_ds = train_ds.with_options(options)
                        except Exception:
                            pass

                        val_ds = cast(Any, tf).data.Dataset.from_tensor_slices(((X_val_num, p_val), y_val.to_numpy())).batch(batch_size).prefetch(cast(Any, tf).data.AUTOTUNE)
                        try:
                            options = cast(Any, tf).data.Options()
                            options.experimental_deterministic = False
                            val_ds = val_ds.with_options(options)
                        except Exception:
                            pass

                        history = model.fit(train_ds, validation_data=val_ds, epochs=tuning_epochs, callbacks=callbacks, verbose=cast(Any, 0))
                    except Exception:
                        history = model.fit([X_train_num, p_train], y_train.to_numpy(), validation_data=([X_val_num, p_val], y_val.to_numpy()), epochs=tuning_epochs, batch_size=batch_size, callbacks=callbacks, verbose=cast(Any, 0))
                else:
                    history = model.fit([X_train_num, p_train], y_train.to_numpy(), validation_data=([X_val_num, p_val], y_val.to_numpy()), epochs=tuning_epochs, batch_size=batch_size, callbacks=callbacks, verbose=cast(Any, 0))

                val_loss = float(history.history.get("val_loss", [np.inf])[-1])
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                raise optuna.TrialPruned(f"fit-failed: {exc}")

            trial.report(val_loss, step=len(history.history.get("loss", [])))
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                best_model = model

            return val_loss

        pruner = None
        if isinstance(pruner_name, str) and pruner_name.lower() == "median":
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(NN_TRAINING.get("pruner_warmup", 1)))

        # Decide sampler: use GridSampler when hyperparameter space is a small discrete grid
        grid_space: dict[str, list] = {}
        for k, v in self.hyperparams.items():
            grid_space[k] = v if isinstance(v, list) else [v]

        total_combinations = 1
        for opts in grid_space.values():
            total_combinations *= max(1, len(opts))

        max_grid = int(NN_TRAINING.get("max_grid_combinations", 2000))
        if total_combinations <= max_grid:
            sampler = optuna.samplers.GridSampler(grid_space)
        else:
            sampler = optuna.samplers.RandomSampler(seed=int(RANDOM_SEED))

        study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        if best_model is None:
            raise RuntimeError("Failed to fit any NeuralNetworkWithEmbeddings models during tuning.")

        self.tuned_params = best_params
        self.model = best_model
        # `X_train` already has the target column removed above, so just
        # record its columns directly instead of attempting to drop again.
        self.feature_columns = X_train.columns.tolist()

        # Store embedding matrix learned on training data for the best model
        try:
            self.best_train_embedding_vectors = self._extract_embedding_df(best_model)
        except Exception:
            self.best_train_embedding_vectors = None

        print(f"NeuralNetworkWithEmbeddings tuning complete. Best val_loss={best_loss:.6f}, params={best_params}")

    def train_final(self) -> None:
        combined = pd.concat([self.train_df, self.val_df], axis=0)
        X = combined.drop(columns=[self.target_col]).copy()
        y = combined[self.target_col]

        # permno mapping from combined dataset ensures we have IDs for all seen permnos
        permnos_combined = self._get_permnos(pd.concat([self.train_df, self.val_df], axis=0))
        unique_permnos = pd.Index(permnos_combined.unique())
        # Reserve index 0 for OOV (unknown permnos); start mapping at 1
        self._permno_mapping = {int(p): i + 1 for i, p in enumerate(unique_permnos)}

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

        # Vocabulary size = len(unique) + 1 (for OOV at index 0)
        model = self._build_embedding_model(X.shape[1], len(unique_permnos) + 1, embedding_dim, params)

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

        # Store embedding matrix after training on combined train+val
        try:
            self.combined_train_val_embedding_vectors = self._extract_embedding_df(model)
        except Exception:
            self.combined_train_val_embedding_vectors = None

        print(f"NeuralNetworkWithEmbeddings final training complete. Params={params}")

    def predict(self) -> pd.Series:
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not trained. Call `train_final()` or `auto_tune()` first.")
            
        target_df = self.test_df_final if self.test_df_final is not None else self.test_df

        X_test = target_df.drop(columns=[self.target_col]).copy()
        if getattr(self, "feature_columns", None):
            X_test = X_test.reindex(columns=self.feature_columns)

        permnos_test = self._get_permnos(target_df)
        p_test = np.array([self._permno_mapping.get(int(x), 0) for x in permnos_test])

        preds = cast(Any, self.model).predict([X_test.to_numpy(), p_test]).ravel()
        return pd.Series(preds, index=target_df.index, name=f"{self.target_col}_pred")
