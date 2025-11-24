"""
This module will provide a unified interface for different machine learning models.

Models included:

- ElasticNet
- GradientBoostingTree (lightgbm)
- NeuralNetwork (tensorflow + keras)
- NeuralNetwork + embeddings layer using permno (tensorflow + keras) 
"""

import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import numpy as np

from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class MlModel(ABC):
    """Abstract base class for machine learning models."""

    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    target_column: str = "retadj_next"
    tuned_params: dict | None = None
    # generic hyperparameters bag for subclasses (e.g. {"alpha_list": [...], "max_iter": 100000})
    hyperparameters: dict | None = None
    fitted_model: Any | None = None

    def __post_init__(self):
        # basic validation
        for name in ("train_data", "val_data", "test_data"):
            df = getattr(self, name)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")

        if not isinstance(self.target_column, str):
            raise TypeError("target_column must be a string")

    # add hyperparameters as needed @abstractmethod
    @abstractmethod
    def auto_tune(self) -> Any:
        pass

    @abstractmethod
    def train(self) -> Any:
        pass

    @abstractmethod
    def predict(self, dataset: str = "test") -> Any:
        """Return predictions for the specified dataset ('train','val','test').

        Subclasses should return a sequence aligned to the requested dataset's index.
        """
        pass

    # Evaluation metrics using sklearn
    def evaluate(self, dataset: str = "test"):
        """Evaluate predictions on a dataset. `dataset` can be 'train','val', or 'test'.

        Returns dict with MSE, MAE, R2.
        """
        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")

        df = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]
        y_true = df[self.target_column]
        y_pred = self.predict(dataset)

        # ensure alignment and shape
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.reindex(df.index).to_numpy()
        else:
            y_pred = np.asarray(y_pred).ravel()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "MAE": mae, "R2": r2}

    # --- Small helpers to reduce duplication across models ---
    def _validate_dataset(self, dataset: str) -> pd.DataFrame:
        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")
        return {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]

    def _get_X_y(self, dataset: str = "train"):
        """Return (X, y) for the requested dataset as (DataFrame, Series)."""
        df = self._validate_dataset(dataset)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    def _scale_train_val(self, X_train, X_val, scaler: StandardScaler | None = None):
        """Fit a scaler on X_train (if not provided) and return (X_train_s, X_val_s, scaler)."""
        if scaler is None:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
        else:
            X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        return X_train_s, X_val_s, scaler

    def _combine_train_val(self):
        combined = pd.concat([self.train_data, self.val_data], axis=0)
        X = combined.drop(columns=[self.target_column])
        y = combined[self.target_column]
        return X, y


class ElasticNet(MlModel):

    def auto_tune(self):
        """Grid search over `alpha_list` and `l1_ratio_list` using
        self.train_data to fit and self.val_data to evaluate. Sets
        `self.tuned_params` to the best pair and `self.fitted_model`
        to the best model fit on the training set (not combined).
        """
        # read hyperparameters bag with sensible defaults
        hp = self.hyperparameters or {}
        alpha_list = list(hp.get("alpha_list", [0.1, 1.0, 10.0]))
        l1_ratio_list = list(hp.get("l1_ratio_list", [0.1, 0.5, 0.9]))
        max_iter = int(hp.get("max_iter", 1_000_000))

        X_train, y_train = self._get_X_y("train")
        X_val, y_val = self._get_X_y("val")

        best_score = float("inf")
        best_params = None
        best_model = None

        for alpha in alpha_list:
            for l1 in l1_ratio_list:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("enet", SKElasticNet(alpha=float(alpha), l1_ratio=float(l1), max_iter=max_iter))
                ])
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                mse = mean_squared_error(y_val, preds)
                if mse < best_score:
                    best_score = mse
                    best_params = {"alpha": alpha, "l1_ratio": l1}
                    best_model = model

        self.tuned_params = best_params
        # keep the model fitted on train only; `train()` will retrain on train+val
        self.fitted_model = best_model
        return {"best_params": best_params, "best_mse": best_score}

    def train(self):
        """Retrain model on combined train + val using tuned hyperparameters.

        If `auto_tune` was not run, uses the first values in the lists
        or sklearn defaults.
        """
        # obtain tuned params or defaults from hyperparameters bag
        hp = self.hyperparameters or {}
        max_iter = int(hp.get("max_iter", 1_000_000))

        if self.tuned_params is None:
            # run auto_tune to get reasonable defaults
            self.auto_tune()

        alpha = None
        l1 = None
        if self.tuned_params:
            alpha = self.tuned_params.get("alpha")
            l1 = self.tuned_params.get("l1_ratio")
        if alpha is None:
            alpha = float(hp.get("alpha", 1.0))
        else:
            alpha = float(alpha)
        if l1 is None:
            l1 = float(hp.get("l1_ratio", 0.5))
        else:
            l1 = float(l1)

        X, y = self._combine_train_val()

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", SKElasticNet(alpha=alpha, l1_ratio=l1, max_iter=max_iter))
        ])
        model.fit(X, y)
        self.fitted_model = model
        return self.fitted_model

    def predict(self, dataset: str = "test"):
        """Use the fitted model to predict on the specified dataset.

        `dataset` may be 'train', 'val', or 'test'. Returns a pd.Series
        aligned to the dataset index.
        """
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Run `auto_tune()` or `train()` first.")

        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")

        df = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]
        X = df.drop(columns=[self.target_column])
        preds = self.fitted_model.predict(X)
        return pd.Series(preds, index=df.index)


class GradientBoostingTree(MlModel):
    """LightGBM regressor wrapper following MlModel contract.

    Expects `self.hyperparameters` to possibly contain lists for grid search:
      - "num_leaves_list"
      - "learning_rate_list"
      - "n_estimators_list"
      - "max_depth_list"
    """

    def auto_tune(self):
        hp = self.hyperparameters or {}
        num_leaves_list = list(hp.get("num_leaves_list", [31, 63]))
        learning_rate_list = list(hp.get("learning_rate_list", [0.1, 0.01]))
        n_estimators_list = list(hp.get("n_estimators_list", [100, 200]))
        max_depth_list = list(hp.get("max_depth_list", [ -1, 10]))

        X_train, y_train = self._get_X_y("train")
        X_val, y_val = self._get_X_y("val")

        best_score = float("inf")
        best_params = None
        best_model = None

        for nl in num_leaves_list:
            for lr in learning_rate_list:
                for n_est in n_estimators_list:
                    for md in max_depth_list:
                        model = lgb.LGBMRegressor(
                            num_leaves=int(nl),
                            learning_rate=float(lr),
                            n_estimators=int(n_est),
                            max_depth=int(md),
                            random_state=0,
                        )
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
                        preds = np.asarray(preds).ravel()
                        mse = mean_squared_error(y_val, preds)
                        if mse < best_score:
                            best_score = mse
                            best_params = {
                                "num_leaves": int(nl),
                                "learning_rate": float(lr),
                                "n_estimators": int(n_est),
                                "max_depth": int(md),
                            }
                            best_model = model

        self.tuned_params = best_params
        self.fitted_model = best_model
        return {"best_params": best_params, "best_mse": best_score}

    def train(self):
        hp = self.hyperparameters or {}
        if self.tuned_params is None:
            self.auto_tune()

        params = self.tuned_params or {}
        num_leaves = int(params.get("num_leaves", hp.get("num_leaves", 31)))
        learning_rate = float(params.get("learning_rate", hp.get("learning_rate", 0.1)))
        n_estimators = int(params.get("n_estimators", hp.get("n_estimators", 100)))
        max_depth = int(params.get("max_depth", hp.get("max_depth", -1)))

        X, y = self._combine_train_val()

        model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
        )
        model.fit(X, y)
        self.fitted_model = model
        return self.fitted_model

    def predict(self, dataset: str = "test"):
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Run `auto_tune()` or `train()` first.")
        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")
        df = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]
        X = df.drop(columns=[self.target_column])
        preds = self.fitted_model.predict(X)
        return pd.Series(np.asarray(preds).ravel(), index=df.index)


class NeuralNetwork(MlModel):
    """Simple feed-forward Keras regressor wrapper.

    Hyperparameters (in `self.hyperparameters`) used for tuning:
      - "hidden_units_list": list of lists/tuples specifying units per layer
      - "learning_rate_list"
      - "epochs"
      - "batch_size"
    """

    def _build_model(self, input_dim: int, hidden_units, learning_rate: float):
        model = keras.Sequential()
        for i, units in enumerate(hidden_units):
            if i == 0:
                model.add(keras.layers.Dense(int(units), activation="relu", input_shape=(input_dim,)))
            else:
                model.add(keras.layers.Dense(int(units), activation="relu"))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        return model

    def auto_tune(self):
        hp = self.hyperparameters or {}
        # allow user to opt-out of optuna by setting optuna=False in hyperparameters
        use_optuna = bool(hp.get("use_optuna", True))
        n_trials = int(hp.get("optuna_n_trials", 10))
        epochs_default = int(hp.get("epochs", 20))

        # fallback grid options (kept for compatibility)
        hidden_units_list = hp.get("hidden_units_list", [[32], [64, 32]])
        learning_rate_list = hp.get("learning_rate_list", [1e-3, 1e-4])
        batch_size = int(hp.get("batch_size", 32))

        X_train, y_train = self._get_X_y("train")
        X_val, y_val = self._get_X_y("val")
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()

        # scale inputs helper (we will re-scale inside objective if using optuna)
        X_train_s_init, X_val_s_init, scaler_init = self._scale_train_val(X_train, X_val)

        # If optuna not available or disabled, run simple grid search (previous behavior)
        if not use_optuna:
            best_score = float("inf")
            best_params = None
            best_model = None
            for hidden in hidden_units_list:
                for lr in learning_rate_list:
                    keras.backend.clear_session()
                    model = self._build_model(X_train_s_init.shape[1], hidden, float(lr))
                    model.fit(X_train_s_init, y_train, epochs=epochs_default, batch_size=batch_size, verbose=0)
                    preds = np.asarray(model.predict(X_val_s_init, verbose=0)).ravel()
                    mse = mean_squared_error(y_val, preds)
                    if mse < best_score:
                        best_score = mse
                        best_params = {"hidden_units": hidden, "learning_rate": float(lr), "epochs": epochs_default}
                        best_model = {"scaler": scaler_init, "model": model}

            self.tuned_params = best_params
            self.fitted_model = best_model
            return {"best_params": best_params, "best_mse": best_score}

        # Try to import optuna; if missing, fall back to grid search above
        try:
            import optuna
        except Exception:
            # optuna not installed in environment; fallback to grid
            best_score = float("inf")
            best_params = None
            best_model = None
            for hidden in hidden_units_list:
                for lr in learning_rate_list:
                    keras.backend.clear_session()
                    model = self._build_model(X_train_s_init.shape[1], hidden, float(lr))
                    model.fit(X_train_s_init, y_train, epochs=epochs_default, batch_size=batch_size, verbose=0)
                    preds = np.asarray(model.predict(X_val_s_init, verbose=0)).ravel()
                    mse = mean_squared_error(y_val, preds)
                    if mse < best_score:
                        best_score = mse
                        best_params = {"hidden_units": hidden, "learning_rate": float(lr), "epochs": epochs_default}
                        best_model = {"scaler": scaler_init, "model": model}

            self.tuned_params = best_params
            self.fitted_model = best_model
            return {"best_params": best_params, "best_mse": best_score}

        # Optuna is available — set up an optimization study
        def _objective(trial: "optuna.trial.Trial"):
            # suggest number of layers and units per layer
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden = []
            for i in range(n_layers):
                hidden.append(trial.suggest_int(f"n_units_l{i}", 4, 64))

            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch = trial.suggest_categorical("batch_size", [8, 16, 32])
            epochs = int(hp.get("epochs", epochs_default))

            # scale inside objective to avoid leaking info
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            keras.backend.clear_session()
            model = self._build_model(X_train_s.shape[1], hidden, float(lr))
            # use early stopping to speed tuning
            callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
            model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=epochs, batch_size=batch, callbacks=callbacks, verbose=0)

            preds = np.asarray(model.predict(X_val_s, verbose=0)).ravel()
            mse = mean_squared_error(y_val, preds)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=n_trials)

        best = study.best_trial.params
        # reconstruct hidden_units list
        n_layers = best.get("n_layers", 1)
        hidden_units = [int(best[f"n_units_l{i}"]) for i in range(n_layers)]
        best_lr = float(best.get("learning_rate", 1e-3))
        best_batch = int(best.get("batch_size", batch_size))

        # retrain final model on train only (auto_tune stores model fit on training set)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        keras.backend.clear_session()
        final_model = self._build_model(X_train_s.shape[1], hidden_units, best_lr)
        final_model.fit(X_train_s, y_train, epochs=epochs_default, batch_size=best_batch, verbose=0)

        tuned_params = {"hidden_units": hidden_units, "learning_rate": best_lr, "batch_size": best_batch, "optuna_best_trial": study.best_trial.number}
        self.tuned_params = tuned_params
        self.fitted_model = {"scaler": scaler, "model": final_model}
        return {"best_params": tuned_params, "best_mse": float(study.best_value)}

    def train(self):
        hp = self.hyperparameters or {}
        if self.tuned_params is None:
            self.auto_tune()

        params = self.tuned_params or {}
        hidden = params.get("hidden_units", hp.get("hidden_units", [32]))
        learning_rate = float(params.get("learning_rate", hp.get("learning_rate", 1e-3)))
        epochs = int(params.get("epochs", hp.get("epochs", 50)))
        batch_size = int(hp.get("batch_size", 32))

        combined = pd.concat([self.train_data, self.val_data], axis=0)
        X = combined.drop(columns=[self.target_column])
        y = combined[self.target_column].to_numpy()

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        keras.backend.clear_session()
        model = self._build_model(X_s.shape[1], hidden, learning_rate)
        model.fit(X_s, y, epochs=epochs, batch_size=batch_size, verbose=0)

        self.fitted_model = {"scaler": scaler, "model": model}
        return self.fitted_model

    def predict(self, dataset: str = "test"):
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Run `auto_tune()` or `train()` first.")
        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")
        scaler = self.fitted_model["scaler"]
        model = self.fitted_model["model"]
        df = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]
        X_test = df.drop(columns=[self.target_column])
        X_test_s = scaler.transform(X_test)
        preds = np.asarray(model.predict(X_test_s, verbose=0)).ravel()
        return pd.Series(preds, index=df.index)

class NeuralNetworkWithEmbedding(MlModel):
    """Keras feed-forward regressor with a learned embedding for `permno`.

    The `permno` identifier is expected to be available as the second level
    of a `MultiIndex` on the DataFrame (level 1). If the DataFrame does not
    have a MultiIndex, the code will look for a column named `'permno'`.

    The model takes two inputs: an integer `permno` index (embedded) and the
    numeric features (scaled). The embedding index 0 is reserved for unknown
    / out-of-vocabulary permnos.

    Hyperparameters expected in `self.hyperparameters`:
      - "embedding_dim_list": list of ints
      - "hidden_units_list": list of lists (e.g. [[32],[64,32]])
      - "learning_rate_list": list of floats
      - "epochs": int
      - "batch_size": int
    """

    def _extract_permno_array(self, df: pd.DataFrame):
        # try MultiIndex level 1 first
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 1:
            return pd.Series(df.index.get_level_values("permno").to_numpy(), index=df.index)
        # fallback to column
        if "permno" in df.columns:
            return pd.Series(df["permno"].to_numpy(), index=df.index)
        raise ValueError("DataFrame must have `permno` as MultiIndex level 1 or a column named 'permno'.")

    def _build_embedding_model(self, num_permnos: int, embedding_dim: int, num_numeric: int, hidden_units, learning_rate: float):
        # permno input
        perm_input = keras.layers.Input(shape=(1,), dtype="int32", name="permno_input")
        embed = keras.layers.Embedding(input_dim=num_permnos + 1, output_dim=embedding_dim, name="perm_embedding")(perm_input)
        embed_flat = keras.layers.Flatten()(embed)

        # numeric input
        num_input = keras.layers.Input(shape=(num_numeric,), dtype="float32", name="numeric_input")

        x = keras.layers.Concatenate()([embed_flat, num_input])
        for units in hidden_units:
            x = keras.layers.Dense(int(units), activation="relu")(x)
        out = keras.layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=[perm_input, num_input], outputs=out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        return model

    def auto_tune(self):
        hp = self.hyperparameters or {}
        embedding_list = list(hp.get("embedding_dim_list", [4, 8]))
        hidden_units_list = list(hp.get("hidden_units_list", [[32], [64, 32]]))
        learning_rate_list = list(hp.get("learning_rate_list", [1e-3]))
        epochs = int(hp.get("epochs", 20))
        batch_size = int(hp.get("batch_size", 32))

        X_train, y_train = self._get_X_y("train")
        X_val, y_val = self._get_X_y("val")

        # extract permnos
        perm_train = self._extract_permno_array(self.train_data)
        perm_val = self._extract_permno_array(self.val_data)

        # build permno mapping from training set (reserve 0 for unknown)
        unique_perm = pd.Series(pd.unique(perm_train)).tolist()
        perm2id = {p: i + 1 for i, p in enumerate(unique_perm)}

        def encode_perm(series):
            return np.array([perm2id.get(p, 0) for p in series.tolist()], dtype="int32")

        perm_train_ids = encode_perm(perm_train)
        perm_val_ids = encode_perm(perm_val)

        # numeric features (drop target and possible permno column if present)
        def _numeric_df(df):
            if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 1:
                # numeric features are columns only
                return df.drop(columns=[self.target_column], errors="ignore")
            if "permno" in df.columns:
                return df.drop(columns=[self.target_column, "permno"], errors="ignore")
            return df.drop(columns=[self.target_column], errors="ignore")

        X_train_num = _numeric_df(self.train_data)
        X_val_num = _numeric_df(self.val_data)

        # scale numeric features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_val_s = scaler.transform(X_val_num)

        best_score = float("inf")
        best_cfg = None
        best_model_bundle = None

        for emb_dim in embedding_list:
            for hidden in hidden_units_list:
                for lr in learning_rate_list:
                    keras.backend.clear_session()
                    model = self._build_embedding_model(num_permnos=len(unique_perm), embedding_dim=int(emb_dim), num_numeric=X_train_s.shape[1], hidden_units=hidden, learning_rate=float(lr))
                    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
                    model.fit([perm_train_ids, X_train_s], y_train.to_numpy(), validation_data=([perm_val_ids, X_val_s], y_val.to_numpy()), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
                    preds = np.asarray(model.predict([perm_val_ids, X_val_s], verbose=0)).ravel()
                    mse = mean_squared_error(y_val.to_numpy(), preds)
                    if mse < best_score:
                        best_score = mse
                        best_cfg = {"embedding_dim": int(emb_dim), "hidden_units": hidden, "learning_rate": float(lr), "epochs": epochs, "batch_size": batch_size}
                        best_model_bundle = {"scaler": scaler, "model": model, "perm2id": perm2id}

        self.tuned_params = best_cfg
        # fitted_model kept as model trained on training set (not combined)
        self.fitted_model = best_model_bundle
        return {"best_params": best_cfg, "best_mse": best_score}

    def train(self):
        hp = self.hyperparameters or {}
        if self.tuned_params is None:
            self.auto_tune()

        params = self.tuned_params or {}
        emb_dim = int(params.get("embedding_dim", hp.get("embedding_dim", 8)))
        hidden = params.get("hidden_units", hp.get("hidden_units", [32]))
        lr = float(params.get("learning_rate", hp.get("learning_rate", 1e-3)))
        epochs = int(params.get("epochs", hp.get("epochs", 50)))
        batch_size = int(params.get("batch_size", hp.get("batch_size", 32)))

        # use combined train+val for final mapping and scaler
        combined = pd.concat([self.train_data, self.val_data], axis=0)
        perm_combined = self._extract_permno_array(combined)
        unique_perm = pd.Series(pd.unique(perm_combined)).tolist()
        perm2id = {p: i + 1 for i, p in enumerate(unique_perm)}

        # numeric features
        if isinstance(combined.index, pd.MultiIndex) and combined.index.nlevels > 1:
            X_num = combined.drop(columns=[self.target_column], errors="ignore")
        else:
            X_num = combined.drop(columns=[self.target_column, "permno"], errors="ignore")
        y = combined[self.target_column].to_numpy()

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_num)

        # encode permnos
        perm_ids = np.array([perm2id.get(p, 0) for p in perm_combined.tolist()], dtype="int32")

        keras.backend.clear_session()
        model = self._build_embedding_model(num_permnos=len(unique_perm), embedding_dim=emb_dim, num_numeric=X_s.shape[1], hidden_units=hidden, learning_rate=lr)
        model.fit([perm_ids, X_s], y, epochs=epochs, batch_size=batch_size, verbose=0)

        self.fitted_model = {"scaler": scaler, "model": model, "perm2id": perm2id}
        return self.fitted_model

    def predict(self, dataset: str = "test"):
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Run `auto_tune()` or `train()` first.")
        if dataset not in {"train", "val", "test"}:
            raise ValueError("dataset must be one of 'train','val','test'")

        df = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[dataset]

        # extract permno series and numeric features
        perm_series = self._extract_permno_array(df)
        perm2id = self.fitted_model["perm2id"]
        perm_ids = np.array([perm2id.get(p, 0) for p in perm_series.tolist()], dtype="int32")

        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 1:
            X_num = df.drop(columns=[self.target_column], errors="ignore")
        else:
            X_num = df.drop(columns=[self.target_column, "permno"], errors="ignore")

        scaler = self.fitted_model["scaler"]
        model = self.fitted_model["model"]
        X_s = scaler.transform(X_num)
        preds = np.asarray(model.predict([perm_ids, X_s], verbose=0)).ravel()
        return pd.Series(preds, index=df.index)
