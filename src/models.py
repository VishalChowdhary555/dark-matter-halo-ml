import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor


FEATURE_COLUMNS = [
    "distance_mpc",
    "n_points",
    "short_curve_flag",
    "r_max",
    "r_mean",
    "vobs_max",
    "vobs_mean",
    "vobs_std",
    "vbar_max",
    "vbar_mean",
    "vgas_max",
    "vdisk_max",
    "vbul_max",
    "err_mean",
    "outer_to_inner_v_ratio",
    "bar_to_obs_max_ratio",
    "dm_proxy_max",
    "dm_proxy_mean",
    "inner_slope",
    "outer_slope",
]


def get_models() -> dict:
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "GaussianProcess": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0),
                normalize_y=True,
                random_state=42,
            )),
        ]),
        "NeuralNetwork": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=3000,
                random_state=42,
            )),
        ]),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_ml_benchmark(ml_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = ml_df[FEATURE_COLUMNS].fillna(0.0).values
    y_mass = ml_df["log10_M200"].values
    y_rs = ml_df["log10_rs"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()

    metrics_rows = []
    pred_store = {}

    for name, model in models.items():
        pred_mass = cross_val_predict(model, X, y_mass, cv=kf)
        pred_rs = cross_val_predict(model, X, y_rs, cv=kf)

        metrics_rows.append({
            "model": name,
            "target": "log10_M200",
            **regression_metrics(y_mass, pred_mass),
        })
        metrics_rows.append({
            "model": name,
            "target": "log10_rs",
            **regression_metrics(y_rs, pred_rs),
        })

        pred_store[(name, "mass")] = pred_mass
        pred_store[(name, "rs")] = pred_rs

    return pd.DataFrame(metrics_rows), pred_store


def fit_final_nn_models(ml_df: pd.DataFrame):
    X = ml_df[FEATURE_COLUMNS].fillna(0.0).values
    y_mass = ml_df["log10_M200"].values
    y_rs = ml_df["log10_rs"].values

    models = get_models()
    mass_model = models["NeuralNetwork"]
    rs_model = models["NeuralNetwork"]

    mass_model.fit(X, y_mass)
    rs_model.fit(X, y_rs)

    ml_df = ml_df.copy()
    ml_df["pred_log10_M200"] = mass_model.predict(X)
    ml_df["pred_log10_rs"] = rs_model.predict(X)

    return ml_df, mass_model, rs_model
