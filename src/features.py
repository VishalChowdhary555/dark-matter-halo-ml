import numpy as np
import pandas as pd

from .physics import vbar_from_components


def summarize_galaxy(df: pd.DataFrame, error_floor: float = 2.0) -> dict:
    r = df["Rad"].values
    vobs = df["Vobs"].values
    err = np.maximum(df["errV"].values, error_floor)
    vgas = np.abs(df["Vgas"].values)
    vdisk = np.abs(df["Vdisk"].values)
    vbul = np.abs(df["Vbul"].values)
    vbar = vbar_from_components(df)

    dm_proxy = np.sqrt(np.clip(vobs**2 - vbar**2, 0, None))

    return {
        "galaxy": df["galaxy"].iloc[0],
        "distance_mpc": df["distance_mpc"].iloc[0],
        "n_points": len(df),
        "short_curve_flag": int(len(df) < 8),
        "r_max": np.max(r),
        "r_mean": np.mean(r),
        "vobs_max": np.max(vobs),
        "vobs_mean": np.mean(vobs),
        "vobs_std": np.std(vobs),
        "vbar_max": np.max(vbar),
        "vbar_mean": np.mean(vbar),
        "vgas_max": np.max(vgas),
        "vdisk_max": np.max(vdisk),
        "vbul_max": np.max(vbul),
        "err_mean": np.mean(err),
        "outer_to_inner_v_ratio": np.mean(vobs[-3:]) / max(np.mean(vobs[:3]), 1e-6),
        "bar_to_obs_max_ratio": np.max(vbar) / max(np.max(vobs), 1e-6),
        "dm_proxy_max": np.max(dm_proxy),
        "dm_proxy_mean": np.mean(dm_proxy),
        "inner_slope": (vobs[min(3, len(vobs)-1)] - vobs[0]) / (r[min(3, len(r)-1)] - r[0] + 1e-6),
        "outer_slope": (vobs[-1] - vobs[max(len(vobs)-4, 0)]) / (r[-1] - r[max(len(r)-4, 0)] + 1e-6),
    }


def build_feature_table(galaxy_dfs: list[pd.DataFrame], fit_df: pd.DataFrame, error_floor: float = 2.0) -> pd.DataFrame:
    feature_df = pd.DataFrame([summarize_galaxy(df, error_floor=error_floor) for df in galaxy_dfs])

    fit_ok = fit_df[fit_df["success"] == True].copy()

    ml_df = feature_df.merge(
        fit_ok[["galaxy", "log10_M200", "log10_rs", "chi2_red", "low_point_fit"]],
        on="galaxy",
        how="inner",
    ).dropna().reset_index(drop=True)

    return ml_df
