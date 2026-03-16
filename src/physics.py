import numpy as np
import pandas as pd


def vbar_from_components(
    df: pd.DataFrame,
    ups_disk: float = 0.5,
    ups_bulge: float = 0.7,
) -> np.ndarray:
    term = (
        np.abs(df["Vgas"].values) * df["Vgas"].values
        + ups_disk * np.abs(df["Vdisk"].values) * df["Vdisk"].values
        + ups_bulge * np.abs(df["Vbul"].values) * df["Vbul"].values
    )
    term = np.clip(term, 0, None)
    return np.sqrt(term)


def r200_from_m200(M200: float, rho_crit: float = 136.0) -> float:
    return (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)


def nfw_rhos_from_m200_rs(M200: float, rs: float, rho_crit: float = 136.0) -> float:
    R200 = r200_from_m200(M200, rho_crit=rho_crit)
    c = R200 / rs
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    return M200 / (4.0 * np.pi * rs**3 * f_c)


def nfw_enclosed_mass(r: np.ndarray, M200: float, rs: float, rho_crit: float = 136.0) -> np.ndarray:
    rho_s = nfw_rhos_from_m200_rs(M200, rs, rho_crit=rho_crit)
    x = np.asarray(r) / rs
    f_x = np.log(1.0 + x) - x / (1.0 + x)
    return 4.0 * np.pi * rho_s * rs**3 * f_x


def nfw_velocity(
    r: np.ndarray,
    M200: float,
    rs: float,
    G: float = 4.30091e-6,
    rho_crit: float = 136.0,
) -> np.ndarray:
    Menc = nfw_enclosed_mass(r, M200, rs, rho_crit=rho_crit)
    return np.sqrt(G * Menc / np.asarray(r))


def nfw_density(r: np.ndarray, M200: float, rs: float, rho_crit: float = 136.0) -> np.ndarray:
    rho_s = nfw_rhos_from_m200_rs(M200, rs, rho_crit=rho_crit)
    x = np.asarray(r) / rs
    return rho_s / (x * (1.0 + x) ** 2)
