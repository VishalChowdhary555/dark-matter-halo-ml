import os
import numpy as np
import matplotlib.pyplot as plt

from .physics import vbar_from_components, nfw_velocity, nfw_density


def plot_sample_rotation_curves(fit_ok, galaxy_dfs, output_path: str, n: int = 9):
    name_to_df = {df["galaxy"].iloc[0]: df for df in galaxy_dfs}
    sample_names = fit_ok.sort_values("chi2_red").head(n)["galaxy"].tolist()

    ncols = 3
    nrows = int(np.ceil(len(sample_names) / ncols))
    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, name in enumerate(sample_names, 1):
        df = name_to_df[name]
        r = df["Rad"].values
        vobs = df["Vobs"].values
        err = df["errV"].values
        vbar = vbar_from_components(df)

        plt.subplot(nrows, ncols, i)
        plt.errorbar(r, vobs, yerr=err, fmt="o", ms=3, label="Observed")
        plt.plot(r, vbar, lw=2, label="Baryons")
        plt.plot(r, np.abs(df["Vgas"].values), "--", lw=1, label="Gas")
        plt.plot(r, np.abs(df["Vdisk"].values), "--", lw=1, label="Disk")
        if np.any(np.abs(df["Vbul"].values) > 0):
            plt.plot(r, np.abs(df["Vbul"].values), "--", lw=1, label="Bulge")

        plt.title(name)
        plt.xlabel("Radius [kpc]")
        plt.ylabel("Velocity [km/s]")
        plt.grid(alpha=0.3)
        if i == 1:
            plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_fit_grid(fit_ok, galaxy_dfs, output_path: str, n: int = 9):
    name_to_df = {df["galaxy"].iloc[0]: df for df in galaxy_dfs}
    name_to_fit = {row["galaxy"]: row for _, row in fit_ok.iterrows()}

    sample_names = fit_ok.sort_values("chi2_red").head(n)["galaxy"].tolist()

    ncols = 3
    nrows = int(np.ceil(len(sample_names) / ncols))
    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, name in enumerate(sample_names, 1):
        df = name_to_df[name]
        fr = name_to_fit[name]
        r = df["Rad"].values
        vobs = df["Vobs"].values
        err = np.maximum(df["errV"].values, 2.0)
        vbar = vbar_from_components(df)
        vdm = nfw_velocity(r, 10**fr["log10_M200"], 10**fr["log10_rs"])
        vtot = np.sqrt(vbar**2 + vdm**2)

        plt.subplot(nrows, ncols, i)
        plt.errorbar(r, vobs, yerr=err, fmt="o", ms=3, label="Observed")
        plt.plot(r, vbar, lw=2, label="Baryons")
        plt.plot(r, vdm, lw=2, label="Dark matter")
        plt.plot(r, vtot, lw=3, label="Total fit")
        plt.title(f"{name}\nlogM={fr['log10_M200']:.2f}, logrs={fr['log10_rs']:.2f}, chi2r={fr['chi2_red']:.2f}")
        plt.xlabel("Radius [kpc]")
        plt.ylabel("Velocity [km/s]")
        plt.grid(alpha=0.3)
        if i == 1:
            plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_ml_pred_vs_actual(ml_df, output_path: str):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(ml_df["log10_M200"], ml_df["pred_log10_M200"], s=18)
    a = min(ml_df["log10_M200"].min(), ml_df["pred_log10_M200"].min())
    b = max(ml_df["log10_M200"].max(), ml_df["pred_log10_M200"].max())
    plt.plot([a, b], [a, b], "--")
    plt.xlabel("Actual log10_M200")
    plt.ylabel("Predicted log10_M200")
    plt.title("Halo mass")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(ml_df["log10_rs"], ml_df["pred_log10_rs"], s=18)
    a = min(ml_df["log10_rs"].min(), ml_df["pred_log10_rs"].min())
    b = max(ml_df["log10_rs"].max(), ml_df["pred_log10_rs"].max())
    plt.plot([a, b], [a, b], "--")
    plt.xlabel("Actual log10_rs")
    plt.ylabel("Predicted log10_rs")
    plt.title("Halo scale radius")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_density_profiles(fit_ok, output_path: str, n: int = 4):
    name_to_fit = {row["galaxy"]: row for _, row in fit_ok.iterrows()}
    sample_names = fit_ok.sort_values("chi2_red").head(n)["galaxy"].tolist()

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(sample_names, 1):
        fr = name_to_fit[name]
        r = np.logspace(-1, 2, 250)
        rho = nfw_density(r, 10**fr["log10_M200"], 10**fr["log10_rs"])

        plt.subplot(2, 2, i)
        plt.loglog(r, rho, lw=2)
        plt.xlabel("Radius [kpc]")
        plt.ylabel(r"Density [M$_\odot$/kpc$^3$]")
        plt.title(name)
        plt.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
