import os
import time
import pandas as pd

from .config import ProjectConfig
from .utils import ensure_directories, reset_directory, save_json, print_and_log
from .data_loader import (
    download_zip,
    extract_zip,
    find_rotmod_files,
    parse_all_galaxies,
)
from .halo_fitting import fit_all_galaxies
from .features import build_feature_table
from .models import run_ml_benchmark, fit_final_nn_models
from .plotting import (
    plot_sample_rotation_curves,
    plot_fit_grid,
    plot_ml_pred_vs_actual,
    plot_density_profiles,
)


def setup_project(config: ProjectConfig) -> None:
    if config.force_clean_run:
        reset_directory(config.data_dir)

    ensure_directories(
        config.data_dir,
        config.raw_dir,
        config.extract_dir,
        config.results_dir,
        config.figures_dir,
    )


def run_pipeline(config: ProjectConfig) -> dict:
    log_path = os.path.join(config.results_dir, f"run_log_{config.output_tag}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SPARC DARK MATTER HALO ML RUN LOG\n\n")

    logger = lambda msg: print_and_log(msg, log_path)

    logger(f"Run mode: {config.run_mode}")
    logger(f"Minimum points per galaxy: {config.min_points_per_galaxy}")

    if config.force_redownload or not os.path.exists(config.zip_path):
        download_zip(config.urls, config.zip_path, logger=logger)
    else:
        logger(f"Using existing zip: {config.zip_path}")

    extract_zip(config.zip_path, config.extract_dir, logger=logger)

    dat_files = find_rotmod_files(config.extract_dir)
    logger(f"Found {len(dat_files)} rotmod files after extraction.")

    if config.use_sample_limit is not None:
        dat_files = dat_files[: config.use_sample_limit]
        logger(f"Sample limit active. Using first {len(dat_files)} files.")

    galaxy_dfs, rejected_files = parse_all_galaxies(
        dat_files,
        min_points=config.min_points_per_galaxy,
    )

    logger(f"Parsed valid galaxies: {len(galaxy_dfs)}")
    logger(f"Rejected files: {len(rejected_files)}")

    if len(galaxy_dfs) == 0:
        raise RuntimeError("No valid galaxies parsed.")

    fit_start = time.time()
    fit_df = fit_all_galaxies(galaxy_dfs, error_floor=config.error_floor)
    fit_elapsed = time.time() - fit_start

    fit_df.to_csv(config.fit_csv, index=False)
    logger(f"Fit stage runtime: {fit_elapsed / 60.0:.4f} minutes")
    logger(f"Fit rows saved: {len(fit_df)}")

    fit_ok = fit_df[fit_df["success"] == True].copy()
    boundary_hits = fit_ok[
        fit_ok["hit_lower_mass"]
        | fit_ok["hit_upper_mass"]
        | fit_ok["hit_lower_rs"]
        | fit_ok["hit_upper_rs"]
    ].copy()
    low_point_fits = fit_ok[fit_ok["low_point_fit"] == True].copy()

    logger(f"Successful fits: {len(fit_ok)}")
    logger(f"Boundary-hit fits: {len(boundary_hits)}")
    logger(f"Low-point fits (<8 points): {len(low_point_fits)}")

    ml_df = build_feature_table(
        galaxy_dfs,
        fit_df,
        error_floor=config.error_floor,
    )
    ml_df.to_csv(config.feature_csv, index=False)
    logger(f"ML dataset rows: {len(ml_df)}")

    ml_metrics_df, _ = run_ml_benchmark(ml_df)
    ml_metrics_df.to_csv(config.metrics_csv, index=False)

    ml_pred_df, _, _ = fit_final_nn_models(ml_df)
    ml_pred_df.to_csv(config.pred_csv, index=False)

    plot_sample_rotation_curves(
        fit_ok,
        galaxy_dfs,
        os.path.join(config.figures_dir, f"sample_rotation_curves_{config.output_tag}.png"),
    )
    plot_fit_grid(
        fit_ok,
        galaxy_dfs,
        os.path.join(config.figures_dir, f"nfw_fit_grid_{config.output_tag}.png"),
    )
    plot_ml_pred_vs_actual(
        ml_pred_df,
        os.path.join(config.figures_dir, f"ml_pred_vs_actual_{config.output_tag}.png"),
    )
    plot_density_profiles(
        fit_ok,
        os.path.join(config.figures_dir, f"density_profiles_{config.output_tag}.png"),
    )

    summary = {
        "run_mode": config.run_mode,
        "min_points_per_galaxy": int(config.min_points_per_galaxy),
        "found_rotmod_files": int(len(dat_files)),
        "parsed_valid_galaxies": int(len(galaxy_dfs)),
        "fit_rows_saved": int(len(fit_df)),
        "successful_fits": int(len(fit_ok)),
        "boundary_hit_fits": int(len(boundary_hits)),
        "low_point_fits": int(len(low_point_fits)),
        "median_log10_M200": float(fit_ok["log10_M200"].median()) if len(fit_ok) else None,
        "median_log10_rs": float(fit_ok["log10_rs"].median()) if len(fit_ok) else None,
        "median_chi2_red": float(fit_ok["chi2_red"].median()) if len(fit_ok) else None,
        "runtime_minutes_fit_stage": float(fit_elapsed / 60.0),
        "rejected_filenames": [os.path.basename(fp) for fp in rejected_files],
    }

    for _, row in ml_metrics_df.iterrows():
        summary[f"{row['model']}_{row['target']}_MAE"] = float(row["MAE"])
        summary[f"{row['model']}_{row['target']}_RMSE"] = float(row["RMSE"])
        summary[f"{row['model']}_{row['target']}_R2"] = float(row["R2"])

    save_json(summary, config.summary_json)
    logger("Final summary:")
    logger(str(summary))

    return summary


def main():
    config = ProjectConfig()
    setup_project(config)
    summary = run_pipeline(config)

    print("\nRun complete.")
    print("Summary:")
    for key, value in summary.items():
        if key != "rejected_filenames":
            print(f"{key}: {value}")

    print("\nSaved files:")
    print(f"- {config.fit_csv}")
    print(f"- {config.feature_csv}")
    print(f"- {config.metrics_csv}")
    print(f"- {config.pred_csv}")
    print(f"- {config.summary_json}")
    print(f"- {config.figures_dir}")


if __name__ == "__main__":
    main()
