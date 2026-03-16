from dataclasses import dataclass
import os


@dataclass
class ProjectConfig:
    run_mode: str = "strict"   # "strict" or "extended"
    force_clean_run: bool = True
    force_redownload: bool = True
    use_sample_limit: int | None = None
    save_every: int = 10

    data_dir: str = "data"
    raw_dir: str = os.path.join(data_dir, "raw")
    extract_dir: str = os.path.join(raw_dir, "extracted")
    results_dir: str = "results"
    figures_dir: str = "figures"

    zip_name: str = "Rotmod_LTG.zip"

    urls: tuple = (
        "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1",
        "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
        "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    )

    G: float = 4.30091e-6
    rho_crit: float = 136.0
    ups_disk: float = 0.5
    ups_bulge: float = 0.7
    error_floor: float = 2.0

    @property
    def min_points_per_galaxy(self) -> int:
        if self.run_mode == "strict":
            return 8
        if self.run_mode == "extended":
            return 4
        raise ValueError("run_mode must be 'strict' or 'extended'")

    @property
    def output_tag(self) -> str:
        return self.run_mode

    @property
    def zip_path(self) -> str:
        return os.path.join(self.raw_dir, self.zip_name)

    @property
    def fit_csv(self) -> str:
        return os.path.join(self.results_dir, f"nfw_fit_results_{self.output_tag}.csv")

    @property
    def feature_csv(self) -> str:
        return os.path.join(self.results_dir, f"galaxy_features_and_targets_{self.output_tag}.csv")

    @property
    def metrics_csv(self) -> str:
        return os.path.join(self.results_dir, f"ml_metrics_5fold_{self.output_tag}.csv")

    @property
    def pred_csv(self) -> str:
        return os.path.join(self.results_dir, f"ml_predictions_full_{self.output_tag}.csv")

    @property
    def summary_json(self) -> str:
        return os.path.join(self.results_dir, f"summary_report_{self.output_tag}.json")
