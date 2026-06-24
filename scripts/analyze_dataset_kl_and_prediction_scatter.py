#!/usr/bin/env python3
"""Generate dataset KL-divergence and prediction scatter visualizations."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr

from visualize_drformer_sota_cases import DATASETS, MODELS, _load_series, select_cases


warnings.filterwarnings("ignore")

DISPLAY_NAMES = {
    "ECL": "ECL",
    "traffic": "Traffic",
    "PEMS03": "PEMS03",
    "ETTh2": "ETTh2",
}

DATASET_SOURCES = {
    "ECL": ("csv", Path("electricity/electricity.csv")),
    "traffic": ("csv", Path("traffic/traffic.csv")),
    "PEMS03": ("npz", Path("PEMS/PEMS03.npz")),
    "ETTh2": ("csv", Path("ETT-small/ETTh2.csv")),
}

PAIR_COLORS = ("#2A6FBB", "#C44E52")


def _load_dataset_values(dataset_dir: Path, dataset: str) -> np.ndarray:
    source_type, relative_path = DATASET_SOURCES[dataset]
    path = dataset_dir / relative_path
    if source_type == "csv":
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=np.float64, copy=False).reshape(-1)
    elif source_type == "npz":
        loaded = np.load(path, allow_pickle=True)
        values = loaded["data"][:, :, 0].astype(np.float64, copy=False).reshape(-1)
    else:
        raise ValueError(f"Unsupported dataset source type: {source_type}")

    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite numeric values found for {dataset}")
    return values


def _prepare_distribution_values(
    values: np.ndarray,
    max_points: int,
    transform: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if values.size > max_points:
        idx = rng.choice(values.size, size=max_points, replace=False)
        values = values[idx]

    if transform == "raw":
        prepared = values
    elif transform == "log1p":
        min_value = float(np.min(values))
        shifted = values - min_value if min_value < 0 else values
        prepared = np.log1p(shifted)
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    prepared = prepared[np.isfinite(prepared)]
    if prepared.size == 0:
        raise ValueError("No finite values remain after transformation")
    return prepared


def _pair_x_grid(data_p: np.ndarray, data_q: np.ndarray) -> np.ndarray:
    lower = float(min(np.percentile(data_p, 0.5), np.percentile(data_q, 0.5)))
    upper = float(max(np.percentile(data_p, 99.5), np.percentile(data_q, 99.5)))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(min(data_p.min(), data_q.min()))
        upper = float(max(data_p.max(), data_q.max()))
    pad = 0.08 * (upper - lower) if upper > lower else 1.0
    return np.linspace(lower - pad, upper + pad, 1000)


def _gaussian_pdf_kl(data_p: np.ndarray, data_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu_p = float(np.mean(data_p))
    mu_q = float(np.mean(data_q))
    sigma_p = max(float(np.std(data_p)), 1e-8)
    sigma_q = max(float(np.std(data_q)), 1e-8)
    x = _pair_x_grid(data_p, data_q)
    pdf_p = stats.norm.pdf(x, mu_p, sigma_p)
    pdf_q = stats.norm.pdf(x, mu_q, sigma_q)
    prob_p = pdf_p / (pdf_p.sum() + 1e-12)
    prob_q = pdf_q / (pdf_q.sum() + 1e-12)
    kl_value = float(np.sum(rel_entr(prob_p + 1e-12, prob_q + 1e-12)))
    return x, pdf_p, pdf_q, kl_value


def compute_pairwise_kl_distributions(
    dataset_dir: Path,
    max_points: int,
    transform: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]]]:
    rng = np.random.default_rng(seed)
    values = {
        dataset: _prepare_distribution_values(
            _load_dataset_values(dataset_dir, dataset),
            max_points,
            transform,
            rng,
        )
        for dataset in DATASETS
    }
    rows: list[dict[str, float | str]] = []
    for dataset_p, dataset_q in combinations(DATASETS, 2):
        x, _, _, kl_value = _gaussian_pdf_kl(values[dataset_p], values[dataset_q])
        del x
        rows.append(
            {
                "dataset_p": DISPLAY_NAMES[dataset_p],
                "dataset_q": DISPLAY_NAMES[dataset_q],
                "kl_p_parallel_q": kl_value,
                "mean_p": float(np.mean(values[dataset_p])),
                "std_p": float(np.std(values[dataset_p])),
                "mean_q": float(np.mean(values[dataset_q])),
                "std_q": float(np.std(values[dataset_q])),
            }
        )
    return values, rows


def plot_pairwise_kl_distributions(
    dataset_values: dict[str, np.ndarray],
    output_stem: Path,
    transform: str,
) -> list[dict[str, float | str]]:
    pairs = list(combinations(DATASETS, 2))
    fig, axes = plt.subplots(
        2,
        len(pairs),
        figsize=(22.5, 7.2),
        dpi=300,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.93, bottom=0.13, wspace=0.22, hspace=0.28)
    rows: list[dict[str, float | str]] = []

    for col_idx, (dataset_p, dataset_q) in enumerate(pairs):
        label_p = DISPLAY_NAMES[dataset_p]
        label_q = DISPLAY_NAMES[dataset_q]
        data_p = dataset_values[dataset_p]
        data_q = dataset_values[dataset_q]
        x, pdf_p, pdf_q, kl_value = _gaussian_pdf_kl(data_p, data_q)
        hist_range = (float(x.min()), float(x.max()))

        hist_ax = axes[0, col_idx]
        pdf_ax = axes[1, col_idx]
        hist_ax.hist(
            data_p,
            bins=70,
            range=hist_range,
            density=True,
            color=PAIR_COLORS[0],
            alpha=0.48,
            label=label_p,
        )
        hist_ax.hist(
            data_q,
            bins=70,
            range=hist_range,
            density=True,
            color=PAIR_COLORS[1],
            alpha=0.42,
            label=label_q,
        )
        hist_ax.set_title(f"{label_p} vs. {label_q}", fontsize=15, fontweight="bold", pad=8)
        xlabel = "Raw value" if transform == "raw" else r"$\log(1+x)$ value"
        hist_ax.set_xlabel(xlabel, fontsize=12)
        if col_idx == 0:
            hist_ax.set_ylabel("Histogram density", fontsize=13)
        hist_ax.tick_params(labelsize=10, length=2.5)
        hist_ax.grid(True, linestyle="--", alpha=0.24, linewidth=0.6)
        hist_ax.legend(frameon=False, fontsize=11, loc="upper right")
        hist_ax.set_box_aspect(1)

        pdf_ax.plot(x, pdf_p, color=PAIR_COLORS[0], lw=1.9, label=f"{label_p} PDF")
        pdf_ax.plot(x, pdf_q, color=PAIR_COLORS[1], lw=1.9, label=f"{label_q} PDF")
        pdf_ax.fill_between(x, pdf_p, color=PAIR_COLORS[0], alpha=0.18)
        pdf_ax.fill_between(x, pdf_q, color=PAIR_COLORS[1], alpha=0.16)
        pdf_ax.set_title(
            rf"$D_{{KL}}({label_p}\parallel {label_q})={kl_value:.2f}$",
            fontsize=15,
            fontweight="bold",
            pad=8,
        )
        pdf_ax.set_xlabel(xlabel, fontsize=12)
        if col_idx == 0:
            pdf_ax.set_ylabel("Gaussian PDF", fontsize=13)
        pdf_ax.tick_params(labelsize=10, length=2.5)
        pdf_ax.grid(True, linestyle="--", alpha=0.24, linewidth=0.6)
        pdf_ax.legend(frameon=False, fontsize=11, loc="upper right")
        pdf_ax.set_box_aspect(1)

        rows.append(
            {
                "dataset_p": label_p,
                "dataset_q": label_q,
                "kl_p_parallel_q": kl_value,
                "mean_p": float(np.mean(data_p)),
                "std_p": float(np.std(data_p)),
                "mean_q": float(np.mean(data_q)),
                "std_q": float(np.std(data_q)),
            }
        )

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_pairwise_kl_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["dataset_p", "dataset_q", "kl_p_parallel_q", "mean_p", "std_p", "mean_q", "std_q"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _forecast_pair(result_dir: Path, sample_id: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    pred, true = _load_series(result_dir, sample_id)
    return true[-pred_len:].astype(float), pred[-pred_len:].astype(float)


def _scatter_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - true
    mse = float(np.mean(residual**2))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((true - true.mean()) ** 2))
    nse = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else float("nan")
    if len(true) > 1 and true.std() > 0 and pred.std() > 0:
        corr, corr_p = stats.pearsonr(true, pred)
    else:
        corr, corr_p = float("nan"), float("nan")
    if len(true) > 1 and true.std() > 0:
        regression = stats.linregress(true, pred)
        slope = float(regression.slope)
        intercept = float(regression.intercept)
        p_value = float(regression.pvalue)
    else:
        slope = float("nan")
        intercept = float("nan")
        p_value = float("nan")
    return {
        "corr": float(corr),
        "corr_p": float(corr_p),
        "nse": nse,
        "mse": mse,
        "rmse": rmse,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
    }


def _density_order(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.vstack([true, pred])
    try:
        density = stats.gaussian_kde(xy)(xy)
        density = (density - density.min()) / (density.max() - density.min() + 1e-12)
    except Exception:
        density = np.zeros_like(true)
    order = np.argsort(density)
    return true[order], pred[order], density[order]


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "nan"
    if p_value < 1e-3:
        return "<1e-3"
    return f"{p_value:.3f}"


def plot_prediction_scatter(
    results_dir: Path,
    output_stem: Path,
    pred_len: int,
) -> tuple[list, list[dict[str, float | str | int]]]:
    result_dirs, selected, _ = select_cases(results_dir, pred_len)
    selected_by_dataset = {score.dataset: score for score in selected}
    metric_rows: list[dict[str, float | str | int]] = []

    fig, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(19.5, 12.2),
        dpi=300,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.925, top=0.91, bottom=0.08, wspace=0.26, hspace=0.28)

    scatter_cmap = "RdBu_r"
    for row_idx, dataset in enumerate(DATASETS):
        sample_id = selected_by_dataset[dataset].sample_id
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            true, pred = _forecast_pair(result_dirs[dataset][model], sample_id, pred_len)
            metrics = _scatter_metrics(true, pred)
            true_s, pred_s, density_s = _density_order(true, pred)

            lo = float(min(true.min(), pred.min()))
            hi = float(max(true.max(), pred.max()))
            pad = 0.06 * (hi - lo) if hi > lo else 1.0
            lo -= pad
            hi += pad

            ax.scatter(
                true_s,
                pred_s,
                c=density_s,
                s=18,
                cmap=scatter_cmap,
                vmin=0,
                vmax=1,
                alpha=0.85,
                edgecolors="none",
            )
            line_x = np.array([lo, hi])
            ax.plot(line_x, line_x, color="#D62728", lw=1.25, ls="--", label="1:1 line")
            if np.isfinite(metrics["slope"]) and np.isfinite(metrics["intercept"]):
                ax.plot(
                    line_x,
                    metrics["slope"] * line_x + metrics["intercept"],
                    color="#111111",
                    lw=1.2,
                    label="Regression",
                )

            if row_idx == 0:
                ax.set_title(model, fontsize=14, fontweight="bold", pad=9)
            if col_idx == 0:
                ax.set_ylabel(DISPLAY_NAMES[dataset], fontsize=13, rotation=0, labelpad=44, va="center")
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Ground truth", fontsize=11)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8.5, length=2.5)
            ax.set_aspect("equal", adjustable="box")

            text = (
                f"Corr={metrics['corr']:.3f}\n"
                f"NSE={metrics['nse']:.3f}\n"
                f"MSE={metrics['mse']:.3g}\n"
                f"RMSE={metrics['rmse']:.3g}\n"
                f"p={_format_p_value(metrics['p_value'])}"
            )
            ax.text(
                0.045,
                0.955,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.2,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#BDBDBD",
                    "linewidth": 0.45,
                    "alpha": 0.9,
                },
            )

            metric_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "model": model,
                    **metrics,
                }
            )

    fig.text(0.018, 0.5, "Prediction", rotation=90, va="center", ha="center", fontsize=14)
    legend_handles = [
        Line2D([0], [0], color="#D62728", lw=1.25, ls="--", label="1:1 line"),
        Line2D([0], [0], color="#111111", lw=1.2, label="Regression"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.975),
    )
    cbar_ax = fig.add_axes([0.945, 0.18, 0.014, 0.62])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=scatter_cmap),
        cax=cbar_ax,
    )
    cbar.set_label("Point density (normalized)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selected, metric_rows


def write_scatter_metrics(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = [
        "dataset",
        "sample_id",
        "model",
        "corr",
        "corr_p",
        "nse",
        "mse",
        "rmse",
        "slope",
        "intercept",
        "p_value",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--results-dir", type=Path, default=Path("test_results_copy"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/kl_scatter_analysis"))
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--max-points", type=int, default=600_000)
    parser.add_argument("--distribution-transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_values, _ = compute_pairwise_kl_distributions(
        dataset_dir=dataset_dir,
        max_points=args.max_points,
        transform=args.distribution_transform,
        seed=args.seed,
    )
    kl_stem = output_dir / "dataset_pairwise_distribution_kl"
    kl_rows = plot_pairwise_kl_distributions(dataset_values, kl_stem, args.distribution_transform)
    write_pairwise_kl_csv(output_dir / "dataset_pairwise_distribution_kl.csv", kl_rows)

    scatter_stem = output_dir / "prediction_scatter_baselines_4x5"
    selected, metric_rows = plot_prediction_scatter(results_dir, scatter_stem, args.pred_len)
    write_scatter_metrics(output_dir / "prediction_scatter_metrics.csv", metric_rows)

    print("Selected samples from test_results_copy:")
    for score in selected:
        print(
            f"  {DISPLAY_NAMES[score.dataset]}: id={score.sample_id}, "
            f"gain_vs_{score.best_other_model}={100 * score.rel_gain_vs_best_other:.2f}%"
        )
    print(f"KL figure:      {kl_stem.with_suffix('.pdf')}")
    print(f"KL PNG:         {kl_stem.with_suffix('.png')}")
    print(f"Scatter figure: {scatter_stem.with_suffix('.pdf')}")
    print(f"Scatter PNG:    {scatter_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
