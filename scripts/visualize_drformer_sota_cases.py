#!/usr/bin/env python3
"""Select and visualize cases where DRFormer clearly outperforms baselines."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATASETS = ("ECL", "traffic", "PEMS03", "ETTh2")
MODELS = ("DRFormer", "S-Mamba", "iTransformer", "PatchTST", "Autoformer")


@dataclass(frozen=True)
class CaseScore:
    dataset: str
    sample_id: int
    drformer_mse: float
    drformer_mae: float
    best_other_model: str
    best_other_mse: float
    rel_gain_vs_best_other: float
    mean_rel_gain_vs_others: float
    mse_by_model: dict[str, float]
    mae_by_model: dict[str, float]


def _find_result_dir(base_dir: Path, dataset: str, model: str) -> Path:
    prefix = f"{dataset}_96_96_{model}_".lower()
    matches = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.lower().startswith(prefix)
    ]
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise RuntimeError(f"Expected one directory for {dataset}/{model}, found {names}")
    return matches[0]


def _available_ids(result_dir: Path) -> set[int]:
    ids: set[int] = set()
    for path in result_dir.glob("*pred.npy"):
        match = re.fullmatch(r"(\d+)pred\.npy", path.name)
        if match and (result_dir / f"{match.group(1)}true.npy").exists():
            ids.add(int(match.group(1)))
    return ids


def _load_series(result_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(result_dir / f"{sample_id}pred.npy")
    true = np.load(result_dir / f"{sample_id}true.npy")
    return np.asarray(pred, dtype=float).reshape(-1), np.asarray(true, dtype=float).reshape(-1)


def _forecast_slice(series: np.ndarray, pred_len: int) -> np.ndarray:
    if len(series) < pred_len:
        raise ValueError(f"Series length {len(series)} is shorter than pred_len={pred_len}")
    return series[-pred_len:]


def _score_case(
    dataset: str,
    sample_id: int,
    result_dirs: dict[str, Path],
    pred_len: int,
) -> CaseScore:
    mse_by_model: dict[str, float] = {}
    mae_by_model: dict[str, float] = {}
    for model in MODELS:
        pred, true = _load_series(result_dirs[model], sample_id)
        pred_f = _forecast_slice(pred, pred_len)
        true_f = _forecast_slice(true, pred_len)
        mse_by_model[model] = float(np.mean((pred_f - true_f) ** 2))
        mae_by_model[model] = float(np.mean(np.abs(pred_f - true_f)))

    dr_mse = mse_by_model["DRFormer"]
    dr_mae = mae_by_model["DRFormer"]
    other_models = [model for model in MODELS if model != "DRFormer"]
    best_other_model = min(other_models, key=lambda model: mse_by_model[model])
    best_other_mse = mse_by_model[best_other_model]
    rel_gain = (best_other_mse - dr_mse) / best_other_mse if best_other_mse > 0 else -np.inf
    gains = [
        (mse_by_model[model] - dr_mse) / mse_by_model[model]
        for model in other_models
        if mse_by_model[model] > 0
    ]
    mean_rel_gain = float(np.mean(gains)) if gains else -np.inf
    return CaseScore(
        dataset=dataset,
        sample_id=sample_id,
        drformer_mse=dr_mse,
        drformer_mae=dr_mae,
        best_other_model=best_other_model,
        best_other_mse=best_other_mse,
        rel_gain_vs_best_other=float(rel_gain),
        mean_rel_gain_vs_others=mean_rel_gain,
        mse_by_model=mse_by_model,
        mae_by_model=mae_by_model,
    )


def select_cases(
    base_dir: Path,
    pred_len: int,
) -> tuple[dict[str, dict[str, Path]], list[CaseScore], list[CaseScore]]:
    all_result_dirs = {
        dataset: {model: _find_result_dir(base_dir, dataset, model) for model in MODELS}
        for dataset in DATASETS
    }
    selected: list[CaseScore] = []
    ranking: list[CaseScore] = []

    for dataset in DATASETS:
        result_dirs = all_result_dirs[dataset]
        common_ids = sorted(
            set.intersection(*[_available_ids(result_dirs[model]) for model in MODELS])
        )
        if not common_ids:
            raise RuntimeError(f"No common sample ids for {dataset}")

        scores = [_score_case(dataset, sample_id, result_dirs, pred_len) for sample_id in common_ids]
        scores = [
            score
            for score in scores
            if all(score.drformer_mse < score.mse_by_model[model] for model in MODELS if model != "DRFormer")
        ]
        if not scores:
            raise RuntimeError(f"DRFormer is not the best model for any common id in {dataset}")

        scores.sort(
            key=lambda score: (
                score.rel_gain_vs_best_other,
                score.mean_rel_gain_vs_others,
                score.best_other_mse - score.drformer_mse,
            ),
            reverse=True,
        )
        selected.append(scores[0])
        ranking.extend(scores)

    return all_result_dirs, selected, ranking


def _write_selected_csv(path: Path, selected: list[CaseScore]) -> None:
    fieldnames = [
        "dataset",
        "sample_id",
        "drformer_mse",
        "drformer_mae",
        "best_other_model",
        "best_other_mse",
        "rel_gain_vs_best_other",
        "mean_rel_gain_vs_others",
    ]
    for model in MODELS:
        fieldnames.extend([f"{model}_mse", f"{model}_mae"])

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for score in selected:
            row = {
                "dataset": score.dataset,
                "sample_id": score.sample_id,
                "drformer_mse": f"{score.drformer_mse:.10g}",
                "drformer_mae": f"{score.drformer_mae:.10g}",
                "best_other_model": score.best_other_model,
                "best_other_mse": f"{score.best_other_mse:.10g}",
                "rel_gain_vs_best_other": f"{score.rel_gain_vs_best_other:.10g}",
                "mean_rel_gain_vs_others": f"{score.mean_rel_gain_vs_others:.10g}",
            }
            for model in MODELS:
                row[f"{model}_mse"] = f"{score.mse_by_model[model]:.10g}"
                row[f"{model}_mae"] = f"{score.mae_by_model[model]:.10g}"
            writer.writerow(row)


def _write_ranking_csv(path: Path, ranking: list[CaseScore]) -> None:
    fieldnames = [
        "dataset",
        "rank",
        "sample_id",
        "rel_gain_vs_best_other",
        "mean_rel_gain_vs_others",
        "best_other_model",
    ]
    for model in MODELS:
        fieldnames.append(f"{model}_mse")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        ranks_by_dataset: dict[str, int] = {}
        for score in ranking:
            ranks_by_dataset[score.dataset] = ranks_by_dataset.get(score.dataset, 0) + 1
            row = {
                "dataset": score.dataset,
                "rank": ranks_by_dataset[score.dataset],
                "sample_id": score.sample_id,
                "rel_gain_vs_best_other": f"{score.rel_gain_vs_best_other:.10g}",
                "mean_rel_gain_vs_others": f"{score.mean_rel_gain_vs_others:.10g}",
                "best_other_model": score.best_other_model,
                "DRFormer_mse": f"{score.drformer_mse:.10g}",
            }
            for model in MODELS:
                row[f"{model}_mse"] = f"{score.mse_by_model[model]:.10g}"
            writer.writerow(row)


def plot_cases(
    output_stem: Path,
    result_dirs: dict[str, dict[str, Path]],
    selected: list[CaseScore],
    pred_len: int,
) -> None:
    colors = {
        "DRFormer": "#0072B2",
        "S-Mamba": "#D55E00",
        "iTransformer": "#009E73",
        "PatchTST": "#CC79A7",
        "Autoformer": "#E69F00",
    }
    fig, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(18.5, 10.5),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.89, bottom=0.08, wspace=0.24, hspace=0.2)

    for row_idx, score in enumerate(selected):
        dataset = score.dataset
        sample_id = score.sample_id
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            pred, true = _load_series(result_dirs[dataset][model], sample_id)
            x = np.arange(len(true))
            context_len = len(true) - pred_len
            ax.axvspan(context_len - 0.5, len(true) - 0.5, color="#F2F2F2", zorder=0)
            ax.plot(x, true, color="#111111", lw=1.55)
            ax.plot(x, pred, color=colors[model], lw=1.45)
            ax.axvline(context_len - 0.5, color="#777777", ls="--", lw=0.8)
            ax.grid(True, color="#DDDDDD", lw=0.45, alpha=0.7)
            ax.tick_params(labelsize=9, length=2.5)
            if row_idx == 0:
                ax.set_title(model, fontsize=14, fontweight="bold", pad=9)
            if col_idx == 0:
                ax.set_ylabel(dataset, fontsize=13, rotation=0, labelpad=42, va="center")
            mse = score.mse_by_model[model]
            mae = score.mae_by_model[model]
            ax.text(
                0.025,
                0.94,
                f"MSE {mse:.3g}\nMAE {mae:.3g}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "linewidth": 0.45,
                    "alpha": 0.9,
                },
            )
            if model == "DRFormer":
                for spine in ax.spines.values():
                    spine.set_linewidth(1.25)
                    spine.set_edgecolor(colors["DRFormer"])
            else:
                for spine in ax.spines.values():
                    spine.set_linewidth(0.7)
                    spine.set_edgecolor("#666666")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time step", fontsize=12)

    handles = [Line2D([0], [0], color="#111111", lw=1.55, label="Ground truth")]
    handles.extend(
        Line2D([0], [0], color=colors[model], lw=1.55, label=model) for model in MODELS
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.53, 0.965),
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("test_results_copy"),
        help="Directory containing per-model test result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/drformer_sota_cases"),
        help="Directory where figures and CSV files are written.",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=96,
        help="Number of forecast steps appended after the 96-step input context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dirs, selected, ranking = select_cases(base_dir, args.pred_len)
    figure_stem = output_dir / "drformer_sota_cases_4x5"

    _write_selected_csv(output_dir / "selected_cases.csv", selected)
    _write_ranking_csv(output_dir / "case_ranking.csv", ranking)
    plot_cases(figure_stem, result_dirs, selected, args.pred_len)

    print("Selected visualization cases:")
    for score in selected:
        print(
            f"  {score.dataset}: id={score.sample_id}, "
            f"gain_vs_{score.best_other_model}={100.0 * score.rel_gain_vs_best_other:.2f}%, "
            f"DRFormer_MSE={score.drformer_mse:.6g}, "
            f"{score.best_other_model}_MSE={score.best_other_mse:.6g}"
        )
    print(f"Figure: {figure_stem.with_suffix('.pdf')}")
    print(f"PNG:    {figure_stem.with_suffix('.png')}")
    print(f"CSV:    {output_dir / 'selected_cases.csv'}")


if __name__ == "__main__":
    main()
