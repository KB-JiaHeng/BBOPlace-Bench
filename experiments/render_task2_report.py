#!/usr/bin/env python3
"""Render report-quality Task 2 figures from frozen analysis artifacts.

The plotting convention mirrors ``render_task1_report.py``: five-seed means are
shown with population standard deviation (ddof=0), evaluation axes use true
objective evaluations, and no smoothing or interpolation is applied beyond
connecting recorded checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

METHODS = ["nsga2", "moead"]
METHOD_LABELS = {"nsga2": "NSGA-II", "moead": "MOEA/D"}
COLORS = {"nsga2": "#1f77b4", "moead": "#d62728", "task1": "#666666", "task2": "#2ca02c"}
MARKERS = {"nsga2": "o", "moead": "s"}
DISPLACEMENT_BINS = ["0", "1-10", "11-100", ">100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "task2_analysis",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")


def read_csv(path: Path, required: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if required:
        require_columns(frame, required, path)
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominated = np.any(
            np.all(values <= point, axis=1)
            & np.any(values < point, axis=1)
        )
        keep[index] = not dominated
    return values[keep]


def mean_std0(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        frame.groupby("evaluation_count", as_index=False)[value]
        .agg(mean="mean", std0=lambda series: float(np.std(series.to_numpy(float), ddof=0)))
        .sort_values("evaluation_count")
    )


def style_evaluation_axis(ax: plt.Axes) -> None:
    ax.set_xlim(20, 10000)
    ax.set_xticks([20, 2000, 4000, 6000, 8000, 10000])
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.08)
    ax.set_xlabel("True objective evaluations")


def render_paired_final_hypervolume(analysis_dir: Path, output_dir: Path) -> None:
    data = read_csv(
        analysis_dir / "paired_hypervolume.csv",
        ["seed", "nsga2_hv", "moead_hv", "nsga2_minus_moead"],
    ).sort_values("seed")
    fig, ax = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
    y = np.arange(len(data))
    for position, row in zip(y, data.itertuples(index=False)):
        ax.plot(
            [row.nsga2_hv, row.moead_hv],
            [position, position],
            color="0.72",
            linewidth=1.5,
            zorder=1,
        )
    ax.scatter(data["nsga2_hv"], y, s=48, marker=MARKERS["nsga2"], color=COLORS["nsga2"], label="NSGA-II", zorder=3)
    ax.scatter(data["moead_hv"], y, s=48, marker=MARKERS["moead"], color=COLORS["moead"], label="MOEA/D", zorder=3)
    right = float(max(data["nsga2_hv"].max(), data["moead_hv"].max()))
    span = float(max(data["nsga2_hv"].max(), data["moead_hv"].max()) - min(data["nsga2_hv"].min(), data["moead_hv"].min()))
    for position, delta in zip(y, data["nsga2_minus_moead"]):
        ax.text(right + 0.035 * span, position, f"Δ={delta:+.3f}", va="center", fontsize=8, color="0.35")
    ax.set_yticks(y, [f"Seed {int(seed)}" for seed in data["seed"]])
    ax.invert_yaxis()
    ax.set_xlabel("Final population hypervolume, reference point (1.1, 1.1)")
    ax.set_title("Paired final hypervolume across five seeds")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(True, axis="x", alpha=0.24)
    ax.grid(False, axis="y")
    ax.text(
        0.01,
        0.015,
        "Δ = NSGA-II - MOEA/D; paired initialization within each seed",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, "paired_final_hypervolume")


def render_hypervolume_convergence(analysis_dir: Path, output_dir: Path) -> None:
    data = read_csv(
        analysis_dir / "hypervolume_convergence.csv",
        ["method", "seed", "evaluation_count", "hv_ref_1.1"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.25), constrained_layout=True)
    for method in METHODS:
        summary = mean_std0(data[data["method"] == method], "hv_ref_1.1")
        x = summary["evaluation_count"].to_numpy(float)
        mean = summary["mean"].to_numpy(float)
        std0 = summary["std0"].to_numpy(float)
        ax.plot(
            x,
            mean,
            marker=MARKERS[method],
            markersize=4.5,
            linewidth=1.75,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax.fill_between(x, mean - std0, mean + std0, color=COLORS[method], alpha=0.14, linewidth=0)
    style_evaluation_axis(ax)
    ax.set_ylabel("Population hypervolume")
    ax.set_title("Hypervolume convergence")
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.012,
        0.02,
        "Lines: 5-seed mean; bands: population std (ddof=0); fixed normalization",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, "hypervolume_convergence")


def render_objective_space_and_task1_coverage(analysis_dir: Path, output_dir: Path) -> None:
    points = read_csv(
        analysis_dir / "final_front_points.csv",
        ["method", "seed", "normalized_hpwl", "normalized_congestion_top10"],
    )
    task1 = read_csv(
        analysis_dir / "task1_reference_points.csv",
        ["normalized_hpwl", "normalized_congestion_top10", "nondominated"],
    )
    common = read_csv(
        analysis_dir / "common_final_front.csv",
        ["normalized_hpwl", "normalized_congestion_top10"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.65), constrained_layout=True, sharex=True, sharey=True)
    left, right = axes
    for method in METHODS:
        subset = points[points["method"] == method]
        left.scatter(
            subset["normalized_hpwl"],
            subset["normalized_congestion_top10"],
            s=22,
            alpha=0.50,
            color=COLORS[method],
            marker=MARKERS[method],
            label=f"{METHOD_LABELS[method]} final fronts",
        )
        front = nondominated(subset[["normalized_hpwl", "normalized_congestion_top10"]].to_numpy(float))
        front = front[np.argsort(front[:, 0])]
        left.plot(front[:, 0], front[:, 1], color=COLORS[method], linewidth=1.6)
    left.set_title("Final fronts from five seeds")
    left.set_xlabel("Normalized HPWL")
    left.set_ylabel(r"Normalized $congestion_{top10}$")
    left.legend(frameon=False, fontsize=7.4)

    task1_nd = task1[task1["nondominated"].astype(str).str.lower().isin(["true", "1"])]
    right.scatter(
        task1["normalized_hpwl"],
        task1["normalized_congestion_top10"],
        s=30,
        color=COLORS["task1"],
        alpha=0.58,
        label="Task 1 mixed saved-best",
    )
    if not task1_nd.empty:
        right.scatter(
            task1_nd["normalized_hpwl"],
            task1_nd["normalized_congestion_top10"],
            s=90,
            marker="*",
            color="black",
            label="Task 1 nondominated reference",
            zorder=4,
        )
    ordered = common.sort_values("normalized_hpwl")
    right.plot(
        ordered["normalized_hpwl"],
        ordered["normalized_congestion_top10"],
        color=COLORS["task2"],
        linewidth=1.7,
    )
    right.scatter(
        ordered["normalized_hpwl"],
        ordered["normalized_congestion_top10"],
        s=28,
        color=COLORS["task2"],
        label="Task 2 common front",
        zorder=3,
    )
    right.set_title("Coverage of available Task 1 reference")
    right.set_xlabel("Normalized HPWL")
    right.legend(frameon=False, fontsize=7.2)
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.text(0.98, 0.02, "Lower-left is better", transform=ax.transAxes, ha="right", fontsize=7.3, color="0.35")
    save_figure(fig, output_dir, "objective_space_and_task1_coverage")


def render_phenotype_diversity(analysis_dir: Path, output_dir: Path) -> None:
    data = read_csv(
        analysis_dir / "phenotype_diversity_over_time.csv",
        ["method", "seed", "evaluation_count", "unique_phenotypes", "max_single_phenotype_fraction"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.55), constrained_layout=True, sharex=True)
    specifications = [
        ("unique_phenotypes", "Unique phenotypes in population", (0, 21)),
        ("max_single_phenotype_fraction", "Largest phenotype share", (0, 1.02)),
    ]
    for ax, (column, ylabel, ylim) in zip(axes, specifications):
        for method in METHODS:
            summary = mean_std0(data[data["method"] == method], column)
            x = summary["evaluation_count"].to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            std0 = summary["std0"].to_numpy(float)
            ax.plot(x, mean, linewidth=1.55, color=COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(x, np.clip(mean - std0, ylim[0], ylim[1]), np.clip(mean + std0, ylim[0], ylim[1]), color=COLORS[method], alpha=0.13, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", alpha=0.24)
        ax.grid(True, which="minor", alpha=0.08)
    style_evaluation_axis(axes[-1])
    axes[0].set_title("Phenotype diversity during search")
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[-1].text(
        0.012,
        0.035,
        "Lines: 5-seed mean; bands: population std (ddof=0)",
        transform=axes[-1].transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, "phenotype_diversity_over_time")


def render_moead_slot_collapse(analysis_dir: Path, output_dir: Path) -> None:
    data = read_csv(
        analysis_dir / "moead_slot_occupancy_over_time.csv",
        ["seed", "evaluation_count", "unique_slot_phenotypes", "largest_phenotype_slot_fraction"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.55), constrained_layout=True, sharex=True)
    columns = [
        ("unique_slot_phenotypes", "Unique phenotypes across 20 slots", (0, 21)),
        ("largest_phenotype_slot_fraction", "Largest phenotype slot share", (0, 1.02)),
    ]
    for ax, (column, ylabel, ylim) in zip(axes, columns):
        for seed, subset in data.groupby("seed"):
            subset = subset.sort_values("evaluation_count")
            ax.plot(subset["evaluation_count"], subset[column], color=COLORS["moead"], alpha=0.20, linewidth=0.75)
        summary = mean_std0(data.assign(method="moead"), column)
        ax.plot(summary["evaluation_count"], summary["mean"], color=COLORS["moead"], linewidth=1.9, label="5-seed mean")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.18)
        ax.axvline(40, color="0.35", linestyle="--", linewidth=0.8)
    axes[0].set_title("MOEA/D weight-slot collapse and recovery")
    axes[0].legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("True objective evaluations (log scale)")
    axes[-1].set_xticks([20, 40, 100, 1000, 10000])
    axes[-1].get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axes[-1].text(
        0.012,
        0.035,
        "Thin lines: individual seeds; dashed line: first completed MOEA/D sweep",
        transform=axes[-1].transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, "moead_slot_collapse")


def weighted_rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (method, displacement_bin), subset in frame.groupby(["method", "displacement_bin"]):
        weights = subset["parent_child_comparisons"].to_numpy(float)
        values = subset[column].to_numpy(float)
        rows.append(
            {
                "method": method,
                "displacement_bin": displacement_bin,
                "value": float(np.average(values, weights=weights)),
                "comparisons": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def render_displacement_outcomes(analysis_dir: Path, output_dir: Path) -> None:
    data = read_csv(
        analysis_dir / "operator_effect_by_displacement.csv",
        ["method", "seed", "displacement_bin", "parent_child_comparisons", "survival_rate", "dominated_by_parent_rate"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.75), constrained_layout=True, sharey=True)
    metrics = [
        ("survival_rate", "Offspring survival rate"),
        ("dominated_by_parent_rate", "Dominated by a parent"),
    ]
    x = np.arange(len(DISPLACEMENT_BINS))
    width = 0.34
    for ax, (column, title) in zip(axes, metrics):
        aggregate = weighted_rates(data, column)
        for offset, method in zip([-width / 2, width / 2], METHODS):
            subset = aggregate[aggregate["method"] == method].set_index("displacement_bin").loc[DISPLACEMENT_BINS]
            values = 100.0 * subset["value"].to_numpy(float)
            bars = ax.bar(x + offset, values, width=width, color=COLORS[method], alpha=0.88, label=METHOD_LABELS[method])
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x, DISPLACEMENT_BINS)
        ax.set_xlabel("Moved macros relative to parent")
        ax.set_title(title)
        ax.set_ylim(0, 72)
        ax.grid(True, axis="y", alpha=0.24)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Rate (%)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].text(
        0.01,
        0.02,
        "Rates are weighted by parent-child comparison counts across five seeds",
        transform=axes[0].transAxes,
        fontsize=7.2,
        color="0.35",
    )
    save_figure(fig, output_dir, "displacement_conditioned_outcomes")


def write_figure_manifest(output_dir: Path) -> None:
    stems = [
        "paired_final_hypervolume",
        "hypervolume_convergence",
        "objective_space_and_task1_coverage",
        "phenotype_diversity_over_time",
        "moead_slot_collapse",
        "displacement_conditioned_outcomes",
    ]
    with (output_dir / "figure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "png", "pdf"])
        writer.writeheader()
        for stem in stems:
            writer.writerow({"stem": stem, "png": f"{stem}.png", "pdf": f"{stem}.pdf"})


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_paired_final_hypervolume(args.analysis_dir, args.output_dir)
    render_hypervolume_convergence(args.analysis_dir, args.output_dir)
    render_objective_space_and_task1_coverage(args.analysis_dir, args.output_dir)
    render_phenotype_diversity(args.analysis_dir, args.output_dir)
    render_moead_slot_collapse(args.analysis_dir, args.output_dir)
    render_displacement_outcomes(args.analysis_dir, args.output_dir)
    write_figure_manifest(args.output_dir)
    print(f"Rendered Task 2 report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
