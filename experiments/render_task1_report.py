#!/usr/bin/env python3
"""Render report-quality Task 1 figures from the completed analysis artifacts.

The report convention matches the original repository: five-seed means with
population standard deviation (ddof=0). Convergence values are plotted only at
true-evaluation checkpoints spaced by 20, with no interpolation or smoothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

MUTATIONS = ["swap", "shift", "random_resetting", "shuffle", "pm"]
CROSSOVERS = ["uniform", "sbx"]
MUTATION_LABELS = {
    "swap": "Swap",
    "shift": "Shift",
    "random_resetting": "Random resetting",
    "shuffle": "Shuffle",
    "pm": "Polynomial mutation",
}
COLORS = {
    "swap": "#1f77b4",
    "shift": "#d62728",
    "random_resetting": "#2ca02c",
    "shuffle": "#9467bd",
    "pm": "#ff7f0e",
    "uniform": "#1f77b4",
    "sbx": "#d62728",
    5.0: "#2ca02c",
    15.0: "#ff7f0e",
    30.0: "#1f77b4",
}
STD0_FACTOR_FOR_N5 = np.sqrt(4.0 / 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "task1_analysis",
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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def hpwl_scale(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1e5


def std0_from_existing_curve(frame: pd.DataFrame) -> np.ndarray:
    # Existing curve CSVs were produced with pandas std (ddof=1), n=5.
    return frame["std"].fillna(0).to_numpy(float) * STD0_FACTOR_FOR_N5 / 1e5


def style_evaluation_axis(ax: plt.Axes) -> None:
    ax.set_xlim(20, 10000)
    ax.set_xticks([20, 2000, 4000, 6000, 8000, 10000])
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax.set_xlabel("True objective evaluations")
    ax.set_ylabel(r"Best-so-far HPWL ($\times 10^5$)")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.08)


def render_convergence(analysis_dir: Path, output_dir: Path, crossover: str) -> None:
    data = pd.read_csv(analysis_dir / "plot_data" / f"{crossover}_mutation_convergence.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.35), constrained_layout=True)
    for mutation in MUTATIONS:
        curve = data[data["mutation"] == mutation].sort_values("evaluation_id")
        x = curve["evaluation_id"].to_numpy(int)
        mean = hpwl_scale(curve["mean"])
        std0 = std0_from_existing_curve(curve)
        ax.plot(
            x,
            mean,
            label=MUTATION_LABELS[mutation],
            color=COLORS[mutation],
            linewidth=1.65,
        )
        ax.fill_between(
            x,
            mean - std0,
            mean + std0,
            color=COLORS[mutation],
            alpha=0.13,
            linewidth=0,
        )
    style_evaluation_axis(ax)
    ax.set_title(f"{crossover.upper()} crossover: mutation convergence")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.text(
        0.012,
        0.02,
        "Lines: 5-seed mean; bands: population std (ddof=0); no smoothing",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, f"{crossover}_mutation_convergence")


def final_summary(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "tables" / "final_summary_both_ddof.csv"
    data = pd.read_csv(path)
    return data


def render_heatmap(analysis_dir: Path, output_dir: Path) -> None:
    summary = final_summary(analysis_dir).set_index(["crossover", "mutation"])
    means = np.array(
        [[summary.loc[(c, m), "mean"] / 1e5 for m in MUTATIONS] for c in CROSSOVERS]
    )
    stds = np.array(
        [[summary.loc[(c, m), "std_population_ddof0"] / 1e5 for m in MUTATIONS] for c in CROSSOVERS]
    )
    fig, ax = plt.subplots(figsize=(7.35, 2.85), constrained_layout=True)
    image = ax.imshow(means, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(MUTATIONS)), [MUTATION_LABELS[m] for m in MUTATIONS])
    ax.set_yticks(range(len(CROSSOVERS)), [c.upper() for c in CROSSOVERS])
    ax.tick_params(axis="x", rotation=18)
    ax.grid(False)
    threshold = (means.min() + means.max()) / 2
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            color = "white" if means[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{means[i, j]:.2f}\n± {stds[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(r"Final HPWL ($\times 10^5$)")
    ax.set_title("Final best-so-far HPWL at 10,000 evaluations (lower is better)")
    save_figure(fig, output_dir, "final_hpwl_heatmap_2_by_5")


def render_interaction(analysis_dir: Path, output_dir: Path) -> None:
    summary = final_summary(analysis_dir).set_index(["crossover", "mutation"])
    x = np.arange(len(MUTATIONS))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for crossover, offset in [("uniform", -0.035), ("sbx", 0.035)]:
        means = np.array([summary.loc[(crossover, m), "mean"] for m in MUTATIONS]) / 1e5
        stds = np.array(
            [summary.loc[(crossover, m), "std_population_ddof0"] for m in MUTATIONS]
        ) / 1e5
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            marker="o",
            markersize=5,
            linewidth=1.7,
            capsize=3,
            capthick=1,
            color=COLORS[crossover],
            label=crossover.upper(),
        )
    ax.set_xticks(x, [MUTATION_LABELS[m] for m in MUTATIONS], rotation=14, ha="right")
    ax.set_ylabel(r"Final best-so-far HPWL ($\times 10^5$)")
    ax.set_xlabel("Mutation operator")
    ax.set_title("Crossover-mutation interaction")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.24)
    ax.grid(False, axis="x")
    save_figure(fig, output_dir, "crossover_mutation_interaction")


def render_distribution(analysis_dir: Path, output_dir: Path) -> None:
    run = pd.read_csv(analysis_dir / "tables" / "run_level_summary.csv")
    summary = final_summary(analysis_dir)
    order = [tuple(row) for row in summary[["crossover", "mutation"]].to_numpy()]
    labels = [f"{c.upper()} + {MUTATION_LABELS[m]}" for c, m in order]
    values = [
        run[(run["crossover"] == c) & (run["mutation"] == m)]["best_at_10000"].to_numpy() / 1e5
        for c, m in order
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    bp = ax.boxplot(
        values,
        vert=False,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        widths=0.56,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
        boxprops={"linewidth": 0.9},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.85")
        patch.set_alpha(0.8)
    rng = np.random.default_rng(20260720)
    for position, arr in enumerate(values, start=1):
        jitter = rng.uniform(-0.10, 0.10, size=len(arr))
        ax.scatter(arr, np.full(len(arr), position) + jitter, s=22, zorder=3, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Final best-so-far HPWL ($\times 10^5$)")
    ax.set_ylabel("Configuration (ordered by mean)")
    ax.set_title("Final HPWL distribution across five seeds")
    ax.grid(True, axis="x", alpha=0.24)
    ax.grid(False, axis="y")
    ax.text(
        0.99,
        0.015,
        "Diamond: mean; line: median; dots: individual seeds",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, "final_hpwl_distribution")


def render_sensitivity(analysis_dir: Path, output_dir: Path, kind: str) -> None:
    data = pd.read_csv(analysis_dir / "plot_data" / f"{kind}_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.25), constrained_layout=True)
    for eta in [5.0, 15.0, 30.0]:
        curve = data[data["eta"] == eta].sort_values("evaluation_id")
        x = curve["evaluation_id"].to_numpy(int)
        mean = hpwl_scale(curve["mean"])
        std0 = std0_from_existing_curve(curve)
        ax.plot(x, mean, label=rf"$\eta={eta:g}$", color=COLORS[eta], linewidth=1.7)
        ax.fill_between(x, mean - std0, mean + std0, color=COLORS[eta], alpha=0.14, linewidth=0)
    style_evaluation_axis(ax)
    if kind == "sbx_eta":
        ax.set_title(r"SBX distribution index sensitivity with Swap mutation")
    else:
        ax.set_title(r"Polynomial-mutation distribution index sensitivity with Uniform crossover")
    ax.legend(frameon=False)
    ax.text(
        0.012,
        0.02,
        "Lines: 5-seed mean; bands: population std (ddof=0); no smoothing",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    save_figure(fig, output_dir, f"{kind}_sensitivity")


def write_figure_manifest(output_dir: Path) -> None:
    stems = [
        "uniform_mutation_convergence",
        "sbx_mutation_convergence",
        "final_hpwl_heatmap_2_by_5",
        "crossover_mutation_interaction",
        "final_hpwl_distribution",
        "sbx_eta_sensitivity",
        "pm_eta_sensitivity",
    ]
    rows = []
    for stem in stems:
        rows.append({"stem": stem, "png": f"{stem}.png", "pdf": f"{stem}.pdf"})
    pd.DataFrame(rows).to_csv(output_dir / "figure_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_convergence(args.analysis_dir, args.output_dir, "uniform")
    render_convergence(args.analysis_dir, args.output_dir, "sbx")
    render_heatmap(args.analysis_dir, args.output_dir)
    render_interaction(args.analysis_dir, args.output_dir)
    render_distribution(args.analysis_dir, args.output_dir)
    render_sensitivity(args.analysis_dir, args.output_dir, "sbx_eta")
    render_sensitivity(args.analysis_dir, args.output_dir, "pm_eta")
    write_figure_manifest(args.output_dir)
    print(f"Rendered report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
