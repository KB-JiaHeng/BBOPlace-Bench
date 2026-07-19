#!/usr/bin/env python3
"""Generate the frozen Task 1 tables and report-ready figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "experiments" / "task1_runs" / "formal"
OUTPUT_DIR = ROOT / "experiments" / "task1_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "plot_data"
TABLE_DIR = OUTPUT_DIR / "tables"

CROSSOVERS = ["uniform", "sbx"]
MUTATIONS = ["swap", "shift", "random_resetting", "shuffle", "pm"]
CHECKPOINTS = [20, 1000, 5000, 10000]
PLOT_STEP = 20


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted(MANIFEST_DIR.glob("seed_*__*.json")):
        if path.name.endswith("initial_hash.json"):
            continue
        with path.open() as f:
            manifest = json.load(f)
        if manifest.get("status") != "complete":
            continue
        config = manifest["configuration"]
        rows.append(
            {
                "seed": int(manifest["seed"]),
                "family": config["family"],
                "crossover": config["crossover"],
                "mutation": config["mutation"],
                "sbx_eta": float(config["sbx_eta"]),
                "pm_eta": float(config["pm_eta"]),
                "result_path": manifest["result_path"],
            }
        )
    runs = pd.DataFrame(rows)
    if len(runs) != 70:
        raise RuntimeError(f"Expected 70 complete formal runs, found {len(runs)}")
    return runs


def load_data(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_frames = []
    generation_frames = []
    for row in runs.itertuples(index=False):
        result = Path(row.result_path)
        trace = pd.read_csv(result / "evaluation_trace.csv")
        if len(trace) != 10000 or trace["evaluation_id"].iloc[-1] != 10000:
            raise RuntimeError(f"Invalid trace: {result}")
        trace = trace.assign(
            seed=row.seed,
            family=row.family,
            crossover=row.crossover,
            mutation=row.mutation,
            sbx_eta=row.sbx_eta,
            pm_eta=row.pm_eta,
            result_path=row.result_path,
        )
        trace_frames.append(trace)

        generations = pd.read_csv(result / "generation_metrics.csv")
        generations = generations.assign(
            seed=row.seed,
            family=row.family,
            crossover=row.crossover,
            mutation=row.mutation,
            sbx_eta=row.sbx_eta,
            pm_eta=row.pm_eta,
            result_path=row.result_path,
        )
        generation_frames.append(generations)

    return pd.concat(trace_frames, ignore_index=True), pd.concat(
        generation_frames, ignore_index=True
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_curve(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    sampled = frame[
        (frame["evaluation_id"] >= 20)
        & (frame["evaluation_id"] % PLOT_STEP == 0)
    ]
    return (
        sampled.groupby([group_column, "evaluation_id"])["best_so_far_hpwl"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_main_convergence(main_trace: pd.DataFrame) -> None:
    sampled_all = main_trace[
        (main_trace["evaluation_id"] >= 20)
        & (main_trace["evaluation_id"] % PLOT_STEP == 0)
    ]
    y_min = sampled_all["best_so_far_hpwl"].min()
    y_max = sampled_all["best_so_far_hpwl"].max()
    padding = 0.03 * (y_max - y_min)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mutation_colors = {
        mutation: color_cycle[index % len(color_cycle)]
        for index, mutation in enumerate(MUTATIONS)
    }

    for crossover in CROSSOVERS:
        frame = main_trace[main_trace["crossover"] == crossover]
        aggregate = aggregate_curve(frame, "mutation")
        aggregate.to_csv(
            DATA_DIR / f"{crossover}_mutation_convergence.csv", index=False
        )
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for mutation in MUTATIONS:
            curve = aggregate[aggregate["mutation"] == mutation]
            x = curve["evaluation_id"].to_numpy()
            mean = curve["mean"].to_numpy()
            std = curve["std"].fillna(0).to_numpy()
            line = ax.plot(
                x,
                mean,
                label=mutation,
                color=mutation_colors[mutation],
            )[0]
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=line.get_color(),
                alpha=0.18,
            )
        ax.set_xlabel("True objective evaluations")
        ax.set_ylabel("Best-so-far HPWL")
        ax.set_title(f"{crossover.upper()} crossover: mutation convergence")
        ax.set_xlim(20, 10000)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Mutation")
        save_figure(fig, f"{crossover}_mutation_convergence")


def build_run_summary(main_trace: pd.DataFrame) -> pd.DataFrame:
    checkpoints = main_trace[main_trace["evaluation_id"].isin(CHECKPOINTS)]
    wide = checkpoints.pivot_table(
        index=["seed", "crossover", "mutation"],
        columns="evaluation_id",
        values="best_so_far_hpwl",
    ).reset_index()
    wide = wide.rename(
        columns={
            20: "best_at_20",
            1000: "best_at_1000",
            5000: "best_at_5000",
            10000: "best_at_10000",
        }
    )
    wide["relative_improvement_percent"] = (
        (wide["best_at_20"] - wide["best_at_10000"])
        / wide["best_at_20"]
        * 100.0
    )
    wide.to_csv(TABLE_DIR / "run_level_summary.csv", index=False)
    return wide


def plot_heatmap_and_interaction(run_summary: pd.DataFrame) -> None:
    grouped = (
        run_summary.groupby(["crossover", "mutation"])["best_at_10000"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    grouped.to_csv(TABLE_DIR / "final_summary.csv", index=False)

    matrix = np.array(
        [
            [
                grouped[
                    (grouped["crossover"] == crossover)
                    & (grouped["mutation"] == mutation)
                ]["mean"].iloc[0]
                for mutation in MUTATIONS
            ]
            for crossover in CROSSOVERS
        ]
    )
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(MUTATIONS)), MUTATIONS, rotation=25, ha="right")
    ax.set_yticks(range(len(CROSSOVERS)), [c.upper() for c in CROSSOVERS])
    ax.set_title("Mean final best-so-far HPWL (lower is better)")
    for row, crossover in enumerate(CROSSOVERS):
        for col, mutation in enumerate(MUTATIONS):
            stats = grouped[
                (grouped["crossover"] == crossover)
                & (grouped["mutation"] == mutation)
            ].iloc[0]
            ax.text(
                col,
                row,
                f"{stats['mean']:.0f}\n± {stats['std']:.0f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax, label="HPWL")
    pd.DataFrame(matrix, index=CROSSOVERS, columns=MUTATIONS).to_csv(
        DATA_DIR / "final_hpwl_heatmap.csv"
    )
    save_figure(fig, "final_hpwl_heatmap_2_by_5")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    interaction_rows = []
    for crossover in CROSSOVERS:
        subset = grouped[grouped["crossover"] == crossover].set_index("mutation")
        means = subset.loc[MUTATIONS, "mean"].to_numpy()
        stds = subset.loc[MUTATIONS, "std"].to_numpy()
        ax.errorbar(
            MUTATIONS,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=crossover.upper(),
        )
        for mutation, mean, std in zip(MUTATIONS, means, stds):
            interaction_rows.append(
                {
                    "crossover": crossover,
                    "mutation": mutation,
                    "mean": mean,
                    "std": std,
                }
            )
    pd.DataFrame(interaction_rows).to_csv(
        DATA_DIR / "crossover_mutation_interaction.csv", index=False
    )
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Final best-so-far HPWL")
    ax.set_title("Crossover–mutation interaction at 10,000 evaluations")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "crossover_mutation_interaction")


def plot_final_distribution(run_summary: pd.DataFrame) -> None:
    labels = [f"{c}\n{m}" for c in CROSSOVERS for m in MUTATIONS]
    data = [
        run_summary[
            (run_summary["crossover"] == crossover)
            & (run_summary["mutation"] == mutation)
        ]["best_at_10000"].to_numpy()
        for crossover in CROSSOVERS
        for mutation in MUTATIONS
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.boxplot(data, labels=labels, showmeans=True)
    rng = np.random.default_rng(20260719)
    for position, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=22)
    ax.set_ylabel("Final best-so-far HPWL")
    ax.set_title("Final HPWL distribution across five seeds")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "final_hpwl_distribution")


def sensitivity_frame(trace: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "sbx_eta":
        extra = trace[trace["family"] == "sbx_eta"]
        main = trace[
            (trace["family"] == "main")
            & (trace["crossover"] == "sbx")
            & (trace["mutation"] == "swap")
        ]
        frame = pd.concat([extra, main], ignore_index=True)
        frame = frame.assign(eta=frame["sbx_eta"])
    elif kind == "pm_eta":
        extra = trace[trace["family"] == "pm_eta"]
        main = trace[
            (trace["family"] == "main")
            & (trace["crossover"] == "uniform")
            & (trace["mutation"] == "pm")
        ]
        frame = pd.concat([extra, main], ignore_index=True)
        frame = frame.assign(eta=frame["pm_eta"])
    else:
        raise ValueError(kind)
    return frame


def plot_sensitivity(trace: pd.DataFrame, kind: str, title: str) -> None:
    frame = sensitivity_frame(trace, kind)
    sampled = frame[
        (frame["evaluation_id"] >= 20)
        & (frame["evaluation_id"] % PLOT_STEP == 0)
    ]
    aggregate = (
        sampled.groupby(["eta", "evaluation_id"])["best_so_far_hpwl"]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(DATA_DIR / f"{kind}_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for eta in [5.0, 15.0, 30.0]:
        curve = aggregate[aggregate["eta"] == eta]
        x = curve["evaluation_id"].to_numpy()
        mean = curve["mean"].to_numpy()
        std = curve["std"].fillna(0).to_numpy()
        line = ax.plot(x, mean, label=f"eta={eta:g}")[0]
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.18,
        )
    ax.set_xlabel("True objective evaluations")
    ax.set_ylabel("Best-so-far HPWL")
    ax.set_title(title)
    ax.set_xlim(20, 10000)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, f"{kind}_sensitivity")


def build_checkpoint_and_duplicate_tables(
    main_trace: pd.DataFrame,
    generations: pd.DataFrame,
) -> None:
    checkpoint_summary = (
        main_trace[main_trace["evaluation_id"].isin(CHECKPOINTS)]
        .groupby(["crossover", "mutation", "evaluation_id"])[
            "best_so_far_hpwl"
        ]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    checkpoint_summary.to_csv(
        TABLE_DIR / "checkpoint_summary_at_20_1000_5000_10000.csv",
        index=False,
    )

    main_generations = generations[generations["family"] == "main"]
    per_run = (
        main_generations.groupby(["seed", "crossover", "mutation"])
        .agg(
            duplicate_rejections=("duplicate_rejections", "sum"),
            raw_generated=("raw_generated", "sum"),
            accepted_unique_offspring=("accepted_unique_offspring", "sum"),
            mating_iterations=("mating_iterations", "sum"),
            generations=("generation", "max"),
        )
        .reset_index()
    )
    per_run["duplicate_rejection_rate"] = (
        per_run["duplicate_rejections"] / per_run["raw_generated"].replace(0, np.nan)
    )
    per_run.to_csv(TABLE_DIR / "duplicate_run_level.csv", index=False)
    duplicate_summary = (
        per_run.groupby(["crossover", "mutation"])
        .agg(
            mean_duplicate_rejections=("duplicate_rejections", "mean"),
            std_duplicate_rejections=("duplicate_rejections", "std"),
            mean_duplicate_rejection_rate=("duplicate_rejection_rate", "mean"),
            mean_generations=("generations", "mean"),
        )
        .reset_index()
    )
    duplicate_summary.to_csv(TABLE_DIR / "duplicate_summary.csv", index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    trace, generations = load_data(runs)
    main_trace = trace[trace["family"] == "main"].copy()
    if len(main_trace) != 50 * 10000:
        raise RuntimeError(f"Unexpected main trace size: {len(main_trace)}")

    run_summary = build_run_summary(main_trace)
    plot_main_convergence(main_trace)
    plot_heatmap_and_interaction(run_summary)
    plot_final_distribution(run_summary)
    plot_sensitivity(trace, "sbx_eta", "SBX eta sensitivity with swap mutation")
    plot_sensitivity(trace, "pm_eta", "PM eta sensitivity with Uniform crossover")
    build_checkpoint_and_duplicate_tables(main_trace, generations)

    manifest = {
        "complete_runs": int(len(runs)),
        "main_runs": int((runs["family"] == "main").sum()),
        "sensitivity_runs": int((runs["family"] != "main").sum()),
        "output_directory": str(OUTPUT_DIR),
    }
    with (OUTPUT_DIR / "analysis_complete.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
