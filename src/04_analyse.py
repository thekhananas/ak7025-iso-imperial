#!/usr/bin/env python3
"""
Stage 4: Analyse results — QWK, statistical tests, visualisations.

Computes:
- Quadratic Weighted Kappa (QWK) per strategy
- Exact / adjacent agreement rates
- Friedman test across strategies + Wilcoxon pairwise post-hoc
- G-Eval dimension analysis
- Publication-ready figures

Usage:
    pixi run analyse
    # or: python src/04_analyse.py
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

from utils import RESULTS_DIR, ensure_dirs, load_config

warnings.filterwarnings("ignore", category=FutureWarning)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
STRATEGY_LABELS = {
    "zero_shot": "S1: Zero-Shot",
    "zero_shot_cot": "S2: Zero-Shot\n+ CoT",
    "few_shot_rubric": "S3: Few-Shot\n+ Rubric",
    "few_shot_cot_rubric": "S4: Few-Shot\n+ CoT + Rubric",
}
STRATEGY_ORDER = list(STRATEGY_LABELS.keys())
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def compute_grading_metrics(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Compute QWK, exact agreement, adjacent agreement, MAE, RMSE per strategy."""
    # Use mode of predicted scores across runs (majority vote)
    agg = (
        gen_df[gen_df["parse_success"] == True]
        .groupby(["sample_id", "strategy"])
        .agg(
            predicted_score=("predicted_score", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()),
            human_score=("human_score", "first"),
        )
        .reset_index()
    )

    rows = []
    for strategy in STRATEGY_ORDER:
        subset = agg[agg["strategy"] == strategy]
        if len(subset) < 2:
            continue

        y_true = subset["human_score"].astype(int)
        y_pred = subset["predicted_score"].astype(int)

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        lwk = cohen_kappa_score(y_true, y_pred, weights="linear")
        exact = (y_true == y_pred).mean()
        adjacent = (abs(y_true - y_pred) <= 1).mean()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        rows.append(
            {
                "strategy": strategy,
                "label": STRATEGY_LABELS[strategy],
                "n_samples": len(subset),
                "QWK": round(qwk, 4),
                "LWK": round(lwk, 4),
                "exact_agreement": round(exact, 4),
                "adjacent_agreement": round(adjacent, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_qwk_ci(
    gen_df: pd.DataFrame, n_resamples: int = 1000, alpha: float = 0.05
) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for QWK per strategy."""
    agg = (
        gen_df[gen_df["parse_success"] == True]
        .groupby(["sample_id", "strategy"])
        .agg(
            predicted_score=("predicted_score", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()),
            human_score=("human_score", "first"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(42)
    rows = []

    for strategy in STRATEGY_ORDER:
        subset = agg[agg["strategy"] == strategy]
        if len(subset) < 5:
            continue

        y_true = subset["human_score"].values.astype(int)
        y_pred = subset["predicted_score"].values.astype(int)
        n = len(y_true)

        boot_qwks = []
        for _ in range(n_resamples):
            idx = rng.choice(n, size=n, replace=True)
            try:
                bqwk = cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic")
                boot_qwks.append(bqwk)
            except Exception:
                pass

        if boot_qwks:
            ci_low = np.percentile(boot_qwks, 100 * alpha / 2)
            ci_high = np.percentile(boot_qwks, 100 * (1 - alpha / 2))
            rows.append(
                {
                    "strategy": strategy,
                    "QWK_mean": round(np.mean(boot_qwks), 4),
                    "QWK_ci_low": round(ci_low, 4),
                    "QWK_ci_high": round(ci_high, 4),
                }
            )

    return pd.DataFrame(rows)


def run_statistical_tests(eval_df: pd.DataFrame) -> str:
    """Run Friedman test + Wilcoxon pairwise comparisons on G-Eval scores."""
    report_lines = ["=" * 60, "STATISTICAL TESTS", "=" * 60, ""]

    valid = eval_df[eval_df["parse_success"] == True].copy()

    # Aggregate: mean eval score per (sample_id, strategy, dimension)
    agg = (
        valid.groupby(["sample_id", "strategy", "dimension"])["eval_score"]
        .mean()
        .reset_index()
    )

    # Compute composite score (mean across dimensions)
    composite = agg.groupby(["sample_id", "strategy"])["eval_score"].mean().reset_index()
    composite.columns = ["sample_id", "strategy", "composite_score"]

    # --- Friedman Test (within-subjects, non-parametric) ---
    report_lines.append("1. FRIEDMAN TEST (composite G-Eval score across strategies)")
    report_lines.append("-" * 40)

    pivot = composite.pivot(index="sample_id", columns="strategy", values="composite_score")
    # Drop samples that don't have all strategies
    pivot = pivot.dropna()

    if len(pivot) >= 3 and len(pivot.columns) >= 3:
        strategy_arrays = [pivot[s].values for s in STRATEGY_ORDER if s in pivot.columns]
        stat, p_value = stats.friedmanchisquare(*strategy_arrays)
        report_lines.append(f"  Chi-squared = {stat:.4f}")
        report_lines.append(f"  p-value = {p_value:.6f}")
        report_lines.append(
            f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at α=0.05"
        )
    else:
        report_lines.append("  Insufficient data for Friedman test")

    # --- Wilcoxon Signed-Rank Pairwise ---
    report_lines.append("")
    report_lines.append("2. WILCOXON SIGNED-RANK PAIRWISE COMPARISONS")
    report_lines.append("-" * 40)

    available_strategies = [s for s in STRATEGY_ORDER if s in pivot.columns]
    n_comparisons = len(list(combinations(available_strategies, 2)))
    bonferroni_alpha = 0.05 / max(n_comparisons, 1)
    report_lines.append(f"  Bonferroni-corrected α = {bonferroni_alpha:.4f} ({n_comparisons} comparisons)")
    report_lines.append("")

    for s1, s2 in combinations(available_strategies, 2):
        a = pivot[s1].values
        b = pivot[s2].values
        try:
            stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
            sig = "***" if p_value < bonferroni_alpha else "n.s."
            effect_r = stat / (len(a) * (len(a) + 1) / 2)  # rank-biserial approx
            report_lines.append(
                f"  {STRATEGY_LABELS[s1].replace(chr(10), ' ')} vs "
                f"{STRATEGY_LABELS[s2].replace(chr(10), ' ')}: "
                f"W={stat:.1f}, p={p_value:.6f} {sig}, r≈{effect_r:.3f}"
            )
        except Exception as e:
            report_lines.append(f"  {s1} vs {s2}: Error — {e}")

    # --- Per-dimension Kruskal-Wallis ---
    report_lines.append("")
    report_lines.append("3. PER-DIMENSION KRUSKAL-WALLIS TESTS")
    report_lines.append("-" * 40)

    for dim in sorted(agg["dimension"].unique()):
        dim_data = agg[agg["dimension"] == dim]
        groups = [
            dim_data[dim_data["strategy"] == s]["eval_score"].dropna().values
            for s in available_strategies
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            stat, p_value = stats.kruskal(*groups)
            sig = "*" if p_value < 0.05 else "n.s."
            report_lines.append(f"  {dim}: H={stat:.3f}, p={p_value:.4f} {sig}")

    report_lines.append("")
    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_qwk_bars(metrics_df: pd.DataFrame, ci_df: pd.DataFrame, fig_dir: Path, config: dict):
    """Bar chart of QWK per strategy with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if not ci_df.empty and "strategy" in ci_df.columns:
        merged = metrics_df.merge(ci_df, on="strategy", how="left")
    else:
        merged = metrics_df.copy()

    x = range(len(merged))
    bars = ax.bar(
        x,
        merged["QWK"],
        color=PALETTE[: len(merged)],
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Error bars from bootstrap
    if "QWK_ci_low" in merged.columns:
        yerr_low = merged["QWK"] - merged["QWK_ci_low"]
        yerr_high = merged["QWK_ci_high"] - merged["QWK"]
        ax.errorbar(
            x, merged["QWK"],
            yerr=[yerr_low, yerr_high],
            fmt="none", ecolor="black", capsize=5, zorder=4,
        )

    # Acceptance threshold line
    threshold = config["analysis"]["qwk_acceptance_threshold"]
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold (QWK={threshold})")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in merged["strategy"]], fontsize=9)
    ax.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax.set_title("Grading Accuracy by Prompt Strategy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    # Annotate bar values
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{merged.iloc[i]['QWK']:.3f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    plt.tight_layout()
    fmt = config["analysis"].get("figure_format", "png")
    fname = f"qwk_by_strategy.{fmt}"
    plt.savefig(fig_dir / fname, dpi=config["analysis"]["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved {fname}")


def plot_geval_radar(eval_df: pd.DataFrame, fig_dir: Path, config: dict):
    """Radar chart of 5 G-Eval dimensions per strategy."""
    valid = eval_df[eval_df["parse_success"] == True]
    means = valid.groupby(["strategy", "dimension"])["eval_score"].mean().unstack()

    dimensions = list(means.columns)
    n_dims = len(dimensions)

    if n_dims < 3:
        print("  ⚠️  Not enough dimensions for radar chart, skipping.")
        return

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, strategy in enumerate(STRATEGY_ORDER):
        if strategy not in means.index:
            continue
        values = means.loc[strategy].values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=PALETTE[i],
                label=STRATEGY_LABELS[strategy].replace("\n", " "))
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", " ").title() for d in dimensions], fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Feedback Quality by Dimension and Strategy", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    fmt = config["analysis"].get("figure_format", "png")
    fname = f"geval_radar.{fmt}"
    plt.savefig(fig_dir / fname, dpi=config["analysis"]["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved {fname}")


def plot_geval_heatmap(eval_df: pd.DataFrame, fig_dir: Path, config: dict):
    """Heatmap of mean G-Eval scores: strategy × dimension."""
    valid = eval_df[eval_df["parse_success"] == True]
    means = valid.groupby(["strategy", "dimension"])["eval_score"].mean().unstack()

    # Reorder
    means = means.reindex(index=[s for s in STRATEGY_ORDER if s in means.index])
    means.index = [STRATEGY_LABELS.get(s, s).replace("\n", " ") for s in means.index]
    means.columns = [c.replace("_", " ").title() for c in means.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        means, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=1, vmax=5, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean G-Eval Score (1-5)"},
    )
    ax.set_title("Feedback Quality Heatmap: Strategy × Dimension")
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.tight_layout()
    fmt = config["analysis"].get("figure_format", "png")
    fname = f"geval_heatmap.{fmt}"
    plt.savefig(fig_dir / fname, dpi=config["analysis"]["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved {fname}")


def plot_score_distribution(gen_df: pd.DataFrame, fig_dir: Path, config: dict):
    """Box plots of predicted vs. human score distributions."""
    valid = gen_df[gen_df["parse_success"] == True].copy()
    valid["strategy_label"] = valid["strategy"].map(
        lambda s: STRATEGY_LABELS.get(s, s).replace("\n", " ")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted score distribution
    sns.boxplot(
        data=valid, x="strategy_label", y="predicted_score",
        palette=PALETTE, ax=axes[0],
    )
    axes[0].set_title("Predicted Score Distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=15)

    # Score error (predicted - human)
    valid["score_error"] = valid["predicted_score"] - valid["human_score"]
    sns.boxplot(
        data=valid, x="strategy_label", y="score_error",
        palette=PALETTE, ax=axes[1],
    )
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Score Error Distribution (Predicted − Human)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Error")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fmt = config["analysis"].get("figure_format", "png")
    fname = f"score_distribution.{fmt}"
    plt.savefig(fig_dir / fname, dpi=config["analysis"]["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = load_config()
    ensure_dirs()
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gen_path = RESULTS_DIR / "generation_results.csv"
    eval_path = RESULTS_DIR / "evaluation_results.csv"

    if not gen_path.exists():
        print("ERROR: No generation results. Run `pixi run generate` first.")
        sys.exit(1)

    gen_df = pd.read_csv(gen_path)
    print(f"Loaded {len(gen_df)} generation records")

    # --- 1. Grading Metrics ---
    print("\n📈 Computing grading metrics...")
    metrics_df = compute_grading_metrics(gen_df)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print(metrics_df[["label", "QWK", "exact_agreement", "adjacent_agreement", "MAE"]].to_string(index=False))

    # --- 2. Bootstrap CIs ---
    print("\n📈 Computing bootstrap confidence intervals...")
    ci_df = bootstrap_qwk_ci(gen_df, n_resamples=config["analysis"]["bootstrap_n_resamples"])
    if not ci_df.empty:
        ci_df.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
        print(ci_df.to_string(index=False))

    # --- 3. QWK Bar Chart ---
    print("\n📊 Generating figures...")
    plot_qwk_bars(metrics_df, ci_df, fig_dir, config)
    plot_score_distribution(gen_df, fig_dir, config)

    # --- 4. G-Eval Analysis (if evaluation data exists) ---
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        print(f"\nLoaded {len(eval_df)} evaluation records")

        # Save G-Eval summary
        valid_evals = eval_df[eval_df["parse_success"] == True]
        geval_summary = valid_evals.groupby(["strategy", "dimension"])["eval_score"].agg(
            ["mean", "std", "count"]
        ).round(3)
        geval_summary.to_csv(RESULTS_DIR / "geval_scores.csv")
        print("\nG-Eval scores (mean ± std):")
        print(geval_summary)

        # Figures
        plot_geval_radar(eval_df, fig_dir, config)
        plot_geval_heatmap(eval_df, fig_dir, config)

        # Statistical tests
        print("\n📊 Running statistical tests...")
        test_report = run_statistical_tests(eval_df)
        report_path = RESULTS_DIR / "statistical_tests.txt"
        report_path.write_text(test_report)
        print(test_report)
    else:
        print("\n⚠️  No evaluation results found. Skipping G-Eval analysis.")
        print("   Run `pixi run evaluate` first for full analysis.")

    print(f"\n✅ Analysis complete! All outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
