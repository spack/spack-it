# ruff: noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pssgplot import BarPlot, PlotEnvironment

from analysis.core import (
    pre,
)

df_all = pre("results/main_results/all.jsonl")
df_all = df_all[(df_all["model"] == "gpt-5")].copy()


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


dep = (
    df_all.groupby("ref")["dependency_score"]
    .agg(score="mean", error=ci95)
    .reset_index()
    .assign(metric="Dependency")
)

var = (
    df_all.groupby("ref")["variants_score"]
    .agg(score="mean", error=ci95)
    .reset_index()
    .assign(metric="Variant")
)

# If some refs are missing one metric, concat will handle it; drop NAs if needed
plot_df = pd.concat([dep, var], ignore_index=True)

# --- 2) Choose x order (e.g., by Dependency score, high → low) ---
order = (dep.sort_values("score", ascending=True)["ref"]).tolist()

# Optional: pretty names for x tick labels
# ref_map = {...}
# plot_df["ref_display"] = plot_df["ref"].map(ref_map).fillna(plot_df["ref"])
# x_col = "ref_display"
x_col = "ref"

# --- 3) Plot grouped bars with hue=metric and pass a single 'error' column ---
with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    bars = BarPlot()
    ax = bars.plot(
        data=plot_df,
        x=x_col,
        y="score",
        hue="metric",  # <-- grouped bars
        title="Similarity Scores by Reference Type",
        xlabel="Reference",
        ylabel="Score",
        order=order,  # <-- enforce ref ordering
        hatch=True,
        legend_title="Score",
        title_fontsize=16,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend=True,
        legend_fontsize=10,
        legend_loc="upper left",
        legend_bbox=(-0.009, 1.15),
    )
    plt.gcf().set_size_inches(7.5, 4)
    plt.tick_params(axis="both", labelsize=11)

    bars.save(
        "output/scores-by-ref-gpt5.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
