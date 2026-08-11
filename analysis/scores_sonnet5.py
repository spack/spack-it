# ruff: noqa: F401
"""Option (A) from changes/05-sc26-reference-type-rewrite.md decision 1:
same chart as the old scores.py (S_d/S_v grouped bars by reference type),
repointed from GPT-5/results/main_results/all.jsonl to Claude Sonnet 5's
sc26 distilled-metadata reference sweep. Preview only -- scores.py (the old
GPT-5 version) is left untouched until the (A) vs (B) decision is made.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pssgplot import BarPlot, PlotEnvironment

from analysis.core import pre

# the six distilled-metadata reference-type configs, same set used for
# fig/sonnet5-main.pdf's ref subset (none/similar1/similar2/randombuildsys2)
# plus the other two (random1, randombuildsys1) to cover the full sweep --
# raw_buildsys/cmake_raw deliberately excluded, that's a different axis
# (metadata form, not reference type).
files = [
    "none_distilled",
    "random1_distilled",
    "randombuildsys1_distilled",
    "randombuildsys2_distilled",
    "similar1_distilled",
    "similar2_distilled",
]
df_all = pd.concat(
    [pre(f"results/sc26/NO-CUI-LC-claude-sonnet-5_{f}.jsonl") for f in files],
    ignore_index=True,
)


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


dep = (
    df_all.groupby("ref", observed=True)["dependency_score"]
    .agg(score="mean", error=ci95)
    .reset_index()
    .assign(metric="Dependency")
)

var = (
    df_all.groupby("ref", observed=True)["variants_score"]
    .agg(score="mean", error=ci95)
    .reset_index()
    .assign(metric="Variant")
)

plot_df = pd.concat([dep, var], ignore_index=True)
order = (dep.sort_values("score", ascending=True)["ref"]).tolist()

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    bars = BarPlot()
    ax = bars.plot(
        data=plot_df,
        x="ref",
        y="score",
        hue="metric",
        title="Similarity Scores by Reference Type (Claude Sonnet 5)",
        xlabel="Reference",
        ylabel="Score",
        order=order,
        hatch=True,
        legend_title="Score",
        title_fontsize=16,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend=True,
        legend_fontsize=10,
        legend_loc="upper left",
        legend_bbox=(0.01, 0.99),
    )
    plt.gcf().set_size_inches(7.5, 4)
    plt.tick_params(axis="both", labelsize=11)

    bars.save(
        "output/scores-by-ref-sonnet5.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
