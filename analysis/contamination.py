# ruff: noqa: F401

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from pssgplot import PlotEnvironment
from pssgplot.lineplot import LinePlot

from analysis.core import (
    cumulative_success_df,
    pre,
)

# this is one part of the analysis, the other will focus on comparing the install pct of the git repos we fetched that aren't in spack...probably not a plot
df_old = pre("results/contamination_leakage/old-pkgs.jsonl")
df_new = pre("results/contamination_leakage/new-pkgs.jsonl")


# doing a smoke test against spack-like packages and just doing installations
df_all = pre("results/main_results/all.jsonl")
df_smoke = pre("results/contamination_leakage/install_smoke.jsonl")

df_all = df_all.copy()[
    (df_all["buildsys"] == "distilled")
    & (df_all["model"] == "gpt-5")
    & (df_all["reference"] == "similar pkg (1)")
]
df_all["available"] = True
df_smoke["available"] = False

combined = pd.concat([df_all, df_smoke], ignore_index=True)

combined = cumulative_success_df(combined, by="available")

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    ax = LinePlot()
    ax.plot(
        data=combined,
        x="attempt",
        y="rate",
        hue="group",
        markers=True,
        legend=True,
        legend_title="Existing Spack Package",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 1),
        title="Generation of Packages Beyond E4S",
        title_fontsize=14,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend_fontsize=14,
        legend_loc="lower right",
    )

    ticks = sorted(combined["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    # same proportions as the scores-by-ref plot so it fits in the two column
    plt.gcf().set_size_inches(6, 3.5)
    plt.tick_params(axis="both", labelsize=11)

    ax.save(
        "output/gpt5-contamination.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
