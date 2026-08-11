# ruff: noqa: F401

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from pssgplot import PlotEnvironment
from pssgplot.lineplot import LinePlot

from analysis.core import (
    cumulative_success_df,
    pre,
)

df_baseline = pre("results/sc26/gpt-5.4-mini_similar1_distilled.jsonl")
df_audit = pre("results/sc26/gpt-5.4-mini_similar1_distilled_audit_2.jsonl")

combined = pd.concat([df_baseline, df_audit], ignore_index=True)

combined = cumulative_success_df(combined, by="audit")

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    ax = LinePlot()
    ax.plot(
        data=combined,
        x="attempt",
        y="rate",
        hue="group",
        markers=True,
        legend=True,
        legend_title="Audit Status",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 0.5),
        title="Impact of Alternative Error Signals on Installation",
        title_fontsize=14,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend_fontsize=14,
        legend_loc="lower right",
        figsize=(6, 4),
    )

    ticks = sorted(combined["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    plt.tick_params(axis="both", labelsize=11)

    ax.save(
        "output/audit.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
