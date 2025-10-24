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

df_all = pre("results/main_results/all.jsonl")
df_audit = pre("results/audited.jsonl")

df_all = df_all.copy()[
    (df_all["buildsys"] == "distilled")
    & (df_all["model"] == "gpt-4.1")
    & (df_all["reference"] == "similar pkg (1)")
]
combined = pd.concat([df_all, df_audit], ignore_index=True)

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
        ylim=(0, 1),
        title="Impact of Alternative Error Signals on Installation",
        title_fontsize=14,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend_fontsize=14,
        legend_loc="center left",
    )

    ticks = sorted(combined["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    # same proportions as the scores-by-ref plot so it fits in the two column
    plt.gcf().set_size_inches(6, 3.5)
    plt.tick_params(axis="both", labelsize=11)

    ax.save(
        "output/audit.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
