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

# Claude Sonnet 5, best-performing model in the sc26 roster.
claude_main_files = [
    "none_distilled",
    "similar1_distilled",
    "similar2_distilled",
    "randombuildsys2_distilled",
]
claude_main = pd.concat(
    [
        pre(f"results/sc26/NO-CUI-LC-claude-sonnet-5_{f}.jsonl")
        for f in claude_main_files
    ],
    ignore_index=True,
)
claude_main = cumulative_success_df(claude_main, by="ref")

# pssgplot's default palette puts black at index 3; with exactly 4 groups
# here, "None" would land on it. Swap in the palette's 5th color (purple)
# instead of changing the library default.
NO_BLACK_PALETTE = ["#D55E00", "#0072B2", "#009E73", "#800080"]

with PlotEnvironment(font_path="fonts/gillsans.ttf", color_palette=NO_BLACK_PALETTE):
    ax = LinePlot()
    ax.plot(
        data=claude_main,
        x="attempt",
        y="rate",
        hue="group",
        markers=True,
        legend=True,
        legend_title="Reference",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 1),
        title="Cumulative Installation Rate With Iterative Repair",
        title_fontsize=14,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend_fontsize=14,
        figsize=(6, 4),
    )

    ticks = sorted(claude_main["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    plt.tick_params(axis="both", labelsize=11)
    a = plt.gca()
    a.set_title(a.get_title(), fontsize=14, pad=12)

    ax.save(
        "output/sonnet5-main.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )

# 30-attempts/k=20 plateau figure: see analysis/attempts.py.
