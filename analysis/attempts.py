# ruff: noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pssgplot import PlotEnvironment
from pssgplot.lineplot import LinePlot

from analysis.core import (
    cumulative_success_df,
    pre,
)

df_all = pre("results/main_results/all.jsonl")


# 30 attempts
attempts30 = pre("results/30_attempts.jsonl")
attempts30 = cumulative_success_df(attempts30)


attempts30["group"] = attempts30["group"].replace(
    {
        "gpt-5": "gpt-5 with Similar Spack Package",
    }
)

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    ax = LinePlot()
    ax.plot(
        data=attempts30,
        x="attempt",
        y="rate",
        # hue="group",
        # markers=True,
        # legend=True,
        # legend_title="Coenfiguration",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 1),
        # title="Plateau of Interative Generation Loop",
        # title_fontsize=16,
    )

    ticks = sorted(attempts30["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    plt.gcf().set_size_inches(6, 4)

    ax.save(
        "output/30attempts.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
