# ruff: noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pssgplot import PlotEnvironment
from pssgplot.lineplot import LinePlot

from analysis.core import (
    cumulative_success_df,
    pre,
)

# k=20 plateau study: gpt-5.4-mini, distilled metadata, one similar reference,
# extended from the standard k=5 budget.
attempts_k20 = pre("results/sc26/gpt-5.4-mini_similar1_distilled_k20.jsonl")
attempts_k20 = cumulative_success_df(attempts_k20)

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    ax = LinePlot()
    ax.plot(
        data=attempts_k20,
        x="attempt",
        y="rate",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 0.5),
        title="Convergence of Iterative Improvement",
        title_fontsize=14,
    )

    ticks = [1, 5, 10, 15, 20]
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
