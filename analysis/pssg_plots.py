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


# gpt41main = df_all.copy()[
#     (df_all["buildsys"] == "distilled")
#     & (df_all["model"] == "gpt-4.1")
#     & (
#         df_all["reference"].isin(
#             ["none", "similar pkg (1)", "random pkg (1) ", "random cmake pkg (1)"]
#         )
#     )
# ]
# gpt41main = cumulative_success_df(gpt41main, by="reference")


# gpt41main["group"] = gpt41main["group"].replace(
#     {
#         "none": "None",
#         "similar pkg (1)": "Similar",
#         "random pkg (1) ": "Random",
#         "random cmake pkg (1)": "Random in Build System",
#     }
# )

# with PlotEnvironment(font_path="fonts/gillsans.ttf"):
#     ax = LinePlot()
#     ax.plot(
#         data=gpt41main,
#         x="attempt",
#         y="rate",
#         hue="group",
#         markers=True,
#         legend=True,
#         legend_title="Reference",
#         xlabel="Generation Attempts",
#         ylabel="Cumulative Installation Rate",
#         ylim=(0, 1),
#         title="Cumulative Installation Rate With Iterative Repair",
#         title_fontsize=16,
#     )

#     ticks = sorted(gpt41main["attempt"].unique())
#     plt.xticks(ticks)  # sets ticks on the current axes
#     plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
#     plt.gcf().set_size_inches(6, 4)

#     ax.save(
#         "output/gpt41-main.pdf",
#         format="pdf",
#         dpi=600,
#         bbox_inches="tight",
#         pad_inches=0.1,
#     )


# GPT 5
gpt5main = df_all.copy()[
    (df_all["buildsys"] == "distilled")
    & (df_all["model"] == "gpt-5")
    & (
        df_all["reference"].isin(
            ["none", "similar pkg (1)", "random pkg (1) ", "similar pkg (2)"]
        )
    )
]
gpt5main = cumulative_success_df(gpt5main, by="ref")


# gpt5main["group"] = gpt5main["group"].replace(
#     {
#         "none": "None",
#         "similar pkg (2)": "2 Similar",
#         "similar pkg (1)": "Similar",
#         "random pkg (1) ": "Random",
#     }
# )

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    ax = LinePlot()
    ax.plot(
        data=gpt5main,
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
    )

    ticks = sorted(gpt5main["attempt"].unique())
    plt.xticks(ticks)  # sets ticks on the current axes
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    # same proportions as the scores-by-ref plot so it fits in the two column
    plt.gcf().set_size_inches(6, 4)
    plt.tick_params(axis="both", labelsize=11)

    ax.save(
        "output/gpt5-main.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )

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
        # disable legend
        # hue="group",
        markers=True,
        legend=None,
        legend_title="Configuration",
        xlabel="Generation Attempts",
        ylabel="Cumulative Installation Rate",
        ylim=(0, 1),
        title="Convergence of Iterative Improvement",
        title_fontsize=14,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend_fontsize=14,
    )

    ticks = [1, 5, 10, 15, 20, 25, 30]
    # ticks = sorted(attempts30["attempt"].unique())
    plt.xticks(ticks)  # only show every 5th tick
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    # same proportions as the scores-by-ref plot so it fits in the two column
    plt.gcf().set_size_inches(6, 3.5)
    plt.tick_params(axis="both", labelsize=11)

    ax.save(
        "output/30attempts.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
