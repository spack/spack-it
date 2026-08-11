# ruff: noqa: F401
"""Option (B) from changes/05-sc26-reference-type-rewrite.md decision 1:
new chart visualizing dependency/variant-score GAIN over each model's
own none/distilled baseline, averaged equally across all 4 sc26-roster
models, grouped by reference type -- same computation as
analysis/config_lift.py (kept as a CLI table there; this is its plot
form). Directly visualizes the cross-model "Effect of reference type"
claim, unlike scores_sonnet5.py (option A), which is Claude-Sonnet-5-only
and (per changes/05-...md) reproduces the old non-significant single-model
pattern rather than the pooled finding. "Gain" matches the wording already
used in the drafted results.tex prose (changes/05-...md's "Effect of
reference type" paragraph says "yield the largest gains"/"smallest gains");
title deliberately avoids leading with "Similarity" since the paper's point
is that reference *similarity* isn't what drives the effect. Preview only,
for the (A) vs (B) decision -- not wired into any figure path yet.
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pssgplot import BarPlot, PlotEnvironment

from analysis.core import pivot_tbl, pre

REF_DISPLAY = {
    "similar pkg (1)": "Similar",
    "random pkg (1) ": "Random",
    "random pkg (1)": "Random",
    "similar pkg (2)": "2 Similar",
    "random cmake pkg (1)": "Random CMake",
    "random cmake pkg (2)": "2 Random CMake",
    "none": "None",
}

files = sorted(
    f
    for f in glob.glob("results/sc26/*_distilled.jsonl")
    if "smoketest" not in f and "k20" not in f and "audit" not in f
)

df = pd.concat([pre(path) for path in files], ignore_index=True)
out = pivot_tbl(df).reset_index()

baseline = out[(out["reference"] == "none") & (out["buildsys"] == "distilled")].set_index(
    "model"
)

metrics = {"dependency_score": "Dependency", "variants_score": "Variant"}
for metric in metrics:
    out[f"{metric}_gain"] = out.apply(
        lambda r, m=metric: r[m] - baseline.loc[r["model"], m], axis=1
    )


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


rows = []
for metric, label in metrics.items():
    agg = (
        out.groupby("reference")[f"{metric}_gain"]
        .agg(score="mean", error=ci95)
        .reset_index()
        .assign(metric=label)
    )
    rows.append(agg)
plot_df = pd.concat(rows, ignore_index=True)
plot_df["ref"] = plot_df["reference"].map(REF_DISPLAY)

dep_order = (
    plot_df[plot_df["metric"] == "Dependency"]
    .sort_values("score", ascending=True)["ref"]
    .tolist()
)

with PlotEnvironment(font_path="fonts/gillsans.ttf"):
    bars = BarPlot()
    ax = bars.plot(
        data=plot_df,
        x="ref",
        y="score",
        hue="metric",
        title="Dependency and Variant Score Gain by Reference Type",
        xlabel="Reference",
        ylabel="Gain over no-reference baseline",
        order=dep_order,
        hatch=True,
        legend_title="Score",
        title_fontsize=15,
        ylabel_fontsize=14,
        xlabel_fontsize=14,
        legend=True,
        legend_fontsize=10,
        legend_loc="upper left",
        legend_bbox=(0.01, 0.99),
    )
    plt.gcf().set_size_inches(7.5, 4.3)
    plt.tick_params(axis="both", labelsize=11)
    plt.axhline(0, color="#606060", linewidth=1, zorder=1)

    bars.save(
        "output/scores-gain-by-ref.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
