# ruff: noqa: F401

import matplotlib.pyplot as plt
import pandas as pd

from analysis.core import (
    coverage_matrix_vs_anchor,
    ground_baseline,
    hist_attempts,
    hm,
    hme,
    pivot_tbl,
    plot_cumulative_success_lines,
    plot_install_scores,
    plot_scores_over_attempts,
    plot_status_hist,
    pre,
)

df_all = pre("results/main_results/all.jsonl")

# plot_install_scores(df_all)
# breakpoint()

# hme(df_all, mode="avg_attempts")
plt.show()

# print(coverage_matrix_vs_anchor(df_all))

print(pivot_tbl(df_all))
out = pivot_tbl(df_all)
cross_model = (
    out.reset_index()
    .groupby("model")
    .apply(
        lambda g: pd.Series(
            {
                "dependency_score": (g["dependency_score"] * g["n"]).sum()
                / g["n"].sum(),
                "variants_score": (g["variants_score"] * g["n"]).sum() / g["n"].sum(),
                # "load": (g["load"] * g["n"]).sum() / g["n"].sum(),
                "n_total": g["n"].sum(),
            }
        )
    )
    .sort_values("dependency_score", ascending=False)
    .round(3)
)

print(cross_model)

breakpoint()

print(
    cross_model.to_latex(
        float_format="%.3f",
        caption="Cross-model installation and concretization (weighted means; k≤5).",
        label="tab:cross-model",
    )
)

exit()

# gpt-4.1 installation rates
fig, axes = plot_cumulative_success_lines(
    df_all,
    by="reference",
    baseline=ground_baseline,
    filter_expr='buildsys == "distilled" and model == "gpt-4.1" and reference in ["none", "similar pkg (1)", "random pkg (1) ", "random cmake pkg (1)"]',
    title="Cumulative Installation Success Rate With Iterative Repair",
    legend_map={
        "none": "None",
        "similar pkg (1)": "Similar",
        "random pkg (1) ": "Random",
        "random cmake pkg (1)": "Random in Build System",
    },
    legend_title="Spack Reference Recipe",
    subtitle="Model: gpt-4.1",
)
plt.show()

# gpt-5 installation rates

fig, axes = plot_cumulative_success_lines(
    df_all,
    by="reference",
    baseline=ground_baseline,
    filter_expr='buildsys == "distilled" and model == "gpt-5" and reference in ["none", "similar pkg (1)", "random pkg (1) ", "similar pkg (2)"]',
    title="Cumulative Installation Success Rate With Iterative Repair",
    legend_map={
        "none": "None",
        "similar pkg (2)": "2 Similar",
        "similar pkg (1)": "Similar",
        "random pkg (1) ": "Random",
    },
    legend_title="Spack Reference Recipe",
    subtitle="Model: gpt-5",
)
plt.show()


fig, axes = plot_cumulative_success_lines(
    df_all,
    by="reference",
    baseline=ground_baseline,
    facet_by="model",
)
plt.show()

hm(df_all, mode="scores")
plt.show()


hist_attempts(df_all)

# plot_scores_over_attempts(df_all)
# plot_status_hist(df_all)
