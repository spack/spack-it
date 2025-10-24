# ruff: noqa: F401

import matplotlib.pyplot as plt

from analysis.core import plot_score_success_scatter, pre

df_all = pre("results/main_results/all.jsonl")

# TODO want to show some sort of analysis about similarity how does it affect other parts of the problem

plot_score_success_scatter(df_all, score="variants")
plt.show()
