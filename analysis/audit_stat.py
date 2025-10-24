# compare_install_audit.py
import numpy as np
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from analysis.core import pre

# -----------------------------
# Load and prepare data
# -----------------------------
df_all = pre("results/main_results/all.jsonl")
df_audit = pre("results/audited.jsonl")

# Add install_success binary flag
for df in (df_all, df_audit):
    df["install_success"] = (df["status"] == "install").astype(int)

# Filter df_all as requested
filter_expr = (
    'reference == "similar pkg (1)" and buildsys == "distilled" and model == "gpt-4.1"'
)
df_all = df_all.query(filter_expr).copy()

print("=== Dataset sizes ===")
print(f"Filtered df_all: {len(df_all)} rows")
print(f"Audited df_audit: {len(df_audit)} rows\n")


# -----------------------------
# Compute rates + confidence intervals
# -----------------------------
def rate_ci(successes, n, alpha=0.05, method="wilson"):
    if n == 0:
        return np.nan, (np.nan, np.nan)
    p = successes / n
    lo, hi = proportion_confint(successes, n, alpha=alpha, method=method)
    return p, (lo, hi)


def show_rate(name, df):
    n = len(df)
    k = int(df["install_success"].sum())
    p, (lo, hi) = rate_ci(k, n)
    print(
        f"{name:<10}  n={n:4d}  installs={k:4d}  rate={p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
    )
    return k, n


print("=== Install success rates ===")
k_all, n_all = show_rate("df_all", df_all)
k_aud, n_aud = show_rate("df_audit", df_audit)

# -----------------------------
# Two-proportion z-test
# -----------------------------
stat, p = proportions_ztest([k_all, k_aud], [n_all, n_aud], alternative="two-sided")
print("\n=== Two-proportion z-test ===")
print(f"z = {stat:.3f}, p = {p:.3g}")

if p < 0.05:
    print("→ Significant difference in install rates (p < 0.05).")
else:
    print("→ No significant difference in install rates (p ≥ 0.05).")

print("\nDone.")
