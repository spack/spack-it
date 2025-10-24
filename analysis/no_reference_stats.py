# stats_simple.py
# Compare scores between has_reference==0 ("none") and has_reference==1 ("with_ref")
# Uses: Welch's t-test, Mann–Whitney U, and simple effect sizes.

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

from analysis.core import pre


def describe(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    q1, q3 = np.percentile(x, [25, 75]) if n else (np.nan, np.nan)
    return dict(
        n=n,
        mean=float(np.mean(x)) if n else np.nan,
        sd=float(np.std(x, ddof=1)) if n > 1 else np.nan,
        median=float(np.median(x)) if n else np.nan,
        q1=float(q1),
        q3=float(q3),
    )


def cohens_d(x, y):
    # d = (mean_y - mean_x) / pooled_sd  (positive ⇒ group y larger)
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float(0.0 if sp == 0 else (np.mean(y) - np.mean(x)) / sp)


def rank_biserial_from_u(u, n1, n2):
    # For U computed with the FIRST group as x:
    # r_rb > 0 ⇒ FIRST group tends to be larger; r_rb < 0 ⇒ second group tends larger
    return float(1 - (2 * u) / (n1 * n2))


# -----------------------------
# Load and split
# -----------------------------
df = pre("results/main_results/all.jsonl")
df["has_reference"] = (df["reference"] != "none").astype(int)
# df = df[(df["status"] == "install")].copy()
# df = df[(df["model"] == "gpt-5")].copy()
df = df.dropna(subset=["dependency_score", "variants_score"]).copy()

none = df[df["has_reference"] == 0]
with_ref = df[df["has_reference"] == 1]

scores = ["dependency_score", "variants_score"]

print("Group sizes:")
print(f"  none (0):     n={len(none)}")
print(f"  with_ref (1): n={len(with_ref)}")

for score in scores:
    print("\n" + "=" * 70)
    print(f"Score: {score}")
    print("=" * 70)

    x = pd.to_numeric(none[score], errors="coerce").dropna()
    y = pd.to_numeric(with_ref[score], errors="coerce").dropna()

    # ---- Descriptives
    d0 = describe(x)
    d1 = describe(y)
    print("Descriptives:")
    print(
        f"  none     n={d0['n']}, mean={d0['mean']:.6g}, sd={d0['sd']:.6g}, "
        f"median={d0['median']:.6g}, IQR=[{d0['q1']:.6g}, {d0['q3']:.6g}]"
    )
    print(
        f"  with_ref n={d1['n']}, mean={d1['mean']:.6g}, sd={d1['sd']:.6g}, "
        f"median={d1['median']:.6g}, IQR=[{d1['q1']:.6g}, {d1['q3']:.6g}]"
    )

    # ---- Quick normality check (same as you did)
    W0, p0 = shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
    W1, p1 = shapiro(y) if len(y) >= 3 else (np.nan, np.nan)
    print("Normality (Shapiro-Wilk):")
    print(f"  none:     W={W0:.3f}, p={p0:.3g}")
    print(f"  with_ref: W={W1:.3f}, p={p1:.3g}")

    # ---- Welch's t-test (means; robust to unequal variances)
    t_stat, t_p = ttest_ind(x, y, equal_var=False)
    print("Welch's t-test:")
    print(f"  t={t_stat:.3f}, p={t_p:.3g}")

    # ---- Mann–Whitney U (distribution / median shift)
    # Note: U is for the FIRST sample (x = none)
    u_stat, mw_p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    print("Man-Whitney U:")
    print(f"  U={u_stat:.3f}, p={mw_p:.3g}")

    # ---- Simple effect sizes
    d = cohens_d(x, y)  # >0 ⇒ with_ref higher than none
    r_rb = rank_biserial_from_u(u_stat, len(x), len(y))  # >0 ⇒ none > with_ref
    print("Effect sizes:")
    print(f"  Cohen's d (with_ref = none) = {d:.3f}")
    print(
        f"  Rank-biserial r (FIRST=none) = {r_rb:.3f}  "
        f"{'(r<0 ⇒ with_ref higher)' if not np.isnan(r_rb) else ''}"
    )

print("\nDone.")
