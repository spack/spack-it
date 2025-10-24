# ruff: noqa: F401


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, MaxNLocator
from pssgplot import PlotEnvironment

from analysis.core import (
    pre,
)

SUCCESS_STAGE = "install"

df_all = pre("results/main_results/all.jsonl")


def plot_heatmap_metric(
    df: pd.DataFrame,
    value: str = "avg_attempts_to_success",
    *,
    compute_avg: bool = False,
    success_value: str = "install",
    status_col: str = "status",
    attempt_col: str = "attempt_num",
    buildsys: str | None = None,
    filter_col: str = "buildsys",
    x: str = "model",
    y: str = "reference",
    cmap: str = "Purples_r",
    annot: bool = True,
    fmt: str = ".2f",
    linewidths: float = 0.6,
    linecolor: str = "lightgray",
    square: bool = True,
    fontsize: int = 15,
    figsize=(8.5, 6.5),
    # scale controls
    vmin_int: int | None = None,
    vmax_int: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    continuous_step: float = 0.2,  # spacing for continuous ticks
    # ordering / labels
    x_order: list[str] | None = None,
    y_order: list[str] | None = None,
    font_path: str | None = "fonts/gillsans.ttf",
    title: str | None = None,
    cbar_label: str | None = None,
    y_label: str | None = None,
    save_path: None,
):
    """
    Single heatmap for any precomputed metric column. When vmin_int/vmax_int are set,
    integer ticks are used. Otherwise, the colorbar is continuous with tick spacing
    defined by `continuous_step` (default 0.2), and the bounds are snapped to that grid.
    """
    d = df.copy()
    if buildsys is not None:
        if filter_col not in d.columns:
            raise ValueError(f"filter_col='{filter_col}' not found.")
        d = d[d[filter_col] == buildsys]

    # compute metric if requested or missing
    if compute_avg or value not in d.columns:
        if {status_col, attempt_col}.issubset(d.columns):
            d[value] = d[attempt_col].where(d[status_col] == success_value)
        else:
            raise ValueError(
                f"Missing '{status_col}' or '{attempt_col}' to compute {value}."
            )

    needed = {x, y, value}
    missing = needed - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pivot = d.pivot_table(values=value, index=y, columns=x, aggfunc="mean")

    # axis order
    if y_order is None:
        y_order = list(pivot.index.dropna())
    if x_order is None:
        x_order = list(pivot.columns.dropna())
    pivot = pivot.reindex(index=y_order, columns=x_order)

    vals = pivot.to_numpy().astype(float)
    finite = np.isfinite(vals)

    # choose mode
    integer_mode = vmin_int is not None and vmax_int is not None
    if integer_mode:
        vmin_use, vmax_use = float(vmin_int), float(vmax_int)
        ticks = list(range(int(vmin_int), int(vmax_int) + 1))
        tick_locator = MaxNLocator(integer=True)
        tick_labels = [str(t) for t in ticks]
    else:
        # continuous: infer bounds if needed
        if finite.any():
            raw_min = float(np.nanmin(vals)) if vmin is None else float(vmin)
            raw_max = float(np.nanmax(vals)) if vmax is None else float(vmax)
        else:
            raw_min, raw_max = 0.0, 1.0
        # snap bounds to the step grid so ticks land on-scale
        step = float(continuous_step)
        lo = np.floor(raw_min / step) * step
        hi = np.ceil(raw_max / step) * step
        if hi <= lo:
            hi = lo + step
        # round to avoid 0.6000000001 type artifacts
        r = max(0, min(6, int(np.ceil(-np.log10(step)))))  # decimals to round
        vmin_use = round(lo, r)
        vmax_use = round(hi, r)
        # ticks on the snapped grid within [vmin_use, vmax_use]
        ticks_arr = np.arange(vmin_use, vmax_use + step / 2, step)
        ticks_arr = np.round(ticks_arr, r)
        ticks = ticks_arr.tolist()
        tick_locator = FixedLocator(ticks)
        tick_labels = [f"{t:.{r}f}" for t in ticks]

    if cbar_label is None:
        cbar_label = value.replace("_", " ").title()

    with PlotEnvironment(font_path=font_path):
        fig, ax = plt.subplots(figsize=figsize)

        heat = sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin_use,
            vmax=vmax_use,
            annot=annot,
            fmt=fmt,
            annot_kws={"fontsize": max(10, fontsize - 3)},
            linewidths=linewidths,
            linecolor=linecolor,
            mask=pivot.isna(),
            cbar=True,
            square=square,
        )

        cbar = heat.collections[0].colorbar
        cbar.outline.set_visible(False)

        # apply ticks/labels
        cbar.set_ticks(ticks)
        if not integer_mode:
            # force exact tick positions and labels for continuous mode
            cbar.ax.yaxis.set_major_locator(tick_locator)
            cbar.ax.set_yticklabels(tick_labels)
        else:
            cbar.ax.yaxis.set_major_locator(tick_locator)

        cbar.ax.tick_params(labelsize=max(10, fontsize - 2))
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=18, fontsize=fontsize)

        ncols = pivot.shape[1]
        ax.set_xticks(np.arange(ncols) + 0.5)
        ax.set_xticklabels(list(pivot.columns))

        # 2) & 3) shift labels left and disable clipping
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_rotation_mode("anchor")
            lbl.set_horizontalalignment("right")
            lbl.set_fontsize(max(8, fontsize - 4))
            lbl.set_clip_on(False)  # <-- allow left overhang

        ax.set_xlabel("")
        ylabel = y_label if y_label is not None else y.replace("_", " ").title()
        ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis="y", labelsize=fontsize - 2)

        if title:
            ax.set_title(title, fontsize=fontsize, pad=15)

        fig.tight_layout()
        if save_path:
            fig.savefig(
                save_path, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
            )


df_all["avg_attempts_to_success"] = df_all["attempt_num"].where(
    df_all["status"] == "install"
)

df_all["install_success"] = (df_all["status"] == "install").astype(int)


plot_heatmap_metric(
    df_all,
    value="avg_attempts_to_success",
    y="ref",  # your mapped/ordered categorical
    vmin_int=1,
    vmax_int=5,  # fixed 1..5 scale with integer ticks
    cbar_label="Attempts to Installation",
    y_label="Reference Type",
    figsize=(4, 4),
    fontsize=10,
    save_path="output/avg_attempts_heatmap.pdf",
)

# TODO explain that this is within the iterative repair look where k=5
plot_heatmap_metric(
    df_all,
    value="install_success",
    y="ref",  # your mapped/ordered categorical
    vmin=0.0,
    vmax=1.0,
    cmap="Reds",
    cbar_label="Installation Rate",
    y_label="Reference Type",
    figsize=(4, 4),
    fontsize=10,
    save_path="output/install_rate_heatmap.pdf",
)

plot_heatmap_metric(
    df_all,
    value="dependency_score",
    y="ref",  # your mapped/ordered categorical
    vmin=0.0,
    vmax=1.0,
    cmap="Greens",
    cbar_label="Dependency Score",
    y_label="Reference Type",
    figsize=(4, 4),
    fontsize=10,
    save_path="output/dependency_score_heatmap.pdf",
)

plot_heatmap_metric(
    df_all,
    value="variants_score",
    y="ref",  # your mapped/ordered categorical
    vmin=0.0,
    vmax=1.0,
    cmap="Blues",
    cbar_label="Variant Score",
    y_label="Reference Type",
    figsize=(4, 4),
    fontsize=10,
    save_path="output/variant_score_heatmap.pdf",
)
