"""Plot model-comparison curves from the benchmark summary CSVs in ``outputs/``.

Reads every ``*_results_turbofan.csv`` file, then for each metric (RMSE, Score),
each CMAPSS sub-dataset (FD001..FD004) and each censored-data percentage, draws a
plot with:

    X axis = percentage of broken data
    Y axis = the metric (log scale)

Every model is drawn as its own coloured line so they can be compared directly.
One image file is produced per (metric x sub-dataset x censored) combination.

Some runs diverged and produced astronomically large or ``inf`` metric values;
a logarithmic Y axis keeps the normal (~15 RMSE / ~200 Score) and divergent
values readable, and non-finite / non-positive points are dropped.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import results_columns

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "comparison_plots")

# Metrics to plot on the Y axis.
METRICS = [results_columns.RMSE, results_columns.SCORE]

# Censored=0.0 only pairs with broken=0.0 (the single uncensored reference run),
# so there is no broken-percentage sweep to draw for it.
CENSORED_LEVELS = [0.6, 0.8, 0.9, 0.98]

# Fixed per-model style (colour follows the entity, never its rank / plot order).
# Colours are the CVD-validated categorical slots from the data-viz palette;
# distinct markers + line styles add a secondary channel (relief rule) so the
# models stay distinguishable in greyscale / for colour-vision-deficient readers.
MODEL_STYLE = {
    "co_training_ensemble":    {"color": "#2a78d6", "marker": "o", "ls": "-"},   # blue
    "co_training_ensemble_v2": {"color": "#1baf7a", "marker": "s", "ls": "--"},  # aqua
    "coprog":                  {"color": "#eda100", "marker": "^", "ls": "-."},  # yellow
    "transformer":             {"color": "#008300", "marker": "D", "ls": "-"},   # green
    "lstm":                    {"color": "#4a3aa7", "marker": "v", "ls": "--"},  # violet
    "autoencoder":             {"color": "#e34948", "marker": "P", "ls": "-."},  # red
}
DEFAULT_STYLE = {"color": "#898781", "marker": "x", "ls": ":"}

# Ink / chrome tokens (light surface).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_all_results() -> pd.DataFrame:
    """Concatenate every benchmark summary CSV into one tidy frame."""
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*_results_turbofan.csv")))
    if not files:
        raise FileNotFoundError(f"No *_results_turbofan.csv files in {OUTPUTS_DIR}")

    cols = [
        results_columns.SUB_DATASET,
        results_columns.CENSORED_PERCENTAGE,
        results_columns.BROKEN_PERCENTAGE,
        results_columns.MODEL,
        results_columns.RMSE,
        results_columns.SCORE,
    ]
    frames = []
    for f in files:
        # read_csv keys by header name, so the transformer file's different
        # column order (Model last) is handled transparently.
        df = pd.read_csv(f)[cols]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def plot_panel(df: pd.DataFrame, metric: str, sub_dataset: str, censored: float) -> None:
    """Draw one (metric, sub-dataset, censored) panel and save it, if it has data."""
    panel = df[
        (df[results_columns.SUB_DATASET] == sub_dataset)
        & (df[results_columns.CENSORED_PERCENTAGE] == censored)
        & (df[results_columns.BROKEN_PERCENTAGE] > 0.0)
    ]
    if panel.empty:
        return

    # Robust, per-panel Y cap so the well-behaved models spread out readably.
    # Diverged / inf / very-high points are drawn as markers at the top edge
    # instead of blowing out the linear scale (MAD is robust to a diverged model
    # as long as the well-behaved ones are the majority).
    finite = panel[metric][np.isfinite(panel[metric]) & (panel[metric] > 0.0)]
    if finite.empty:
        return
    median = finite.median()
    mad = (finite - median).abs().median()
    cap = median + max(6.0 * 1.4826 * mad, 0.05 * abs(median))
    min_val = finite.min()
    bottom = max(0.0, min_val - 0.08 * (cap - min_val))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    plotted_any = False
    any_clipped = False
    for model in MODEL_STYLE:  # fixed model order -> stable legend & colours
        style = MODEL_STYLE.get(model, DEFAULT_STYLE)
        sub = panel[panel[results_columns.MODEL] == model].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(results_columns.BROKEN_PERCENTAGE)
        x = sub[results_columns.BROKEN_PERCENTAGE].to_numpy()
        y = sub[metric].to_numpy()

        in_range = np.isfinite(y) & (y > 0.0) & (y <= cap)
        # too-high (finite but above cap) or inf -> shown as a top-edge marker.
        above = (np.isposinf(y)) | (np.isfinite(y) & (y > cap))

        # Line + markers only where the value is on-scale.
        ax.plot(
            x, np.where(in_range, y, np.nan),
            label=model,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        if above.any():
            ax.scatter(
                x[above], np.full(above.sum(), cap),
                marker="^", s=110,
                color=style["color"], edgecolor="white", linewidth=0.8,
                zorder=5, clip_on=False,
            )
            any_clipped = True
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_ylim(bottom, cap * 1.04)
    ax.set_xlabel("Broken data percentage", color=INK)
    ax.set_ylabel(metric, color=INK)
    ax.set_title(
        f"{metric} vs broken percentage — {sub_dataset}, censored {censored:g}",
        color=INK,
    )
    if any_clipped:
        # Below the axes so it never collides with the top-edge ▲ markers.
        fig.text(
            0.5, 0.005, "▲ = value above chart top (very high / diverged / inf)",
            ha="center", va="bottom", fontsize=8, color=MUTED,
        )

    broken_ticks = sorted(panel[results_columns.BROKEN_PERCENTAGE].unique())
    ax.set_xticks(broken_ticks)

    ax.grid(True, which="both", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(title="Model", frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(
        PLOTS_DIR, f"{metric}_{sub_dataset}_censored-{censored:.2f}.png"
    )
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_all_results()

    for metric in METRICS:
        for sub_dataset in sorted(df[results_columns.SUB_DATASET].unique()):
            for censored in CENSORED_LEVELS:
                plot_panel(df, metric, sub_dataset, censored)

    n = len(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
    print(f"Done. {n} plot files in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
