"""Plot per-iteration test-metric curves of each ensemble member.

Each co-training ensemble run writes a ``metrics_per_stage.csv`` recording metrics
at every training stage (``initial`` -> ``iteration_1..N`` -> ``final``). This
script draws, for each single (model, sub-dataset, censored %, broken %) config,
two plots against the iteration:

  * RMSE  — the four members' ``test_rmse_0..3`` plus ``weighted_test_rmse``
  * Score — the four members' ``test_score_0..3`` plus ``weighted_test_score``

each series in its own colour (shared between the RMSE and Score plots).

Divergent / very large values are handled exactly like ``plot_model_comparison.py``:
a robust per-plot cap keeps the well-behaved curves readable and any value above the
cap (or ``inf``) is drawn as a ▲ marker at the top edge instead of blowing up the
scale.

Output goes to ``outputs/iteration_plots/`` (regenerated each run); the cross-model
comparison plots in ``outputs/comparison_plots/`` are never touched.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_model_results import OUTPUTS_DIR, parse_config

PLOTS_DIR = os.path.join(OUTPUTS_DIR, "iteration_plots")

# (top-level folder glob, model label used in titles / filenames).
# The literal "-turbofan" after "ensemble" keeps the v1 glob from matching "_v2-".
MODEL_SPECS = [
    ("model-co_training_ensemble_v2-turbofan-FD*", "co_training_ensemble_v2"),
    ("model-co_training_ensemble-turbofan-FD*", "co_training_ensemble"),
]

# Fixed style per series, shared across the RMSE and Score plots. Distinct
# categorical colours (not a blue gradation) + marker keep them separable in
# colour, greyscale and for CVD. "^" is reserved for the above-cap indicator, so
# no series uses it. "weighted" is drawn thickest as the headline metric.
MEMBER_STYLES = [
    ("model 0", "#2a78d6", "o"),  # blue
    ("model 1", "#eda100", "s"),  # yellow
    ("model 2", "#008300", "D"),  # green
    ("model 3", "#e34948", "v"),  # red
]
WEIGHTED_STYLE = ("weighted", "#4a3aa7", "*")  # violet (headline)


def build_series(member_tmpl: str, weighted_col: str) -> list[dict]:
    """Series list (4 members + weighted) for a given metric's column names."""
    series = [
        {"label": label, "col": member_tmpl.format(i), "color": color,
         "marker": marker, "ls": "-", "lw": 1.8}
        for i, (label, color, marker) in enumerate(MEMBER_STYLES)
    ]
    label, color, marker = WEIGHTED_STYLE
    series.append({"label": label, "col": weighted_col, "color": color,
                   "marker": marker, "ls": "-", "lw": 3.0})
    return series


# One entry per metric to plot; each produces its own set of PNGs.
METRIC_SPECS = [
    {
        "prefix": "RMSE", "word": "test RMSE", "ylabel": "test RMSE",
        "member_cols": [f"test_rmse_{i}" for i in range(4)],
        "series": build_series("test_rmse_{}", "weighted_test_rmse"),
    },
    {
        "prefix": "Score", "word": "test score", "ylabel": "test score",
        "member_cols": [f"test_score_{i}" for i in range(4)],
        "series": build_series("test_score_{}", "weighted_test_score"),
    },
]

# Ink / chrome tokens (light surface).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def clean_stage(stage: str) -> str:
    """`iteration_3` -> `3`; `initial` / `final` kept as-is for x-tick labels."""
    return re.sub(r"iteration_", "", str(stage))


def plot_config(model_label: str, sub_dataset: str, censored: float, broken: float,
                df: pd.DataFrame, spec: dict) -> bool:
    """Draw one (model, sub-dataset, censored, broken) plot for one metric spec."""
    series = spec["series"]
    present = [s["col"] for s in series if s["col"] in df.columns]
    if df.empty or not present:
        return False

    values = df[present].apply(pd.to_numeric, errors="coerce")

    # Robust, per-plot Y cap (same MAD approach as plot_model_comparison.py).
    # The cap is derived from the *member* columns only: the weighted aggregate
    # diverges together with a diverged member, so including it would tip the pool
    # ~50/50 and defeat the robust cap. Tying the scale to the members keeps the
    # interesting range readable; a diverged aggregate is then simply clipped to a
    # top marker like anything else.
    member_cols = [c for c in spec["member_cols"] if c in values.columns]
    cap_cols = member_cols if member_cols else present
    flat = values[cap_cols].to_numpy().ravel()
    finite = flat[np.isfinite(flat) & (flat > 0.0)]
    if finite.size == 0:
        return False
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    cap = median + max(6.0 * 1.4826 * mad, 0.05 * abs(median))
    min_val = finite.min()
    bottom = max(0.0, min_val - 0.08 * (cap - min_val))

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    any_clipped = False

    for s in series:
        col = s["col"]
        if col not in values.columns:
            continue
        y = values[col].to_numpy()
        in_range = np.isfinite(y) & (y > 0.0) & (y <= cap)
        above = np.isposinf(y) | (np.isfinite(y) & (y > cap))

        ax.plot(
            x, np.where(in_range, y, np.nan),
            label=s["label"],
            color=s["color"], marker=s["marker"], linestyle=s["ls"],
            linewidth=s["lw"], markersize=8 if s["label"] == "weighted" else 7,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=4 if s["label"] == "weighted" else 3,
        )
        if above.any():
            ax.scatter(
                x[above], np.full(above.sum(), cap),
                marker="^", s=100,
                color=s["color"], edgecolor="white", linewidth=0.8,
                zorder=5, clip_on=False,
            )
            any_clipped = True

    ax.set_ylim(bottom, cap * 1.04)
    ax.set_xlabel("Iteration", color=INK)
    ax.set_ylabel(spec["ylabel"], color=INK)
    ax.set_title(
        f"Member & weighted {spec['word']} vs iteration — {model_label}\n"
        f"{sub_dataset}, censored {censored:g}, broken {broken:g}",
        color=INK, fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([clean_stage(s) for s in df["stage"]])

    if any_clipped:
        fig.text(
            0.5, 0.005, "▲ = value above chart top (very high / diverged / inf)",
            ha="center", va="bottom", fontsize=8, color=MUTED,
        )

    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(title="Series", frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(
        PLOTS_DIR,
        f"{spec['prefix']}_iterations_{model_label}_{sub_dataset}"
        f"_censored-{censored:.2f}_broken-{broken:.2f}.png",
    )
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return True


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Regenerate from scratch so no stale files from a previous plotting scheme remain.
    for old in glob.glob(os.path.join(PLOTS_DIR, "*.png")):
        os.remove(old)

    n = 0
    for model_glob, model_label in MODEL_SPECS:
        for model_dir in sorted(glob.glob(os.path.join(OUTPUTS_DIR, model_glob))):
            fd_match = re.search(r"turbofan-(FD\d+)", os.path.basename(model_dir))
            if not fd_match:
                continue
            sub_dataset = fd_match.group(1)

            for config_dir in sorted(glob.glob(os.path.join(model_dir, "censored-*-broken-*"))):
                csv_path = os.path.join(config_dir, "metrics_per_stage.csv")
                if not os.path.isfile(csv_path):
                    continue
                censored, broken = parse_config(os.path.basename(config_dir))
                df = pd.read_csv(csv_path)
                for spec in METRIC_SPECS:
                    if plot_config(model_label, sub_dataset, censored, broken, df, spec):
                        n += 1

    print(f"Done. {n} plot files in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
