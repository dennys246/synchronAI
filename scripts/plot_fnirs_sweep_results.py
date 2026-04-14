#!/usr/bin/env python3
"""Generate publication-quality figures from fNIRS child/adult sweep results.

Reads history.json files from sweep and optional ablation directories,
producing four figures: training curves, final performance bars,
overfitting analysis, and temporal modeling comparison.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12
LEGEND_SIZE = 8

BN_COLOR = "#1f77b4"   # blue family
MS_COLOR = "#ff7f0e"   # orange family
RND_COLOR = "#d62728"  # red

BEST_MODEL = "bn_lstm64"

# Run ordering (bottleneck first, then multiscale)
BN_RUNS = [
    "bn_linear", "bn_mlp32", "bn_mlp64_proj",
    "bn_lstm64", "bn_lstm_proj", "bn_mlp32_overlap",
]
MS_RUNS = [
    "ms_linear", "ms_mlp32", "ms_mlp64_proj",
    "ms_mlp64_hvreg", "ms_mlp128",
]
RND_RUNS = ["bn_lstm64"]  # ablation random-init variant

MEAN_POOLED = {
    "bn_linear", "bn_mlp32", "bn_mlp64_proj",
    "ms_linear", "ms_mlp32", "ms_mlp64_proj", "ms_mlp64_hvreg", "ms_mlp128",
    "bn_mlp32_overlap",
}
LSTM_MODELS = {"bn_lstm64", "bn_lstm_proj"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_histories(sweep_dir: Path, ablation_dir: Path | None):
    """Return dict of {run_name: history_dict} and group membership."""
    histories: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}  # run_name -> "bn" | "ms" | "random"

    # Sweep runs
    for run_name in BN_RUNS + MS_RUNS:
        hpath = sweep_dir / run_name / "history.json"
        if hpath.exists():
            with open(hpath) as f:
                histories[run_name] = json.load(f)
            groups[run_name] = "bn" if run_name.startswith("bn_") else "ms"

    # Ablation random runs
    if ablation_dir is not None:
        for run_name in RND_RUNS:
            hpath = ablation_dir / run_name / "history.json"
            if hpath.exists():
                rnd_key = f"random_{run_name}"
                with open(hpath) as f:
                    histories[rnd_key] = json.load(f)
                groups[rnd_key] = "random"

    return histories, groups


def _group_color(group: str) -> str:
    return {"bn": BN_COLOR, "ms": MS_COLOR, "random": RND_COLOR}[group]


def _line_style(group: str) -> dict:
    if group == "bn":
        return {"linestyle": "-", "linewidth": 1.2}
    elif group == "ms":
        return {"linestyle": "--", "linewidth": 1.2}
    else:  # random
        return {"linestyle": ":", "linewidth": 2.5, "color": RND_COLOR}


def _cmap_for_runs(names: list[str], groups: dict[str, str]):
    """Assign a unique colour per run, loosely grouped by family."""
    bn_cmap = plt.cm.Blues
    ms_cmap = plt.cm.Oranges
    bn_names = [n for n in names if groups.get(n) == "bn"]
    ms_names = [n for n in names if groups.get(n) == "ms"]
    rnd_names = [n for n in names if groups.get(n) == "random"]
    colors = {}
    for i, n in enumerate(bn_names):
        colors[n] = bn_cmap(0.35 + 0.55 * i / max(len(bn_names) - 1, 1))
    for i, n in enumerate(ms_names):
        colors[n] = ms_cmap(0.35 + 0.55 * i / max(len(ms_names) - 1, 1))
    for n in rnd_names:
        colors[n] = RND_COLOR
    return colors


def _best_epoch(vals: list[float], higher_better: bool = True) -> int:
    if higher_better:
        return int(np.argmax(vals))
    return int(np.argmin(vals))


def _save(fig: plt.Figure, output_dir: Path, name: str):
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 1: Training curves (2x2)
# ---------------------------------------------------------------------------

def fig_training_curves(histories, groups, output_dir):
    print("Generating Figure 1: Training curves ...")
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), constrained_layout=True)
    ax_auc, ax_acc = axes[0]
    ax_tloss, ax_vloss = axes[1]

    names = list(histories.keys())
    colors = _cmap_for_runs(names, groups)

    panels = [
        (ax_auc, "val_aucs", "Validation AUC", True),
        (ax_acc, "val_accs", "Validation Accuracy", True),
        (ax_tloss, "train_losses", "Training Loss", False),
        (ax_vloss, "val_losses", "Validation Loss", False),
    ]

    for ax, key, title, higher_better in panels:
        for run_name in names:
            h = histories[run_name]
            vals = h.get(key, [])
            if not vals:
                continue
            epochs = np.arange(1, len(vals) + 1)
            grp = groups[run_name]
            style = _line_style(grp)
            color = style.pop("color", colors[run_name])
            ax.plot(epochs, vals, color=color, label=run_name, **style)

            # Star marker on best epoch for bn_lstm64
            if run_name == BEST_MODEL and key == "val_aucs":
                best_ep = _best_epoch(vals, higher_better)
                ax.plot(
                    best_ep + 1, vals[best_ep], marker="*", markersize=14,
                    color=colors[run_name], markeredgecolor="k",
                    markeredgewidth=0.6, zorder=5,
                )

        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        ax.set_ylabel(title.split()[-1], fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Shared legend below the figure
    handles, labels = ax_auc.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center",
            ncol=min(4, len(handles)), fontsize=LEGEND_SIZE,
            bbox_to_anchor=(0.5, -0.08), frameon=True,
        )

    _save(fig, output_dir, "fig1_training_curves")


# ---------------------------------------------------------------------------
# Figure 2: Final performance bar chart
# ---------------------------------------------------------------------------

def fig_final_performance(histories, groups, output_dir):
    print("Generating Figure 2: Final performance bar chart ...")

    # Gather final metrics, sort by best val AUC descending
    records = []
    for name, h in histories.items():
        aucs = h.get("val_aucs", [])
        accs = h.get("val_accs", [])
        f1s = h.get("val_f1s", [])
        if not aucs:
            continue
        best_ep = _best_epoch(aucs)
        records.append({
            "name": name,
            "auc": aucs[best_ep],
            "acc": accs[best_ep] if best_ep < len(accs) else accs[-1],
            "f1": f1s[best_ep] if best_ep < len(f1s) else (f1s[-1] if f1s else 0),
            "group": groups[name],
        })
    records.sort(key=lambda r: r["auc"], reverse=True)

    if not records:
        print("  No data -- skipping.")
        return

    n = len(records)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)

    bar_colors_map = {"bn": "#4a90d9", "ms": "#e8943a", "random": "#d62728"}
    auc_bars = ax.bar(
        x - width, [r["auc"] for r in records], width, label="AUC",
        color=[bar_colors_map[r["group"]] for r in records], edgecolor="k", linewidth=0.4,
    )
    acc_bars = ax.bar(
        x, [r["acc"] for r in records], width, label="Accuracy",
        color=[bar_colors_map[r["group"]] for r in records], edgecolor="k", linewidth=0.4,
        alpha=0.7,
    )
    f1_bars = ax.bar(
        x + width, [r["f1"] for r in records], width, label="F1",
        color=[bar_colors_map[r["group"]] for r in records], edgecolor="k", linewidth=0.4,
        alpha=0.45,
    )

    # Chance line
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=0, label="Chance")

    # Star on best model
    for i, r in enumerate(records):
        if r["name"] == BEST_MODEL or (r["name"].endswith(BEST_MODEL) and r["group"] == "bn"):
            ax.plot(i - width, r["auc"] + 0.02, marker="*", markersize=14,
                    color="gold", markeredgecolor="k", markeredgewidth=0.6, zorder=5)
            break

    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in records], rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    # Group legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4a90d9", edgecolor="k", label="Bottleneck"),
        Patch(facecolor="#e8943a", edgecolor="k", label="Multiscale"),
        Patch(facecolor="#d62728", edgecolor="k", label="Random init"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Chance"),
    ]
    metric_elements = [
        Patch(facecolor="gray", edgecolor="k", alpha=1.0, label="AUC"),
        Patch(facecolor="gray", edgecolor="k", alpha=0.7, label="Accuracy"),
        Patch(facecolor="gray", edgecolor="k", alpha=0.45, label="F1"),
    ]
    ax.legend(
        handles=legend_elements + metric_elements, fontsize=LEGEND_SIZE,
        loc="upper right", ncol=2, frameon=True,
    )
    ax.set_title("Final Performance by Model (sorted by AUC)", fontsize=TITLE_SIZE)

    _save(fig, output_dir, "fig2_final_performance")


# ---------------------------------------------------------------------------
# Figure 3: Overfitting analysis
# ---------------------------------------------------------------------------

def fig_overfitting(histories, groups, output_dir):
    print("Generating Figure 3: Overfitting analysis ...")

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    group_colors = {"bn": BN_COLOR, "ms": MS_COLOR, "random": RND_COLOR}

    for name, h in histories.items():
        aucs = h.get("val_aucs", [])
        train_accs = h.get("train_accs", [])
        val_accs = h.get("val_accs", [])
        if not aucs or not train_accs or not val_accs:
            continue
        best_ep = _best_epoch(aucs)
        ta = train_accs[best_ep] if best_ep < len(train_accs) else train_accs[-1]
        va = val_accs[best_ep] if best_ep < len(val_accs) else val_accs[-1]
        grp = groups[name]
        ax.scatter(ta, va, c=group_colors[grp], s=60, edgecolors="k",
                   linewidths=0.5, zorder=3)
        ax.annotate(
            name, (ta, va), fontsize=7, textcoords="offset points",
            xytext=(4, 4), ha="left",
        )

    # y = x diagonal
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Train Accuracy (best AUC epoch)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Val Accuracy (best AUC epoch)", fontsize=LABEL_SIZE)
    ax.set_title("Overfitting Analysis", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BN_COLOR, edgecolor="k", label="Bottleneck"),
        Patch(facecolor=MS_COLOR, edgecolor="k", label="Multiscale"),
        Patch(facecolor=RND_COLOR, edgecolor="k", label="Random init"),
        plt.Line2D([0], [0], color="k", linestyle="--", alpha=0.5, label="Perfect gen."),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="lower right")

    _save(fig, output_dir, "fig3_overfitting")


# ---------------------------------------------------------------------------
# Figure 4: Temporal modeling comparison (box + swarm)
# ---------------------------------------------------------------------------

def fig_temporal_comparison(histories, groups, output_dir):
    print("Generating Figure 4: Temporal modeling comparison ...")

    categories = {"Mean-pooled": [], "LSTM": [], "Random init": []}
    labels_map = {"Mean-pooled": [], "LSTM": [], "Random init": []}

    for name, h in histories.items():
        aucs = h.get("val_aucs", [])
        if not aucs:
            continue
        best_auc = max(aucs)
        grp = groups[name]

        if grp == "random":
            categories["Random init"].append(best_auc)
            labels_map["Random init"].append(name)
        elif name in LSTM_MODELS:
            categories["LSTM"].append(best_auc)
            labels_map["LSTM"].append(name)
        else:
            categories["Mean-pooled"].append(best_auc)
            labels_map["Mean-pooled"].append(name)

    # Remove empty categories
    cats_to_plot = [k for k in ["Mean-pooled", "LSTM", "Random init"] if categories[k]]
    if not cats_to_plot:
        print("  No data -- skipping.")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)

    data = [categories[c] for c in cats_to_plot]
    positions = np.arange(1, len(cats_to_plot) + 1)

    cat_colors = {
        "Mean-pooled": "#6baed6",
        "LSTM": "#2171b5",
        "Random init": RND_COLOR,
    }

    bp = ax.boxplot(
        data, positions=positions, widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops={"color": "k", "linewidth": 1.5},
    )
    for patch, cat in zip(bp["boxes"], cats_to_plot):
        patch.set_facecolor(cat_colors[cat])
        patch.set_alpha(0.5)
        patch.set_edgecolor("k")
        patch.set_linewidth(0.8)

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for pos, cat in zip(positions, cats_to_plot):
        vals = categories[cat]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter, vals, c=cat_colors[cat], s=50,
            edgecolors="k", linewidths=0.5, zorder=3,
        )
        for v, j, lbl in zip(vals, jitter, labels_map[cat]):
            ax.annotate(
                lbl, (pos + j, v), fontsize=6, textcoords="offset points",
                xytext=(5, 2), ha="left",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(cats_to_plot, fontsize=LABEL_SIZE)
    ax.set_ylabel("Best Validation AUC", fontsize=LABEL_SIZE)
    ax.set_title("Temporal Modeling Comparison", fontsize=TITLE_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    _save(fig, output_dir, "fig4_temporal_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot fNIRS child/adult sweep results.",
    )
    parser.add_argument(
        "--sweep-dir", type=str,
        default="runs/fnirs_child_adult_sweep",
        help="Directory containing sweep run subdirectories (default: runs/fnirs_child_adult_sweep)",
    )
    parser.add_argument(
        "--ablation-dir", type=str,
        default="runs/fnirs_ablation_random",
        help="Directory containing random-init ablation runs (default: runs/fnirs_ablation_random)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Output directory for figures (default: <sweep-dir>/figures)",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else None
    if ablation_dir is not None and not ablation_dir.exists():
        print(f"Ablation dir {ablation_dir} not found -- skipping ablation runs.")
        ablation_dir = None

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep dir:    {sweep_dir}")
    print(f"Ablation dir: {ablation_dir or '(none)'}")
    print(f"Output dir:   {output_dir}")
    print()

    histories, groups = load_histories(sweep_dir, ablation_dir)
    if not histories:
        print("ERROR: No history.json files found. Check --sweep-dir path.")
        return

    print(f"Loaded {len(histories)} runs: {', '.join(sorted(histories.keys()))}\n")

    fig_training_curves(histories, groups, output_dir)
    fig_final_performance(histories, groups, output_dir)
    fig_overfitting(histories, groups, output_dir)
    fig_temporal_comparison(histories, groups, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
