"""Generate README figures from the completed values reported in the final PDF.

Only final-report-confirmed values are encoded here.
"""

from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "B0": "#4C78A8",
    "B1": "#F58518",
    "B2": "#54A24B",
    "B3": "#E45756",
}

EXPORTS = ROOT / "results" / "exports"


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


b0_rows = load_csv(EXPORTS / "b0_b3_results.csv")
b0_b3 = {}
labels = {"B0": "B0 text only", "B1": "B1 raw descriptions",
          "B2": "B2 denoised", "B3": "B3 denoised + SNR"}
for configuration in ("B0", "B1", "B2", "B3"):
    label = labels[configuration]
    b0_b3[label] = {"IEMOCAP": [], "MELD": []}
    for dataset in ("IEMOCAP", "MELD"):
        rows = [row for row in b0_rows
                if row["dataset"] == dataset and row["configuration"] == configuration]
        rows.sort(key=lambda row: int(row["epoch"]))
        b0_b3[label][dataset] = [float(row["weighted_f1"]) for row in rows]


def plot_b0_b3():
    epochs = np.array([1, 2, 3])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, dataset in zip(axes, ["IEMOCAP", "MELD"]):
        for label, values in b0_b3.items():
            key = label.split()[0]
            ax.plot(epochs, values[dataset], marker="o", linewidth=2.3,
                    markersize=6, label=label, color=COLORS[key])
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_ylabel("Weighted F1 (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(60, 74)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("B0–B3 Results Across Training Epochs", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIGURES / "b0_b3_weighted_f1.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Final-report Table VIII component changes.
def plot_ablation():
    components = ["Descriptions\n(B1-B0)", "Denoising\n(B2-B1)",
                  "SNR-aware\n(B3-B2)", "Net effect\n(B3-B1)"]
    iemocap = [1.03, -0.73, 0.90, 0.17]
    meld = [0.92, -2.38, 2.04, -0.34]
    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, iemocap, width, label="IEMOCAP",
                   color="#4C78A8")
    bars2 = ax.bar(x + width / 2, meld, width, label="MELD",
                   color="#E45756")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhspan(-0.5, 0.5, color="#E6B655", alpha=0.12)

    ax.set_title("Component Effects on Weighted F1")
    ax.set_ylabel("Change in weighted F1 (points)")
    ax.set_xticks(x, components)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (0.08 if value >= 0 else -0.16),
                    f"{value:+.2f}", ha="center", va="bottom",
                    fontsize=9)

    fig.text(0.5, 0.01,
             "Shaded band: approximately ±0.5 F1, where single-seed differences are inconclusive.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(FIGURES / "component_ablation.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Final-report Table VII completed PEFT results.
def plot_peft():
    methods = ["LoRA r16", "LoRA r8", "LoRA r32", "DoRA r16"]
    values = [72.50, 72.98, 74.07, 73.73]
    params = [12.6, 6.3, 25.2, 13.0]
    colors = ["#4C78A8", "#72B7B2", "#59A14F", "#E45756"]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(methods, np.nan_to_num(values, nan=0), color=colors,
                   edgecolor="#444444", linewidth=0.6)
    ax1.set_title("PEFT Comparison on IEMOCAP B3 (Epoch 3)")
    ax1.set_ylabel("Weighted F1 (%)")
    ax1.set_ylim(0, 78)
    ax1.grid(axis="y", alpha=0.25)

    for bar, value, count in zip(bars, values, params):
        if np.isnan(value):
            ax1.text(bar.get_x() + bar.get_width() / 2, 1.0,
                     "TBD", ha="center", va="bottom", fontsize=10,
                     fontweight="bold", color="#666666")
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                     f"{value:.2f}", ha="center", fontsize=10,
                     fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 max(0, bar.get_height()) - 4,
                 f"{count:.1f}M params",
                 ha="center", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(FIGURES / "peft_comparison.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_b0_b3()
    plot_ablation()
    plot_peft()
    print("Generated figures:")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"- {path.relative_to(ROOT)}")
