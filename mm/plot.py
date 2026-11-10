"""Small multiples: one panel per model. Each row is an attribute; gray = humans (Awad et al. 2018, global),
blue = what the model does (revealed, position bias removed), orange = what it says it should do (stated)."""

import matplotlib.pyplot as plt
import pandas as pd

from mm.analyze import ATTRS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUMAN, DOES, SAYS = "#8a8984", "#2a78d6", "#eb6834"
LABELS = {"Intervention": "Inaction over swerving", "Relation to AV": "Pedestrians over passengers",
          "Law": "Lawful over jaywalkers", "Gender": "Women over men", "Fitness": "Fit over large",
          "Social Status": "High over low status", "Age": "Young over elderly", "No. Characters": "More lives over fewer",
          "Species": "Humans over pets"}


def main():
    eff = pd.read_csv("results/effects.csv")
    eff = eff[eff.persona == "none"]
    models = sorted(eff.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models) + 1.6, 4.6), sharey=True,
                             facecolor=SURFACE)
    order = ATTRS[::-1]
    for ax, m in zip(axes, models):
        d = eff[eff.model == m].set_index("attribute").reindex(order)
        y = range(len(order))
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK2, lw=1)
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.plot([r.lo, r.hi], [yi, yi], color=DOES, lw=2, alpha=0.5, solid_capstyle="round")
        ax.scatter(d.human_global, y, s=64, color=HUMAN, edgecolor=SURFACE, lw=2, zorder=3, label="Humans")
        if "stated_should" in d:
            ax.scatter(d.stated_should, y, s=64, color=SAYS, edgecolor=SURFACE, lw=2, zorder=4, marker="D",
                       label="Model says (should)")
        ax.scatter(d.effect, y, s=72, color=DOES, edgecolor=SURFACE, lw=2, zorder=5, label="Model does")
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["−1", "−.5", "0", ".5", "1"])
        ax.set_title(m, color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(order)), [LABELS[a] for a in order], color=INK, fontsize=9)
    axes[len(models) // 2].set_xlabel("Change in probability of being spared (ΔP)", color=INK2, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.6, 1.0))
    fig.suptitle("What small AI models do vs. what they say, against 40M human Moral Machine decisions",
                 x=0.01, ha="left", y=1.07, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig("results/moral_machine_models.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("wrote results/moral_machine_models.png")


if __name__ == "__main__":
    main()
