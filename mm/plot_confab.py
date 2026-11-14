"""One chart for the post: how often each model claims its choice saves lives, when that claim is false
(equal-size groups) vs. when it could be true (different sizes)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EQUAL, DIFF = "#eb6834", "#2a78d6"
MODELS = ["mistral:7b", "gemma3:4b", "phi4-mini:3.8b"]


def main():
    a = pd.read_csv("results/confab/coded.csv")
    rng = np.random.default_rng(0)
    rows = []
    for m in MODELS:
        g = a[a.model == m]
        for label, sub in (("equal", g[g.equal_n]), ("different", g[~g.equal_n])):
            v = sub.cites_count.values.astype(float)
            boots = [rng.choice(v, len(v)).mean() for _ in range(4000)]
            rows.append((m, label, v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(v)))
    d = pd.DataFrame(rows, columns=["model", "groups", "rate", "lo", "hi", "n"])

    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(MODELS))
    h = 0.34
    for off, label, color, name in ((-h / 2 - 0.02, "equal", EQUAL, "Groups the same size (claim is false)"),
                                    (h / 2 + 0.02, "different", DIFF, "Groups differ in size")):
        sub = d[d.groups == label].set_index("model").reindex(MODELS)
        ax.barh(y + off, sub.rate, height=h, color=color, label=name, zorder=3)
        ax.errorbar(sub.rate, y + off, xerr=[sub.rate - sub.lo, sub.hi - sub.rate], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=0, zorder=4)
        for yi, (_, r) in zip(y + off, sub.iterrows()):
            ax.text(r.hi + 0.02, yi, f"{r.rate:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, MODELS, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Explanations claiming the choice means fewer deaths", color=INK2, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK,
              handlelength=1.2, borderaxespad=0.2)
    fig.suptitle("AI models say they \"minimize loss of life\" even when both groups are the same size",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.905, "Moral Machine dilemmas, 100 per model, 95% bootstrap intervals", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig("results/confab/confabulation.png", dpi=220, facecolor=SURFACE)
    print(d.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
