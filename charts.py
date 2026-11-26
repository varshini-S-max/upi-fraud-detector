"""Neon-dark matplotlib charts shared by the app and train.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

BG, FG = "#0b1026", "#cfe8ff"
CYAN, PURPLE, PINK, GREEN, RED = "#00f0ff", "#a855f7", "#ff4ecd", "#2ee59d", "#ff4d6d"


def _fig(w=6.4, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color("#2a3560")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    return fig, ax


def importance_fig(importances):
    imp = importances.sort_values()
    fig, ax = _fig()
    ax.barh(imp.index, imp.values, color=CYAN, edgecolor=PURPLE)
    ax.set_title("Which features detect fraud?", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", color="#1d2650")
    fig.tight_layout()
    return fig


def confusion_fig(m, threshold):
    """2x2 grid; each cell has its own colour so small numbers stay readable."""
    from matplotlib.patches import FancyBboxPatch
    cells = [  # (row, col, label, value, colour)
        (0, 0, "True OK", m["tn"], "#12665a"),
        (0, 1, "False alarm", m["fp"], "#b8741a"),
        (1, 0, "Missed fraud", m["fn"], "#a8324a"),
        (1, 1, "Caught fraud", m["tp"], "#1f6fd1"),
    ]
    fig, ax = _fig(5.4, 4.4)
    for r, c, label, value, color in cells:
        ax.add_patch(FancyBboxPatch((c + 0.04, 1 - r + 0.04), 0.92, 0.92,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=color, edgecolor="#cfe8ff", linewidth=1.2))
        ax.text(c + 0.5, 1 - r + 0.56, f"{value:,}", ha="center", va="center",
                color="white", fontsize=20, fontweight="bold")
        ax.text(c + 0.5, 1 - r + 0.30, label, ha="center", va="center",
                color="white", fontsize=11)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Predicted OK", "Predicted fraud"])
    ax.set_yticks([0.5, 1.5], ["Actually fraud", "Actually OK"])
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})", fontweight="bold")
    fig.tight_layout()
    return fig


def pr_fig(y_true, p, threshold):
    prec, rec, thr = precision_recall_curve(y_true, p)
    idx = int(np.argmin(np.abs(thr - threshold)))
    fig, ax = _fig()
    ax.plot(rec, prec, color=CYAN, linewidth=2.5)
    ax.fill_between(rec, prec, color=CYAN, alpha=0.12)
    ax.scatter([rec[idx]], [prec[idx]], color=PINK, s=90, zorder=5,
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Recall (frauds caught)")
    ax.set_ylabel("Precision (alerts that are real)")
    ax.set_title("Precision-Recall trade-off", fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#1d2650")
    leg = ax.legend(facecolor=BG, edgecolor="#2a3560")
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    return fig
