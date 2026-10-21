import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.4, "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COL = {"A": "#E08A1E", "C": "#2F6DB5", "W": "#7A4FB8", "L3p": "#C0392B", "M": "#2E8B57"}
FILL = {"A": "#F5CC94", "C": "#A9C8EE", "W": "#CDB8EB", "L3p": "#EFA9A1", "M": "#B7DFC5"}
INK = "#1F2A36"
from pathlib import Path as _P
OUT = str(_P(__file__).resolve().parents[1] / "figs") + "/"
_P(OUT).mkdir(exist_ok=True)
def label(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight"); fig.savefig(OUT + name + ".png", bbox_inches="tight", dpi=150)
