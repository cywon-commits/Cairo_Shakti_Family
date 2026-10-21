import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts"); sys.path.insert(0, "/mnt/user-data/outputs/cairo_asi")
import numpy as np, json
from figstyle import *
from matplotlib.colors import ListedColormap
names = ["A", "C", "M", "W", "L3p"]
S = {X: {float(k): v for k, v in json.load(open(f"/mnt/user-data/outputs/cairo_asi/lattice_sums_{X}.json")).items()} for X in ("A", "M", "W", "L3p")}
def coef(X, d, key):
    if X == "C": return 0.0
    D = sorted(S[X]); return np.interp(d, D, [S[X][x][key] for x in D])
def E(X, d, mu): return coef(X, d, "SS") + mu*coef(X, d, "LS") + mu*mu*coef(X, d, "LL")
dg = np.linspace(0.137, 0.25, 400); mg = np.linspace(0.5, 4.5, 400)
W = np.array([[int(np.argmin([E(X, d, mu) for X in names])) for d in dg] for mu in mg])
fig, ax = plt.subplots(figsize=(4.9, 3.5))
ax.pcolormesh(dg, mg, W, cmap=ListedColormap([FILL[X] for X in names]), vmin=-0.5, vmax=4.5, shading="auto", rasterized=True)
# realistic fixed-gap islands
from cand_response import CairoLattice4
def area(L, w=100.0): return (L - w)*w + np.pi*(w/2)**2
dd = np.linspace(0.137, 0.247, 12); mud = []
for d in dd:
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)]); sc = 472/ref.elen[is34].mean()
    mud.append(area(ref.elen[~is34].mean()*sc - 44)/area(472 - 172))
ax.plot(dd, mud, "k-", lw=1.8, label="realistic islands, fixed gaps")
ax.axhline(2.0, color="k", ls="--", lw=1.1, label="ideal needles ($\\mu=2$)")
pts = [(0.25, 1.33, "C"), (0.25, 1.77, "~"), (0.25, 2.30, "W"), (0.25, 3.01, "W"), (0.20, 1.17, "C"), (0.20, 1.57, "~"), (0.20, 2.04, "W"),
       (0.25, 2.90, "W"), (0.20, 2.40, "W"), (0.175, 1.60, "W"), (0.175, 1.89, "W"), (0.20, 0.70, "W"), (0.20, 0.85, "W"),
       (0.175, 2.30, "A"), (0.175, 2.70, "A"), (0.20, 2.60, "W"), (0.20, 3.00, "A"), (0.20, 3.40, "A"), (0.25, 3.20, "W"), (0.25, 3.60, "A"), (0.25, 4.00, "A"),
       (0.150, 0.6, "L3p"), (0.150, 0.9, "A"), (0.150, 1.2, "A"), (0.155, 0.6, "L3p"), (0.155, 0.9, "L3p"), (0.155, 1.2, "A"), (0.155, 1.5, "A"),
       (0.160, 0.6, "W"), (0.160, 0.9, "C"), (0.160, 1.2, "L3p"), (0.160, 1.5, "A"), (0.165, 0.7, "W"), (0.165, 1.0, "C"), (0.165, 1.3, "C")]
for d, mu, w in pts: ax.plot(d, mu, "o", ms=4.6, mfc="white" if w == "~" else COL[w], mec="k", mew=0.6, clip_on=False, zorder=5)
for txt, x, y, c in (("A", 0.152, 3.6, "A"), ("W", 0.222, 2.35, "W"), ("C", 0.222, 1.27, "C"), ("W", 0.222, 0.68, "W")):
    ax.text(x, y, txt, fontsize=10, fontweight="bold", color=COL[c], ha="center", va="center")
ax.annotate("L3$'$", xy=(0.1535, 0.72), xytext=(0.140, 1.05), fontsize=10, fontweight="bold", color=COL["L3p"], ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=COL["L3p"], lw=1.0))
ax.plot([], [], "o", ms=5, mfc="0.6", mec="k", label="realistic-island checks (colour = winner, white = W$|$C boundary)")
ax.set_xlim(0.137, 0.25); ax.set_ylim(0.5, 4.5); ax.set_xlabel("$d$"); ax.set_ylabel("moment ratio $\\mu=m_{\\rm long}/m_{\\rm short}$")
ax.legend(loc="upper center", fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.16), ncol=2)
save(fig, "fig7_dmu")
print("ok")
