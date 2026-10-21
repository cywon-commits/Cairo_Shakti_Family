import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from cairo_asi import *

def draw(ax, L, s, title, win=(0.3, 4.3)):
    chi = L.chirality(s)
    norm = TwoSlopeNorm(0, -1, 1)
    cmap = plt.get_cmap("bwr_r")          # +1 (CCW) blue, -1 (CW) red
    for f, pts, c in zip(L.faces, L.face_pts, chi):
        ax.add_patch(Polygon(pts, closed=True, fc=cmap(norm(c)), ec="0.35", lw=0.5, alpha=0.85))
    m = L.edir * s[:, None] * L.elen[:, None] * 0.7
    ax.quiver(L.epos[:, 0] - m[:, 0]/2, L.epos[:, 1] - m[:, 1]/2, m[:, 0], m[:, 1],
              angles="xy", scale_units="xy", scale=1, width=0.006, headwidth=4, color="k")
    q = L.charges(s)
    for z, mk in [(3, "o"), (4, "s")]:
        sel = L.vtype == z
        ax.scatter(*L.vpos[sel].T, c=q[sel], cmap="PiYG", vmin=-2, vmax=2, marker=mk, s=18, zorder=3, edgecolors="k", linewidths=0.3)
    ax.set_xlim(*win); ax.set_ylim(*win); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

L = CairoLattice(8, 8)
s_gs = np.load("gs_cell2x2.npy"); s_sa = np.load("sa_best_dip6_8.npy")
J = L.coupling_matrix(6.0)
e = lambda s: 0.5*s@J@s/L.N
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
draw(ax[0], L, s_gs, f"(a) 2x2-periodic GS candidate, E/N={e(s_gs):.4f}")
draw(ax[1], L, s_sa, f"(b) best SA state (8x8), E/N={e(s_sa):.4f}")
res = json.load(open("sa_results_8.json"))
ax[2].hist(res["dip6"]["sa"], bins=12, color="0.6", label="SA final states (20 runs)")
ax[2].axvline(e(s_gs), color="r", label="2x2 GS candidate")
ax[2].set_xlabel("E/N  (D/a³)"); ax[2].set_ylabel("count"); ax[2].legend(fontsize=8)
ax[2].set_title("(c) dipolar model, rc = 6a", fontsize=10)
plt.tight_layout(); plt.savefig("cairo_states.png", dpi=140)

# chirality statistics per pentagon orientation class (4 pentagons / cell)
for name, s in [("GS", s_gs), ("SA", s_sa)]:
    chi = L.chirality(s)
    print(name, "distinct |chirality| values:", np.unique(np.round(np.abs(chi), 2), return_counts=True))
