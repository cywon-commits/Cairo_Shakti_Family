"""Draw the lowest-T PT state at d = 0.150, n = 12 (below the A energy) as dimers on the pentagon faces (as plot_phases.py)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from cairo_asi import CairoLattice
from dimer_map import DimerMap
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from hybrid_coupling import local_needle_coupling
from plot_phases import tile
n = 12; fig, axs = plt.subplots(1, 2, figsize=(15, 7.6))
cmap, norm = plt.get_cmap("bwr_r"), TwoSlopeNorm(0, -1, 1)
for ax, (title, s) in zip(axs, [("A reference, d = 0.150", tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)),
                                ("lowest PT state, d = 0.150, T = 0.03 (e - e_A = -4.4e-4)", np.load("fss/lowT_state_d0.1500_n12.npy"))]):
    L = CairoLattice(n, n, d=0.150); D = DimerMap(L, local_needle_coupling(L)); chi = L.chirality(s); dm = D.dimers(s)
    for pts, c in zip(L.face_pts, chi): ax.add_patch(Polygon(pts, fc=cmap(norm(c)), ec="0.45", lw=0.3, alpha=0.8))
    for k in dm:
        f1, f2 = D.e2f[k]; p1 = L.face_center[f1]; p2 = p1 + L.mic(L.face_center[f2] - p1)
        ax.plot(*np.array([p1, p2]).T, color="gold", lw=3, solid_capstyle="round"); ax.plot(*np.array([p1, p2]).T, "k", lw=0.6)
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig("fss_lowT_d0.1500_n12.png", dpi=90)
