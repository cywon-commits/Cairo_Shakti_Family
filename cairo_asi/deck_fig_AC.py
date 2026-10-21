import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from cairo_asi import CairoLattice
from dimer_map import DimerMap
from plot_phases import tile
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from hybrid_coupling import local_needle_coupling
n = 8; fig, axs = plt.subplots(1, 3, figsize=(18, 6.6))
cmap, norm = plt.get_cmap("bwr_r"), TwoSlopeNorm(0, -1, 1)
cases = [("A  (Cairo side, d = 0.137)", 0.1372, tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)),
         ("C  (Shakti side, d = 0.25)", 0.25, tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, n)),
         ("W  (wall crystal, d = 0.20)", 0.20, np.load("pt_best_d0.2000_n8_A3.npy"))]
for ax, (name, d, s) in zip(axs, cases):
    L = CairoLattice(n, n, d=d); D = DimerMap(L, local_needle_coupling(L)); chi = L.chirality(s); dm = D.dimers(s)
    for pts, c in zip(L.face_pts, chi): ax.add_patch(Polygon(pts, fc=cmap(norm(c)), ec="0.45", lw=0.4, alpha=0.85))
    for k in dm:
        f1, f2 = D.e2f[k]; p1 = L.face_center[f1]; p2 = p1 + L.mic(L.face_center[f2] - p1)
        ax.plot(*np.array([p1, p2]).T, color="#f2c230", lw=6, solid_capstyle="round"); ax.plot(*np.array([p1, p2]).T, "k", lw=1)
    ax.set_xlim(0.3, 4.3); ax.set_ylim(0.3, 4.3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(name, fontsize=18)
plt.tight_layout(); plt.savefig("deckfig/ACW_real.png", dpi=105, facecolor="white")
