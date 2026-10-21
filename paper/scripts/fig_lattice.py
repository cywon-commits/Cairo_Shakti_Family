import sys, os; from pathlib import Path; _H = Path(__file__).resolve().parent; _CODE = _H.parents[1] / "cairo_asi"
sys.path.insert(0, str(_H)); sys.path.insert(0, str(_CODE)); os.chdir(_CODE)   # data files are read relative to cairo_asi/
import numpy as np
from figstyle import *
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import TwoSlopeNorm
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from chessboard import chess_coords
from plot_phases import tile
# ---------- Fig. 1: the Cairo(d) family ----------
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.15))
for ax, (d, name), lab in zip(axs, [(0.03, "$d\\to 0$\nsquare ice"), (D_EQ, "$d=0.137$\nequilateral Cairo"), (0.25, "$d=1/4$\nShakti"), (0.40, "$d=0.40$\nconcave")], "abcd"):
    L = CairoLattice(3, 3, d=d)
    for k, (u, v) in enumerate(L.edges):
        p = L.vpos[u]; q = p + L.mic(L.vpos[v] - p)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=1.3, solid_capstyle="round")
    for v, z in enumerate(L.vtype):
        ax.plot(*L.vpos[v], "o", ms=4.5 if z == 4 else 3.2, color=COL["A"] if z == 4 else COL["C"], zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0.25, 2.75); ax.set_ylim(0.25, 2.75)
    ax.set_title(f"({lab}) " + name, fontsize=8.5, pad=4, loc="left")
h1, = axs[0].plot([], [], "o", ms=5, color=COL["A"]); h2, = axs[0].plot([], [], "o", ms=4, color=COL["C"])
fig.legend([h1, h2], ["$z=4$ vertex", "$z=3$ vertex"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
plt.subplots_adjust(wspace=0.18, bottom=0.12); save(fig, "fig1_family")
# ---------- helpers for state drawings ----------
cmap, norm = plt.get_cmap("bwr_r"), TwoSlopeNorm(0, -1, 1)
def draw_real(ax, L, D, s, xlim=(0.3, 3.3)):
    chi = L.chirality(s)
    for pts, c in zip(L.face_pts, chi): ax.add_patch(Polygon(pts, fc=cmap(norm(c)), ec="0.55", lw=0.3, alpha=0.85))
    lo, hi = xlim[0] + 0.05, xlim[1] - 0.05
    for k in range(L.N):
        p, e = L.epos[k], L.edir[k]*s[k]*L.elen[k]*0.36
        if not all(lo < q < hi for q in (*(p + e), *(p - e))): continue
        ax.annotate("", xy=p + e, xytext=p - e, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.15", mutation_scale=6), annotation_clip=True)
    for k in D.dimers(s):
        f1, f2 = D.e2f[k]; p1 = L.face_center[f1]; p2 = p1 + L.mic(L.face_center[f2] - p1)
        ax.plot(*np.array([p1, p2]).T, color="#F2C230", lw=3.6, solid_capstyle="round", zorder=4); ax.plot(*np.array([p1, p2]).T, color="k", lw=0.6, zorder=5)
    ax.set_xlim(*xlim); ax.set_ylim(*xlim); ax.set_aspect("equal"); ax.axis("off")
def draw_chess(ax, L, D, s, win=5):
    pos, _ = chess_coords(L, D); chi = L.chirality(s); dims = {}
    for k in D.dimers(s):
        f1, f2 = D.e2f[k]; c1, c2 = pos[f1], pos[f2]
        if abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) == 1: dims[frozenset([c1, c2])] = (np.sign(chi[f1]), c1, c2, f1, f2)
    for (x, y) in pos.values():
        if -win <= x < win and -win <= y < win: ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="#EDEDED" if (x + y) % 2 else "#D2D2D2", ec="none"))
    for dd, (sg, c1, c2, f1, f2) in dims.items():
        (x1, y1), (x2, y2) = c1, c2
        if not all(-win <= q < win for q in (x1, x2, y1, y2)): continue
        x0, y0 = min(x1, x2), min(y1, y2); w, h = (2, 1) if y1 == y2 else (1, 2)
        ax.add_patch(Rectangle((x0 - .5 + .08, y0 - .5 + .08), w - .16, h - .16, fc=COL["C"] if sg > 0 else "#D9534F", ec="k", lw=0.5, alpha=0.8))
    at = {c: f for f, c in pos.items()}
    for (x, y) in pos.values():
        if not (-win <= x < win - 1 and -win <= y < win - 1): continue
        h2 = frozenset([(x, y), (x+1, y)]) in dims and frozenset([(x, y+1), (x+1, y+1)]) in dims
        v2 = frozenset([(x, y), (x, y+1)]) in dims and frozenset([(x+1, y), (x+1, y+1)]) in dims
        if h2 or v2: ax.add_patch(Rectangle((x - .5 + .02, y - .5 + .02), 1.96, 1.96, fc="none", ec=COL["C"] if (x + y) % 2 == 0 else COL["A"], lw=1.4))
    ax.set_xlim(-win - .5, win - .5); ax.set_ylim(-win - .5, win - .5); ax.set_aspect("equal"); ax.axis("off")
# ---------- Fig. 2: A, C, W in real space and on the chessboard ----------
cases = [("A ($d=0.137$)", 0.1372, tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, 8)), ("C ($d=0.25$)", 0.25, tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, 8)),
         ("W ($d=0.20$)", 0.20, np.load("pt_best_d0.2000_n8_A3.npy"))]
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.75))
for c, (name, d, s) in enumerate(cases):
    L = CairoLattice(8, 8, d=d); D = DimerMap(L, local_needle_coupling(L))
    draw_real(axs[0, c], L, D, s); axs[0, c].set_title(f"({'abc'[c]}) " + name, fontsize=9, pad=4, loc="left")
    draw_chess(axs[1, c], L, D, s); axs[1, c].set_title(f"({'def'[c]}) " + name.split(" ")[0] + " on the chessboard", fontsize=9, pad=4, loc="left")
plt.subplots_adjust(hspace=0.16, wspace=0.08); save(fig, "fig2_ACW")
print("ok")
