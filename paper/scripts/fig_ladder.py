import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts"); sys.path.insert(0, "/mnt/user-data/outputs/cairo_asi")
import numpy as np, json, glob
from figstyle import *
import matplotlib.gridspec as gridspec
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from plot_phases import tile
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
from chessboard import chess_coords
exec(open("/mnt/user-data/outputs/paper/fig_lattice.py").read().split("# ---------- Fig. 2")[0].split("# ---------- helpers for state drawings ----------")[1])   # draw_real, draw_chess
R = {}
for f in glob.glob("/mnt/user-data/outputs/cairo_asi/cand_response12x4_*.json"): R.update({float(k): v for k, v in json.load(open(f)).items() if "L3p" in v})
D = np.array(sorted(R)); names = ["A", "C", "W", "M", "L3p"]; lab = {"A": "A", "C": "C", "W": "W", "M": "M (columnar)", "L3p": "L3$'$ (ladder)"}
fig = plt.figure(figsize=(7.0, 2.6)); gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.25)
ax = fig.add_subplot(gs[0]); Emin = np.array([min(R[d][X]["E0"] for X in names) for d in D])
for X in names:
    ax.plot(D, (np.array([R[d][X]["E0"] for d in D]) - Emin)*1e3, "o-", ms=3, color=COL[X], label=lab[X], lw=1.3)
ax.set_ylim(-0.6, 16); ax.set_xlabel("$d$"); ax.set_ylabel("$E-E_{\\min}$  ($10^{-3}\\,D/a^3$ per island)")
ax.axvspan(0.1495, 0.157, color=FILL["L3p"], alpha=0.5, lw=0)
ax.annotate("L3$'$ window", xy=(0.1545, 0.6), xytext=(0.19, 6.0), fontsize=7.5, color=COL["L3p"], ha="left",
            arrowprops=dict(arrowstyle="->", color=COL["L3p"], lw=0.8))
ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95)
ax.set_title("(a) zero-field energies, ideal islands", fontsize=8.5, loc="left", pad=4)
L = CairoLattice(12, 12, d=0.153); Dm = DimerMap(L, local_needle_coupling(L)); s = np.load("/mnt/user-data/outputs/cairo_asi/L3alt_d0.153_n12.npy")
ax = fig.add_subplot(gs[1]); draw_chess(ax, L, Dm, s, win=5); ax.set_title("(b) ladder crystal L3$'$", fontsize=8.5, loc="left", pad=4)
from cairo2 import S0
from selection_T import FastDimers
L4 = CairoLattice(4, 2, d=0.20); F = FastDimers(L4, local_needle_coupling(L4)); ps = np.array([F.psi(x) for x in S0])
M = S0[np.flatnonzero((np.abs(np.abs(ps[:, 0]) - 1) < 1e-9) & (np.abs(np.abs(ps[:, 1]) - 1) < 1e-9))[0]]
L8 = CairoLattice(8, 8, d=0.153); D8 = DimerMap(L8, local_needle_coupling(L8))
ax = fig.add_subplot(gs[2]); draw_chess(ax, L8, D8, tile(M, 4, 2, 8), win=5); ax.set_title("(c) columnar crystal M", fontsize=8.5, loc="left", pad=4)
save(fig, "fig4_ladder")
print("ok")
