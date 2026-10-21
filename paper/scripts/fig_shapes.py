import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts"); sys.path.insert(0, "/mnt/user-data/outputs/cairo_asi")
import numpy as np, json, glob
from figstyle import *
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
from chessboard import chess_coords
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from plot_phases import tile
from cand_response import CairoLattice4
exec(open("/mnt/user-data/outputs/paper/scripts/fig_lattice.py").read().split("# ---------- Fig. 2")[0].split("# ---------- helpers for state drawings ----------")[1])
# ======================= Fig. 4: shapes of M and L3' =======================
from cairo2 import S0
from selection_T import FastDimers
L4 = CairoLattice(4, 2, d=0.20); F = FastDimers(L4, local_needle_coupling(L4)); ps = np.array([F.psi(x) for x in S0])
M4 = S0[np.flatnonzero((np.abs(np.abs(ps[:, 0]) - 1) < 1e-9) & (np.abs(np.abs(ps[:, 1]) - 1) < 1e-9))[0]]
LM = CairoLattice(8, 8, d=0.153); DM = DimerMap(LM, local_needle_coupling(LM)); sM = tile(M4, 4, 2, 8)
LL = CairoLattice(12, 12, d=0.153); DL = DimerMap(LL, local_needle_coupling(LL)); sL = np.load("/mnt/user-data/outputs/cairo_asi/L3alt_d0.153_n12.npy")
R = {}
for f in glob.glob("/mnt/user-data/outputs/cairo_asi/cand_response12x4_*.json"): R.update({float(k): v for k, v in json.load(open(f)).items() if "L3p" in v})
D = np.array(sorted(R)); names = ["A", "C", "W", "M", "L3p"]; lab = {"A": "A", "C": "C", "W": "W", "M": "M (columnar)", "L3p": "L3$'$ (ladder)"}
fig = plt.figure(figsize=(7.0, 5.1)); gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.05], wspace=0.48, hspace=0.42)
ax = fig.add_subplot(gs[0, 0]); draw_real(ax, LM, DM, sM); ax.set_title("(a) M in real space", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[0, 1]); draw_real(ax, LL, DL, sL); ax.set_title("(b) L3$'$ in real space", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[0, 2]); Emin = np.array([min(R[d][X]["E0"] for X in names) for d in D])
for X in names: ax.plot(D, (np.array([R[d][X]["E0"] for d in D]) - Emin)*1e3, "o-", ms=2.6, color=COL[X], label=lab[X], lw=1.1)
ax.set_ylim(-0.6, 16); ax.set_xlabel("$d$"); ax.set_ylabel("$E-E_{\\min}$ ($10^{-3}D/a^3$)", fontsize=8)
ax.axvspan(0.1495, 0.157, color=FILL["L3p"], alpha=0.5, lw=0)
ax.annotate("L3$'$ window", xy=(0.1545, 0.6), xytext=(0.185, 6.0), fontsize=7, color=COL["L3p"], arrowprops=dict(arrowstyle="->", color=COL["L3p"], lw=0.8))
ax.legend(loc="upper right", fontsize=6.3, frameon=True, framealpha=0.95); ax.set_title("(c) zero-field energies", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[1, 0]); draw_chess(ax, LM, DM, sM, win=5); ax.set_title("(d) M on the chessboard", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[1, 1]); draw_chess(ax, LL, DL, sL, win=5); ax.set_title("(e) L3$'$ on the chessboard", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[1, 2]); y = np.arange(0, 12)
hM = np.full(12, 0.5); hL = np.array([0.5 if (k // 3) % 2 == 0 else 2.5 for k in range(12)])
ax.step(y, hM + 0.06, where="mid", color=COL["M"], lw=1.6, label="M: uniform $\\bar h=\\frac{1}{2}$")
ax.step(y, hL, where="mid", color=COL["L3p"], lw=1.6, label="L3$'$: layers at $\\frac{1}{2}$ and $\\frac{5}{2}$")
pass
ax.annotate("", xy=(6.1, 2.45), xytext=(6.1, 0.55), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
ax.text(6.35, 1.5, "wall\n$\\Delta\\bar h=2$", fontsize=6.5, color="0.35", va="center", ha="left")
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3]); ax.set_yticklabels(["0", "", "1", "", "2", "", "3"]); ax.set_ylim(-0.25, 3.3)
ax.set_xlabel("chessboard row"); ax.set_ylabel("local mean height $\\bar h$")
ax.text(0.2, -0.12, "M: uniform, $\\bar h=\\frac{1}{2}$", fontsize=6.8, color=COL["M"], va="bottom")
ax.text(0.2, 2.62, "L3$'$", fontsize=6.8, color=COL["L3p"], va="bottom")
ax.set_title("(f) local height (schematic)", fontsize=8.5, loc="left", pad=3)
save(fig, "fig4_ladder")
# ======================= Fig. 5: the three deformations =======================
def draw_lat(ax, L, color, lw, z=1, alpha=1.0, mk=True):
    for k, (u, v) in enumerate(L.edges):
        p = L.vpos[u]; q = p + L.mic(L.vpos[v] - p)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="round", zorder=z, alpha=alpha)
    if mk:
        for v, zt in enumerate(L.vtype): ax.plot(*L.vpos[v], "o", ms=4.2 if zt == 4 else 3.0, color=COL["A"] if zt == 4 else COL["C"], zorder=z + 1, alpha=alpha)
cases = [("(a) undeformed", {}, "$p4g$: rotation $R$\nand mirrors $M$"),
         ("(b) anisotropy\n$\\varepsilon=d_1-d_2$", {"eps": 0.08}, "breaks $R$\nA and M respond"),
         ("(c) twist $\\theta$\nof each partner pair", {"theta": np.radians(14)}, "breaks $M$\nC and M respond"),
         ("(d) axial strain $\\eta$", {"eta": 0.22}, "breaks $x\\leftrightarrow y$\nM and L3$'$ respond")]
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.9))
d0 = 0.16; ctr = np.array([1.5, 1.5])
for ax, (title, kw, note) in zip(axs, cases):
    L0 = CairoLattice4(3, 3, d0); L1 = CairoLattice4(3, 3, d0, **kw)
    if "eta" in kw:
        sc = np.array([1 + kw["eta"]/2, 1 - kw["eta"]/2]); L1.vpos = L1.vpos + ctr - ctr*sc       # strain about the window centre
    if kw:
        draw_lat(ax, L0, "0.80", 1.0, z=1, mk=False)
        if "eta" not in kw:
            for v in range(len(L0.vtype)):
                if L0.vtype[v] == 3:
                    p0, p1 = L0.vpos[v], L1.vpos[v]
                    if all(0.45 < q < 2.55 for q in (*p0, *p1)) and np.linalg.norm(p1 - p0) > 0.012:
                        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COL["L3p"], mutation_scale=7), zorder=6)
        else:
            for a, b in (((1.5, 1.5), (2.55, 1.5)), ((1.5, 1.5), (0.45, 1.5))):
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COL["L3p"], mutation_scale=10), zorder=6)
            for a, b in (((1.5, 2.6), (1.5, 2.2)), ((1.5, 0.4), (1.5, 0.8))):
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COL["L3p"], mutation_scale=10), zorder=6)
    draw_lat(ax, L1, INK, 1.2, z=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0.35, 2.65); ax.set_ylim(0.35, 2.65)
    ax.set_title(title, fontsize=8.2, loc="left", pad=3); ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top", fontsize=7)
plt.subplots_adjust(wspace=0.16, bottom=0.2, top=0.82); save(fig, "fig5_deformations")
print("ok")
