import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from cairo_asi import CairoLattice, D_EQ
plt.rcParams.update({"font.size": 15})
INK, BLUE, ORANGE, GREY = "#1f2a36", "#2f6db5", "#e08a1e", "#9aa3ad"
# 1. family geometry
fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
for ax, (d, name) in zip(axs, [(0.03, "d → 0\nsquare ice"), (D_EQ, "d = 0.137\nequilateral Cairo"), (0.25, "d = 1/4\nShakti"), (0.40, "d = 0.40\nconcave")]):
    L = CairoLattice(3, 3, d=d)
    for k, (u, v) in enumerate(L.edges):
        p = L.vpos[u]; q = p + L.mic(L.vpos[v] - p)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=2.4, solid_capstyle="round")
    for v, z in enumerate(L.vtype):
        ax.plot(*L.vpos[v], "o", ms=9 if z == 4 else 6, color=ORANGE if z == 4 else BLUE, zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0.2, 2.8); ax.set_ylim(0.2, 2.8); ax.set_title(name, fontsize=17)
h1, = axs[0].plot([], [], "o", ms=9, color=ORANGE); h2, = axs[0].plot([], [], "o", ms=6, color=BLUE)
fig.legend([h1, h2], ["z = 4 vertex", "z = 3 vertex"], loc="lower center", ncol=2, fontsize=15, frameon=False)
plt.subplots_adjust(top=0.84, bottom=0.12, left=0.01, right=0.99, wspace=0.05); plt.savefig("deckfig/family.png", dpi=110, transparent=False, facecolor="white")
# 2. height clock: four basketweave placements on the chessboard
def bw(ax, ox, oy, colr, title):
    for x in range(8):
        for y in range(8): ax.add_patch(Rectangle((x, y), 1, 1, fc="#eeeeee" if (x + y) % 2 else "#d4d4d4", ec="none"))
    for bx in range(-1, 5):
        for by in range(-1, 5):
            x0, y0 = 2*bx + ox, 2*by + oy
            if not (0 <= x0 and x0 + 2 <= 8 and 0 <= y0 and y0 + 2 <= 8): continue
            ax.add_patch(Rectangle((x0 + .04, y0 + .04), 1.92, 1.92, fc=colr, alpha=.28, ec=colr, lw=2))
            if (bx + by) % 2 == 0:
                for yy in (y0, y0 + 1): ax.add_patch(Rectangle((x0 + .1, yy + .1), 1.8, .8, fc="white", ec=INK, lw=1.4))
            else:
                for xx in (x0, x0 + 1): ax.add_patch(Rectangle((xx + .1, y0 + .1), .8, 1.8, fc="white", ec=INK, lw=1.4))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=17, color=colr)
fig = plt.figure(figsize=(12, 12))
pos = {0: (0.36, 0.66), 1: (0.66, 0.36), 2: (0.36, 0.06), 3: (0.06, 0.36)}
lab = {0: ((0, 0), BLUE, "h = 0 : C (z4-B centred)"), 1: ((1, 0), ORANGE, "h = 1 : A (partner bond 1)"),
       2: ((1, 1), BLUE, "h = 2 : C (z4-A centred)"), 3: ((0, 1), ORANGE, "h = 3 : A (partner bond 2)")}
for h, (x, y) in pos.items():
    ax = fig.add_axes([x, y, 0.28, 0.28]); (ox, oy), c, t = lab[h]; bw(ax, ox, oy, c, t)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for a, b in [((0.62, 0.82), (0.8, 0.66)), ((0.8, 0.34), (0.62, 0.2)), ((0.36, 0.2), (0.2, 0.34)), ((0.2, 0.66), (0.36, 0.82))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=30, lw=2.5, color=INK, connectionstyle="arc3,rad=-0.25"))
ax.text(0.5, 0.5, "height plateau\nh mod 4\n\nΦ = cos πh\nψ_C = cos(πh/2)\nψ_A = sin(πh/2)", ha="center", va="center", fontsize=20, color=INK)
plt.savefig("deckfig/height_clock.png", dpi=100, facecolor="white")
# 3. Saccone: T_A vs T_eff
fig, ax = plt.subplots(figsize=(10, 6))
c = np.array([376, 450, 500]); TA = [322, 432, 562]; Teff = [2100, 1330, 1365]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, TA, w, color=BLUE, label="T_A  (ordering, manifold MC)")
ax.bar(x + w/2, Teff, w, color=ORANGE, label="T_eff  (from measured z4 vertex populations)")
ax.axhline(130, color=INK, ls="--", lw=1.5); ax.text(2.45, 150, "T_B = 130 K", ha="right", fontsize=14)
ax.axhline(295, color=GREY, ls=":", lw=1.5); ax.text(2.45, 315, "room temperature", ha="right", fontsize=14, color="#5b636b")
for i in range(3): ax.text(i, max(TA[i], Teff[i]) + 60, f"T_A/T_eff = {TA[i]/Teff[i]:.2f}", ha="center", fontsize=14)
ax.set_xticks(x); ax.set_xticklabels([f"c = {v} nm" for v in c]); ax.set_ylabel("temperature (K)"); ax.set_ylim(0, 2500)
ax.legend(frameon=False, fontsize=13, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("deckfig/saccone_T.png", dpi=110, facecolor="white")
