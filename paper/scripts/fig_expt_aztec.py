import sys, os; from pathlib import Path; _H = Path(__file__).resolve().parent; _CODE = _H.parents[1] / "cairo_asi"
sys.path.insert(0, str(_H)); sys.path.insert(0, str(_CODE)); os.chdir(_CODE)   # data files are read relative to cairo_asi/
import numpy as np
from figstyle import *
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
# ---------- Fig. 9: Saccone et al. samples ----------
fig, ax = plt.subplots(figsize=(3.6, 2.6))
c = ["376", "450", "500"]; TA = [322, 432, 562]; Teff = [2100, 1330, 1365]; x = np.arange(3); w = 0.36
ax.bar(x - w/2, TA, w, color=COL["A"], label="$T_A$ (equilibrium A ordering)")
ax.bar(x + w/2, Teff, w, color="0.55", label="$T_{\\rm eff}$ from z4 vertex populations")
ax.axhline(130, color=INK, ls="--", lw=1.0, label="blocking temperature $T_B=130$ K")
for i in range(3): ax.text(x[i], max(TA[i], Teff[i]) + 70, f"$T_A/T_{{\\rm eff}}={TA[i]/Teff[i]:.2f}$", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"$c={v}$ nm" for v in c]); ax.set_ylabel("temperature (K)"); ax.set_ylim(0, 3300); ax.set_xlim(-0.6, 2.6)
ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95)
save(fig, "fig9_saccone")
# ---------- Fig. 10: small Aztec diamonds ----------
fig = plt.figure(figsize=(7.0, 3.6)); gs = gridspec.GridSpec(2, 4, height_ratios=[1.35, 1], hspace=0.45, wspace=0.18)
for col, n in enumerate([6, 7, 8, 9]):
    ax = fig.add_subplot(gs[0, col]); part = np.load(f"aztecS_gs_n{n}_d+0.000_s5.npy"); S = part.shape[0]; off = n + 1
    inside = np.array([[abs(a - off + 0.5) + abs(b - off + 0.5) <= n for b in range(S)] for a in range(S)])
    for a in range(S):
        for b in range(S):
            if inside[a, b]: ax.add_patch(Rectangle((a - off, b - off), 1, 1, fc="#EDEDED" if (a + b) % 2 else "#D2D2D2", ec="none"))
    for a in range(S):
        for b in range(S):
            if part[a, b] == 0: ax.add_patch(Rectangle((a - off + .1, b - off + .1), 1.8, .8, fc="white", ec=INK, lw=0.5))
            elif part[a, b] == 2: ax.add_patch(Rectangle((a - off + .1, b - off + .1), .8, 1.8, fc="white", ec=INK, lw=0.5))
    ne = no = 0
    for a in range(S - 1):
        for b in range(S - 1):
            if not all(inside[a + i, b + j] for i in (0, 1) for j in (0, 1)): continue
            if (part[a, b] == 0 and part[a, b+1] == 0) or (part[a, b] == 2 and part[a+1, b] == 2):
                ev = ((a - off) + (b - off)) % 2 == 0; ne += ev; no += (not ev); cc = COL["C"] if ev else COL["A"]
                ax.add_patch(Rectangle((a - off + .03, b - off + .03), 1.94, 1.94, fc=cc, alpha=0.35, ec=cc, lw=0.9))
    ax.plot(0, 0, "k+", ms=7, mew=1.2); ax.set_xlim(-n - 1.2, n + 1.2); ax.set_ylim(-n - 1.2, n + 1.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"({'abcd'[col]}) $n={n}$: " + ("C (even)" if ne > no else "A (odd)"), fontsize=8.5, loc="left", pad=2, color=COL["C"] if ne > no else COL["A"])
ax = fig.add_subplot(gs[1, :2]); nn = np.array([6, 7, 8, 9]); ds = np.array([0.250, 0.200, 0.167, 0.167])
ax.plot(nn, ds, "o", ms=6, mec="k", mew=0.6, color="0.3", zorder=3); ax.plot([n_ for n_ in nn], [ds[i] for i in range(4)], lw=0)
for n_, v in zip(nn, ds): ax.plot(n_, v, "o", ms=6, mec="k", mew=0.6, color=COL["A"] if n_ % 2 == 0 else COL["C"], zorder=4)
xx = np.linspace(5.6, 10, 100); ax.plot(xx, 1.4/xx, "k--", lw=1.0, label="$1.4/n$")
ax.set_xlabel("Aztec order $n$"); ax.set_ylabel("$|\\delta^*(n)|$"); ax.set_xlim(5.6, 10); ax.set_ylim(0.12, 0.27); ax.legend(loc="upper right", fontsize=7.5, frameon=False)
ax.set_title("(e) boundary-induced shift of the A$|$C point", fontsize=8.5, loc="left", pad=3)
ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
ax.text(0.0, 0.92, "(f) boundary pins the plateau", fontsize=8.5, transform=ax.transAxes, va="top")
rows = [("$n$ mod 4", "pinned $\\bar h$", "crystal"), ("0", "$\\pm 1$", "A (two variants)"), ("1", "2", "C, blocks (1,1)"), ("2", "$\\pm 1$", "A (two variants)"), ("3", "0", "C, blocks (0,0)")]
for r, row in enumerate(rows):
    for cidx, txt in enumerate(row):
        ax.text([0.02, 0.30, 0.58][cidx], 0.72 - 0.15*r, txt, fontsize=7.8, transform=ax.transAxes, va="center", fontweight="bold" if r == 0 else "normal")
ax.plot([0.0, 1.0], [0.655, 0.655], transform=ax.transAxes, color="0.5", lw=0.6)
save(fig, "fig10_aztec")
print("ok")
