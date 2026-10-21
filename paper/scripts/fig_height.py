import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts"); sys.path.insert(0, "/mnt/user-data/outputs/cairo_asi")
import numpy as np, json, glob
from figstyle import *
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
# ---------- Fig. 3: height plateaus ----------
def bw(ax, ox, oy, colr):
    for x in range(8):
        for y in range(8): ax.add_patch(Rectangle((x, y), 1, 1, fc="#EDEDED" if (x + y) % 2 else "#D2D2D2", ec="none"))
    for bx in range(-1, 5):
        for by in range(-1, 5):
            x0, y0 = 2*bx + ox, 2*by + oy
            if not (0 <= x0 and x0 + 2 <= 8 and 0 <= y0 and y0 + 2 <= 8): continue
            ax.add_patch(Rectangle((x0 + .05, y0 + .05), 1.9, 1.9, fc=colr, alpha=0.25, ec=colr, lw=1.2))
            if (bx + by) % 2 == 0:
                for yy in (y0, y0 + 1): ax.add_patch(Rectangle((x0 + .12, yy + .12), 1.76, .76, fc="white", ec=INK, lw=0.7))
            else:
                for xx in (x0, x0 + 1): ax.add_patch(Rectangle((xx + .12, y0 + .12), .76, 1.76, fc="white", ec=INK, lw=0.7))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
fig = plt.figure(figsize=(7.0, 4.6)); gs = gridspec.GridSpec(2, 4, height_ratios=[1, 0.9], hspace=0.55, wspace=0.18)
for h, (ox, oy, ph, lab) in enumerate([(0, 0, "C", "a"), (1, 0, "A", "b"), (1, 1, "C", "c"), (0, 1, "A", "d")]):
    ax = fig.add_subplot(gs[0, h]); bw(ax, ox, oy, COL[ph])
    ax.set_title(f"({lab}) $\\bar h \\equiv {h}$: {ph}\n" + ("blocks on $z{=}4$ sites" if ph == "C" else "blocks on bonds"), fontsize=8, pad=3, loc="left", color=COL[ph])
ax = fig.add_subplot(gs[1, :]); hh = np.linspace(-0.25, 4.25, 600)
for delta, ls, lab in ((+0.35, "-", "$\\delta>0$ (Shakti side): even plateaus = C"), (-0.35, "--", "$\\delta<0$ (Cairo side): odd plateaus = A")):
    V = -np.cos(2*np.pi*hh) - delta*np.cos(np.pi*hh); ax.plot(hh, V, ls, color=INK, lw=1.3, label=lab)
for h0 in range(5):
    ph = "C" if h0 % 2 == 0 else "A"; ax.axvline(h0, color=COL[ph], lw=0.8, alpha=0.6)
    ax.text(h0, 1.68, ph if h0 < 4 else "C", ha="center", va="bottom", fontsize=8, color=COL[ph], fontweight="bold")
for h0 in (0.5, 1.5, 2.5, 3.5): ax.axvline(h0, color=COL["M"], lw=0.8, ls=":", alpha=0.9)
ax.text(1.5, 1.68, "M", ha="center", va="bottom", fontsize=8, color=COL["M"], fontweight="bold")
ax.set_xlim(-0.25, 4.25); ax.set_ylim(-1.6, 1.62); ax.set_xticks(range(5)); ax.set_xlabel("coarse-grained height $\\bar h$ (mod 4)")
ax.set_ylabel("$V(\\bar h)$ (schematic)"); ax.set_yticks([]); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False, fontsize=7.5)
ax.set_title("(e) locking potential  $V=-u\\cos 2\\pi\\bar h-\\delta\\cos\\pi\\bar h$: integer heights = basketweave, half-integer = columnar", fontsize=8, loc="left", pad=14)
save(fig, "fig3_height")
print("ok")
