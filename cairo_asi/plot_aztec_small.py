import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
EVEN, ODD = "#1D9E75", "#EF9F27"
fig = plt.figure(figsize=(19, 9.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.28, wspace=0.12)
for col, n in enumerate([6, 7, 8, 9]):
    ax = fig.add_subplot(gs[0, col]); part = np.load(f"aztecS_gs_n{n}_d+0.000_s5.npy"); S = part.shape[0]; off = n + 1
    inside = np.array([[abs(a - off + 0.5) + abs(b - off + 0.5) <= n for b in range(S)] for a in range(S)])
    for a in range(S):
        for b in range(S):
            if inside[a, b]:
                x, y = a - off, b - off
                ax.add_patch(Rectangle((x, y), 1, 1, fc="#e6e6e6" if (x + y) % 2 else "#c8c8c8", ec="none"))
    for a in range(S):
        for b in range(S):
            if part[a, b] == 0:   ax.add_patch(Rectangle((a - off + .08, b - off + .08), 1.84, .84, fc="white", ec="k", lw=1.1))
            elif part[a, b] == 2: ax.add_patch(Rectangle((a - off + .08, b - off + .08), .84, 1.84, fc="white", ec="k", lw=1.1))
    ne = no = 0
    for a in range(S - 1):
        for b in range(S - 1):
            if not (inside[a, b] and inside[a+1, b] and inside[a, b+1] and inside[a+1, b+1]): continue
            flip = (part[a, b] == 0 and part[a, b+1] == 0) or (part[a, b] == 2 and part[a+1, b] == 2)
            if flip:
                ev = ((a - off) + (b - off)) % 2 == 0; ne += ev; no += (not ev)
                ax.add_patch(Rectangle((a - off + .02, b - off + .02), 1.96, 1.96, fc=EVEN if ev else ODD, alpha=0.45, ec=EVEN if ev else ODD, lw=2.2))
    ax.plot(0, 0, "k+", ms=12, mew=2)                                   # centre vertex (the central plaquette)
    ax.set_xlim(-n - 1.5, n + 1.5); ax.set_ylim(-n - 1.5, n + 1.5); ax.set_aspect("equal"); ax.axis("off")
    pref = "even (= centre class)" if ne > no else "odd"
    ax.set_title(f"n = {n}: {ne if ne > no else no} blocks, all {pref}\n⌊(n+1)²/4⌋ = {(n+1)**2//4}", fontsize=12)
# delta*(n) panel
ax = fig.add_subplot(gs[1, 0:2])
nn = np.array([6, 7, 8, 9]); dsign = np.array([0.250, 0.200, 0.167, 0.167]); dpure = np.array([0.143, 0.143, 0.111, 0.111])
cols = [ODD if k % 2 == 0 else EVEN for k in nn]
ax.scatter(nn, dsign, s=110, c=cols, edgecolors="k", zorder=3, label="Φ_GS changes sign")
ax.scatter(nn, dpure, s=90, c=cols, marker="s", edgecolors="k", zorder=3, label="pure preferred crystal lost")
x = np.linspace(5.5, 12, 100); ax.plot(x, 1.4/x, "k--", lw=1, label="1.4 / n")
ax.set_xlabel("Aztec order n"); ax.set_ylabel("|δ| needed to overturn the boundary preference")
ax.set_title("boundary-induced shift of the A|C transition (upper bounds)", fontsize=11); ax.legend(fontsize=9); ax.set_xlim(5.5, 10)
ax.text(0.03, 0.06, "colour = class preferred at δ = 0 (orange = odd for even n, green = even for odd n)", fontsize=9, transform=ax.transAxes)
# schematic: signed preference on a delta axis
ax = fig.add_subplot(gs[1, 2:4])
for k, n in enumerate([6, 7, 8, 9]):
    sgn = -1 if n % 2 == 0 else 1; d = dsign[k]
    ax.barh(k, sgn*d, color=EVEN if sgn > 0 else ODD, edgecolor="k", height=0.55)
    ax.text(sgn*d + 0.01*sgn, k, f"n = {n}", va="center", ha="left" if sgn > 0 else "right", fontsize=10)
ax.axvline(0, color="k", lw=1.2); ax.set_yticks([]); ax.set_xlim(-0.32, 0.32)
ax.set_xlabel("effective transition point δ*(n)  (bulk: δ* = 0)")
ax.set_title("finite diamond: effective A|C point alternates with the parity of n", fontsize=11)
ax.text(0.02, 0.02, "bar = |δ| up to which the boundary-preferred class keeps winning", fontsize=9, transform=ax.transAxes)
fig.suptitle("Small Aztec diamonds at the bulk A|C point (δ = 0): ground-state basketweave and its class\n"
             "green = block on even plaquette (class of the central plaquette), orange = odd; + = centre", fontsize=13)
plt.savefig("aztec_small_boundary.png", dpi=115, bbox_inches="tight")
