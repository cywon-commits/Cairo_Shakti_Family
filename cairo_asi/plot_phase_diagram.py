import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cairo_asi import D_EQ
fig, ax = plt.subplots(figsize=(10, 6.2))
Tmin, Tmax = 1e-3, 3
# measured points (8x8, manifold MC / PT, rc = 64a)
TA = [(0.100, 1.80), (0.120, 0.95), (0.137, 0.39), (0.145, 0.115)]            # |psi_A| = 0.5
TC = [(0.145, 0.51), (0.150, 0.54), (0.155, 0.54), (0.160, 0.58), (0.250, 0.32)]  # |psi_C| = 0.5 (local C structure)
TS = [(0.250, 0.03)]                                                              # stacking order (walls -> 0)
TSup = [(0.180, 0.003), (0.200, 0.004)]                                           # upper bounds on stacking order of W
# regions (schematic, from the points)
dA = np.array([0.100, 0.120, 0.137, 0.145, 0.1527]); tA = np.array([1.80, 0.95, 0.39, 0.115, 1e-3])
ax.fill_between(dA, Tmin, tA, color="#E24B4A", alpha=0.25, lw=0)
dC = np.array([0.145, 0.150, 0.155, 0.160, 0.25, 0.34]); tC = np.array([0.51, 0.54, 0.54, 0.58, 0.32, 0.32])
ax.fill_between(dC, Tmin, tC, where=None, color="#378ADD", alpha=0.18, lw=0)
ax.fill_between([0.17, 0.21], Tmin, 0.004, color="#1D9E75", alpha=0.45, lw=0)
ax.fill_between([0.215, 0.34], Tmin, 0.03, color="#185FA5", alpha=0.35, lw=0)
ax.plot(*zip(*TA), "o-", color="#A32D2D", label="A order  (|ψ_A| = 0.5)")
ax.plot(*zip(*TC), "s--", color="#185FA5", label="local C structure  (|ψ_C| = 0.5)")
ax.plot(*zip(*TS), "D", color="k", ms=8, label="stacking crossover, 8x8 (finite-size, ~2σL/ln L)")
ax.errorbar(*zip(*TSup), yerr=[[0.001, 0.0013], [0, 0]], uplims=True, fmt="v", color="#0F6E56", ms=8, label="stacking crossover of W, 8x8: upper bound")
ax.axvline(0.1527, color="k", ls=":", lw=1.2); ax.text(0.1535, 1.8e-3, "A|C at T=0\n(d≈0.153)", fontsize=8)
for x, t in [(D_EQ, "equilateral Cairo"), (0.25, "Shakti")]:
    ax.axvline(x, color="0.4", lw=0.8); ax.text(x, 3.3, t, ha="center", va="bottom", fontsize=9)
ax.text(0.11, 0.02, "A", fontsize=18, color="#A32D2D", ha="center")
ax.text(0.19, 0.0018, "W", fontsize=14, color="#0F6E56", ha="center")
ax.text(0.28, 0.008, "C", fontsize=16, color="#0C447C", ha="center")
ax.text(0.26, 0.09, "local C; rigid straight walls\n(stacking frozen kinetically)", fontsize=9, color="#185FA5", ha="center")
ax.text(0.215, 1.5, "dimer liquid (manifold, disordered)", fontsize=9, color="0.3", ha="center")
ax.text(0.147, 0.25, "re-entrant\nC-like", fontsize=7, color="#185FA5", ha="center")
ax.set_yscale("log"); ax.set_ylim(Tmin, Tmax); ax.set_xlim(0.09, 0.34)
ax.set_xlabel("d   (Cairo ← → Shakti)"); ax.set_ylabel("T   (D/a³)")
ax.set_title("DRAFT (d, T) phase diagram inside the dimer manifold — 8x8, needle model, rc = 64a\n"
             "points = measured; shaded = schematic. Stacking lines are finite-size crossovers, not intrinsic phase boundaries", fontsize=10, pad=28)
ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.36))
plt.tight_layout(); plt.savefig("phase_diagram_draft.png", dpi=130)
