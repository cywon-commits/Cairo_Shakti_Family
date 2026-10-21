import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cand_response import CairoLattice4
import real_islands as RI
names = ["A", "C", "M", "W", "L3p"]; cols = {"A": "#f3c38b", "C": "#9ec3ea", "M": "#b9dfa7", "W": "#c9b3e6", "L3": "#e8a0a0", "L3p": "#e8a0a0"}
S = {X: {float(k): v for k, v in json.load(open(f"lattice_sums_{X}.json")).items()} for X in ("A", "M", "W", "L3p")}
def coef(X, d, key):
    if X == "C": return 0.0
    D = sorted(S[X]); return np.interp(d, D, [S[X][x][key] for x in D])
def E(X, d, mu): return coef(X, d, "SS") + mu*coef(X, d, "LS") + mu*mu*coef(X, d, "LL")
dg = np.linspace(0.137, 0.25, 300); mg = np.linspace(0.5, 4.5, 300)
W = np.array([[int(np.argmin([E(X, d, mu) for X in names])) for d in dg] for mu in mg])
fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.pcolormesh(dg, mg, W, cmap=ListedColormap([cols[X] for X in names]), vmin=-0.5, vmax=4.5, shading="auto")
# realistic fixed-gap islands: mu(d)
def area(L): return (L - RI.WID)*RI.WID + np.pi*(RI.WID/2)**2
mud = []
for d in dg[::10]:
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)]); sc = 472/ref.elen[is34].mean()
    mud.append(area(ref.elen[~is34].mean()*sc - 2*22)/area(472 - 2*86))
ax.plot(dg[::10], mud, "k-", lw=2.5, label="realistic islands, fixed gaps (Saccone-type)")
ax.axhline(2.0, color="k", ls="--", lw=1.5, label="ideal needle (mu = 2, extended islands)")
# verification points (realistic, partner gap tuned): W-C sign
pts = [(0.25, 1.33, "C"), (0.25, 1.77, "~"), (0.25, 2.30, "W"), (0.25, 3.01, "W"), (0.20, 1.17, "C"), (0.20, 1.57, "~"), (0.20, 2.04, "W")]
for d, mu, w in pts: ax.plot(d, mu, "o", ms=11, mfc={"C": "#1f5fa8", "W": "#6a3fb0", "~": "white"}[w], mec="k", mew=1.5)
ax.plot([], [], "o", ms=10, mfc="white", mec="k", label="realistic checks (colour = W vs C winner; white = at the boundary)")
for X in names: ax.plot([], [], "s", ms=13, color=cols[X], label=X.replace("L3p", "L3′"))
ax.set_xlabel("d", fontsize=13); ax.set_ylabel("moment ratio  μ = m_long / m_short", fontsize=13)
ax.set_title("Analytic (d, μ) phase diagram from point-dipole lattice sums: E_X − E_C = S_SS + μ S_LS + μ² S_LL", fontsize=12)
ax.legend(fontsize=9.5, loc="upper left"); plt.tight_layout(); plt.savefig("dmu_phase_analytic.png", dpi=110)
# print the boundaries along a few d
for d in (0.15, 0.175, 0.20, 0.25):
    seq = []; prev = None
    for mu in np.linspace(0.5, 4.5, 801):
        w = names[int(np.argmin([E(X, d, mu) for X in names]))]
        if w != prev: seq.append(f"{w}@{mu:.2f}"); prev = w
    print(f"d={d}: " + " -> ".join(seq))
