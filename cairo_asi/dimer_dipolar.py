import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from collections import Counter
from cairo_asi import *
from dimer_map import DimerMap

L = CairoLattice(8, 8); D = DimerMap(L); Jl = D.J
E0_local = -3.74497
res = {}
for name, f in [("dipolar 2x2 GS candidate", "gs_cell2x2.npy"), ("best SA (dipolar)", "sa_best_dip6_8.npy"),
                ("vertex-local GS (tiled 2x1)", None)]:
    if f is None:
        Ls = CairoLattice(2, 1); _, s0 = exact_ground_state_fast(Ls.coupling_matrix(None))
        s = np.array([s0[(i % 2)*10 + b] for i in range(8) for j in range(8) for b in range(10)])
    else:
        s = np.load(f)
    dm = D.dimers(s); chi = L.chirality(s)
    # dimer orientation: vector between pentagon centres
    ang = []
    pairs = Counter()
    for k in dm:
        f1, f2 = D.e2f[k]
        v = L.mic(L.face_center[f2] - L.face_center[f1]); a = np.degrees(np.arctan2(v[1], v[0])) % 180
        ang.append(round(a, 0))
        c1, c2 = sorted([abs(chi[f1]), abs(chi[f2])])
        pairs[(round(c1, 1), round(c2, 1), "same" if np.sign(chi[f1]) == np.sign(chi[f2]) else "opp")] += 1
    unhappy = sum(D.level(v, s) for v in np.where(L.vtype == 3)[0])
    print(f"[{name}] E_local/N={0.5*s@Jl@s/L.N:.5f} (manifold {E0_local}); unhappy z3 {unhappy}/{(L.vtype==3).sum()}; "
          f"dimers {len(dm)}, perfect {D.is_perfect(dm)}")
    print("   dimer orientations (deg):", dict(Counter(ang)))
    print("   dimer (|chi1|,|chi2|,sign):", dict(pairs))
    res[name] = (s, dm)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
cmap, norm = plt.get_cmap("bwr_r"), TwoSlopeNorm(0, -1, 1)
for ax, name in zip(axs, ["dipolar 2x2 GS candidate", "best SA (dipolar)"]):
    s, dm = res[name]; chi = L.chirality(s)
    for pts, c in zip(L.face_pts, chi):
        ax.add_patch(Polygon(pts, fc=cmap(norm(c)), ec="0.4", lw=0.4, alpha=0.8))
    for k in dm:
        f1, f2 = D.e2f[k]; p1 = L.face_center[f1]; p2 = p1 + L.mic(L.face_center[f2] - p1)
        ax.plot(*np.array([p1, p2]).T, color="gold", lw=5, solid_capstyle="round")
        ax.plot(*np.array([p1, p2]).T, color="k", lw=1)
    unh = [v for v in np.where(L.vtype == 3)[0] if D.level(v, s) == 1]
    ax.scatter(*L.vpos[unh].T, s=25, c="lime", edgecolors="k", zorder=5, label="unhappy z=3")
    ax.set_xlim(0.3, 5.3); ax.set_ylim(0.3, 5.3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, fontsize=11); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig("cairo_dimers.png", dpi=140)
