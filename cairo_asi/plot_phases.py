import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from collections import Counter
from cairo_asi import CairoLattice, D_EQ
from dimer_map import DimerMap
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling

def tile(s0, cx, cy, n):
    s = np.empty(n*n*10)
    for i in range(n):
        for j in range(n):
            for b in range(10): s[(i*n+j)*10+b] = s0[((i % cx)*cy + (j % cy))*10+b]
    return s
cases = [("A (Cairo side)", 0.1372, "scan3_min_d0.1372.npy"), ("B (intermediate)", 0.1531, "scan3_min_d0.1531.npy"),
         ("C (Shakti side)", 0.2500, "scan3_min_d0.2500.npy")]
if __name__ == '__main__':
    n = 8; fig, axs = plt.subplots(1, 3, figsize=(18, 6.3))
    cmap, norm = plt.get_cmap("bwr_r"), TwoSlopeNorm(0, -1, 1)
    for ax, (name, d, f) in zip(axs, cases):
        s = tile(np.load(f)[0], 4, 2, n)
        L = CairoLattice(n, n, d=d); D = DimerMap(L, needle_coupling(L, None)); chi = L.chirality(s); dm = D.dimers(s)
        assert D.is_perfect(dm)
        for pts, c in zip(L.face_pts, chi): ax.add_patch(Polygon(pts, fc=cmap(norm(c)), ec="0.45", lw=0.4, alpha=0.8))
        for k in dm:
            f1, f2 = D.e2f[k]; p1 = L.face_center[f1]; p2 = p1 + L.mic(L.face_center[f2]-p1)
            ax.plot(*np.array([p1, p2]).T, color="gold", lw=5, solid_capstyle="round"); ax.plot(*np.array([p1, p2]).T, "k", lw=1)
        # dimer pair classes
        cls = Counter()
        for k in dm:
            f1, f2 = D.e2f[k]; a, b = sorted([abs(chi[f1]), abs(chi[f2])])
            cls[(round(a, 1), round(b, 1), "same" if np.sign(chi[f1]) == np.sign(chi[f2]) else "opp")] += 1
        ax.set_xlim(0.3, 4.3); ax.set_ylim(0.3, 4.3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}, d = {d}", fontsize=12)
        print(name, "dimer (|chi|,|chi|,sign) classes:", dict(cls), " |chi| histogram:", dict(Counter(np.round(np.abs(chi), 1))))
    plt.tight_layout(); plt.savefig("phases_ABC.png", dpi=120)
    