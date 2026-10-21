import numpy as np, collections, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from chessboard import draw, chess_coords
n = 12; L = CairoLattice(n, n, d=0.150); D = DimerMap(L, local_needle_coupling(L))
s0 = np.load("fss/lowT_state_d0.1500_n12.npy")
pos, conf = chess_coords(L, D); dm = D.dimers(s0)
# orientation statistics and parallel-domino ladders on the chessboard
dims = {}
for k in dm:
    f1, f2 = D.e2f[k]; c1, c2 = pos[f1], pos[f2]
    if abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) == 1: dims[frozenset([c1, c2])] = "H" if c1[1] == c2[1] else "V"
H = sum(v == "H" for v in dims.values()); V = len(dims) - H
# ladder length: maximal run of parallel dominoes stacked on top of each other (H dominoes stacked in y, V in x)
def run_len(dom):
    a, b = sorted(dom); o = dims[dom]; L_ = 1
    step = (0, 1) if o == "H" else (1, 0)
    for sgn in (1, -1):
        cur = (a, b)
        while True:
            nxt = frozenset([(cur[0][0] + sgn*step[0], cur[0][1] + sgn*step[1]), (cur[1][0] + sgn*step[0], cur[1][1] + sgn*step[1])])
            if nxt in dims and dims[nxt] == o: L_ += 1; cur = tuple(sorted(nxt))
            else: break
    return L_
runs = collections.Counter(min(run_len(dd), 12) for dd in dims)
print(f"coordinate conflicts {conf}; dominoes H {H}, V {V}; distribution of parallel-ladder lengths (domino count, capped at 12): {dict(sorted(runs.items()))}")
fig, ax = plt.subplots(figsize=(7.5, 7.8))
draw(ax, L, D, s0, "d = 0.150 lowest state (e - e_A = -4.4e-4) on the pentagon chessboard")
plt.tight_layout(); plt.savefig("superstructure_d0150_chessboard.png", dpi=110)
