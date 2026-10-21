"""Cell-resolved local plateau map of the d = 0.150 low-T state: each Cairo cell has one plaquette of each kind."""
import numpy as np, collections, sys
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from block_order import BlockOrder
from plot_phases import tile
n, d = 12, 0.150
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); D = DimerMap(L, Jl); B = BlockOrder(L, D)
PLAT = {"z4B": 0, "p1": 1, "z4A": 2, "p2": 3}
# assign every chessboard plaquette to the Cairo cell of its centre (faces' cell index = face vertex cell)
cellof = {}
for cyc, org, kind, _ in B.plaq:
    c = np.mean([L.face_center[f] for f in cyc], axis=0)
    cellof[(org, kind)] = (int(np.floor(c[0] + 1e-9)) % n, int(np.floor(c[1] + 1e-9)) % n)
def cmap(s):
    grid = [[[] for _ in range(n)] for _ in range(n)]
    for o, k in B.flippable(s):
        i, j = cellof[(o, k)]; grid[i][j].append(PLAT[k])
    return grid
def show(s, name):
    g = cmap(s); print(name)
    for j in range(n - 1, -1, -1):
        row = ""
        for i in range(n):
            v = g[i][j]; row += (".", str(v[0]) if len(v) == 1 else "*")[min(len(v), 2) if len(v) <= 1 else 1] if len(v) <= 1 else "*"
            row += " "
        print("   " + row)
    cnt = collections.Counter(len(g[i][j]) for i in range(n) for j in range(n)); print("   blocks per cell:", dict(cnt))
    return g
s0 = np.load("fss/lowT_state_d0.1500_n12.npy")
g = show(s0, "lowest state (e - e_A = -4.4e-4): local plateau per Cairo cell (columns = x, rows = y)")
show(tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n), "A crystal reference")
