import numpy as np, itertools
from collections import Counter
from cairo_asi import *
from shakti import ShaktiLattice
from count_degeneracy import enumerate_gs
from dimer_map import DimerMap
from compare_shakti import setup_from, levels, perfect_matchings

S = ShaktiLattice(1, 1); JS = S.coupling_matrix(None)
print("Shakti vertex levels (end-segment dipoles):")
for z in (3, 4):
    lv, dg = levels(S, JS, np.where(S.vtype == z)[0][0]); print(f"  z={z}: {np.round(lv,3)} deg {dg}")
for nx, ny in [(2,2),(3,2)]:
    S = ShaktiLattice(nx, ny); J, ze, zc, part, P = setup_from(S)
    E, W = enumerate_gs(S.N, ze, zc, P)
    nunh = (S.vtype == 3).sum() // 2
    line = f"Shakti {nx}x{ny}: N={S.N}, W={W}, ln W / unhappy z3 = {np.log(W)/nunh:.4f}"
    if nx*ny <= 4:
        pm, _ = perfect_matchings(S); line += f", face-graph perfect matchings (all sectors) = {pm}"
    print(line, flush=True)
