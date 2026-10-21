import numpy as np
from scipy.stats import spearmanr
from cairo_asi import CairoLattice, D_EQ
from needle import needle_coupling
from manifold_select import setup, enum_manifold
from dimer_map import DimerMap

nx, ny = 4, 2
data = {}
for d in (D_EQ, 0.25):
    L = CairoLattice(nx, ny, d=d); Jl = needle_coupling(L, None); Jd = needle_coupling(L, 6.0)
    ze, zc, part = setup(L, Jl)
    _, S = enum_manifold(Jl, ze, zc, part, 200000); S = S.astype(float)
    S = np.vstack([S, -S])
    Ed = 0.5*np.einsum('ij,jk,ik->i', S, Jd, S)/L.N
    data[d] = {tuple(s.astype(int)): e for s, e in zip(S, Ed)}
kC, kS = set(data[D_EQ]), set(data[0.25])
print("same manifold (as spin sets) at both ends:", kC == kS, len(kC))
keys = sorted(kC)
eC = np.array([data[D_EQ][k] for k in keys]); eS = np.array([data[0.25][k] for k in keys])
print("Spearman rank correlation of dipolar energies over the manifold:", round(spearmanr(eC, eS).correlation, 4))
minC = {k for k in keys if data[D_EQ][k] < eC.min() + 1e-7}
minS = {k for k in keys if data[0.25][k] < eS.min() + 1e-7}
print("# minimisers: Cairo", len(minC), " Shakti", len(minS), " identical sets:", minC == minS)
kc = next(iter(minC)); ks = next(iter(minS))
print("Cairo's choice evaluated at Shakti geometry: dE/N above Shakti min =", round(data[0.25][kc] - eS.min(), 5))
print("Shakti's choice evaluated at Cairo geometry: dE/N above Cairo min  =", round(data[D_EQ][ks] - eC.min(), 5))
# ranks
order_S = np.argsort(eS); rank_S = {keys[i]: r for r, i in enumerate(order_S)}
print("rank of Cairo-selected state in Shakti ordering:", rank_S[kc], "of", len(keys))
np.save("sel_cairo_4x2.npy", np.array(kc, float)); np.save("sel_shakti_4x2.npy", np.array(ks, float))
