import numpy as np, itertools, json, sys, time
import needle
needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling, subdipoles
from cairo_asi import CairoLattice, D_EQ
from manifold_select import setup, enum_manifold
nx, ny = 4, 2
L0 = CairoLattice(nx, ny, d=D_EQ); Jl0 = needle_coupling(L0, None)
ze, zc, part = setup(L0, Jl0)
_, S = enum_manifold(Jl0, ze, zc, part, 200000); S = S.astype(float); S = np.vstack([S, -S])
ds = [float(x) for x in sys.argv[1:]]
for d in ds:
    t0 = time.time()
    L = CairoLattice(nx, ny, d=d); Jd = needle_coupling(L, 6.0); Jl = needle_coupling(L, None)
    v3 = np.where(L.vtype == 3)[0][0]; ks = [k for k, _ in L.incidence[v3]]
    lv = sorted(set(np.round([sum(Jl[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(3), 2))
                             for c in itertools.product([-1, 1], repeat=3)], 3)))
    Ed = 0.5*np.einsum('ij,jk,ik->i', S, Jd, S)/L.N
    Emin = Ed.min(); idx = np.where(Ed < Emin + 1e-7)[0]
    second = Ed[Ed > Emin + 1e-7].min()
    np.save(f"scan3_min_d{d:.4f}.npy", S[idx])
    print(f"d={d:.4f}: z3 gap {lv[1]-lv[0]:.3f}; min E/N {Emin:.5f}; #minimisers {len(idx)}; gap to next {second-Emin:.5f}  [{time.time()-t0:.0f}s]", flush=True)
