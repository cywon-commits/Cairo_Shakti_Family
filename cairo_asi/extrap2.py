import numpy as np, itertools
from collections import Counter
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice, D_EQ, anneal, quench
from manifold_select import setup, enum_manifold
ref = {}
for lab, f in [("A", "scan3_min_d0.1372.npy"), ("B", "scan3_min_d0.1531.npy"), ("C", "scan3_min_d0.2500.npy")]:
    ref[frozenset(tuple(r.astype(int)) for r in np.load(f))] = lab
L0 = CairoLattice(4, 2, d=D_EQ); Jl0 = needle_coupling(L0, None); ze0, zc0, p0 = setup(L0, Jl0)
_, S0 = enum_manifold(Jl0, ze0, zc0, p0, 200000); S0 = S0.astype(float); S0 = np.vstack([S0, -S0])
print("--- large d: phase labels ---")
for d in np.round(np.arange(0.330, 0.3601, 0.003), 3).tolist() + [0.40, 0.45, 0.48]:
    L = CairoLattice(4, 2, d=d); Jd = needle_coupling(L, 6.0)
    Ed = 0.5*np.einsum('ij,jk,ik->i', S0, Jd, S0)/L.N; Emin = Ed.min()
    mins = frozenset(tuple(S0[i].astype(int)) for i in np.where(Ed < Emin + 1e-7)[0])
    if mins not in ref: ref[mins] = chr(65 + len(ref)); np.save(f"phase_{ref[mins]}_d{d:.3f}.npy", np.array(sorted(mins), float))
    gap = Ed[Ed > Emin + 1e-7].min() - Emin
    print(f"  d={d:.3f}: phase {ref[mins]}, gap to next {gap:.5f}")
print("--- small d: what does the SA ground state look like? (4x2) ---")
for d in [0.01, 0.03, 0.05, 0.07]:
    L = CairoLattice(4, 2, d=d); Jd = needle_coupling(L, 6.0)
    best = None
    for sd in range(20):
        q = quench(Jd, anneal(Jd, np.random.choice([-1., 1.], L.N), 40.0, 0.05, 3000, sd)[0]); e = 0.5*q@Jd@q/L.N
        if best is None or e < best[0]: best = (e, q)
    s = best[1]; qv = L.charges(s)
    pairs = [tuple(L.edges[k]) for k in range(L.N) if tuple(L.vtype[L.edges[k]]) == (3, 3)]
    pq = Counter(int(qv[u] + qv[v]) for u, v in pairs)            # charge of merged 'effective z4' vertex
    z4q = Counter(int(x) for x in qv[L.vtype == 4])
    Ed = 0.5*np.einsum('ij,jk,ik->i', S0, Jd, S0)/L.N
    print(f"  d={d:.2f}: SA best {best[0]:.4f} vs dimer-manifold min {Ed.min():.4f}; z4 charges {dict(z4q)}; merged-pair charges {dict(pq)}")
