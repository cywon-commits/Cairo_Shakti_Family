import numpy as np, itertools, sys, time
from numba import njit
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice, D_EQ, anneal, quench, exact_ground_state_fast
from manifold_select import setup, enum_manifold

# reference phase sets from the smooth scan
ref = {}
for lab, f in [("A", "scan3_min_d0.1372.npy"), ("B", "scan3_min_d0.1531.npy"), ("C", "scan3_min_d0.2500.npy")]:
    ref[frozenset(tuple(r.astype(int)) for r in np.load(f))] = lab
L0 = CairoLattice(4, 2, d=D_EQ); Jl0 = needle_coupling(L0, None); ze0, zc0, part0 = setup(L0, Jl0)
_, S0 = enum_manifold(Jl0, ze0, zc0, part0, 200000); S0 = S0.astype(float); S0 = np.vstack([S0, -S0])
man0 = {tuple(s.astype(int)) for s in S0}

def levels(L, J, z):
    v = np.where(L.vtype == z)[0][0]; ks = [k for k, _ in L.incidence[v]]
    t = [sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))]
    lv = sorted(set(np.round(t, 3))); return lv, [int(sum(abs(np.array(t)-l) < 1e-3)) for l in lv]

for d in [float(x) for x in sys.argv[1:]]:
    t0 = time.time()
    L = CairoLattice(4, 2, d=d); Jl = needle_coupling(L, None); Jd = needle_coupling(L, 6.0)
    lv3, dg3 = levels(L, Jl, 3)
    ze, zc, part = setup(L, Jl); _, S = enum_manifold(Jl, ze, zc, part, 200000)
    same = (2*len(S) == len(man0)) and all(tuple(s) in man0 for s in S.astype(int))
    Ed = 0.5*np.einsum('ij,jk,ik->i', S0, Jd, S0)/L.N; Emin = Ed.min()
    mins = frozenset(tuple(S0[i].astype(int)) for i in np.where(Ed < Emin + 1e-7)[0])
    lab = ref.get(mins, f"new({len(mins)})")
    # exact full-dipolar GS on a 2x1 cell vs its manifold minimum
    L2 = CairoLattice(2, 1, d=d); Jd2 = needle_coupling(L2, 6.0); Jl2 = needle_coupling(L2, None)
    e_exact, s_ex = exact_ground_state_fast(Jd2)
    z2, c2, p2 = setup(L2, Jl2); _, S2 = enum_manifold(Jl2, z2, c2, p2, 10000); S2 = S2.astype(float)
    e_man2 = min(0.5*s@Jd2@s for s in S2)
    # SA on the 4x2 cell with full dipolar
    esa = min(0.5*q@Jd@q/L.N for q in [quench(Jd, anneal(Jd, np.random.choice([-1., 1.], L.N), 40.0, 0.05, 3000, sd)[0]) for sd in range(12)])
    print(f"d={d:.3f}: z3 levels deg {dg3}, gap {lv3[1]-lv3[0]:7.3f} | manifold same: {same} | selected: {lab} | "
          f"2x1 exact GS below manifold by {(e_man2-e_exact)/L2.N:.4f}/spin | 4x2: SA {esa:.4f} vs manifold min {Emin:.4f}  [{time.time()-t0:.0f}s]", flush=True)
