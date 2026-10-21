"""Island-model dependence: manifold validity and dipolar selection (4x2 exact) for point-dipole models."""
import numpy as np, itertools, sys, glob
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import point_coupling, local_needle_coupling
from manifold_select import setup, enum_manifold
model = sys.argv[1]; unit = (model == "P2"); ds = [float(x) for x in sys.argv[2].split(",")]; rc = float(sys.argv[3])
L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0)
ze, zc, part = setup(L0, Jn); _, S0 = enum_manifold(Jn, ze, zc, part, 200000); S0 = S0.astype(float)
man0 = {tuple(s.astype(int)) for s in S0} | {tuple(-s.astype(int)) for s in S0}; S0 = np.vstack([S0, -S0])
ref = {}
for lab, f in [("A", "scan3_min_d0.1372.npy"), ("C", "scan3_min_d0.2500.npy")] + [(f.split("_")[1], f) for f in sorted(glob.glob("phase_[A-Z]_d*.npy"))]:
    ref[frozenset(tuple(r.astype(int)) for r in np.load(f))] = lab
for d in ds:
    L = CairoLattice(4, 2, d=d); Jl = point_coupling(L, None, unit)
    v3 = np.where(L.vtype == 3)[0][0]; ks = [k for k, _ in L.incidence[v3]]
    t = [sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(3), 2)) for c in itertools.product([-1, 1], repeat=3)]
    lv = sorted(set(np.round(t, 5))); dg = [int(sum(abs(np.array(t) - l) < 1e-4)) for l in lv]
    z, c, p = setup(L, Jl); _, S = enum_manifold(Jl, z, c, p, 400000)
    same = (2*len(S) == len(man0)) and all(tuple(s) in man0 for s in S.astype(int))
    line = f"{model} d={d:.4f}: z3 levels deg {dg} gap {lv[1]-lv[0]:.3f}; manifold same as needle: {same} (size {2*len(S)})"
    if same:
        J = point_coupling(L, rc, unit); E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N; Emin = E.min()
        mins = frozenset(tuple(S0[i].astype(int)) for i in np.where(E < Emin + 1e-7)[0])
        if mins not in ref: ref[mins] = f"new{len(ref)}"; np.save(f"phase_{model}_{ref[mins]}_d{d:.4f}.npy", np.array(sorted(mins), float))
        line += f"; selected: {ref[mins]}"
    print(line, flush=True)
