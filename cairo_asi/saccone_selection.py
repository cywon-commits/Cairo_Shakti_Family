import numpy as np, itertools, sys, json, time
from saccone_geometry import SacconeArray, coupling_matrices, K_PER_EV
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import local_needle_coupling
import needle
from manifold_select import setup, enum_manifold
needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0)
ze, zc, part = setup(L0, Jn); _, S0 = enum_manifold(Jn, ze, zc, part, 200000); S0 = S0.astype(float)
man0 = {tuple(s.astype(int)) for s in S0} | {tuple(-s.astype(int)) for s in S0}; S0 = np.vstack([S0, -S0])
A_set = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.1372.npy")}; C_set = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.2500.npy")}
iA = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in A_set]; iC = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in C_set]
out = {}
for c in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time(); Arr = SacconeArray(4, 2, c, nl=16, nw=3); L = Arr.L
    Jl, J = coupling_matrices(Arr)
    lv = {}
    for z in (3, 4):
        v = np.where(L.vtype == z)[0][0]; ks = [k for k, _ in L.incidence[v]]
        t = [sum(Jl[ks[x], ks[y]]*cc[x]*cc[y] for x, y in itertools.combinations(range(len(ks)), 2)) for cc in itertools.product([-1, 1], repeat=len(ks))]
        u = sorted(set(np.round(t, 5))); lv[z] = ([x - u[0] for x in u], [int(sum(abs(np.array(t) - q) < 1e-4)) for q in u])
    z, cz, p = setup(L, Jl); _, S = enum_manifold(Jl, z, cz, p, 400000)
    same = (2*len(S) == len(man0)) and all(tuple(s) in man0 for s in S.astype(int))
    rec = dict(z3_levels_K=[round(x*K_PER_EV) for x in lv[3][0]], z3_deg=lv[3][1], z4_levels_K=[round(x*K_PER_EV) for x in lv[4][0]], z4_deg=lv[4][1], manifold_same=same)
    if same:
        E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N
        imin = int(np.argmin(E)); lab = "A" if imin in iA else ("C" if imin in iC else "other")
        rec.update(selected=lab, EA_minus_EC_K=float((E[iA].min() - E[iC].min())*K_PER_EV), spread_K=float((E.max() - E.min())*K_PER_EV),
                   gap_to_next_K=float((np.sort(np.unique(np.round(E, 9)))[1] - E.min())*K_PER_EV))
    out[c] = rec
    print(f"c={c:.0f} nm: z3 levels (K above lowest) {rec['z3_levels_K']} deg {rec['z3_deg']}; z4 levels {rec['z4_levels_K']} deg {rec['z4_deg']}; manifold same: {same}"
          + (f"\n    selected {rec['selected']}; E_A - E_C = {rec['EA_minus_EC_K']:+.1f} K/island; manifold spread {rec['spread_K']:.1f} K/island; gap to next state {rec['gap_to_next_K']:.2f} K/island" if same else "")
          + f"  [{time.time()-t0:.0f}s]", flush=True)
json.dump({str(k): v for k, v in out.items()}, open("saccone_selection.json", "w"), indent=1)
