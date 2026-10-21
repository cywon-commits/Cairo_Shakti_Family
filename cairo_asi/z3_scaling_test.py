"""Step 1: scale only the z3-vertex pair couplings (J1-type, J3-type) by lam; check manifold, z3 gap, E_A-E_C, T_A."""
import numpy as np, itertools, sys, json, time, os
from saccone_geometry import SacconeArray, coupling_matrices, K_PER_EV
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import local_needle_coupling
import needle
from manifold_select import setup, enum_manifold
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from plot_phases import tile
needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
lam = float(sys.argv[1]); cs = [float(x) for x in sys.argv[2].split(",")]; doTA = len(sys.argv) > 3
L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0)
ze, zc, part = setup(L0, Jn); _, S0 = enum_manifold(Jn, ze, zc, part, 200000); S0 = S0.astype(float)
man0 = {tuple(s.astype(int)) for s in S0} | {tuple(-s.astype(int)) for s in S0}; S0 = np.vstack([S0, -S0])
A_set = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.1372.npy")}; C_set = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.2500.npy")}
iA = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in A_set]; iC = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in C_set]
def scale_z3(L, Jl, J, lam):
    Jl, J = Jl.copy(), J.copy()
    for v in np.where(L.vtype == 3)[0]:
        ks = [k for k, _ in L.incidence[v]]
        for i, j in itertools.combinations(ks, 2):
            dv = (lam - 1.0)*Jl[i, j]
            Jl[i, j] += dv; Jl[j, i] += dv; J[i, j] += dv; J[j, i] += dv
    return Jl, J
def levels(L, Jl, z):
    v = np.where(L.vtype == z)[0][0]; ks = [k for k, _ in L.incidence[v]]
    t = [sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))]
    u = sorted(set(np.round(t, 3))); return [round(x - u[0]) for x in u], [int(sum(abs(np.array(t) - q) < 1e-2)) for q in u]
for c in cs:
    t0 = time.time()
    A = SacconeArray(4, 2, c, nl=16, nw=3); L = A.L
    Jl, J = coupling_matrices(A); Jl *= K_PER_EV; J *= K_PER_EV
    res = {}
    for lab, l in [("orig", 1.0), (f"lam={lam}", lam)]:
        Jl2, J2 = scale_z3(L, Jl, J, l)
        z, cz, p = setup(L, Jl2); _, S = enum_manifold(Jl2, z, cz, p, 400000)
        same = (2*len(S) == len(man0)) and all(tuple(s) in man0 for s in S.astype(int))
        E = 0.5*np.einsum('ij,jk,ik->i', S0, J2, S0)/L.N
        El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl2, S0)
        res[lab] = dict(z3=levels(L, Jl2, 3), same=same, EAC=float(E[iA].min() - E[iC].min()), spread=float(E.max() - E.min()),
                        Elocal_const=bool(El.max() - El.min() < 1e-6))
    o, m = res["orig"], res[f"lam={lam}"]
    print(f"c={c:.0f}: z3 levels orig {o['z3'][0]} -> {m['z3'][0]} (deg {m['z3'][1]}); manifold same {o['same']}/{m['same']}; "
          f"E_A-E_C {o['EAC']:+.2f} -> {m['EAC']:+.2f} K/island; spread {o['spread']:.2f} -> {m['spread']:.2f}; local E constant over manifold {m['Elocal_const']}  [{time.time()-t0:.0f}s]", flush=True)
if doTA:
    c = cs[0]; z = np.load(f"Jsac_c{c:.0f}_n8.npz"); Arr = SacconeArray(8, 8, c, nl=16, nw=3); L = Arr.L
    Jl2, J2 = scale_z3(L, z["Jl"], z["J"], lam); F = FastDimers(L, Jl2); arr = mc_arrays(L)
    s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, 8); rows = []
    for iT, T in enumerate(np.geomspace(2000, 5, 26)):
        s, _, _ = mc(s, J2, Jl2, *arr, T, T, 800, 17*iT + 1); ps = []
        for b in range(60):
            s, _, _ = mc(s, J2, Jl2, *arr, T, T, 10, 5000 + 100*iT + b); ps.append(abs(F.psi(s)[0]))
        rows.append((T, np.mean(ps)))
    TA = next((float(np.exp(np.log(T1) + (0.5-p1)/(p2-p1)*(np.log(T2)-np.log(T1)))) for (T1, p1), (T2, p2) in zip(rows, rows[1:]) if p1 < 0.5 <= p2), None)
    print(f"c={c:.0f}, lam={lam}: T_A = {TA:.1f} K (original 321.6 K)")
