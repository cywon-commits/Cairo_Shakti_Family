"""Step 1 of the z3-coupling check: scale only the couplings of island pairs sharing a z3 vertex (J1, J3) by lam.
Usage: python saccone_z3scale.py c lam mode(exact4x2|TA|cool)"""
import numpy as np, itertools, sys, os, time, json
from saccone_geometry import SacconeArray, coupling_matrices, K_PER_EV
from cairo_asi import CairoLattice, D_EQ, anneal
from hybrid_coupling import local_needle_coupling
import needle
from manifold_select import setup, enum_manifold
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from plot_phases import tile
c = float(sys.argv[1]); lam = float(sys.argv[2]); mode = sys.argv[3]; t0 = time.time()
def z3_part(L, Jl):
    M = np.zeros_like(Jl)
    for v in np.where(L.vtype == 3)[0]:
        ks = [k for k, _ in L.incidence[v]]
        for x, y in itertools.combinations(ks, 2): M[x, y] = Jl[x, y]; M[y, x] = Jl[y, x]
    return M
def z3_levels(L, Jl):
    v = np.where(L.vtype == 3)[0][0]; ks = [k for k, _ in L.incidence[v]]
    t = sorted(set(np.round([sum(Jl[ks[x], ks[y]]*cc[x]*cc[y] for x, y in itertools.combinations(range(3), 2)) for cc in itertools.product([-1, 1], repeat=3)], 3)))
    return [round(x - t[0]) for x in t]
if mode == "exact4x2":
    needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
    L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0)
    ze, zc, part = setup(L0, Jn); _, S0 = enum_manifold(Jn, ze, zc, part, 200000); S0 = S0.astype(float)
    man0 = {tuple(s.astype(int)) for s in S0} | {tuple(-s.astype(int)) for s in S0}; S0 = np.vstack([S0, -S0])
    iA = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in {tuple(r.astype(int)) for r in np.load("scan3_min_d0.1372.npy")}]
    iC = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in {tuple(r.astype(int)) for r in np.load("scan3_min_d0.2500.npy")}]
    Arr = SacconeArray(4, 2, c, nl=16, nw=3); L = Arr.L; Jl, J = coupling_matrices(Arr); Jl *= K_PER_EV; J *= K_PER_EV
    M = z3_part(L, Jl); Jl2 = Jl - (1 - lam)*M; J2 = J - (1 - lam)*M
    z, cz, p = setup(L, Jl2); _, S = enum_manifold(Jl2, z, cz, p, 400000)
    same = (2*len(S) == len(man0)) and all(tuple(s) in man0 for s in S.astype(int))
    E = 0.5*np.einsum('ij,jk,ik->i', S0, J2, S0)/L.N; El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl2, S0)/L.N
    sel = "A" if int(np.argmin(E)) in iA else ("C" if int(np.argmin(E)) in iC else "other")
    print(f"c={c:.0f} lam={lam}: z3 levels (K) {z3_levels(L, Jl2)}; manifold same {same}; local-energy spread over manifold {El.max()-El.min():.2e} K; "
          f"selected {sel}; E_A-E_C {(E[iA].min()-E[iC].min()):+.2f} K/island; manifold spread {E.max()-E.min():.2f} K/island  [{time.time()-t0:.0f}s]")
else:
    n = 8; z = np.load(f"Jsac_c{c:.0f}_n{n}.npz"); Jl, J = z["Jl"], z["J"]; L = SacconeArray(n, n, c, nl=16, nw=3).L
    M = z3_part(L, Jl); Jl = Jl - (1 - lam)*M; J = J - (1 - lam)*M
    F = FastDimers(L, Jl); arr = mc_arrays(L); sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); EA = 0.5*sA@J@sA/L.N; El0 = 0.5*sA@Jl@sA
    def st(s):
        bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
        lev = F.lev[np.arange(len(code)), code]; unh = lev == 1; lab = np.zeros(len(code))
        for q in range(3):
            heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
            lab[heal] = np.where(np.isin(F.etype[heal, q], [6, 7, 8, 9]), 1, -1)
        return float(abs(lab[unh].mean())) if unh.any() else 0.0, float(unh.mean()), float((0.5*s@Jl@s - El0)/L.N)
    if mode == "TA":
        Ts = np.geomspace(2000, 5, 26); s = sA.copy(); rows = []
        for iT, T in enumerate(Ts):
            s, _, _ = mc(s, J, Jl, *arr, T, T, 800, 17*iT + 1); ps = []
            for b in range(60):
                s, _, _ = mc(s, J, Jl, *arr, T, T, 10, 5000 + 100*iT + b); ps.append(st(s)[0])
            rows.append((float(T), float(np.mean(ps))))
        TA = next((float(np.exp(np.log(T1) + (0.5-p1)/(p2-p1)*(np.log(T2)-np.log(T1)))) for (T1, p1), (T2, p2) in zip(rows, rows[1:]) if p1 < 0.5 <= p2), None)
        print(f"c={c:.0f} lam={lam}: T_A = {TA:.0f} K  [{time.time()-t0:.0f}s]")
    else:
        Tg = np.geomspace(3000, 150, int(np.log10(3000/150)*20) + 1)
        for sd in range(2):
            s = np.random.default_rng(sd).choice([-1., 1.], L.N)
            for T in Tg: s, _ = anneal(J, s, T, T, 1000, 300 + 1000*sd + int(T) % 997)
            p, fb, ex = st(s)
            print(f"c={c:.0f} lam={lam}: single-flip cooling to 150 K seed {sd}: |psi_A| {p:.2f}, frac B {fb:.3f}, local excess {ex:.1f} K/island  [{time.time()-t0:.0f}s]", flush=True)
