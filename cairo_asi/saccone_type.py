"""Saccone-type islands (z3-z4 gap g34, partner gap gp) across the Cairo-Shakti family: manifold validity, zero-field
candidate energies (12x4), vertex-only linear responses, and formula thresholds k* = dE / lambda_vertex."""
import numpy as np, itertools, json, sys, time
import needle
G34, GP = float(sys.argv[1]), float(sys.argv[2]); needle.GAP, needle.GAP_P = G34, GP; needle.SUBDIV = 8; needle.FIXED_M = True
from cand_response import CairoLattice4, STATES, names, NX, NY
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from cairo2 import S0
def levels(L, Jl):
    out = {}
    for z, b in ((3, 2), (4, 0)):
        v = [x for x in range(len(L.vtype)) if x % 6 == b][0]; ks = [k for k, _ in L.incidence[v]]
        t = np.round([sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))], 6)
        u = sorted(set(t)); out[z] = [int(np.sum(t == q)) for q in u]
    return out
H = {"eps": 0.002, "theta": 0.01, "eta": 0.004}; res = {}
for db in [float(x) for x in sys.argv[3].split(",")]:
    t0 = time.time()
    L42 = CairoLattice4(4, 2, db); Jl42 = local_needle_coupling(L42); lv = levels(L42, Jl42)
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl42, S0)/80; valid = lv[3][:2] == [2, 4] and lv[4][0] == 2 and El.max() - El.min() < 1e-8
    if not valid:
        print(f"d={db:.3f}: manifold NOT valid (z3 {lv[3]}, z4 {lv[4]}, spread {El.max()-El.min():.1e})"); res[db] = dict(valid=False); continue
    L = CairoLattice4(NX, NY, db); J = hybrid_coupling(L, 64.0)
    E0 = {X: float(0.5*STATES[X]@J@STATES[X]/L.N) for X in names}
    lam = {}
    for X in names:
        s = STATES[X]; lam[X] = {}
        for key, h in H.items():
            ep = 0.5*s@local_needle_coupling(CairoLattice4(NX, NY, db, **{key: h}))@s/L.N; em = 0.5*s@local_needle_coupling(CairoLattice4(NX, NY, db, **{key: -h}))@s/L.N
            lam[X][key] = float(abs(ep - em)/(2*h))
    gs = min(E0, key=E0.get)
    res[db] = dict(valid=True, E0=E0, lam=lam, gs=gs)
    print(f"d={db:.3f}: GS {gs}; E0 - E_GS: " + ", ".join(f"{X} {E0[X]-E0[gs]:+.5f}" for X in names) + f"  [{time.time()-t0:.0f}s]", flush=True)
    print("        vertex responses: " + "; ".join(f"{X} ε {lam[X]['eps']:.2f} θ {lam[X]['theta']:.3f} η {lam[X]['eta']:.3f}" for X in names), flush=True)
json.dump(res, open(f"saccone_type_g{G34}_{GP}.json", "w"), indent=1)
