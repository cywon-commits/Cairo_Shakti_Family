"""Saccone-type islands across the family: z3-z4 islands gap g34 = 0.18 a, partner islands gap gp = 0.047 a (fixed gaps).
For each d: manifold validity, zero-field energies of A, C, M, W, L3 on 12x4, and vertex-only responses to eps, theta, eta."""
import numpy as np, itertools, json, sys, time
import needle; needle.SUBDIV = 8; needle.FIXED_M = True; needle.GAP = 0.18; needle.GAP_P = 0.047
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from cand_response import CairoLattice4, STATES, names, NX, NY
from cairo2 import S0
H = {"eps": 0.002, "theta": 0.01, "eta": 0.004}
out = {}
for d in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    L42 = CairoLattice4(4, 2, d); Jl42 = local_needle_coupling(L42)
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl42, S0); valid = (El.max() - El.min()) < 1e-6
    lv = {}
    for z in (3, 4):
        v = np.where(L42.vtype == z)[0][0]; ks = [k for k, _ in L42.incidence[v]]
        t = [sum(Jl42[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))]
        u = sorted(set(np.round(t, 6))); lv[z] = ([round(x - u[0], 4) for x in u][:3], [int(sum(abs(np.array(t) - q) < 1e-5)) for q in u][:3])
    rec = dict(valid=bool(valid), z3=lv[3], z4=lv[4])
    if valid:
        L = CairoLattice4(NX, NY, d); J = hybrid_coupling(L, 64.0)
        rec["E0"] = {X: float(0.5*STATES[X]@J@STATES[X]/L.N) for X in names}
        rec["lam"] = {}
        for key, h in H.items():
            Jp = local_needle_coupling(CairoLattice4(NX, NY, d, **{key: h})); Jm = local_needle_coupling(CairoLattice4(NX, NY, d, **{key: -h}))
            rec["lam"][key] = {X: float(abs(0.5*STATES[X]@Jp@STATES[X] - 0.5*STATES[X]@Jm@STATES[X])/L.N/(2*h)) for X in names}
    out[d] = rec
    msg = f"d={d:.3f}: z3 levels {lv[3][0]} deg {lv[3][1]}, manifold valid {valid}"
    if valid:
        e = rec["E0"]; m = min(e.values())
        msg += " | E0 - min: " + ", ".join(f"{X} {e[X]-m:+.5f}" for X in names)
        msg += " | vertex lambda eps/theta/eta: " + "; ".join(f"{X} {rec['lam']['eps'][X]:.2f}/{rec['lam']['theta'][X]:.3f}/{rec['lam']['eta'][X]:.3f}" for X in names)
    print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(f"saccone_islands_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
