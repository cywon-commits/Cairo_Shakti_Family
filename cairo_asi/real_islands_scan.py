import numpy as np, itertools, json, sys, time
from real_islands import RealArray, vertex_J, full_J
from cand_response import STATES, names, NX, NY
from cairo2 import S0
H = {"eps": 0.002, "theta": 0.01, "eta": 0.004}
fn = "real_islands_scan.json"
try: out = json.load(open(fn))
except Exception: out = {}
for d in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    A42 = RealArray(4, 2, d); Jl42 = vertex_J(A42); L42 = A42.L
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl42, S0)
    v = np.where(L42.vtype == 3)[0][0]; ks = [k for k, _ in L42.incidence[v]]
    t = [sum(Jl42[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(3), 2)) for c in itertools.product([-1, 1], repeat=3)]
    u = sorted(set(np.round(t, 3))); deg = [int(sum(abs(np.array(t) - q) < 1e-2)) for q in u]
    valid = deg[:2] == [2, 4] and (El.max() - El.min()) < 1e-3
    rec = dict(z3_levels_K=[round(x - u[0], 1) for x in u], z3_deg=deg, valid=bool(valid))
    if valid:
        A = RealArray(NX, NY, d); Jl, J = full_J(A)
        rec["E0"] = {X: float(0.5*STATES[X]@J@STATES[X]/A.L.N) for X in names}; rec["lam"] = {}
        for key, h in H.items():
            Jp = vertex_J(RealArray(NX, NY, d, **{key: h})); Jm = vertex_J(RealArray(NX, NY, d, **{key: -h}))
            rec["lam"][key] = {X: float(abs(0.5*STATES[X]@Jp@STATES[X] - 0.5*STATES[X]@Jm@STATES[X])/A.L.N/(2*h)) for X in names}
    out[f"{d:.4f}"] = rec; json.dump(out, open(fn, "w"), indent=1)
    msg = f"d={d:.3f}: z3 levels (K) {rec['z3_levels_K']} deg {deg}, valid {valid}"
    if valid:
        e = rec["E0"]; m = min(e.values())
        msg += "\n    E0 - min (K/island): " + ", ".join(f"{X} {e[X]-m:+.2f}" for X in names)
        msg += "\n    vertex lambda (K/island per unit) eps/theta/eta: " + "; ".join(f"{X} {rec['lam']['eps'][X]:.0f}/{rec['lam']['theta'][X]:.1f}/{rec['lam']['eta'][X]:.1f}" for X in names)
    print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
