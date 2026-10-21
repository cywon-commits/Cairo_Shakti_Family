"""Step 2 (chunked): manifold annealing on 12x12 with cached couplings; compare with the best candidate."""
import numpy as np, json, sys, time, os
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from cand_response import CairoLattice4
PTS = {"P1": (0.153, "eta", 0.004), "P2": (0.150, "eta", 0.002), "P3": (0.137, "theta", 0.02), "P4": (0.25, "eps", 0.01)}
name, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]; db, key, val = PTS[name]; N = 12
info = json.load(open("candidates.json")); names = list(info)
def tile_xy(s0, cx, cy, nx, ny):
    t = np.empty(nx*ny*10)
    for i in range(nx):
        for j in range(ny): t[(i*ny + j)*10:(i*ny + j)*10 + 10] = s0[((i % cx)*cy + (j % cy))*10:((i % cx)*cy + (j % cy))*10 + 10]
    return t
ST = {k: tile_xy(np.load(f"cand_{k}.npy"), info[k]["px"], info[k]["py"], N, N) for k in names}
zp, zm = np.load(f"vmcJ_{name}_p.npz"), np.load(f"vmcJ_{name}_m.npz"); J, Jl, Jm = zp["J"], zp["Jl"], zm["J"]
from hybrid_coupling import local_needle_coupling
L = CairoLattice4(N, N, db, **{key: val}); arr = mc_arrays(L)
Jl0 = local_needle_coupling(CairoLattice4(N, N, db))     # manifold constraint: undeformed vertex couplings (topology)
F = FastDimers(L, Jl0)
cand = {X: min(0.5*ST[X]@J@ST[X], 0.5*ST[X]@Jm@ST[X])/L.N for X in names}; best = min(cand, key=cand.get)
fn = "verify_mc.json"; res = json.load(open(fn)) if os.path.exists(fn) else {}
res.setdefault(name, dict(d=db, knob={key: val}, candidates=cand, runs={}))
print(f"{name} (d={db}, {key}={val}): best candidate {best} {cand[best]:.5f}; others " + ", ".join(f"{X} {v - cand[best]:+.5f}" for X, v in cand.items() if X != best), flush=True)
for sd in seeds:
    t0 = time.time(); s = ST["C"].copy()
    s, _, _ = mc(s, J, Jl0, *arr, 1.0, 1.0, 400, 100 + sd)
    for T in np.geomspace(0.5, 0.002, 40): s, _, _ = mc(s, J, Jl0, *arr, T, T, 100, int(1e4*T) + 7*sd)
    e = 0.5*s@J@s/L.N; pa, pc = F.psi(s)
    man = F.psi(s)  # noqa
    res[name]["runs"][str(sd)] = dict(E=float(e), diff=float(e - cand[best]), psiA=float(pa), psiC=float(pc))
    np.save(f"verify_mc_{name}_s{sd}.npy", s); json.dump(res, open(fn, "w"), indent=1)
    print(f"   seed {sd}: E/N {e:.5f}, E - best candidate = {e - cand[best]:+.5f}; psi_A {pa:+.2f}, psi_C {pc:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
