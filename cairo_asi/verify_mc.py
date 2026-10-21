"""Step 2: unbiased manifold annealing on 12x12 at deformed points; compare with the best candidate on the same torus."""
import numpy as np, json, sys, time, os
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from cand_response import CairoLattice4, names
from candidates import sub  # noqa
import json as _j
info = _j.load(open("candidates.json")); N = 12
def tile_xy(s0, cx, cy, nx, ny):
    t = np.empty(nx*ny*10)
    for i in range(nx):
        for j in range(ny): t[(i*ny + j)*10:(i*ny + j)*10 + 10] = s0[((i % cx)*cy + (j % cy))*10:((i % cx)*cy + (j % cy))*10 + 10]
    return t
ST = {k: tile_xy(np.load(f"cand_{k}.npy"), info[k]["px"], info[k]["py"], N, N) for k in names}
points = [("P1", 0.153, {"eta": 0.004}), ("P2", 0.150, {"eta": 0.002}), ("P3", 0.137, {"theta": 0.02}), ("P4", 0.25, {"eps": 0.01})]
res = json.load(open("verify_mc.json")) if os.path.exists("verify_mc.json") else {}
for name, db, kw in points:
    if name in res: continue
    t0 = time.time()
    key, val = list(kw.items())[0]
    L = CairoLattice4(N, N, db, **kw); J = hybrid_coupling(L, 64.0); Jl = local_needle_coupling(L)
    Lm = CairoLattice4(N, N, db, **{key: -val}); Jm = hybrid_coupling(Lm, 64.0)
    cand = {X: min(0.5*ST[X]@J@ST[X], 0.5*ST[X]@Jm@ST[X])/L.N for X in names}
    print(f"{name}: couplings ready [{time.time()-t0:.0f}s]; candidates " + ", ".join(f"{X} {v:.5f}" for X, v in cand.items()), flush=True)
    arr = mc_arrays(L); F = FastDimers(L, Jl); runs = []
    for sd in range(3):
        s = ST["C"].copy()
        s, _, _ = mc(s, J, Jl, *arr, 1.0, 1.0, 200, 100 + sd)                 # melt inside the manifold
        for T in np.geomspace(0.5, 0.002, 30): s, _, _ = mc(s, J, Jl, *arr, T, T, 60, int(1e4*T) + 7*sd)
        e = 0.5*s@J@s/L.N; pa, pc = F.psi(s); runs.append((float(e), float(pa), float(pc)))
        print(f"   seed {sd}: E/N {e:.5f} (best candidate {min(cand.values()):.5f}, diff {e - min(cand.values()):+.5f}); psi_A {pa:+.2f} psi_C {pc:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
        np.save(f"verify_mc_{name}_s{sd}.npy", s)
    res[name] = dict(d=db, knob=kw, candidates=cand, runs=runs); json.dump(res, open("verify_mc.json", "w"), indent=1)
print("done")
