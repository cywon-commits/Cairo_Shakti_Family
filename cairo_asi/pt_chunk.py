"""Checkpointed parallel tempering in the dimer manifold. Each call advances `chunk` rounds.
Usage: python pt_chunk.py d n start seed chunk total"""
import numpy as np, sys, os, json, time
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from pt import wall_profile
from plot_phases import tile
d = float(sys.argv[1]); n = int(sys.argv[2]); start = sys.argv[3]; seed = int(sys.argv[4]); chunk = int(sys.argv[5]); total = int(sys.argv[6])
tag = f"d{d:.4f}_n{n}_{start}{seed}"; ck = f"ptck_{tag}.npz"; t0 = time.time()
L = CairoLattice(n, n, d=d)
jf = f"J_d{d:.4f}_n{n}.npz"
if os.path.exists(jf): z = np.load(jf); J, Jl = z["J"], z["Jl"]
else: Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); np.savez(jf, J=J, Jl=Jl)
arr = mc_arrays(L); F = FastDimers(L, Jl)
Ts = np.geomspace(0.5, 0.003, 30); M = len(Ts); every = 5
if os.path.exists(ck):
    z = np.load(ck, allow_pickle=True); reps = list(z["reps"]); rnd0 = int(z["rnd"]); meas = z["meas"].item(); rng_state = z["rng"].item()
    rng = np.random.default_rng(); rng.bit_generator.state = rng_state
else:
    s0 = tile(np.load("scan3_min_d0.1372.npy" if start == "A" else "scan3_min_d0.2500.npy")[0], 4, 2, n)
    reps = [s0.copy() for _ in range(M)]; rnd0 = 0; rng = np.random.default_rng(seed)
    meas = {m: dict(E=[], pC=[], pA=[], walls=[]) for m in range(M)}; meas["acc"] = np.zeros(M-1); meas["att"] = np.zeros(M-1)
E = np.array([0.5*r@J@r for r in reps])
rnd = rnd0
while rnd < min(total, rnd0 + chunk) and time.time() - t0 < 250:
    for m in range(M):
        reps[m], Etr, _ = mc(reps[m], J, Jl, *arr, Ts[m], Ts[m], every, (seed*1000003 + rnd*97 + m) % (2**31))
        E[m] = Etr[-1]
    for m in range(rnd % 2, M - 1, 2):
        meas["att"][m] += 1; x = (1/Ts[m] - 1/Ts[m+1])*(E[m] - E[m+1])
        if x >= 0 or rng.random() < np.exp(x):
            reps[m], reps[m+1] = reps[m+1], reps[m]; E[m], E[m+1] = E[m+1], E[m]; meas["acc"][m] += 1
    if rnd >= total // 2 and rnd % 5 == 0:
        for m in range(M):
            a, c = F.psi(reps[m]); meas[m]["E"].append(E[m]/L.N); meas[m]["pA"].append(abs(a)); meas[m]["pC"].append(abs(c))
            if Ts[m] < 0.2: meas[m]["walls"].append(wall_profile(F, reps[m], n))
    rnd += 1
np.savez(ck, reps=np.array(reps), rnd=rnd, meas=np.array(meas, dtype=object), rng=np.array(rng.bit_generator.state, dtype=object))
lo = int(np.argmin(E)); np.save(f"pt_best_{tag}.npy", reps[lo])
sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); sC = tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, n)
msg = f"{tag}: rounds {rnd}/{total}; best E/N now {E.min()/L.N:.5f} (tiled C {0.5*sC@J@sC/L.N:.5f}); lowest-T replica walls {wall_profile(F, reps[-1], n)}, |psiC| {abs(F.psi(reps[-1])[1]):.2f}; min swap acc {np.min(meas['acc']/np.maximum(meas['att'],1)):.2f}  [{time.time()-t0:.0f}s]"
if rnd >= total:
    rows = []
    for m in range(M):
        w = [x for x in meas[m]["walls"] if x is not None]
        rows.append(dict(T=float(Ts[m]), E=float(np.mean(meas[m]["E"])), psiA=float(np.mean(meas[m]["pA"])), psiC=float(np.mean(meas[m]["pC"])),
                         walls=float(np.mean(w)) if w else None))
    json.dump(dict(d=d, n=n, rows=rows, Ebest=float(E.min()/L.N)), open(f"pt_{tag}.json", "w"), indent=1)
    msg += "  DONE"
print(msg, flush=True)
