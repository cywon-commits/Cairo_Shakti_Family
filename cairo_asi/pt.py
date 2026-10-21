"""Parallel tempering within the dimer manifold (manifold MC moves + replica exchange).
Usage: python pt.py d n nrounds start(A|C|mix) seed"""
import numpy as np, sys, json, time, os
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from plot_phases import tile

def wall_profile(F, s, n):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    unh = F.lev[np.arange(len(code)), code] > 0; sub = np.zeros(len(code))
    for t in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - t))] == 0) & F.is34[:, t]
        sub[heal] = np.where(np.isin(F.etype[heal, t], [3, 5, 7, 9]), 1, -1)
    cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n
    best = None
    for key in ((ci + cj) % n, (ci - cj) % n):
        prof = np.round(np.array([sub[(key == v) & unh].mean() for v in range(n)]), 6)
        if not np.all(np.isin(prof, [-1, 0, 1])): continue
        nz = int(np.sum(prof == 0)); ns = sum(1 for t in range(n) if prof[t] != 0 and prof[(t+1) % n] != 0 and prof[t] != prof[(t+1) % n])
        if best is None or nz + ns < best: best = nz + ns
    return best                                           # None = not a pure stripe state

if __name__ == "__main__":
    d = float(sys.argv[1]); n = int(sys.argv[2]); nround = int(sys.argv[3]); start = sys.argv[4]; seed = int(sys.argv[5])
    Ts = np.geomspace(0.8, 0.003, 28); M = len(Ts); every = 5
    t0 = time.time()
    L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
    sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); sC = tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, n)
    EC = 0.5*sC@J@sC/L.N; EA = 0.5*sA@J@sA/L.N
    reps = [(sA if (start == "A" or (start == "mix" and m % 2)) else sC).copy() for m in range(M)]
    E = np.array([0.5*r@J@r for r in reps]); acc = np.zeros(M - 1); att = np.zeros(M - 1)
    rng = np.random.default_rng(seed)
    meas = {m: dict(E=[], pA=[], pC=[], walls=[]) for m in range(M)}
    for rnd in range(nround):
        for m in range(M):
            reps[m], Etr, _ = mc(reps[m], J, Jl, *arr, Ts[m], Ts[m], every, seed*100000 + rnd*64 + m)
            E[m] = Etr[-1]
        for m in range(rnd % 2, M - 1, 2):
            att[m] += 1
            x = (1/Ts[m] - 1/Ts[m+1]) * (E[m] - E[m+1])
            if x >= 0 or rng.random() < np.exp(x):
                reps[m], reps[m+1] = reps[m+1], reps[m]; E[m], E[m+1] = E[m+1], E[m]; acc[m] += 1
        if rnd >= nround // 2 and rnd % 5 == 0:
            for m in range(M):
                a, c = F.psi(reps[m]); meas[m]["E"].append(E[m]/L.N); meas[m]["pA"].append(abs(a)); meas[m]["pC"].append(abs(c))
                if Ts[m] < 0.2: meas[m]["walls"].append(wall_profile(F, reps[m], n))
    res = []
    for m in range(M):
        w = [x for x in meas[m]["walls"] if x is not None]
        res.append(dict(T=float(Ts[m]), E=float(np.mean(meas[m]["E"])), Evar=float(np.var(meas[m]["E"])*L.N/Ts[m]**2),
                        psiA=float(np.mean(meas[m]["pA"])), psiC=float(np.mean(meas[m]["pC"])),
                        walls=float(np.mean(w)) if w else None, frac_stripe=len(w)/max(1, len(meas[m]["walls"])) if meas[m]["walls"] else None))
    lo = np.argmin(E); best = reps[int(lo)]
    np.save(f"pt_best_d{d:.4f}_n{n}_{start}{seed}.npy", best)
    out = dict(d=d, n=n, EA=EA, EC=EC, Ebest=float(E.min()/L.N), swap_acc=(acc/np.maximum(att, 1)).tolist(), rows=res)
    json.dump(out, open(f"pt_d{d:.4f}_n{n}_{start}{seed}.json", "w"), indent=1)
    r0 = res[-1]
    print(f"d={d:.4f} n={n} start={start}: lowest T={r0['T']:.3f} <E/N>={r0['E']:.5f} (EA {EA:.5f}, EC {EC:.5f}), |psiA| {r0['psiA']:.2f}, |psiC| {r0['psiC']:.2f}, "
          f"walls {r0['walls']}; best E/N seen {E.min()/L.N:.5f}; min swap acc {min(acc/np.maximum(att,1)):.2f}  [{time.time()-t0:.0f}s]", flush=True)
