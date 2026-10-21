"""PT in the dimer manifold recording joint samples of (psi_A, psi_C) (signed) per temperature.
Usage: python pt_joint.py d nround seed"""
import numpy as np, sys, json, time
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from plot_phases import tile
d = float(sys.argv[1]); nround = int(sys.argv[2]); seed = int(sys.argv[3]); n = 8; t0 = time.time()
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
Ts = np.geomspace(0.8, 0.02, 24); M = len(Ts)
sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); sC = tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, n)
reps = [(sA if m % 2 else sC).copy() for m in range(M)]; E = np.array([0.5*r@J@r for r in reps])
rng = np.random.default_rng(seed); samples = {m: [] for m in range(M)}; acc = np.zeros(M-1); att = np.zeros(M-1)
for rnd in range(nround):
    for m in range(M):
        reps[m], Etr, _ = mc(reps[m], J, Jl, *arr, Ts[m], Ts[m], 5, seed*100000 + rnd*64 + m); E[m] = Etr[-1]
    for m in range(rnd % 2, M-1, 2):
        att[m] += 1; x = (1/Ts[m] - 1/Ts[m+1])*(E[m] - E[m+1])
        if x >= 0 or rng.random() < np.exp(x):
            reps[m], reps[m+1] = reps[m+1], reps[m]; E[m], E[m+1] = E[m+1], E[m]; acc[m] += 1
    if rnd >= nround//3 and rnd % 3 == 0:
        for m in range(M):
            a, c = F.psi(reps[m]); samples[m].append((float(a), float(c), float(E[m]/L.N)))
json.dump(dict(d=d, T=Ts.tolist(), samples={str(m): v for m, v in samples.items()}, swap=(acc/np.maximum(att,1)).tolist()),
          open(f"ptjoint_d{d:.4f}_s{seed}.json", "w"))
print(f"d={d}: done [{time.time()-t0:.0f}s], min swap acc {np.min(acc/np.maximum(att,1)):.2f}")
for m in range(0, M, 2):
    s = np.array(samples[m]); pa, pc = np.abs(s[:, 0]), np.abs(s[:, 1])
    fA = np.mean((pa > 0.5) & (pc < 0.3)); fC = np.mean((pc > 0.5) & (pa < 0.3)); fmix = np.mean((pa > 0.3) & (pc > 0.3)); fdis = np.mean((pa < 0.3) & (pc < 0.3))
    print(f"  T={Ts[m]:.3f}: <|psiA|> {pa.mean():.2f} <|psiC|> {pc.mean():.2f} | fraction of samples: A-like {fA:.2f}, C-like {fC:.2f}, both>0.3 {fmix:.2f}, neither {fdis:.2f}")
