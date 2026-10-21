"""Quench-and-hold coarsening test inside the dimer manifold.
Usage: python coarsening.py d n T_hold which(A|C) nseed nhold"""
import numpy as np, sys, time, json
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling, point_coupling
from selection_T import FastDimers
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile
d = float(sys.argv[1]); n = int(sys.argv[2]); Th = float(sys.argv[3]); which = sys.argv[4]; nseed = int(sys.argv[5]); nhold = int(sys.argv[6])
t0 = time.time()
L = CairoLattice(n, n, d=d); Jl = point_coupling(L, None); J = point_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n
setA, setC = [6, 7, 8, 9], [3, 5, 7, 9]
def local_field(s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    unh = F.lev[np.arange(len(code)), code] > 0; lab = np.zeros(len(code))
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        lab[heal] = np.where(np.isin(F.etype[heal, q], setA if which == "A" else setC), 1, -1)
    m = np.zeros((n, n))
    for a in range(len(code)):
        if unh[a]: m[ci[a], cj[a]] += lab[a]
    return np.sign(m), lab[unh].mean()
def metrics(s):
    m, psi = local_field(s)
    diff = (m != np.roll(m, 1, 0)).sum() + (m != np.roll(m, 1, 1)).sum() + (m != np.roll(np.roll(m, 1, 0), 1, 1)).sum() + (m != np.roll(np.roll(m, 1, 0), -1, 1)).sum()
    return abs(psi), diff / (4*n*n)
start = tile(np.load("scan3_min_d0.1372.npy" if which == "A" else "scan3_min_d0.2500.npy")[0], 4, 2, n)
out = []
for sd in range(nseed):
    s, _, _ = mc(start.copy(), J, Jl, *arr, 1.0, 1.0, 300, 11 + sd)          # disorder at high T
    s, _, _ = mc(s, J, Jl, *arr, 1.0, Th, 20, 211 + sd)                      # fast quench
    traj = [metrics(s)]
    for k in range(nhold // 50):
        s, _, _ = mc(s, J, Jl, *arr, Th, Th, 50, 5000 + 100*sd + k)
        traj.append(metrics(s))
    out.append(traj)
    tp = [traj[0], traj[len(traj)//10], traj[len(traj)//3], traj[-1]]
    print(f"{which} d={d} n={n} T={Th} seed {sd}: (|psi|, wall density) after quench {np.round(tp[0],2).tolist()}, "
          f"t=10% {np.round(tp[1],2).tolist()}, 33% {np.round(tp[2],2).tolist()}, end ({nhold} sweeps) {np.round(tp[3],2).tolist()}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"coarsenP1_{which}_d{d:.4f}_n{n}_T{Th}.json", "w"))
