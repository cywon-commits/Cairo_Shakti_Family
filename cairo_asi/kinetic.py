"""Single-island-flip (Metropolis) kinetics with the full dipolar energy, starting from multi-domain manifold states.
Usage: python kinetic.py d n which(A|C) Ts(comma) nsweep nseed"""
import numpy as np, sys, time, json
from cairo_asi import CairoLattice, anneal
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from selection_T import FastDimers
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile
d = float(sys.argv[1]); n = int(sys.argv[2]); which = sys.argv[3]; Ts = [float(x) for x in sys.argv[4].split(",")]
nsweep = int(sys.argv[5]); nseed = int(sys.argv[6]); t0 = time.time()
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
gs = tile(np.load("scan3_min_d0.1372.npy" if which == "A" else "scan3_min_d0.2500.npy")[0], 4, 2, n)
El0 = 0.5*gs@Jl@gs
# single-flip spectrum from the ground state
h = J @ gs; dE1 = np.sort(2*gs*h); print(f"{which} d={d}: single-flip dE from ground state: min {dE1[0]:.2f}, 10th pct {np.percentile(dE1,10):.2f}, median {np.median(dE1):.2f}")
cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n
setA, setC = [6, 7, 8, 9], [3, 5, 7, 9]
def metrics(s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    lev = F.lev[np.arange(len(code)), code]; unh = lev == 1; lab = np.zeros(len(code))
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        lab[heal] = np.where(np.isin(F.etype[heal, q], setA if which == "A" else setC), 1, -1)
    m = np.zeros((n, n))
    np.add.at(m, (ci[unh], cj[unh]), lab[unh]); m = np.sign(m)
    wall = ((m != np.roll(m, 1, 0)).sum() + (m != np.roll(m, 1, 1)).sum() + (m != np.roll(np.roll(m, 1, 0), 1, 1)).sum()
            + (m != np.roll(np.roll(m, 1, 0), -1, 1)).sum()) / (4*n*n)
    excess = (0.5*s@Jl@s - El0) / L.N            # vertex-local energy above the manifold, per spin
    return abs(lab[unh].mean()) if unh.any() else 0.0, wall, excess
res = {}
for T in Ts:
    rows = []
    for sd in range(nseed):
        s, _, _ = mc(gs.copy(), J, Jl, *arr, 1.0, 1.0, 300, 31 + sd)       # disordered dimer liquid
        s, _, _ = mc(s, J, Jl, *arr, 1.0, 0.1, 20, 331 + sd)               # quench inside manifold -> multi-domain
        traj = [metrics(s)]
        chunk = nsweep // 10
        for k in range(10):
            s, _ = anneal(J, s, T, T, chunk, 9000 + 100*sd + k)            # single-island flips only
            traj.append(metrics(s))
        rows.append(traj)
        print(f"  T={T}: seed {sd}: (|psi|, walls, excess local E/N) start {np.round(traj[0],3).tolist()} -> 10% {np.round(traj[1],3).tolist()} -> end {np.round(traj[-1],3).tolist()}  [{time.time()-t0:.0f}s]", flush=True)
    res[str(T)] = rows
json.dump(res, open(f"kinetic_{which}_d{d:.4f}_n{n}.json", "w"))
