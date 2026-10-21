import numpy as np, sys, time, json
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling
from manifold_mc import lattice_arrays, loop_mc, periods
from plot_phases import tile

d = float(sys.argv[1]); n = int(sys.argv[2]); rc = float(sys.argv[3]); nrun = int(sys.argv[4]); nsweep = int(sys.argv[5])
t0 = time.time()
L = CairoLattice(n, n, d=d); Jl = needle_coupling(L, None); J = hybrid_coupling(L, rc)
ia = lattice_arrays(L)
print(f"d={d} n={n} N={L.N} couplings {time.time()-t0:.0f}s", flush=True)
# reference: best 4x2-periodic states (from manifold enumeration) tiled, if n divisible by 4
refs = {}
import glob
for lab, f in [("A", "scan3_min_d0.1372.npy"), ("C", "scan3_min_d0.2500.npy")]:
    if n % 4 == 0:
        es = []
        for s0 in np.load(f):
            for rot in (0,):
                s = tile(s0, 4, 2, n); es.append(0.5*s@J@s/L.N)
        refs[lab] = min(es)
print("tiled reference E/N:", {k: round(v, 5) for k, v in refs.items()})
start = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n) if n % 4 == 0 else None
res = []
for r in range(nrun):
    if start is None:
        from plot_phases import tile as tl
        s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, 4)  # placeholder not used
    s = tile(np.load(["scan3_min_d0.1372.npy", "scan3_min_d0.2500.npy"][r % 2])[0], 4, 2, n) if n % 4 == 0 else tile2(n)
    s, Etr, tried, lok, acc = loop_mc(s.copy(), J, Jl, *ia, 3.0, 0.005, nsweep, 100 + r, 4 * L.N)
    El = 0.5*s@Jl@s/L.N; E = 0.5*s@J@s/L.N
    per = periods(s, n)
    res.append(E)
    print(f"run {r} (start {'AC'[r%2]}): final E/N {E:.5f}, local E/N {El:.5f}, loops tried {tried}, local-preserving {lok}, accepted {acc}, "
          f"translation periods {per[:4]}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(f"mmc_d{d:.4f}_n{n}_r{r}.npy", s)
