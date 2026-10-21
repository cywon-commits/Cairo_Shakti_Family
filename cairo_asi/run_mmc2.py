import numpy as np, sys, time
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from manifold_mc import periods
from plot_phases import tile
d = float(sys.argv[1]); n = int(sys.argv[2]); rc = float(sys.argv[3]); nrun = int(sys.argv[4]); nsweep = int(sys.argv[5]); T0 = float(sys.argv[6])
t0 = time.time()
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, rc); arr = mc_arrays(L)
refs = {lab: min(0.5*tile(s0, 4, 2, n)@J@tile(s0, 4, 2, n)/L.N for s0 in np.load(f))
        for lab, f in [("A", "scan3_min_d0.1372.npy"), ("C", "scan3_min_d0.2500.npy")]} if n % 4 == 0 else {}
El0 = 0.5*tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)@Jl@tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)/L.N if n % 4 == 0 else None
print(f"d={d} n={n} N={L.N} rc={rc}  [{time.time()-t0:.0f}s]  tiled refs: " + str({k: round(v, 5) for k, v in refs.items()}), flush=True)
starts = [np.load("scan3_min_d0.1372.npy")[0], np.load("scan3_min_d0.2500.npy")[0]]
for r in range(nrun):
    s = tile(starts[r % 2], 4, 2, n) if n % 4 == 0 else None
    s, Etr, st = mc(s.copy(), J, Jl, *arr, T0, 0.002, nsweep, 7 + r)
    E = 0.5*s@J@s/L.N; El = 0.5*s@Jl@s/L.N
    print(f" run {r} start {'AC'[r%2]}: E/N {E:.5f}; local E/N {El:.5f} (should stay {El0:.5f}); moves tried {st[0]}, manifold-preserving {st[1]} "
          f"({100*st[1]/st[0]:.1f}%), accepted {st[2]}; E at T0: {Etr[0]/L.N:.4f}; periods {periods(s, n)[:3]}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(f"mmc2_d{d:.4f}_n{n}_r{r}.npy", s)
