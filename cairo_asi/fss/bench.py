import numpy as np, time, sys
sys.path.insert(0, '.')
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile
for n in [8, 12]:
    t=time.time(); L = CairoLattice(n, n, d=0.145); Jl = local_needle_coupling(L); t1=time.time()
    J = hybrid_coupling(L, 64.0); t2=time.time(); arr = mc_arrays(L); t3=time.time()
    s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)
    mc(s.copy(), J, Jl, *arr, 0.12, 0.12, 2, 1)
    t4=time.time(); s2,Etr,st = mc(s.copy(), J, Jl, *arr, 0.12, 0.12, 200, 1); t5=time.time()
    print(n, L.N, f"Jl {t1-t:.1f}s J {t2-t1:.1f}s arr {t3-t2:.1f}s sweep {(t5-t4)/200*1e3:.2f}ms", st, Etr[-1]/L.N, flush=True)
