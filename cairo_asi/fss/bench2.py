import numpy as np, time, sys
sys.path.insert(0, '.')
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from manifold_mc2 import mc_arrays
from plot_phases import tile
from fss_mc import mc_keep, csr, couplings
d = 0.145
for n in [int(x) for x in sys.argv[1:]]:
    t = time.time(); L = CairoLattice(n, n, d=d); J, Jl = couplings(L, d, n); t1 = time.time()
    arr = mc_arrays(L); lp, li, lv = csr(Jl)
    for T in [0.12, 0.25]:
        s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); h = J @ s; hl = Jl @ s; Etr = np.empty(200)
        mc_keep(s, h, hl, 0.5*s@h, J, lp, li, lv, *arr, T, 2, 1, Etr)
        t2 = time.time(); E, nacc = mc_keep(s, h, hl, 0.5*s@h, J, lp, li, lv, *arr, T, 200, -1, Etr); t3 = time.time()
        print(n, L.N, f"J {t1-t:.0f}s  T={T} sweep {(t3-t2)/200*1e3:.2f}ms acc/sweep {nacc/200:.1f} E/N {E/L.N:.5f}", flush=True)
