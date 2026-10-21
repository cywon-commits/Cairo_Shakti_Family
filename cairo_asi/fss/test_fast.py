import numpy as np, time, sys
sys.path.insert(0, '.')
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile
from fss_mc import hybrid_coupling_fast, mc_keep, csr
n = 8; L = CairoLattice(n, n, d=0.145); Jl = local_needle_coupling(L)
t = time.time(); J0 = hybrid_coupling(L, 64.0); t1 = time.time(); J1 = hybrid_coupling_fast(L, 64.0); t2 = time.time()
print("J max diff", np.abs(J0 - J1).max(), "rel", np.abs(J0-J1).max()/np.abs(J0).max(), f"{t1-t:.1f}s vs {t2-t1:.1f}s")
arr = mc_arrays(L); lp, li, lv = csr(Jl)
for T in [0.08, 0.12, 0.3]:
    s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)
    sa, Ea, _ = mc(s.copy(), J0, Jl, *arr, T, T, 300, 7)
    s2 = s.copy(); h = J0 @ s2; hl = Jl @ s2; Etr = np.empty(300)
    E, nacc = mc_keep(s2, h, hl, 0.5*s2@h, J0, lp, li, lv, *arr, T, 300, 7, Etr)
    print(T, "same state:", np.array_equal(sa, s2), "E diff", abs(Ea[-1]-E), "drift", abs(E-0.5*s2@J0@s2), "h drift", np.abs(h-J0@s2).max(), nacc)
