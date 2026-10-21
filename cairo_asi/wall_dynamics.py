"""Do walls bend or move at finite T? Fixed-T manifold MC from a straight 2-wall state; snapshots every 10 sweeps."""
import numpy as np, sys
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from selection_T import FastDimers
from manifold_mc2 import mc_arrays, mc
d = float(sys.argv[1]); n = 8; s = np.load(sys.argv[2]); Ts = [float(x) for x in sys.argv[3].split(",")]
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n; keys = [(ci + cj) % n, (ci - cj) % n]
def profile(t):
    bits = (t[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    unh = F.lev[np.arange(len(code)), code] > 0; sub = np.zeros(len(code))
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        sub[heal] = np.where(np.isin(F.etype[heal, q], [3, 5, 7, 9]), 1, -1)
    best = None
    for key in keys:
        p = np.round([sub[(key == v) & unh].mean() for v in range(n)], 6)
        imp = int(np.sum(~np.isin(p, [-1, 0, 1])))            # lines with mixed labels = bent wall / defect
        if best is None or imp < best[0]: best = (imp, tuple(p))
    return best
E0 = 0.5*s@J@s; p0 = profile(s)
print(f"d={d}: start profile {p0[1]}")
for T in Ts:
    t = s.copy(); straight = moved = bent = 0; nsnap = 300; excess = []
    for k in range(nsnap):
        t, Etr, _ = mc(t, J, Jl, *arr, T, T, 10, 9000 + k)
        imp, p = profile(t); excess.append((0.5*t@J@t - E0))
        if imp > 0: bent += 1
        elif p != p0[1]: moved += 1
        else: straight += 1
    print(f"  T={T:.3f}: snapshots straight&unchanged {straight}, walls moved (still straight) {moved}, bent/mixed lines {bent}; "
          f"<E - E_start> = {np.mean(excess):.3f}", flush=True)
