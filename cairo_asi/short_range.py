"""Short-range dimer-label correlations in frozen single-flip states vs the random dimer liquid.
g_A: orientation label (C3/C4-pair edges +1, C1/C2-pair -1); g_C: sublattice label (z3-B +1, z3-A -1);
averaged over pairs of dimers in the same or neighbouring unit cells (8 neighbours)."""
import numpy as np, glob
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from selection_T import FastDimers
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile
import needle, os, sys
needle.GAP = float(os.environ.get('GAP', 0.0)); TAG = os.environ.get('TAG', 'needle_d')
n = 8
def corr(L, F, s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    unh = F.lev[np.arange(len(code)), code] == 1; et = np.full(len(code), -1)
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        et[heal] = F.etype[heal, q]
    ok = et >= 0; cell = (F.ks[:, 0] // 10)[ok]; e = et[ok]
    oA = np.where(np.isin(e, [6, 7, 8, 9]), 1, -1); oC = np.where(np.isin(e, [3, 5, 7, 9]), 1, -1)
    ci, cj = cell // n, cell % n
    di = (ci[:, None] - ci[None, :] + n//2) % n - n//2; dj = (cj[:, None] - cj[None, :] + n//2) % n - n//2
    near = (np.abs(di) <= 1) & (np.abs(dj) <= 1) & ~np.eye(len(e), dtype=bool)
    return (oA[:, None]*oA[None, :])[near].mean(), (oC[:, None]*oC[None, :])[near].mean()
for d, which in [(0.1372, "A"), (0.25, "C")]:
    L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 32.0); F = FastDimers(L, Jl); arr = mc_arrays(L)
    gs = tile(np.load("scan3_min_d0.1372.npy" if which == "A" else "scan3_min_d0.2500.npy")[0], 4, 2, n)
    # random dimer liquid baseline: manifold MC at very high T
    base = []
    s = gs.copy()
    for k in range(30):
        s, _, _ = mc(s, J, Jl, *arr, 1e4, 1e4, 20, 50 + k); base.append(corr(L, F, s))
    frozen = [corr(L, F, np.load(f)) for f in sorted(glob.glob(f"coolstate_{which}_{TAG}{d:.4f}_n8_spd20000_s*.npy"))]
    b = np.mean(base, 0); fz = np.array(frozen)
    print(f"d={d} ({'Cairo' if which=='A' else 'Shakti'}): ground state (g_A, g_C) = {np.round(corr(L, F, gs), 3)}; random dimer liquid {np.round(b, 3)} "
          f"(sd {np.round(np.std(base, 0), 3)}); frozen single-flip states: " + ", ".join(f"({x:.3f}, {y:.3f})" for x, y in fz) +
          f"; mean {np.round(fz.mean(0), 3)}")
