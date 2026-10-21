"""Domain-wall tension in the C region: low-T manifold MC states consist of diagonal stripes with pure sublattice
choice (+-1) separated by wall lines (choice 0). sigma = (E_state - E_pureC) / (#wall lines x n cells), per cell of wall."""
import numpy as np, sys, json, os, time
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from selection_T import FastDimers
from plot_phases import tile
n = 8; nrun = int(sys.argv[2]); ds = [float(x) for x in sys.argv[1].split(",")]
fn = "wall_tension.json"; out = json.load(open(fn)) if os.path.exists(fn) else {}
for d in ds:
    t0 = time.time()
    L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); arr = mc_arrays(L); F = FastDimers(L, Jl)
    EC = min(0.5*tile(s0, 4, 2, n)@J@tile(s0, 4, 2, n) for s0 in np.load("scan3_min_d0.2500.npy"))
    cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n
    pts = []
    for r in range(nrun):
        s = tile(np.load(["scan3_min_d0.1372.npy", "scan3_min_d0.2500.npy"][r % 2])[0], 4, 2, n)
        s, _, _ = mc(s, J, Jl, *arr, 3.0, 0.002, 12000, 700 + r)
        bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
        unh = F.lev[np.arange(len(code)), code] > 0; sub = np.zeros(len(code))
        for t in range(3):
            heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - t))] == 0) & F.is34[:, t]
            sub[heal] = np.where(np.isin(F.etype[heal, t], [3, 5, 7, 9]), 1, -1)
        best = None
        for key in ((ci + cj) % n, (ci - cj) % n):
            prof = np.round(np.array([sub[(key == v) & unh].mean() for v in range(n)]), 6)
            if not np.all(np.isin(prof, [-1, 0, 1])): continue
            nz = int(np.sum(prof == 0)); nzv = prof[prof != 0]
            ns = 0
            for t in range(n):                                   # sharp walls: adjacent +-1 lines with opposite sign
                if prof[t] != 0 and prof[(t + 1) % n] != 0 and prof[t] != prof[(t + 1) % n]: ns += 1
            if best is None or nz + ns < sum(best): best = (nz, ns)
        E = 0.5*s@J@s
        pts.append(dict(dE=float(E - EC), walls=best, psiC=float(sub[unh].mean()))); np.save(f"wt_d{d:.4f}_r{r}.npy", s)
    X, y = [], []
    for p in pts:
        if p["walls"] is not None: X.append([p["walls"][0]*n, p["walls"][1]*n]); y.append(p["dE"])
    X, y = np.array(X, float), np.array(y)
    cols = [c for c in range(2) if X[:, c].any()]
    sig = [None, None]
    if cols:
        sol, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        for c, v in zip(cols, sol): sig[c] = float(v)
        resid = float(np.abs(X[:, cols] @ sol - y).max())
    else: resid = 0.0
    out[f"{d:.4f}"] = dict(points=pts, sigma_wide=sig[0], sigma_sharp=sig[1], max_resid=resid)
    json.dump(out, open(fn, "w"), indent=1)
    print(f"d={d:.4f}: (wide, sharp, dE): {[(p['walls'][0], p['walls'][1], round(p['dE'],4)) if p['walls'] else ('?', round(p['dE'],4)) for p in pts]}"
          f" -> sigma_wide {sig[0] if sig[0] is None else round(sig[0],5)}, sigma_sharp {sig[1] if sig[1] is None else round(sig[1],5)} per wall-cell, fit resid {resid:.4f}  [{time.time()-t0:.0f}s]", flush=True)
