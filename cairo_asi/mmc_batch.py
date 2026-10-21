import numpy as np, sys, time, json, os
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from manifold_mc2 import mc_arrays, mc
from manifold_mc import periods
from plot_phases import tile
n = int(sys.argv[1]); nrun = int(sys.argv[2]); nsweep = int(sys.argv[3]); T0 = float(sys.argv[4])
ds = [float(x) for x in sys.argv[5].split(",")]
refs_files = [("A", "scan3_min_d0.1372.npy"), ("C", "scan3_min_d0.2500.npy")]
out = json.load(open(f"mmc_batch_n{n}.json")) if os.path.exists(f"mmc_batch_n{n}.json") else {}
for d in ds:
    t0 = time.time(); rc = float(os.environ.get("RC", 32.0 if d < 0.34 else 128.0))
    L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, rc); arr = mc_arrays(L)
    refs = {lab: min(0.5*tile(s0, 4, 2, n)@J@tile(s0, 4, 2, n)/L.N for s0 in np.load(f)) for lab, f in refs_files}
    best = None; Es = []
    for r in range(nrun):
        s0 = tile(np.load(refs_files[r % 2][1])[0], 4, 2, n)
        s, Etr, st = mc(s0.copy(), J, Jl, *arr, T0, 0.002, nsweep, 1000 + r)
        E = 0.5*s@J@s/L.N; Es.append(E)
        if best is None or E < best[0]: best = (E, s.copy())
    np.save(f"mmcb_n{n}_d{d:.4f}.npy", best[1])
    lab = min(refs, key=refs.get); gapref = best[0] - refs[lab]
    verdict = f"= tiled {lab}" if abs(gapref) < 1e-6 else (f"BELOW tiled {lab} by {-gapref:.5f}" if gapref < 0 else f"above tiled {lab} by {gapref:.5f}")
    print(f"n={n} d={d:.4f} rc={rc:g}: MC best {best[0]:.5f} (runs {np.round(Es,5).tolist()}); tiled A {refs['A']:.5f}, C {refs['C']:.5f} -> {verdict}; "
          f"periods {periods(best[1], n)[:2]}  [{time.time()-t0:.0f}s]", flush=True)
    out[f"{d:.4f}"] = dict(best=best[0], runs=Es, A=refs["A"], C=refs["C"])
    json.dump(out, open(f"mmc_batch_n{n}.json", "w"), indent=1)
