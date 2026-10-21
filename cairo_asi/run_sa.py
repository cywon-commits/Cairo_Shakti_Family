"""SA on an n x n Cairo ASI; compare with tiled exact small-cell ground states."""
import numpy as np, time, json, sys
from cairo_asi import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nrun = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out = {}
for model, rc in [("local", None), ("dip6", 6.0)]:
    L = CairoLattice(n, n); J = L.coupling_matrix(rc)
    # tiled candidates from exact 1x1 and 2x1 / 1x2 periodic ground states
    cand = {}
    for cx, cy in [(1,1),(2,1),(1,2),(2,2)]:
        Ls = CairoLattice(cx, cy); Js = Ls.coupling_matrix(rc)
        if Ls.N <= 30:
            _, s0 = exact_ground_state_fast(Js)
        else:   # 2x2: best of many anneals on the small cell
            best = None
            for sd in range(200):
                s1, _ = anneal(Js, np.random.choice([-1.,1.], Ls.N), 5.0, 0.01, 400, sd); s1 = quench(Js, s1)
                e1 = 0.5*s1@Js@s1
                if best is None or e1 < best[0]: best = (e1, s1.copy())
            s0 = best[1]
        # tile: edge k in cell (i,j) basis b maps to small-cell edge (i%cx, j%cy, b)
        s = np.empty(L.N)
        for i in range(n):
            for j in range(n):
                for b in range(10):
                    s[(i*n + j)*10 + b] = s0[((i % cx)*cy + (j % cy))*10 + b]
        cand[f"tile{cx}x{cy}"] = float(0.5*s@J@s/L.N)
    # simulated annealing
    E, stats = [], []
    t = time.time()
    for sd in range(nrun):
        s, tr = anneal(J, np.random.choice([-1.,1.], L.N), 5.0, 0.01, 3000, sd)
        s = quench(J, s)
        E.append(0.5*s@J@s/L.N); stats.append(L.vertex_stats(s))
        if sd == int(np.argmin(E)): np.save(f"sa_best_{model}_{n}.npy", s)
    E = np.array(E)
    print(f"\n=== {model}, {n}x{n} cells, N={L.N} ({time.time()-t:.0f}s) ===")
    print("tiled exact-cell candidates E/N:", {k: round(v,5) for k,v in cand.items()})
    print(f"SA ({nrun} runs): min {E.min():.5f}  mean {E.mean():.5f}  std {E.std():.5f}")
    print("SA best vertex stats:", stats[int(np.argmin(E))])
    out[model] = dict(cand=cand, sa=E.tolist())
json.dump(out, open(f"sa_results_{n}.json","w"), indent=1)
