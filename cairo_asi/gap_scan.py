"""(d, g) scan: manifold validity, A/C selection (4x2 exact, rc = 64a), energy scales. Usage: python gap_scan.py gs ds"""
import numpy as np, itertools, sys, json, os, time
import needle
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from manifold_select import setup, enum_manifold
gaps = [float(x) for x in sys.argv[1].split(",")]; ds = [float(x) for x in sys.argv[2].split(",")]
fn = "gap_scan.json"; out = json.load(open(fn)) if os.path.exists(fn) else {}
needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0)
ze, zc, part = setup(L0, Jn); _, S0 = enum_manifold(Jn, ze, zc, part, 200000); S0 = S0.astype(float); S0 = np.vstack([S0, -S0])
A = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.1372.npy")}; C = {tuple(r.astype(int)) for r in np.load("scan3_min_d0.2500.npy")}
iA = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in A]; iC = [i for i, s in enumerate(S0) if tuple(s.astype(int)) in C]
for g in gaps:
    for d in ds:
        t0 = time.time(); needle.GAP = g
        L = CairoLattice(4, 2, d=d); Jl = local_needle_coupling(L)
        lv = {}
        for z in (3, 4):
            v = np.where(L.vtype == z)[0][0]; ks = [k for k, _ in L.incidence[v]]
            t = [sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))]
            u = sorted(set(np.round(t, 5))); lv[z] = (u, [int(sum(abs(np.array(t) - q) < 1e-4)) for q in u])
        El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl, S0)
        valid = lv[3][1][:2] == [2, 4] and lv[4][1][0] == 2 and (El.max() - El.min()) < 1e-6
        rec = dict(z3gap=float(lv[3][0][1] - lv[3][0][0]), z3deg=lv[3][1], valid=bool(valid))
        if valid:
            J = hybrid_coupling(L, 64.0); E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N
            imin = int(np.argmin(E)); lab = "A" if imin in iA else ("C" if imin in iC else "other")
            s = S0[imin]; cost = np.sort(-2*s*(J @ s))
            rec.update(sel=lab, EA_minus_EC=float(E[iA].min() - E[iC].min()), spread=float(E.max() - E.min()),
                       flip_min=float(cost[0]), flip_p10=float(np.percentile(cost, 10)))
        out[f"{g:.2f}_{d:.4f}"] = rec; json.dump(out, open(fn, "w"), indent=1)
        print(f"g={g:.2f} d={d:.4f}: z3 deg {rec['z3deg']} gap {rec['z3gap']:.3f} valid {valid}" +
              (f" | selected {rec['sel']}, E_A-E_C {rec['EA_minus_EC']:+.4f}/spin, manifold spread {rec['spread']:.3f}, single-flip min {rec['flip_min']:.2f} p10 {rec['flip_p10']:.2f}" if valid else "")
              + f"  [{time.time()-t0:.1f}s]", flush=True)
needle.GAP = 0.0
