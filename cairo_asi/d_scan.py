import numpy as np, itertools, json
from cairo_asi import CairoLattice, D_EQ, anneal, quench
from needle import needle_coupling
from manifold_select import setup, enum_manifold
nx, ny = 4, 2
L0 = CairoLattice(nx, ny, d=D_EQ); Jl0 = needle_coupling(L0, None)
ze, zc, part = setup(L0, Jl0)
_, S = enum_manifold(Jl0, ze, zc, part, 200000); S = S.astype(float)
keyset = {tuple(s.astype(int)) for s in S} | {tuple(-s.astype(int)) for s in S}
ds = np.round(np.linspace(D_EQ, 0.25, 13), 4); out = []
prev = None
for d in ds:
    L = CairoLattice(nx, ny, d=d); Jl = needle_coupling(L, None); Jd = needle_coupling(L, 6.0)
    # manifold still the same? recompute with this d's vertex model
    ze2, zc2, part2 = setup(L, Jl); _, S2 = enum_manifold(Jl, ze2, zc2, part2, 200000)
    same = len(S2) == len(S) and all(tuple(s) in keyset for s in S2.astype(int))
    v3 = np.where(L.vtype == 3)[0][0]; ks = [k for k, _ in L.incidence[v3]]
    t = sorted(set(np.round([sum(Jl[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(3), 2))
                            for c in itertools.product([-1, 1], repeat=3)], 3)))
    Ed = 0.5*np.einsum('ij,jk,ik->i', S, Jd, S)/L.N
    order = np.argsort(Ed); i = order[0]
    best = tuple(S[i].astype(int)); gap = Ed[order[np.argmax(Ed[order] > Ed[i] + 1e-7)]] - Ed[i]
    # SA check (full dipolar) for states below the manifold minimum
    esa = min(0.5*q@Jd@q/L.N for q in [quench(Jd, anneal(Jd, np.random.choice([-1., 1.], L.N), 8.0, 0.01, 2000, sd)[0]) for sd in range(20)])
    changed = prev is not None and best != prev and tuple(-np.array(best)) != prev
    print(f"d={d:.4f}: manifold unchanged {same}; z3 gap {t[1]-t[0]:.2f}; manifold min E/N {Ed[i]:.5f}; "
          f"gap to next {gap:.5f}; best SA {esa:.5f}; selected state changed: {changed}", flush=True)
    out.append(dict(d=float(d), Emin=float(Ed[i]), sa=float(esa), changed=bool(changed)))
    np.save(f"scan_best_d{d:.4f}.npy", np.array(best, float)); prev = best
json.dump(out, open("d_scan.json", "w"), indent=1)
