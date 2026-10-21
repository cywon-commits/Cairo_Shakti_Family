"""Elementary manifold moves on top of a straight-wall state: energy cost and whether they displace the wall."""
import numpy as np, itertools, sys, json
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from selection_T import FastDimers
from manifold_mc2 import mc_arrays
d = float(sys.argv[1]); n = int(sys.argv[2]); fstate = sys.argv[3]
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); F = FastDimers(L, Jl)
z4e, z3end, part_of, nba, partners = mc_arrays(L)
s0 = np.load(fstate); E0 = 0.5*s0@J@s0; El0 = 0.5*s0@Jl@s0
def sublabels(s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    unh = F.lev[np.arange(len(code)), code] > 0; sub = np.zeros(len(code))
    for t in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - t))] == 0) & F.is34[:, t]
        sub[heal] = np.where(np.isin(F.etype[heal, t], [3, 5, 7, 9]), 1, -1)
    return sub, unh
sub0, unh0 = sublabels(s0)
cell = F.ks[:, 0] // 10; ci, cj = cell // n, cell % n
keys = {"i+j": (ci + cj) % n, "i-j": (ci - cj) % n}
# find stripe direction and wall lines
for kname, key in keys.items():
    prof = np.round([sub0[(key == v) & unh0].mean() for v in range(n)], 6)
    if np.all(np.isin(prof, [-1, 0, 1])) and len(set(prof)) > 1: break
line = key
walls = [v for v in range(n) if prof[v] == 0 or prof[(v+1) % n] * prof[v] < 0]
near_wall = set()
for v in range(n):
    if prof[v] == 0: near_wall |= {v, (v-1) % n, (v+1) % n}
    elif prof[(v+1) % n]*prof[v] < 0: near_wall |= {v, (v+1) % n}
print(f"state: stripe direction {kname}, profile {prof.tolist()}, lines next to walls {sorted(near_wall)}")
res = []
def try_move(verts):
    s = s0.copy(); F_ = []
    for v in verts:
        for k in z4e[v]: s[k] = -s[k]
    aff = sorted({part_of[z3end[k]] for v in verts for k in z4e[v]})
    opts = []
    for p in aff:
        hp = Jl[p] @ s - Jl[p, p]*s[p]
        opts.append([-1., 1.] if abs(hp) < 1e-9 else ([-1.] if hp > 0 else [1.]))
    out = []
    for combo in itertools.product(*opts):
        t = s.copy()
        for p, c in zip(aff, combo): t[p] = c
        if abs(0.5*t@Jl@t - El0) > 1e-7: continue
        sub, unh = sublabels(t)
        changed = np.where((sub != sub0) | (unh != unh0))[0]
        lines = set(line[changed].tolist())
        out.append(dict(dE=float(0.5*t@J@t - E0), nchg=len(changed), lines=sorted(lines), wall=bool(lines & near_wall) if lines else False,
                        prof_change=bool(len(changed) and not lines.isdisjoint(near_wall))))
    return out
nz = z4e.shape[0]
for v in range(nz):
    for r in try_move([v]): r["move"] = "1"; res.append(r)
    for w in nba[v]:
        if w > v:
            for r in try_move([v, w]): r["move"] = "2"; res.append(r)
valid = [r for r in res]
wallm = [r for r in valid if r["wall"]]; bulk = [r for r in valid if not r["wall"] and r["nchg"] > 0]; null = [r for r in valid if r["nchg"] == 0]
print(f"valid manifold moves: {len(valid)} (wall-touching {len(wallm)}, bulk {len(bulk)}, no dimer change {len(null)})")
for name, grp in [("wall-touching", wallm), ("bulk", bulk)]:
    if grp:
        es = sorted(r["dE"] for r in grp)
        print(f"  {name}: min dE {es[0]:.4f}, 5 lowest {np.round(es[:5],4).tolist()}, median {np.median(es):.4f}")
json.dump(res, open(f"kink_d{d:.4f}_n{n}.json", "w"))
