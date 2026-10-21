"""Proper 3-colourings of the Cairo dual (pentagon adjacency): R = CW-closed, B = CCW-closed, G = determined.
For every colouring the spins follow from the closed pentagons; check vertex states and dipolar energy."""
import numpy as np, itertools, sys, collections
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from manifold_select import setup, enum_manifold
from plot_phases import tile
def colourings(L, cap=200000):
    F = len(L.faces); adj = [set() for _ in range(F)]
    owner = collections.defaultdict(list)
    for f, fe in enumerate(L.face_edges):
        for k, o in fe: owner[k].append(f)
    for k, fs in owner.items():
        a, b = fs; adj[a].add(b); adj[b].add(a)
    order = sorted(range(F), key=lambda f: -len(adj[f])); col = [-1]*F; out = []
    def bt(i):
        if len(out) >= cap: return
        if i == F: out.append(col.copy()); return
        f = order[i]
        for c in range(3):
            if all(col[g] != c for g in adj[f]): col[f] = c; bt(i + 1); col[f] = -1
    bt(0); return out, owner
def spins(L, col, owner):
    s = np.zeros(L.N)
    for k, fs in owner.items():
        vals = []
        for f in fs:
            o = dict(L.face_edges[f])[k]
            if col[f] == 1: vals.append(o)        # B: CCW closed
            elif col[f] == 0: vals.append(-o)     # R: CW closed
        if not vals or (len(vals) == 2 and vals[0] != vals[1]): return None
        s[k] = vals[0]
    return s
def levels(L, Jl):
    out = {}
    for z in (3, 4):
        v = np.where(L.vtype == z)[0][0]; ks = [k for k, _ in L.incidence[v]]
        t = sorted(set(np.round([sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in itertools.product([-1, 1], repeat=len(ks))], 6)))
        out[z] = t
    return out
def vstate(L, Jl, s, lv):
    cnt = {3: collections.Counter(), 4: collections.Counter()}
    for v in range(len(L.vtype)):
        z = L.vtype[v]; ks = [k for k, _ in L.incidence[v]]
        e = round(sum(Jl[ks[x], ks[y]]*s[ks[x]]*s[ks[y]] for x, y in itertools.combinations(range(len(ks)), 2)), 6)
        cnt[z][lv[z].index(min(lv[z], key=lambda q: abs(q - e)))] += 1
    return cnt
nx, ny = int(sys.argv[1]), int(sys.argv[2])
for d, refname, reff in [(0.25, "C", "scan3_min_d0.2500.npy"), (0.1372, "A", "scan3_min_d0.1372.npy")]:
    L = CairoLattice(nx, ny, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); lv = levels(L, Jl)
    cols, owner = colourings(L)
    ref = np.load(reff)[0].astype(float) if (nx, ny) == (4, 2) else (tile(np.load(reff)[0], 4, 2, nx) if (nx == ny and nx % 4 == 0) else None)
    Eref = 0.5*ref@J@ref/L.N if ref is not None else None; Elref = 0.5*ref@Jl@ref/L.N if ref is not None else None
    res = []
    for col in cols:
        s = spins(L, col, owner)
        if s is None: continue
        chi = L.chirality(s); ncl = int(np.sum(np.abs(np.abs(chi) - 1) < 1e-9))
        vs = vstate(L, Jl, s, lv)
        res.append((0.5*s@J@s/L.N, 0.5*s@Jl@s/L.N, ncl, dict(vs[3]), dict(vs[4]), collections.Counter(col)))
    res.sort(key=lambda r: r[0])
    print(f"d={d} cell {nx}x{ny}: {len(cols)} proper 3-colourings, {len(res)} consistent; faces {len(L.faces)}")
    for r in res[:3]:
        line = f"   E/N {r[0]:+.4f}, local E/N {r[1]:+.4f}, closed pentagons {r[2]}/{len(L.faces)}, z3 levels {r[3]}, z4 levels {r[4]}, colours {dict(r[5])}"
        if Eref is not None: line += f" | vs {refname}: dE/N {r[0]-Eref:+.4f}, local dE/N {r[1]-Elref:+.4f}"
        print(line)
