import numpy as np, itertools, sys
from collections import Counter
from cairo_asi import *
from shakti import ShaktiLattice
from count_degeneracy import enumerate_gs
from dimer_map import DimerMap

def setup_from(L):
    J = L.coupling_matrix(None)
    z4 = np.where(L.vtype == 4)[0]; ze, zc = [], []
    for v in z4:
        ks = [k for k, _ in L.incidence[v]]
        best = min(itertools.product([-1, 1], repeat=4),
                   key=lambda c: sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(4), 2)))
        ze.append(ks); zc.append(best)
    part = [k for k in range(L.N) if tuple(L.vtype[L.edges[k]]) == (3, 3)]
    P = []
    for p in part:
        row = []
        for v in L.edges[p]:
            o = [k for k, _ in L.incidence[v] if k != p]
            row += [o[0], o[1], J[p, o[0]], J[p, o[1]], J[o[0], o[1]]]
        P.append(row)
    return J, np.array(ze), np.array(zc, float), np.array(part), np.array(P)

def levels(L, J, v):
    ks = [k for k, _ in L.incidence[v]]
    t = [sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(len(ks)), 2))
         for c in itertools.product([-1, 1], repeat=len(ks))]
    lv = sorted(set(np.round(t, 4)))
    return lv, [sum(abs(x - l) < 1e-3 for x in t) for l in lv]

def perfect_matchings(L):
    """count perfect matchings of the face graph using only z3-z4 shared edges"""
    D = DimerMap(L); nf = len(L.faces)
    adj = [[] for _ in range(nf)]
    for k, fs in D.e2f.items():
        if D.is34[k]: adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])
    used = [False]*nf
    def dfs():
        try: u = used.index(False)
        except ValueError: return 1
        used[u] = True; tot = 0
        for w in adj[u]:
            if not used[w]: used[w] = True; tot += dfs(); used[w] = False
        used[u] = False; return tot
    return dfs(), Counter(len(a) for a in adj)

S = ShaktiLattice(1, 1); JS = S.coupling_matrix(None)
print("Shakti vertex levels (point dipoles, vertex-local):")
for z in (3, 4):
    lv, dg = levels(S, JS, np.where(S.vtype == z)[0][0]); print(f"  z={z}: {np.round(lv,3)} deg {dg}")
for nx, ny in [(1,1),(2,1),(2,2)]:
    S = ShaktiLattice(nx, ny); J, ze, zc, part, P = setup_from(S)
    E, W = enumerate_gs(S.N, ze, zc, P)
    Ez4 = sum(sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(4), 2)) for ks, c in zip(ze, zc))
    pm, deg = perfect_matchings(S)
    print(f"Shakti {nx}x{ny}: N={S.N}, E0/N={(E+Ez4)/S.N:.5f}, W={W}, face-graph degree {dict(deg)}, "
          f"perfect matchings (all sectors) {pm}, ln W per unhappy z3 = {np.log(W)/((S.vtype==3).sum()/2):.4f}", flush=True)
