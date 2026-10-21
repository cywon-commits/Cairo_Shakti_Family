"""Inverse of the spin -> dimer map on the manifold: each vertex's allowed configurations given its status
(z4: type I; z3 happy; z3 unhappy healed by a given z3-z4 edge) form one time-reversal pair, so one spin fixes everything."""
import numpy as np, itertools
from collections import deque
def vertex_tables(L, Jl):
    tabs = []
    for v in range(len(L.vtype)):
        ks = [k for k, _ in L.incidence[v]]
        confs = list(itertools.product([-1, 1], repeat=len(ks)))
        E = [sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(len(ks)), 2)) for c in confs]
        lv = {e: r for r, e in enumerate(sorted(set(np.round(E, 6))))}; level = [lv[round(e, 6)] for e in E]
        tabs.append((ks, confs, level))
    return tabs
def spins_from_dimers(L, Jl, dimer_edges, tabs=None):
    tabs = tabs or vertex_tables(L, Jl); dset = set(dimer_edges)
    is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)])
    allowed = []
    for v, (ks, confs, level) in enumerate(tabs):
        if L.vtype[v] == 4: ok = [c for c, l in zip(confs, level) if l == 0]
        else:
            heal = [k for k in ks if k in dset]
            if not heal: ok = [c for c, l in zip(confs, level) if l == 0]
            else:
                q = ks.index(heal[0]); ok = []
                for c, l in zip(confs, level):
                    if l != 1: continue
                    c2 = list(c); c2[q] = -c2[q]
                    if level[confs.index(tuple(c2))] == 0: ok.append(c)
        allowed.append((ks, ok))
    s = np.zeros(L.N); edge_vs = [[] for _ in range(L.N)]
    for v, (ks, ok) in enumerate(allowed):
        for k in ks: edge_vs[k].append(v)
    s[0] = 1; Q = deque([0]); done_v = set()
    while Q:
        k = Q.popleft()
        for v in edge_vs[k]:
            if v in done_v: continue
            ks, ok = allowed[v]; q = ks.index(k)
            match = [c for c in ok if c[q] == s[k]]
            if len(match) != 1: return None
            for kk, val in zip(ks, match[0]):
                if s[kk] == 0: s[kk] = val; Q.append(kk)
                elif s[kk] != val: return None
            done_v.add(v)
    return s if np.all(s != 0) else None
if __name__ == "__main__":
    import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
    from cairo_asi import CairoLattice
    from hybrid_coupling import local_needle_coupling
    from dimer_map import DimerMap
    from cand_response import STATES
    L = CairoLattice(12, 4, d=0.20); Jl = local_needle_coupling(L); D = DimerMap(L, Jl); tabs = vertex_tables(L, Jl)
    for k, s in STATES.items():
        r = spins_from_dimers(L, Jl, D.dimers(s), tabs)
        print(k, "round trip exact (up to global flip):", r is not None and (np.array_equal(r, s) or np.array_equal(r, -s)))
