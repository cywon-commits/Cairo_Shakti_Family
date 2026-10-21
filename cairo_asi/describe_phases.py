import numpy as np, itertools
from collections import Counter
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice
from dimer_map import DimerMap
from plot_phases import tile, cases
n = 8
for name, d, f in cases:
    s = tile(np.load(f)[0], 4, 2, n)
    L = CairoLattice(n, n, d=d); D = DimerMap(L, needle_coupling(L, None)); dm = set(D.dimers(s)); chi = L.chirality(s)
    # 1) which of the 8 z3-z4 edge types (topological, same for all d) carry dimers
    etype = Counter((k % 10) for k in dm)
    # 2) square faces of the dimer lattice (4-cycles of pentagons via z3-z4 edges); flippable = 2 parallel dimers
    adj = {}
    for k, (f1, f2) in D.e2f.items():
        if D.is34[k]: adj.setdefault(f1, {})[f2] = k; adj.setdefault(f2, {})[f1] = k
    cycles = set()
    for a in adj:
        for b, c in itertools.combinations(adj[a], 2):
            for x in set(adj[b]) & set(adj[c]):
                if x != a: cycles.add(frozenset([a, b, c, x]))
    flip = 0
    for cyc in cycles:
        es = [adj[p][q] for p, q in itertools.combinations(cyc, 2) if q in adj[p]]
        if len(es) == 4 and sum(e in dm for e in es) == 2:
            # the two dimer edges must be disjoint (opposite sides)
            used = [set(D.e2f[e]) for e in es if e in dm]
            if not used[0] & used[1]: flip += 1
    # 3) same-sign chirality clusters (connected via any shared edge)
    parent = list(range(len(chi)))
    def fnd(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k, (f1, f2) in D.e2f.items():
        if np.sign(chi[f1]) == np.sign(chi[f2]): parent[fnd(f1)] = fnd(f2)
    cl = Counter(Counter(fnd(x) for x in range(len(chi))).values())
    # 4) net moment and net "toroidal" (sum of chirality)
    m = (L.edir * s[:, None] * L.elen[:, None]).sum(0) / L.N
    print(f"{name}: dimer edge types {dict(sorted(etype.items()))}; square plaquettes {len(cycles)}, flippable {flip}; "
          f"same-sign clusters (size:count) {dict(cl)}; net moment/spin {np.round(m,3)}; sum chi {chi.sum():.1f}")
