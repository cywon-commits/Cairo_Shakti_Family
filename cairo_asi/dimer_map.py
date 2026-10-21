"""Spin state (vertex-local GS manifold) -> dimer covering of the pentagon square lattice.
Rule: every unhappy z3 vertex has exactly one z3-z4 spin whose flip would make it happy;
the two pentagons sharing that edge form a dimer (domino)."""
import numpy as np, itertools
from collections import Counter
from cairo_asi import *

class DimerMap:
    def __init__(self, L, J=None):
        self.L = L; self.J = L.coupling_matrix(None) if J is None else J
        self.e2f = {}
        for fi, fe in enumerate(L.face_edges):
            for k, o in fe: self.e2f.setdefault(k, []).append(fi)
        self.is34 = np.array([tuple(sorted(L.vtype[L.edges[k]])) == (3, 4) for k in range(L.N)])
        # per-vertex energy tables
        self.tab = []
        for v in range(len(L.vpos)):
            ks = [k for k, _ in L.incidence[v]]
            t = {c: sum(self.J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(len(ks)), 2))
                 for c in itertools.product([-1, 1], repeat=len(ks))}
            lv = sorted(set(np.round(list(t.values()), 4)))
            self.tab.append((ks, {c: lv.index(np.round(e, 4)) for c, e in t.items()}))
    def level(self, v, s):
        ks, t = self.tab[v]; return t[tuple(int(s[k]) for k in ks)]
    def dimers(self, s):
        L, out = self.L, []
        for v in np.where(L.vtype == 3)[0]:
            if self.level(v, s) != 1: continue
            for k, _ in L.incidence[v]:
                if not self.is34[k]: continue
                t = s.copy(); t[k] = -t[k]
                if self.level(v, t) == 0: out.append(k)
        return out
    def is_perfect(self, dm):
        c = Counter(f for k in dm for f in self.e2f[k])
        return len(c) == len(self.L.faces) and all(v == 1 for v in c.values())
