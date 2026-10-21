"""Sublattice-parity order parameters of the dimer (domino) manifold on the pentagon chessboard.
Each chessboard plaquette (4-cycle of pentagons through z3-z4 edges) is centred on a Cairo object: a z4 vertex
(sublattice A or B) or a z3-z3 partner bond (orientation 1 = C1C2 pair, 2 = C3C4 pair).
Order parameters over flippable plaquettes (two parallel dimers):
  Psi_x, Psi_y : chessboard parities of the block origin
  Phi   = (N_z4 - N_partner)/N_flip          (+1 C, -1 A)   -- phase-distinguishing, symmetry-invariant
  psiC_ = (N_z4B - N_z4A)/N_z4              (C variants)
  psiA_ = (N_p2 - N_p1)/N_partner           (A variants)"""
import numpy as np, collections, itertools
from chessboard import chess_coords

class BlockOrder:
    def __init__(self, L, D):
        self.L, self.D = L, D
        self.pos, conf = chess_coords(L, D); assert conf == 0
        self.at = {c: f for f, c in self.pos.items()}
        # enumerate chessboard plaquettes as 4-cycles of faces, identify the Cairo object at their centre
        fv = [set(f) for f in L.faces]
        self.plaq = []                          # (faces, origin (x,y), kind)
        e2f = D.e2f
        adj = collections.defaultdict(dict)
        for k, (f1, f2) in e2f.items():
            if D.is34[k]: adj[f1][f2] = k; adj[f2][f1] = k
        seen = set()
        for a in adj:
            for b, c in itertools.combinations(adj[a], 2):
                for x in set(adj[b]) & set(adj[c]):
                    if x == a: continue
                    cyc = frozenset([a, b, c, x])
                    if cyc in seen: continue
                    seen.add(cyc)
                    common = set.intersection(*[fv[f] for f in cyc])
                    if common:
                        v = common.pop(); kind = "z4A" if (v % 6) == 0 else "z4B"          # basis index 0 = A, 1 = B
                    else:
                        # partner bond shared by exactly two of the four faces
                        kind = None
                        for f, g in itertools.combinations(cyc, 2):
                            sh = [k for k, (p, q) in e2f.items() if {p, q} == {f, g} and not D.is34[k]]
                            if sh: kind = "p1" if sh[0] % 10 == 0 else "p2"; break
                    cs = [self.pos[f] for f in cyc]
                    # origin in chessboard coordinates (skip plaquettes cut by the BFS seam)
                    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
                    if max(xs) - min(xs) != 1 or max(ys) - min(ys) != 1: continue
                    self.plaq.append((tuple(cyc), (min(xs), min(ys)), kind, [adj[f] for f in cyc]))
        self.cyc_edges = []
        for cyc, org, kind, _ in self.plaq:
            es = [adj[f][g] for f, g in itertools.combinations(cyc, 2) if g in adj[f]]
            self.cyc_edges.append(es)

    def flippable(self, s):
        dm = set(self.D.dimers(s)); out = []
        for (cyc, org, kind, _), es in zip(self.plaq, self.cyc_edges):
            on = [e for e in es if e in dm]
            if len(on) == 2 and not (set(self.D.e2f[on[0]]) & set(self.D.e2f[on[1]])): out.append((org, kind))
        return out

    def order(self, s):
        fl = self.flippable(s); n = len(fl)
        if n == 0: return dict(n=0)
        px = np.mean([(-1)**o[0] for o, _ in fl]); py = np.mean([(-1)**o[1] for o, _ in fl])
        cnt = collections.Counter(k for _, k in fl)
        nz4 = cnt["z4A"] + cnt["z4B"]; npb = cnt["p1"] + cnt["p2"]
        return dict(n=n, Psi_x=px, Psi_y=py, PsixPsiy=px*py, Phi=(nz4 - npb)/n,
                    psiC_=(cnt["z4B"] - cnt["z4A"])/nz4 if nz4 else 0.0, psiA_=(cnt["p2"] - cnt["p1"])/npb if npb else 0.0,
                    classes=dict(cnt))
