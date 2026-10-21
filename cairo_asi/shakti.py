"""Shakti lattice (SH version: the central long island is ONE Ising spin), same framework as CairoLattice.
Unit: short-island length = 1. Unit cell 4x4 = 2x2 blocks of 2x2 squares; block (p,q) has a horizontal
long island if p+q even, vertical otherwise. z4 = block corners, z3 = block side midpoints."""
import numpy as np
from cairo_asi import CairoLattice

class ShaktiLattice(CairoLattice):
    def __init__(self, nx, ny):
        self.nx, self.ny, self.d = nx, ny, None
        self.box = np.array([4*nx, 4*ny], float)
        self._build_vertices(); self._build_edges(); self._build_faces()

    def _build_vertices(self):
        pos, vt, idx = [], [], {}
        W, H = 2*self.nx, 2*self.ny                     # block grid
        for p in range(W):
            for q in range(H):
                for name, off, z in [("c", (0, 0), 4), ("b", (1, 0), 3), ("l", (0, 1), 3)]:
                    idx[(p, q, name)] = len(pos); pos.append((2*p + off[0], 2*q + off[1])); vt.append(z)
        self.vpos, self.vtype, self._vidx, self.W, self.H = np.array(pos, float), np.array(vt), idx, W, H

    def _v(self, p, q, name):
        return self._vidx[(p % self.W, q % self.H, name)]

    def _build_edges(self):
        e = []
        for p in range(self.W):
            for q in range(self.H):
                # short islands, oriented z3 -> z4
                e.append((self._v(p, q, "b"), self._v(p, q, "c")))
                e.append((self._v(p, q, "b"), self._v(p + 1, q, "c")))
                e.append((self._v(p, q, "l"), self._v(p, q, "c")))
                e.append((self._v(p, q, "l"), self._v(p, q + 1, "c")))
                # long island
                if (p + q) % 2 == 0:
                    e.append((self._v(p, q, "l"), self._v(p + 1, q, "l")))
                else:
                    e.append((self._v(p, q, "b"), self._v(p, q + 1, "b")))
        self.edges = np.array(e)
        u, v = self.edges[:, 0], self.edges[:, 1]
        vec = self.mic(self.vpos[v] - self.vpos[u])
        self.elen = np.linalg.norm(vec, axis=1); self.edir = vec / self.elen[:, None]
        self.epos = self.vpos[u] + 0.5*vec; self.N = len(e)
        inc = [[] for _ in range(len(self.vpos))]
        for k, (a, b) in enumerate(self.edges):
            inc[a].append((k, -1)); inc[b].append((k, +1))
        self.incidence = inc

    def coupling_matrix(self, rc=None):
        """Vertex-local model with each island's END segment at the vertex: at vertex v, island k is a unit
        dipole at distance 1/2 (half a short island) from v along the island. Long islands thus couple at both
        ends like short ones (equivalent to the mSH picture with a rigid central z=2 vertex)."""
        assert rc is None, "only the vertex-local model is implemented for Shakti"
        N = self.N; J = np.zeros((N, N))
        for v, inc in enumerate(self.incidence):
            pts, dirs, ks = [], [], []
            for k, sg in inc:
                u = self.edges[k][0] if self.edges[k][1] == v else self.edges[k][1]
                out = self.mic(self.vpos[u] - self.vpos[v]); out /= np.linalg.norm(out)
                pts.append(self.vpos[v] + 0.5*out); dirs.append(self.edir[k]); ks.append(k)
            for a in range(len(ks)):
                for b in range(a + 1, len(ks)):
                    r = pts[b] - pts[a]; d = np.linalg.norm(r)
                    J[ks[a], ks[b]] += (dirs[a] @ dirs[b] - 3*(dirs[a] @ r)*(dirs[b] @ r)/d**2)/d**3
                    J[ks[b], ks[a]] = J[ks[a], ks[b]]
        return J
