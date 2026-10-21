"""Anisotropic Cairo family Cairo(d1, d2): C1-C2 partner pairs spaced by d1, C3-C4 pairs by d2.
Exact selection on the 4x2 manifold, classifying states into pure A1 (dimers on C1C2-side edges), A2 (C3C4-side), C (z4 sublattice)."""
import numpy as np, sys, json, time, os
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice, D_EQ
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from manifold_select import setup, enum_manifold
from selection_T import FastDimers

class CairoLattice2(CairoLattice):
    def __init__(self, nx, ny, d1, d2):
        self.d1, self.d2 = d1, d2
        super().__init__(nx, ny, d=0.5*(d1 + d2))
    def _build_vertices(self):
        d1, d2 = self.d1, self.d2
        basis = np.array([[0, 0], [0.5, 0.5], [0.5 + d1, d1], [0.5 - d1, -d1], [-d2, 0.5 + d2], [d2, 0.5 - d2]])
        self.vtype = []; pos, idx = [], {}
        for i in range(self.nx):
            for j in range(self.ny):
                for b, p in enumerate(basis):
                    idx[(i, j, b)] = len(pos); pos.append(p + [i, j]); self.vtype.append(4 if b < 2 else 3)
        self.vpos = np.array(pos); self.vtype = np.array(self.vtype); self._vidx = idx

_cache = "manifold_4x2_S0.npy"
if os.path.exists(_cache): S0 = np.load(_cache)
else:
    L0 = CairoLattice(4, 2, d=D_EQ); Jn = local_needle_coupling(L0); ze, zc, part = setup(L0, Jn)
    _, S = enum_manifold(Jn, ze, zc, part, 200000); S = S.astype(float); S0 = np.vstack([S, -S]); np.save(_cache, S0)

def energies(d1, d2, rc=64.0):
    L = CairoLattice2(4, 2, d1, d2); Jl = local_needle_coupling(L); J = hybrid_coupling(L, rc)
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl, S0)/L.N
    E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N
    F = FastDimers(L, Jl); ps = np.array([F.psi(s) for s in S0])
    pa, pc = ps[:, 0], ps[:, 1]
    cls = {"A1": (np.abs(pa + 1) < 1e-9), "A2": (np.abs(pa - 1) < 1e-9), "C": (np.abs(np.abs(pc) - 1) < 1e-9)}
    out = {k: float(E[m].min()) for k, m in cls.items()}
    out.update({"El_" + k: float(El[m][np.argmin(E[m])]) for k, m in cls.items()})
    out["_El"], out["_pa"] = El, pa
    imin = int(np.argmin(E)); out["global"] = float(E[imin]); out["global_psi"] = [float(pa[imin]), float(pc[imin])]
    return out
if __name__ == "__main__":
    print("orientation check at d1 = d2 = 0.137 (A1 and A2 must be degenerate):", energies(0.1372, 0.1372))
