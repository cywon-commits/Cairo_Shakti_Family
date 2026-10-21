"""Deformed Cairo family: anisotropy eps (= d1 - d2) and twist theta (each z3 pair rotated about its centre by theta,
same sense for both pair sets -> keeps the 90-degree rotation about z4, breaks the mirrors)."""
import numpy as np, sys, time
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from selection_T import FastDimers
from cairo2 import S0
def rot(v, t): c, s = np.cos(t), np.sin(t); return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])
class CairoLattice3(CairoLattice):
    def __init__(self, nx, ny, dbar, eps=0.0, theta=0.0):
        self.d1, self.d2, self.theta = dbar + eps/2, dbar - eps/2, theta
        super().__init__(nx, ny, d=dbar)
    def _build_vertices(self):
        d1, d2, t = self.d1, self.d2, self.theta
        c1, c2 = np.array([0.5, 0.0]), np.array([0.0, 0.5])
        o1, o2 = rot(np.array([d1, d1]), t), rot(np.array([-d2, d2]), t)
        basis = np.array([[0, 0], [0.5, 0.5], c1 + o1, c1 - o1, c2 + o2, c2 - o2])
        self.vtype = []; pos, idx = [], {}
        for i in range(self.nx):
            for j in range(self.ny):
                for b, p in enumerate(basis):
                    idx[(i, j, b)] = len(pos); pos.append(p + [i, j]); self.vtype.append(4 if b < 2 else 3)
        self.vpos = np.array(pos); self.vtype = np.array(self.vtype); self._vidx = idx
def energies(dbar, eps=0.0, theta=0.0, rc=64.0):
    L = CairoLattice3(4, 2, dbar, eps, theta); Jl = local_needle_coupling(L); J = hybrid_coupling(L, rc)
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl, S0)/L.N; E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N
    F = FastDimers(L, Jl); ps = np.array([F.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
    cls = {"A1": np.abs(pa + 1) < 1e-9, "A2": np.abs(pa - 1) < 1e-9, "C-": np.abs(pc + 1) < 1e-9, "C+": np.abs(pc - 1) < 1e-9}
    out = {k: float(E[m].min()) for k, m in cls.items()}
    out["_El"], out["_pa"], out["_pc"] = El, pa, pc
    return out
if __name__ == "__main__":
    r0 = energies(0.20); print("theta = 0: A1-A2 %.1e, C+ - C- %.1e" % (r0["A1"] - r0["A2"], r0["C+"] - r0["C-"]))
    r = energies(0.20, theta=0.01)
    El, pa, pc = r["_El"], r["_pa"], r["_pc"]
    X = np.vstack([pc, pa, np.ones_like(pc)]).T; coef, *_ = np.linalg.lstsq(X, El, rcond=None)
    print(f"theta = 0.01 rad, d_bar = 0.20: vertex energy on manifold = {coef[2]:.4f} + ({coef[0]:+.4f}) psi_C + ({coef[1]:+.4f}) psi_A ; "
          f"max residual {np.max(np.abs(El - X@coef)):.1e}")
    print(f"   A1 - A2 = {r['A1'] - r['A2']:+.2e} (should be 0: rotation kept); C+ - C- = {r['C+'] - r['C-']:+.4f}")
