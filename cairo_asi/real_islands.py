"""Realistic islands on (deformed) Cairo lattices: stadium islands, width 100 nm, thickness 3 nm, Ms = 790 kA/m,
fixed vertex gaps (z3-z4 islands 86 nm each end, partner islands 22 nm), z3-z4 edge = 472 nm in the undeformed lattice.
Energies in kelvin."""
import numpy as np, itertools
from saccone_geometry import MU0, MS, E_CH, K_PER_EV, coupling_matrices
from cand_response import CairoLattice4
WID, THK, G34, GP, A_NM = 100.0, 3.0, 86.0, 22.0, 472.0
WID_L = None   # optional width of partner (long) islands
class RealArray:
    def __init__(self, nx, ny, dbar, **deform):
        ref = CairoLattice4(2, 2, dbar); is34r = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)])
        self.scale = A_NM/ref.elen[is34r].mean()                       # nm per lattice unit, fixed by the undeformed lattice
        nl_ref = {True: max(8, int(round((ref.elen[is34r].mean()*self.scale - 2*G34)/19.0))),
                  False: max(8, int(round((ref.elen[~is34r].mean()*self.scale - 2*GP)/19.0)))}   # fixed subdivision per island type
        self.L = L = CairoLattice4(nx, ny, dbar, **deform)
        is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)]); self.is34 = is34
        self.box = L.box*self.scale*1e-9; pts, dirs, own, wts = [], [], [], []
        for k in range(L.N):
            e = L.edir[k]; perp = np.array([-e[1], e[0]]); length = L.elen[k]*self.scale - 2*(G34 if is34[k] else GP)
            w_ = WID if (is34[k] or WID_L is None) else WID_L
            centre = L.epos[k]*self.scale; area = (length - w_)*w_ + np.pi*(w_/2)**2; mom = MS*area*THK*1e-27
            nl = nl_ref[bool(is34[k])]; cell = []
            for i in range(nl):
                for j in range(3):
                    x = -length/2 + (i + 0.5)*length/nl; y = -w_/2 + (j + 0.5)*w_/3
                    if abs(x) > (length - w_)/2:
                        cx = np.sign(x)*(length - w_)/2
                        if (x - cx)**2 + y**2 > (w_/2)**2: continue
                    cell.append(centre + x*e + y*perp)
            pts += cell; dirs += [e]*len(cell); own += [k]*len(cell); wts += [mom/len(cell)]*len(cell)
        self.P = np.array(pts)*1e-9; self.D = np.array(dirs); self.own = np.array(own); self.w = np.array(wts)
    def pair_energy(self, i, j):
        Pi, Pj = self.P[self.own == i], self.P[self.own == j]
        r = Pj[None, :, :] - Pi[:, None, :]; r -= self.box*np.round(r/self.box); d = np.linalg.norm(r, axis=2)
        mi, mj = self.D[self.own == i][0], self.D[self.own == j][0]
        E = MU0/(4*np.pi)*(mi@mj - 3*(r@mi)*(r@mj)/d**2)/d**3
        return float(np.sum(E*self.w[self.own == i][:, None]*self.w[self.own == j][None, :]))/E_CH
def vertex_J(A):
    """vertex-local couplings only (K)"""
    L = A.L; Jl = np.zeros((L.N, L.N))
    for inc in L.incidence:
        ks = [k for k, _ in inc]
        for x, y in itertools.combinations(ks, 2): Jl[x, y] = Jl[y, x] = A.pair_energy(x, y)
    return Jl*K_PER_EV
def full_J(A):
    Jl, J = coupling_matrices(A); return Jl*K_PER_EV, J*K_PER_EV
