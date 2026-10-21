"""Saccone et al., PRMaterials 3, 104402 (2019) geometry in physical units.
b = 344 nm (z3-z3 partner edge), a = 472 nm (z3-z4 edge); islands 300 x 100 x 3 nm, Ms = 790 kA/m.
Partner islands centred on their edge; z3-z4 islands centred at distance c/2 from the z4 vertex.
Each island = grid of point sub-dipoles (nl along length x nw across width). Energies in eV."""
import numpy as np
from scipy.optimize import brentq
from cairo_asi import CairoLattice
MU0 = 4e-7*np.pi; MS = 790e3; T_ISL = 3e-9; LEN = 300e-9; WID = 100e-9; E_CH = 1.602176634e-19
A_NM, B_NM = 472.0, 344.0

def d_from_ratio(r=B_NM/A_NM):
    f = lambda d: 2*np.sqrt(2)*d/np.sqrt((0.5 - d)**2 + d**2) - r
    return brentq(f, 1e-4, 0.45)

class SacconeArray:
    def __init__(self, nx, ny, c_nm, nl=24, nw=3, d=None, gap_partner=None):
        self.d = d_from_ratio() if d is None else d
        self.L = CairoLattice(nx, ny, d=self.d); L = self.L
        is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)])
        self.is34 = is34; self.scale = A_NM / L.elen[is34].mean()          # nm per lattice unit
        self.box = L.box*self.scale*1e-9
        pts, dirs, own = [], [], []
        area = (LEN - WID)*WID + np.pi*(WID/2)**2; mom = MS*area*T_ISL      # stadium volume x Ms
        for k in range(L.N):
            u, v = L.edges[k]; e = L.edir[k]; perp = np.array([-e[1], e[0]])
            if is34[k]:
                w = u if L.vtype[u] == 4 else v                             # z4 end
                sgn = 1.0 if w == u else -1.0                               # edir points u->v
                centre = L.vpos[w]*self.scale + sgn*e*(c_nm/2)
            else:
                centre = L.epos[k]*self.scale
            for i in range(nl):
                for j in range(nw):
                    x = (-LEN/2 + (i + 0.5)*LEN/nl)*1e9; y = (-WID/2 + (j + 0.5)*WID/nw)*1e9
                    if abs(x) > (LEN - WID)/2*1e9:                          # stadium caps
                        cx = np.sign(x)*(LEN - WID)/2*1e9
                        if (x - cx)**2 + y**2 > (WID/2*1e9)**2: continue
                    pts.append(centre + x*e + y*perp); dirs.append(e); own.append(k)
        self.P = np.array(pts)*1e-9; self.D = np.array(dirs); self.own = np.array(own)
        cnt = np.bincount(self.own, minlength=L.N); self.w = mom / cnt[self.own]  # moment per sub-dipole

    def pair_energy(self, i, j):
        """interaction energy (eV) of islands i, j with sigma_i = sigma_j = +1 (minimum image)"""
        Pi, Pj = self.P[self.own == i], self.P[self.own == j]
        r = Pj[None, :, :] - Pi[:, None, :]; r -= self.box*np.round(r/self.box)
        d = np.linalg.norm(r, axis=2); mi, mj = self.D[i if False else self.own == i][0], self.D[self.own == j][0]
        E = MU0/(4*np.pi)*(mi@mj - 3*(r@mi)*(r@mj)/d**2)/d**3
        wi, wj = self.w[self.own == i], self.w[self.own == j]
        return float(np.sum(E*wi[:, None]*wj[None, :]))/E_CH

    def couplings_J(self):
        """J1: partner-(z3z4) at a z3 vertex; J3: (z3z4)-(z3z4) at a z3 vertex;
        z4 vertex: perpendicular pair and collinear pair (distance c)."""
        L = self.L; v3 = np.where(L.vtype == 3)[0][0]; ks = [k for k, _ in L.incidence[v3]]
        p = [k for k in ks if not self.is34[k]][0]; e = [k for k in ks if self.is34[k]]
        J1 = abs(self.pair_energy(p, e[0])); J3 = abs(self.pair_energy(e[0], e[1]))
        v4 = np.where(L.vtype == 4)[0][0]; k4 = [k for k, _ in L.incidence[v4]]
        perp, coll = [], []
        for x in range(4):
            for y in range(x + 1, 4):
                (coll if abs(abs(L.edir[k4[x]] @ L.edir[k4[y]]) - 1) < 1e-6 else perp).append(abs(self.pair_energy(k4[x], k4[y])))
        return J1, J3, float(np.mean(perp)), float(np.mean(coll))


K_PER_EV = 11604.518

def coupling_matrices(A, rc_a=24.0, near_a=2.5):
    """J_local (islands sharing a vertex) and J_full (sub-dipole near field for centre distance < near_a*a,
    point dipoles beyond, periodic images up to rc_a*a), eV."""
    L = A.L; N = L.N; a = A_NM*1e-9
    Jl = np.zeros((N, N))
    for inc in L.incidence:
        ks = [k for k, _ in inc]
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                Jl[ks[x], ks[y]] = Jl[ks[y], ks[x]] = A.pair_energy(ks[x], ks[y])
    C = np.array([A.P[A.own == k].mean(0) for k in range(N)]); m = L.edir
    mom = np.array([A.w[A.own == k].sum() for k in range(N)])
    box = A.box; J = np.zeros((N, N))
    nix = int(np.ceil(rc_a*a/box[0])) + 1; niy = int(np.ceil(rc_a*a/box[1])) + 1
    idx = [np.where(A.own == k)[0] for k in range(N)]
    for sx in range(-nix, nix + 1):
        for sy in range(-niy, niy + 1):
            sh = np.array([sx, sy])*box
            r = C[None, :, :] - C[:, None, :] + sh; dd = np.linalg.norm(r, axis=2)
            eye = np.eye(N, dtype=bool)
            far = (dd >= near_a*a) & (dd < rc_a*a) & ~eye
            if far.any():
                mi = np.einsum('id,ijd->ij', m, r); mj = np.einsum('jd,ijd->ij', m, r)
                with np.errstate(divide='ignore', invalid='ignore'):
                    v = MU0/(4*np.pi)*((m@m.T) - 3*mi*mj/dd**2)/dd**3*mom[:, None]*mom[None, :]/E_CH
                J += np.where(far, v, 0.0)
            for i, j in zip(*np.where((dd < near_a*a) & ~eye)):
                Pi, Pj = A.P[idx[i]], A.P[idx[j]] + sh
                rr = Pj[None, :, :] - Pi[:, None, :]; d = np.linalg.norm(rr, axis=2)
                E = MU0/(4*np.pi)*(m[i]@m[j] - 3*(rr@m[i])*(rr@m[j])/d**2)/d**3
                J[i, j] += np.sum(E*A.w[idx[i]][:, None]*A.w[idx[j]][None, :])/E_CH
    np.fill_diagonal(J, 0.0)
    return Jl, J
