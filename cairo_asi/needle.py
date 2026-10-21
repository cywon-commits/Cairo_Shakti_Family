"""Needle (sub-dipole chain) couplings for the Cairo family CairoLattice(d): d=D_EQ Cairo ... d=1/4 Shakti.
Each island of length l is split into m = round(l/h) equal segments (h = z3z4-edge length / 2), each a point
dipole of moment l/m at the segment centre (moment ~ length). Energies in units of D/a^3, a = z3-z4 edge length."""
import numpy as np
from cairo_asi import CairoLattice
GAP3 = None      # optional z3-end gap of z3-z4 islands (with GAP4); None -> symmetric GAP
GAP4 = None      # optional z4-end gap of z3-z4 islands
GAP_P = None     # optional separate gap for partner (z3-z3) islands; None -> same as GAP
GAP = 0.0        # vertex gap g in units of a (z3-z4 edge length); islands shortened by 2g
FIXED_M = False  # True: fixed #segments per island type -> energies smooth in d
SUBDIV = 2   # sub-dipoles per short (z3-z4) island; set larger for a smooth needle

def subdipoles(L):
    is34 = np.array([tuple(sorted(L.vtype[L.edges[k]])) == (3, 4) for k in range(L.N)])
    a = L.elen[is34].mean(); h = a / SUBDIV
    P, Dr, Wt, own = [], [], [], []
    for k in range(L.N):
        if is34[k] and GAP3 is not None:                      # asymmetric: edges are oriented z3 -> z4
            l = L.elen[k] - (GAP3 + GAP4)*a
            centre = L.epos[k] + 0.5*(GAP3 - GAP4)*a*L.edir[k]
        else:
            g = GAP if (is34[k] or GAP_P is None) else GAP_P
            l = L.elen[k] - 2*g*a; centre = L.epos[k]
        m = (SUBDIV if is34[k] else 2*SUBDIV) if FIXED_M else max(1, int(round(l / h)))
        start = centre - 0.5 * l * L.edir[k]
        for t in range(m):
            P.append(start + (t + 0.5) * (l / m) * L.edir[k]); Dr.append(L.edir[k]); Wt.append(l / m / a); own.append(k)
    return np.array(P) / a, np.array(Dr), np.array(Wt), np.array(own), a, is34

def needle_coupling(L, rc=None):
    P, Dr, Wt, own, a, is34 = subdipoles(L)
    M, N = len(P), L.N
    Wm = np.zeros((M, N)); Wm[np.arange(M), own] = Wt
    K = np.zeros((M, M))
    same = own[:, None] == own[None, :]
    if rc is None:                               # vertex-local: islands sharing a vertex
        share = np.zeros((N, N), bool)
        for inc in L.incidence:
            ks = [k for k, _ in inc]
            for x in ks:
                for y in ks:
                    share[x, y] = x != y
        pair_ok = share[own][:, own]
        shifts = [(0, 0)]; R = np.inf
    else:
        pair_ok = ~same; R = rc
        box = L.box / a
        nix = int(np.ceil(R / box[0])) + 1; niy = int(np.ceil(R / box[1])) + 1
        shifts = [(i, j) for i in range(-nix, nix + 1) for j in range(-niy, niy + 1)]
    dd0 = P[None, :, :] - P[:, None, :]
    if rc is None:
        box = L.box / a; dd0 = dd0 - box * np.round(dd0 / box)
    mm = Dr @ Dr.T
    for sx, sy in shifts:
        r = dd0 + (np.array([sx, sy]) * L.box / a if rc is not None else 0)
        d = np.linalg.norm(r, axis=2)
        mask = (d < R) & (d > 1e-9) & (pair_ok if (sx, sy) == (0, 0) or rc is None else ~same | True)
        if rc is not None and (sx, sy) != (0, 0):
            mask &= ~same                          # self-images of one island: constant, drop
        if not mask.any(): continue
        mi = np.einsum('id,ijd->ij', Dr, r); mj = np.einsum('jd,ijd->ij', Dr, r)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (mm - 3 * mi * mj / d**2) / d**3
        K += np.where(mask, v, 0.0)
    K *= Wt[:, None] * Wt[None, :] / (Wt[:, None] * Wt[None, :])   # weights applied via Wm below
    J = Wm.T @ K @ Wm
    np.fill_diagonal(J, 0.0)
    return J
