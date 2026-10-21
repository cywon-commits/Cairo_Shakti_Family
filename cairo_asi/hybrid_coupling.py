"""Island-level cutoff dipolar couplings: needle model (sub-dipoles) for island pairs whose centres are closer than r0,
point dipoles (moment = island length / a) beyond r0, all periodic images with centre distance < rc. Units a = z3-z4 length."""
import numpy as np
import needle
from needle import subdipoles

def hybrid_coupling(L, rc, r0=6.0):
    needle.SUBDIV = 8; needle.FIXED_M = True      # GAP is left as set by the caller
    P, Dr, Wt, own, a, is34 = subdipoles(L)
    N = L.N; box = L.box / a
    C = L.epos / a; m = L.edir; mom = (L.elen - 2*needle.GAP*a) / a
    J = np.zeros((N, N))
    nix = int(np.ceil(rc / box[0])) + 1; niy = int(np.ceil(rc / box[1])) + 1
    dC0 = C[None, :, :] - C[:, None, :]
    mm = m @ m.T
    near_pairs = []
    for sx in range(-nix, nix + 1):
        for sy in range(-niy, niy + 1):
            sh = np.array([sx, sy]) * box
            r = dC0 + sh; d = np.linalg.norm(r, axis=2)
            self_img = np.eye(N, dtype=bool)                 # island with its own image: constant, drop
            far = (d >= r0) & (d < rc) & ~self_img
            if far.any():
                mi = np.einsum('id,ijd->ij', m, r); mj = np.einsum('jd,ijd->ij', m, r)
                with np.errstate(divide='ignore', invalid='ignore'):
                    v = (mm - 3*mi*mj/d**2) / d**3 * mom[:, None]*mom[None, :]
                J += np.where(far, v, 0.0)
            near = (d < min(r0, rc)) & ~self_img
            for i, j in zip(*np.where(near)):
                near_pairs.append((i, j, sh))
    # needle near field
    idx = [np.where(own == k)[0] for k in range(N)]
    for i, j, sh in near_pairs:
        Pi, Pj = P[idx[i]], P[idx[j]] + sh
        r = Pj[None, :, :] - Pi[:, None, :]; d = np.linalg.norm(r, axis=2)
        v = (m[i] @ m[j] - 3*(r @ m[i])*(r @ m[j])/d**2) / d**3
        J[i, j] += np.sum(v * Wt[idx[i]][:, None] * Wt[idx[j]][None, :])
    np.fill_diagonal(J, 0.0)
    return J


def local_needle_coupling(L):
    """Vertex-local needle couplings (islands sharing a vertex), memory-light version of needle_coupling(L, None)."""
    needle.SUBDIV = 8; needle.FIXED_M = True      # GAP is left as set by the caller
    P, Dr, Wt, own, a, is34 = subdipoles(L)
    idx = [np.where(own == k)[0] for k in range(L.N)]
    box = L.box / a; m = L.edir
    J = np.zeros((L.N, L.N)); done = set()
    for inc in L.incidence:
        ks = [k for k, _ in inc]
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                i, j = ks[x], ks[y]
                if (i, j) in done: continue
                done.add((i, j)); done.add((j, i))
                Pi = P[idx[i]]; Pj = P[idx[j]]
                r = Pj[None, :, :] - Pi[:, None, :]; r -= box * np.round(r / box)
                d = np.linalg.norm(r, axis=2)
                v = (m[i] @ m[j] - 3*(r @ m[i])*(r @ m[j])/d**2) / d**3
                J[i, j] = J[j, i] = np.sum(v * Wt[idx[i]][:, None] * Wt[idx[j]][None, :])
    return J


def point_coupling(L, rc=None, unit_moment=False):
    """Point dipoles at island centres. moment ~ island length (P1) or all equal (P2, unit_moment=True).
    rc=None: vertex-local (islands sharing a vertex, minimum image); else all image pairs with centre distance < rc (units a)."""
    is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)])
    a = L.elen[is34].mean(); C = L.epos / a; m = L.edir; box = L.box / a
    mom = np.ones(L.N) if unit_moment else L.elen / a
    J = np.zeros((L.N, L.N))
    if rc is None:
        for inc in L.incidence:
            ks = [k for k, _ in inc]
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    i, j = ks[x], ks[y]; r = C[j] - C[i]; r -= box*np.round(r/box); d = np.linalg.norm(r)
                    J[i, j] = J[j, i] = (m[i]@m[j] - 3*(m[i]@r)*(m[j]@r)/d**2)/d**3*mom[i]*mom[j]
        return J
    nix = int(np.ceil(rc/box[0])) + 1; niy = int(np.ceil(rc/box[1])) + 1
    dC0 = C[None, :, :] - C[:, None, :]; mm = m @ m.T; eye = np.eye(L.N, dtype=bool)
    for sx in range(-nix, nix + 1):
        for sy in range(-niy, niy + 1):
            r = dC0 + np.array([sx, sy])*box; d = np.linalg.norm(r, axis=2)
            mask = (d < rc) & ~eye
            if not mask.any(): continue
            mi = np.einsum('id,ijd->ij', m, r); mj = np.einsum('jd,ijd->ij', m, r)
            with np.errstate(divide='ignore', invalid='ignore'):
                v = (mm - 3*mi*mj/d**2)/d**3*mom[:, None]*mom[None, :]
            J += np.where(mask, v, 0.0)
    np.fill_diagonal(J, 0.0); return J
