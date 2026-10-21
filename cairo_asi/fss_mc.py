"""Faster manifold MC for large n: same moves/acceptance as manifold_mc2.mc, but
 - local fields h = J s, hl = Jl s are kept between calls (no O(N^2) J @ s per call),
 - Jl is used only through a sparse (CSR) copy (partner field, hl update, dE_local) -> no dense Jl in memory.
mc_keep(...) with the same seed reproduces manifold_mc2.mc exactly (checked in fss_pt.py --selftest).
Also: coupling cache (npz) and a vectorised needle near field for hybrid_coupling."""
import os, numpy as np
from numba import njit
import needle
from needle import subdipoles


def csr(Jl):
    ptr = np.zeros(Jl.shape[0] + 1, np.int64); idx = []; val = []
    for i in range(Jl.shape[0]):
        nz = np.nonzero(Jl[i])[0]; idx.extend(nz); val.extend(Jl[i, nz]); ptr[i + 1] = ptr[i] + len(nz)
    return ptr, np.array(idx, np.int64), np.array(val, float)


@njit(cache=True)
def mc_keep(s, h, hl, E, J, lp, li, lv, z4e, z3end, part_of, nba, partners, T, nsweep, seed, Etr):
    """nsweep sweeps at fixed T; s, h, hl modified in place; returns (E, n_accepted). Etr[sw] = E after sweep sw.
    seed >= 0 reseeds numba's RNG (use only for tests / first call)."""
    if seed >= 0: np.random.seed(seed)
    nz = z4e.shape[0]
    F = np.zeros(64, np.int64); tmp = s.copy(); nacc = 0
    for sw in range(nsweep):
        for _ in range(nz):
            mv = np.random.randint(3); nF = 0
            if mv < 2:
                verts = np.empty(2, np.int64); nvx = 1; verts[0] = np.random.randint(nz)
                if mv == 1:
                    cnt = 0
                    for t in range(nba.shape[1]):
                        if nba[verts[0], t] >= 0: cnt += 1
                    verts[1] = nba[verts[0], np.random.randint(cnt)]; nvx = 2
                for q in range(nvx):
                    for t in range(4):
                        F[nF] = z4e[verts[q], t]; nF += 1
                for a in range(nF): tmp[F[a]] = -s[F[a]]
                nbase = nF
                for a in range(nbase):
                    p = part_of[z3end[F[a]]]
                    dup = False
                    for b in range(nF):
                        if F[b] == p: dup = True
                    if dup: continue
                    hp = 0.0
                    for q in range(lp[p], lp[p + 1]): hp += lv[q]*tmp[li[q]]
                    if abs(hp) < 1e-9: newp = s[p] if np.random.random() < 0.5 else -s[p]
                    else: newp = -1.0 if hp > 0 else 1.0
                    if newp != s[p]:
                        F[nF] = p; nF += 1; tmp[p] = newp
            else:
                p = partners[np.random.randint(partners.size)]; F[0] = p; nF = 1
            dE = 0.0; dEl = 0.0
            for a in range(nF):
                i = F[a]; dE += -2.0*s[i]*h[i]; dEl += -2.0*s[i]*hl[i]
                for b in range(nF):
                    j = F[b]; dE += 2.0*s[i]*J[i, j]*s[j]
                    jl = 0.0                                   # Jl[i, j] from the CSR copy (no dense Jl kept)
                    for q in range(lp[i], lp[i + 1]):
                        if li[q] == j: jl = lv[q]; break
                    dEl += 2.0*s[i]*jl*s[j]
            if abs(dEl) < 1e-7:
                if dE <= 0 or np.random.random() < np.exp(-dE/T):
                    nacc += 1
                    for a in range(nF):
                        i = F[a]; s[i] = -s[i]; si2 = 2.0*s[i]
                        for j in range(s.size): h[j] += si2*J[i, j]   # J symmetric: row access
                        for q in range(lp[i], lp[i + 1]): hl[li[q]] += si2*lv[q]
                    E += dE
            for a in range(nF): tmp[F[a]] = s[F[a]]
        Etr[sw] = E
    return E, nacc


def hybrid_coupling_fast(L, rc, r0=6.0):
    """Same as hybrid_coupling.hybrid_coupling, near-field needle part vectorised over pairs."""
    needle.SUBDIV = 8; needle.FIXED_M = True
    P, Dr, Wt, own, a, is34 = subdipoles(L)
    N = L.N; box = L.box / a
    C = L.epos / a; m = L.edir; mom = (L.elen - 2*needle.GAP*a) / a
    J = np.zeros((N, N))
    nix = int(np.ceil(rc / box[0])) + 1; niy = int(np.ceil(rc / box[1])) + 1
    dC0 = C[None, :, :] - C[:, None, :]; mm = m @ m.T; eye = np.eye(N, dtype=bool)
    NI, NJ, SH = [], [], []
    for sx in range(-nix, nix + 1):
        for sy in range(-niy, niy + 1):
            sh = np.array([sx, sy]) * box
            r = dC0 + sh; d = np.sqrt((r**2).sum(2))
            far = (d >= r0) & (d < rc) & ~eye
            if far.any():
                ii, jj = np.nonzero(far); rr = r[ii, jj]; dd = d[ii, jj]
                v = (mm[ii, jj] - 3*(rr*m[ii]).sum(1)*(rr*m[jj]).sum(1)/dd**2)/dd**3*mom[ii]*mom[jj]
                np.add.at(J, (ii, jj), v)
            near = (d < min(r0, rc)) & ~eye
            ii, jj = np.nonzero(near)
            NI.append(ii); NJ.append(jj); SH.append(np.repeat(sh[None], len(ii), 0))
    NI = np.concatenate(NI); NJ = np.concatenate(NJ); SH = np.concatenate(SH)
    # all islands have the same number of sub-dipoles per type (FIXED_M); group pairs by (m_i, m_j)
    order = np.argsort(own, kind="stable"); cnt = np.bincount(own, minlength=N); start = np.r_[0, np.cumsum(cnt)[:-1]]
    for ci in np.unique(cnt):
        for cj in np.unique(cnt):
            sel = np.nonzero((cnt[NI] == ci) & (cnt[NJ] == cj))[0]
            for b0 in range(0, len(sel), 20000):
                s_ = sel[b0:b0 + 20000]; i = NI[s_]; j = NJ[s_]
                Ii = order[start[i][:, None] + np.arange(ci)]; Ij = order[start[j][:, None] + np.arange(cj)]
                Pi = P[Ii]; Pj = P[Ij] + SH[s_][:, None, :]
                r = Pj[:, None, :, :] - Pi[:, :, None, :]; d2 = (r**2).sum(3)
                ri = (r*m[i][:, None, None, :]).sum(3); rj = (r*m[j][:, None, None, :]).sum(3)
                v = ((m[i]*m[j]).sum(1)[:, None, None] - 3*ri*rj/d2)/d2**1.5
                v = (v*Wt[Ii][:, :, None]*Wt[Ij][:, None, :]).sum((1, 2))
                np.add.at(J, (i, j), v)
    np.fill_diagonal(J, 0.0)
    return J


def couplings(L, d, n, rc=64.0, cache_dir="couplings"):
    """(J, Jl) for the needle model, g = needle.GAP (must be 0 here), cached as npz."""
    from hybrid_coupling import local_needle_coupling
    assert needle.GAP == 0.0
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"J_d{d:.4f}_n{n}_rc{rc:g}_g0.npz")
    if os.path.exists(fn):
        z = np.load(fn); return z["J"], z["Jl"]
    Jl = local_needle_coupling(L); J = hybrid_coupling_fast(L, rc)
    tmp = fn + ".tmp.npz"; np.savez(tmp, J=J, Jl=Jl); os.replace(tmp, fn)
    return J, Jl
