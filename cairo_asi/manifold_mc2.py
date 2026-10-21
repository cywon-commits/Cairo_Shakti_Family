"""Manifold-restricted MC. Moves:
 (1) flip all 4 spins of one z4 vertex (stays Type-I), then reset affected partner (z3-z3) spins to their local optimum;
 (2) same for two z4 vertices sharing a z3 vertex;
 (3) flip one partner spin (only useful when its local field is zero).
Accepted only if the vertex-local energy is unchanged (manifold -> manifold); Metropolis on the full dipolar energy."""
import numpy as np
from numba import njit

def mc_arrays(L):
    z4 = np.where(L.vtype == 4)[0]; nv = len(L.vpos)
    z4e = np.array([[k for k, _ in L.incidence[v]] for v in z4], np.int64)
    part_of = -np.ones(nv, np.int64)                      # z3 vertex -> its partner edge
    for k in range(L.N):
        a, b = L.edges[k]
        if L.vtype[a] == 3 and L.vtype[b] == 3: part_of[a] = k; part_of[b] = k
    other = np.zeros((L.N, 2), np.int64)
    for k in range(L.N): other[k] = L.edges[k]
    # for each z4 edge: its z3 end
    z3end = np.array([[u for u in L.edges[k] if L.vtype[u] == 3][0] if L.vtype[L.edges[k][0]] + L.vtype[L.edges[k][1]] == 7 else -1
                      for k in range(L.N)], np.int64)
    # z4 neighbours (share a z3)
    idx = {v: i for i, v in enumerate(z4)}
    nb = [[] for _ in z4]
    for v3 in np.where(L.vtype == 3)[0]:
        zs = [u for k, _ in L.incidence[v3] for u in L.edges[k] if L.vtype[u] == 4]
        for x in zs:
            for y in zs:
                if x != y and idx[y] not in nb[idx[x]]: nb[idx[x]].append(idx[y])
    nbm = max(len(x) for x in nb)
    nba = -np.ones((len(z4), nbm), np.int64)
    for i, x in enumerate(nb): nba[i, :len(x)] = x
    partners = np.array(sorted(set(part_of[part_of >= 0])), np.int64)
    return z4e, z3end, part_of, nba, partners

@njit(cache=True)
def _apply_eval(F, nF, s, h, hl, J, Jl):
    dE = 0.0; dEl = 0.0
    for a in range(nF):
        i = F[a]; dE += -2.0*s[i]*h[i]; dEl += -2.0*s[i]*hl[i]
        for b in range(nF):
            j = F[b]; dE += 2.0*s[i]*J[i, j]*s[j]; dEl += 2.0*s[i]*Jl[i, j]*s[j]
    return dE, dEl

@njit(cache=True)
def mc(s, J, Jl, z4e, z3end, part_of, nba, partners, T0, T1, nsweep, seed):
    np.random.seed(seed); N = s.size; nz = z4e.shape[0]
    h = J @ s; hl = Jl @ s; E = 0.5*s@h
    F = np.zeros(64, np.int64); tmp = s.copy()
    stats = np.zeros(4, np.int64)                   # tried, local-ok, accepted, -
    Etr = np.empty(nsweep)
    for sw in range(nsweep):
        T = T0*(T1/T0)**(sw/max(nsweep - 1, 1))
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
                # trial state in tmp: flip F, then reset affected partners to local optimum of the LOCAL model
                for a in range(nF): tmp[F[a]] = -s[F[a]]
                nbase = nF
                for a in range(nbase):
                    p = part_of[z3end[F[a]]]
                    dup = False
                    for b in range(nF):
                        if F[b] == p: dup = True
                    if dup: continue
                    # local field on p in trial state
                    hp = 0.0
                    for j in range(N):
                        if Jl[p, j] != 0.0: hp += Jl[p, j]*tmp[j]
                    if abs(hp) < 1e-9: newp = s[p] if np.random.random() < 0.5 else -s[p]
                    else: newp = -1.0 if hp > 0 else 1.0
                    if newp != s[p]:
                        F[nF] = p; nF += 1; tmp[p] = newp
            else:
                p = partners[np.random.randint(partners.size)]; F[0] = p; nF = 1
            stats[0] += 1
            dE, dEl = _apply_eval(F, nF, s, h, hl, J, Jl)
            ok = abs(dEl) < 1e-7
            if ok:
                stats[1] += 1
                if dE <= 0 or np.random.random() < np.exp(-dE/T):
                    stats[2] += 1
                    for a in range(nF):
                        i = F[a]; s[i] = -s[i]
                        for j in range(N):
                            h[j] += 2.0*J[j, i]*s[i]; hl[j] += 2.0*Jl[j, i]*s[i]
                    E += dE
            for a in range(nF): tmp[F[a]] = s[F[a]]
        Etr[sw] = E
    return s, Etr, stats
