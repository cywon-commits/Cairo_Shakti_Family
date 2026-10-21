"""Loop MC restricted to the vertex-local GS (dimer) manifold, Metropolis on the full dipolar energy.
A move = flip a directed (head-to-tail) loop of spins; accepted only if the vertex-local energy is unchanged."""
import numpy as np
from numba import njit

def lattice_arrays(L):
    nv = len(L.vpos)
    inc_e = -np.ones((nv, 4), np.int64); inc_s = np.zeros((nv, 4), np.int64); deg = np.zeros(nv, np.int64)
    for v, inc in enumerate(L.incidence):
        for t, (k, sg) in enumerate(inc):
            inc_e[v, t] = k; inc_s[v, t] = sg
        deg[v] = len(inc)
    return inc_e, inc_s, deg, L.edges.astype(np.int64)

@njit(cache=True)
def loop_mc(s, J, Jl, inc_e, inc_s, deg, edges, T0, T1, nsweep, seed, maxlen):
    np.random.seed(seed)
    N = s.size; nv = deg.size
    h = J @ s; hl = Jl @ s
    E = 0.5 * s @ h
    visit = -np.ones(nv, np.int64)
    path_v = np.zeros(maxlen + 1, np.int64); path_e = np.zeros(maxlen + 1, np.int64)
    acc = 0; tried = 0; local_ok = 0
    Etr = np.empty(nsweep)
    inF = np.zeros(N, np.bool_)
    for sw in range(nsweep):
        T = T0 * (T1 / T0) ** (sw / max(nsweep - 1, 1))
        for _ in range(N // 4):
            v = np.random.randint(nv); n = 0; closed = -1
            ok = True
            while True:
                visit[v] = n; path_v[n] = v
                # outgoing edges at v: spin points out  <=>  sg*s < 0
                cand = np.empty(4, np.int64); nc = 0
                for t in range(deg[v]):
                    k = inc_e[v, t]
                    if inc_s[v, t] * s[k] < 0 and (n == 0 or k != path_e[n - 1]):
                        cand[nc] = k; nc += 1
                if nc == 0 or n >= maxlen:
                    ok = False; break
                k = cand[np.random.randint(nc)]
                path_e[n] = k
                w = edges[k, 0] if edges[k, 1] == v else edges[k, 1]
                n += 1
                if visit[w] >= 0:
                    closed = visit[w]; break
                v = w
            for t in range(n + 1):
                visit[path_v[t]] = -1
            if not ok:
                continue
            tried += 1
            # loop = edges path_e[closed .. n-1]
            dEl = 0.0; dE = 0.0
            for a in range(closed, n):
                i = path_e[a]; inF[i] = True
            for a in range(closed, n):
                i = path_e[a]
                dEl += -2.0 * s[i] * hl[i]; dE += -2.0 * s[i] * h[i]
                for b in range(closed, n):
                    j = path_e[b]
                    dEl += 2.0 * s[i] * Jl[i, j] * s[j]; dE += 2.0 * s[i] * J[i, j] * s[j]
            for a in range(closed, n):
                inF[path_e[a]] = False
            if abs(dEl) > 1e-7:
                continue
            local_ok += 1
            if dE <= 0 or np.random.random() < np.exp(-dE / T):
                for a in range(closed, n):
                    i = path_e[a]
                    s[i] = -s[i]
                    for j in range(N):
                        h[j] += 2.0 * J[j, i] * s[i]; hl[j] += 2.0 * Jl[j, i] * s[i]
                E += dE; acc += 1
        Etr[sw] = E
    return s, Etr, tried, local_ok, acc

def shift_state(s, n, dx, dy):
    t = np.empty_like(s)
    for i in range(n):
        for j in range(n):
            t[(((i + dx) % n) * n + (j + dy) % n) * 10:(((i + dx) % n) * n + (j + dy) % n) * 10 + 10] = s[(i * n + j) * 10:(i * n + j) * 10 + 10]
    return t

def periods(s, n):
    out = []
    for dx in range(1, n + 1):
        for dy in range(0, n):
            if (dx, dy) != (n, 0) and np.array_equal(shift_state(s, n, dx % n, dy), s):
                out.append((dx % n, dy))
    return out
