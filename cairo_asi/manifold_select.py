"""For a given d: enumerate ALL states of the vertex-local (needle) GS manifold on periodic nx x ny cells,
evaluate the full dipolar (needle) energy of each exactly, report the minimum. Usage: python manifold_select.py d"""
import numpy as np, itertools, sys, time, json
from numba import njit
from cairo_asi import CairoLattice, D_EQ
from needle import needle_coupling

def setup(L, J):
    z4 = np.where(L.vtype == 4)[0]; ze, zc = [], []
    for v in z4:
        ks = [k for k, _ in L.incidence[v]]
        best = min(itertools.product([-1, 1], repeat=4),
                   key=lambda c: sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(4), 2)))
        ze.append(ks); zc.append(best)
    part = np.array([k for k in range(L.N) if tuple(L.vtype[L.edges[k]]) == (3, 3)])
    return np.array(ze), np.array(zc, float), part

@njit(cache=True)
def enum_manifold(J, ze, zc, part, maxstates):
    N = J.shape[0]; nz = ze.shape[0]; npart = part.size
    out = np.zeros((maxstates, N), np.int8); n = 0; E0 = 1e18
    s = np.zeros(N)
    for bits in range(1 << (nz - 1)):
        for t in range(nz):
            sg = 1.0 - 2.0*((bits >> t) & 1)
            for q in range(4): s[ze[t, q]] = zc[t, q]*sg
        for p in range(npart): s[part[p]] = 0.0
        # partner options
        opt = np.zeros((npart, 2)); nopt = np.zeros(npart, np.int64)
        for p in range(npart):
            k = part[p]; hk = 0.0
            for j in range(N): hk += J[k, j]*s[j]
            # E contribution of s_k = sk*hk (other partners are zero here, partners never share a vertex)
            if abs(hk) < 1e-9: opt[p, 0] = -1.0; opt[p, 1] = 1.0; nopt[p] = 2
            else: opt[p, 0] = -1.0 if hk > 0 else 1.0; nopt[p] = 1
        tot = 1
        for p in range(npart): tot *= nopt[p]
        for c in range(tot):
            cc = c
            for p in range(npart):
                s[part[p]] = opt[p, cc % nopt[p]]; cc //= nopt[p]
            E = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    if J[i, j] != 0.0: E += J[i, j]*s[i]*s[j]
            if E < E0 - 1e-7:
                E0 = E; n = 0
            if abs(E - E0) < 1e-7 and n < maxstates:
                for i in range(N): out[n, i] = int(s[i])
                n += 1
    return E0, out[:n]

if __name__ == "__main__":
    d = float(sys.argv[1]); cells = [(2,2),(3,2),(2,3),(4,2),(2,4),(3,3)]
    res = {}
    for nx, ny in cells:
        t = time.time()
        L = CairoLattice(nx, ny, d=d); Jl = needle_coupling(L, None); Jd = needle_coupling(L, 6.0)
        ze, zc, part = setup(L, Jl)
        E0, S = enum_manifold(Jl, ze, zc, part, 200000)
        S = S.astype(float); Ed = 0.5*np.einsum('ij,jk,ik->i', S, Jd, S)/L.N
        i = int(np.argmin(Ed)); nbest = int(np.sum(Ed < Ed[i] + 1e-7))
        np.save(f"sel_d{d:.4f}_{nx}x{ny}.npy", S[i])
        res[f"{nx}x{ny}"] = dict(Emin=float(Ed[i]), nstates=2*len(S), nbest=nbest, spread=float(Ed.max()-Ed.min()))
        print(f"d={d:.4f} {nx}x{ny}: manifold size {2*len(S)} (x2 time rev.), dipolar E/N min {Ed[i]:.5f} "
              f"(degenerate minima {nbest}), max-min {Ed.max()-Ed.min():.4f}  [{time.time()-t:.0f}s]", flush=True)
    json.dump(res, open(f"sel_d{d:.4f}.json", "w"), indent=1)
