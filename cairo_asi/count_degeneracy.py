"""Exact GS degeneracy of the vertex-local model on periodic nx x ny cells.
E = sum of vertex energies. All z4 vertices in their 2-fold lowest config fixes every z3-z4 spin;
each partner (z3-z3) spin then only enters its two endpoint z3 vertices and is optimised independently."""
import numpy as np, itertools, sys
from numba import njit
from cairo_asi import *

def setup(nx, ny):
    L = CairoLattice(nx, ny); J = L.coupling_matrix(None)
    z4 = np.where(L.vtype == 4)[0]
    z4edges, z4conf = [], []
    for v in z4:
        ks = [k for k, _ in L.incidence[v]]
        best = min(itertools.product([-1, 1], repeat=4),
                   key=lambda c: sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(4), 2)))
        z4edges.append(ks); z4conf.append(best)
    partner = [k for k in range(L.N) if tuple(L.vtype[L.edges[k]]) == (3, 3)]
    # for each partner spin: its two z3 end vertices, each with 2 other edges and the 3 couplings
    P = []
    for p in partner:
        row = []
        for v in L.edges[p]:
            others = [k for k, _ in L.incidence[v] if k != p]
            row += [others[0], others[1], J[p, others[0]], J[p, others[1]], J[others[0], others[1]]]
        P.append(row)
    return L, np.array(z4edges), np.array(z4conf, float), np.array(partner), np.array(P)

@njit(cache=True)
def enumerate_gs(N, z4edges, z4conf, P):
    nz = z4edges.shape[0]
    Emin, W = 1e18, 0
    s = np.zeros(N)
    for bits in range(1 << (nz - 1)):              # fix first z4 vertex: time reversal
        for t in range(nz):
            sg = 1.0 - 2.0 * ((bits >> t) & 1)
            for q in range(4):
                s[z4edges[t, q]] = z4conf[t, q] * sg
        E = 0.0; w = 1
        for r in range(P.shape[0]):
            e = np.zeros(2)
            for c in range(2):
                sp = -1.0 if c == 0 else 1.0
                for h in range(2):
                    a = int(P[r, 5*h]); b = int(P[r, 5*h+1])
                    e[c] += sp*s[a]*P[r, 5*h+2] + sp*s[b]*P[r, 5*h+3] + s[a]*s[b]*P[r, 5*h+4]
            if abs(e[0]-e[1]) < 1e-9:
                w *= 2; E += e[0]
            else:
                E += min(e[0], e[1])
        if E < Emin - 1e-7:
            Emin, W = E, w
        elif abs(E - Emin) < 1e-7:
            W += w
    return Emin, 2 * W                               # restore time reversal

if __name__ == "__main__":
    sizes = [(1,1),(2,1),(2,2),(3,2),(4,2),(3,3),(4,3),(4,4)]
    with open("degeneracy.txt", "w") as f:
        for nx, ny in sizes:
            L, ze, zc, part, P = setup(nx, ny)
            e, W = enumerate_gs(L.N, ze, zc, P)
            line = f"{nx}x{ny}: N={L.N:4d}  W={W:>10d}  ln W / cell = {np.log(W)/(nx*ny):.4f}"
            print(line, flush=True); f.write(line + "\n"); f.flush()
