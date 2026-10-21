"""lambda_A = Delta'(d)/10 from a single z3 vertex vs the manifold fit (local couplings only)."""
import numpy as np, itertools, sys
import needle
from cairo3 import CairoLattice3
from hybrid_coupling import local_needle_coupling
from selection_T import FastDimers
from cairo2 import S0
def z3_gap(d, b=2):
    L = CairoLattice3(4, 2, d); Jl = local_needle_coupling(L)
    v = [x for x in range(len(L.vtype)) if x % 6 == b][0]; ks = [k for k, _ in L.incidence[v]]
    t = sorted(set(np.round([sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(3), 2)) for c in itertools.product([-1, 1], repeat=3)], 8)))
    return t[1] - t[0]
def manifold_fit(d, eps=0.002):
    L = CairoLattice3(4, 2, d, eps=eps); Jl = local_needle_coupling(L); F = FastDimers(L, Jl)
    El = 0.5*np.einsum('ij,jk,ik->i', S0, Jl, S0)/L.N; pa = np.array([F.psi(s)[0] for s in S0])
    c, *_ = np.linalg.lstsq(np.vstack([pa, np.ones_like(pa)]).T, El, rcond=None); return -c[0]/eps
if __name__ == "__main__":
    g = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; needle.GAP = g; needle.SUBDIV = 8; needle.FIXED_M = True
    print(f"g = {g}:  d  | Delta(d) | Delta'(d)/10 (single vertex) | lambda_A (manifold fit)")
    for d in [float(x) for x in sys.argv[2].split(",")]:
        h = 0.001; dp = (z3_gap(d + h) - z3_gap(d - h))/(2*h)
        print(f"   {d:.3f} | {z3_gap(d):8.3f} | {dp/10:8.3f} | {manifold_fit(d):8.3f}", flush=True)
