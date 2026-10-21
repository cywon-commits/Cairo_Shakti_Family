"""Columnar stripes on the chessboard: m layers of vertical dominoes (2 rows each) + one horizontal row, period 2m+1.
Spins by the inverse map; wall energy per horizontal-row cell relative to the uniform columnar state on the same torus,
split into point-dipole SS, LS, LL parts (mu-dependence)."""
import numpy as np, sys, json, time
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cand_response import CairoLattice4
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from chess_torus import torus_coords, dimers_from_map
from dimer_inverse import spins_from_dimers, vertex_tables
from stripe_sums import dipole_matrix
def pattern(n, m, phases):
    M = 2*n; P = 2*m + 1; assert M % P == 0; g = np.empty((M, M), dtype=object); h = 0
    for y in range(M):
        r = y % P
        if r == 0:
            ph = phases[h % len(phases)]; h += 1
            for x in range(M): g[x, y] = ("r" if x % 2 == 0 else "l") if ph == 0 else ("l" if x % 2 == 0 else "r")
        else:
            for x in range(M): g[x, y] = "u" if r % 2 == 1 else "d"
    return g, M//P
def uniform(n):
    M = 2*n; g = np.empty((M, M), dtype=object)
    for y in range(M):
        for x in range(M): g[x, y] = "u" if y % 2 == 0 else "d"
    return g
if __name__ == '__main__':
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.153
    cases = [tuple(map(int, c.split('x'))) for c in sys.argv[2].split(',')] if len(sys.argv) > 2 else [(1, 12), (2, 10), (3, 14)]
    for phases_name, phases in (("same phase", [0]), ("alternating", [0, 1])):
        print(f"--- horizontal rows: {phases_name}, d = {d}")
        for m, n in cases:
            if phases_name == "alternating" and ((2*n)//(2*m + 1)) % 2: continue      # alternating needs an even number of rows
            t0 = time.time(); L = CairoLattice4(n, n, d); Jl = local_needle_coupling(L); D = DimerMap(L, Jl); xy = torus_coords(L, D, n); tabs = vertex_tables(L, Jl)
            g, nh = pattern(n, m, phases); s = spins_from_dimers(L, Jl, dimers_from_map(D, xy, g, n), tabs)
            u = spins_from_dimers(L, Jl, dimers_from_map(D, xy, uniform(n), n), tabs)
            if s is None or u is None: print(f"   m={m} on {n}x{n}: not a manifold state in this sector (stripe {s is not None}, uniform {u is not None})"); continue
            J, is34 = dipole_matrix(L); Dm = 0.5*J*(np.outer(s, s) - np.outer(u, u))/L.N; S = is34.astype(float); G = 1 - S
            f = L.N/(nh*2*n)                                      # per horizontal-row cell (each row has 2n chessboard cells)
            a, b, c = float(S@Dm@S)*f, float(2*S@Dm@G)*f, float(G@Dm@G)*f
            print(f"   m={m} on {n}x{n} ({nh} walls): sigma_col(mu) = {a:+.4f} {b:+.4f} mu {c:+.4f} mu^2 | at mu=1: {a+b+c:+.4f}, mu=2: {a+2*b+4*c:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
