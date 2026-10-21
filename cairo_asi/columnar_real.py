import numpy as np, json, sys, time
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
import real_islands as RI
from cand_response import CairoLattice4
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from chess_torus import torus_coords, dimers_from_map
from dimer_inverse import spins_from_dimers, vertex_tables
from columnar_stripes import pattern, uniform
from stripe_sums import dipole_matrix
UNIT_K = 1e-7*(RI.MS*((300 - 100)*100 + np.pi*50**2)*3e-27)**2/(472e-9)**3/1.380649e-23
def area(L): return (L - RI.WID)*RI.WID + np.pi*(RI.WID/2)**2
for d in [float(x) for x in sys.argv[1].split(",")]:
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)]); sc = 472/ref.elen[is34].mean()
    mu = area(ref.elen[~is34].mean()*sc - 2*22)/area(472 - 2*86)
    for m, n in ((1, 12), (2, 10)):
        t0 = time.time(); L = CairoLattice4(n, n, d); Jl = local_needle_coupling(L); D = DimerMap(L, Jl); xy = torus_coords(L, D, n); tabs = vertex_tables(L, Jl)
        g, nh = pattern(n, m, [0, 1]); s = spins_from_dimers(L, Jl, dimers_from_map(D, xy, g, n), tabs); u = spins_from_dimers(L, Jl, dimers_from_map(D, xy, uniform(n), n), tabs)
        A = RI.RealArray(n, n, d); _, J = RI.full_J(A); f = L.N/(nh*2*n)
        real = (0.5*s@J@s - 0.5*u@J@u)/L.N*f
        Jp, is34n = dipole_matrix(L); Dm = 0.5*Jp*(np.outer(s, s) - np.outer(u, u))/L.N; S = is34n.astype(float); G = 1 - S
        point = (float(S@Dm@S) + mu*float(2*S@Dm@G) + mu*mu*float(G@Dm@G))*f*UNIT_K
        print(f"d={d} (realistic mu = {mu:.2f}), m={m} on {n}x{n}: columnar wall energy per wall cell: realistic {real:+.2f} K, point formula {point:+.2f} K  [{time.time()-t0:.0f}s]", flush=True)
