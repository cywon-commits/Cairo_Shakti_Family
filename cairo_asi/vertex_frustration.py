"""Vertex-local model = sum of independent vertex energies -> check for vertex frustration (Shakti mechanism)."""
import numpy as np, itertools
from collections import Counter
from cairo_asi import *

L = CairoLattice(4, 4); J = L.coupling_matrix(None)
a = L.elen.mean()
def vertex_energy_table(v):
    ks = [k for k, _ in L.incidence[v]]
    tab = {}
    for conf in itertools.product([-1, 1], repeat=len(ks)):
        s = dict(zip(ks, conf))
        e = sum(J[i, j]*s[i]*s[j] for i, j in itertools.combinations(ks, 2))
        tab[conf] = e
    return ks, tab
for z in (3, 4):
    v = np.where(L.vtype == z)[0][0]
    ks, tab = vertex_energy_table(v)
    levels = sorted(set(np.round(list(tab.values()), 4)))
    print(f"z={z} vertex energy levels:", levels, " degeneracies:", [sum(abs(e-l)<1e-3 for e in tab.values()) for l in levels])

def vertex_levels(s):
    out = {3: Counter(), 4: Counter()}
    for v in range(len(L.vpos)):
        ks, tab = vertex_energy_table(v)
        e = tab[tuple(int(s[k]) for k in ks)]
        lv = sorted(set(np.round(list(tab.values()), 4)))
        out[L.vtype[v]][lv.index(np.round(e, 4))] += 1
    return {z: dict(sorted(c.items())) for z, c in out.items()}

# vertex-local GS: tile exact 2x1 state onto 4x4
Ls = CairoLattice(2, 1); _, s0 = exact_ground_state_fast(Ls.coupling_matrix(None))
s = np.array([s0[((i % 2)*1 + 0)*10 + b] for i in range(4) for j in range(4) for b in range(10)])
print("\nvertex-local GS (tiled 2x1): E/N =", round(0.5*s@J@s/L.N, 5), " vertex level occupancy (0 = lowest):", vertex_levels(s))
s2 = np.load("gs_cell2x2.npy")
L8 = CairoLattice(8, 8)
print("dipolar 2x2 GS candidate  :", end=" ")
s2_44 = np.array([s2[((i)*8 + j)*10 + b] for i in range(4) for j in range(4) for b in range(10)])
print("vertex level occupancy:", vertex_levels(s2_44), " E_local/N =", round(0.5*s2_44@J@s2_44/L.N, 5))

# degeneracy of the vertex-local GS manifold: many anneals on 4x4, count distinct states at E_min
E0 = 0.5*s@J@s
found = set(); hits = 0
for sd in range(300):
    t, _ = anneal(J, np.random.choice([-1., 1.], L.N), 3.0, 0.005, 1500, 5000 + sd); t = quench(J, t)
    if abs(0.5*t@J@t - E0) < 1e-6:
        hits += 1
        key = tuple((t*t[0]).astype(int))
        found.add(key)
print(f"\n4x4 vertex-local: {hits}/300 anneals reach E_min, {len(found)} distinct GS (mod time reversal)")
np.save("local_gs_4x4.npy", np.array([np.array(k, float) for k in found]))
