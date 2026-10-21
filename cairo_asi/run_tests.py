import numpy as np, time
from cairo_asi import *
for model, rc in [("vertex-local", None), ("dipolar rc=6a", 6.0)]:
    print(f"\n=== {model} ===")
    for nx, ny in [(1,1),(2,1)]:
        L = CairoLattice(nx, ny); J = L.coupling_matrix(rc)
        e, s = exact_ground_state_fast(J)
        print(f"exact {nx}x{ny}: E/N = {e/L.N:.5f}  vertex {L.vertex_stats(s)}")
        print("   pentagon chirality:", np.round(L.chirality(s),2))
