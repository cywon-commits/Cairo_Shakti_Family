import numpy as np, sys, time, os
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from hybrid_coupling import hybrid_coupling, local_needle_coupling
from cand_response import CairoLattice4
name, db, key, val, sign = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
f = f"vmcJ_{name}_{'p' if sign > 0 else 'm'}.npz"
if os.path.exists(f): print("exists", f); sys.exit()
t0 = time.time(); L = CairoLattice4(12, 12, db, **{key: sign*val}); J = hybrid_coupling(L, 64.0)
out = dict(J=J)
if sign > 0: out["Jl"] = local_needle_coupling(L)
np.savez(f, **out); print(f"saved {f} [{time.time()-t0:.0f}s]")
