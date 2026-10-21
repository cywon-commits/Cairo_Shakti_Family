import numpy as np, glob, json, time, sys
import needle; needle.SUBDIV = 8; needle.FIXED_M = True
from needle import needle_coupling
from cairo_asi import CairoLattice, D_EQ
from manifold_select import setup, enum_manifold
from hybrid_coupling import hybrid_coupling

L0 = CairoLattice(4, 2, d=D_EQ); Jl0 = needle_coupling(L0, None)
ze, zc, part = setup(L0, Jl0); _, S = enum_manifold(Jl0, ze, zc, part, 200000); S = S.astype(float); S = np.vstack([S, -S])
key = {tuple(s.astype(int)): i for i, s in enumerate(S)}
ref = {}
for lab, f in [("A", "scan3_min_d0.1372.npy"), ("B", "scan3_min_d0.1531.npy"), ("C", "scan3_min_d0.2500.npy")] + \
              [(f.split("_")[1], f) for f in sorted(glob.glob("phase_*_d*.npy"))]:
    ref[frozenset(tuple(r.astype(int)) for r in np.load(f))] = lab
rcs = [float(x) for x in sys.argv[1].split(",")]
ds = [float(x) for x in sys.argv[2].split(",")]
out = {}
for rc in rcs:
    t = time.time(); row = []
    for d in ds:
        L = CairoLattice(4, 2, d=d); J = hybrid_coupling(L, rc)
        E = 0.5*np.einsum('ij,jk,ik->i', S, J, S)/L.N; Emin = E.min()
        mins = frozenset(tuple(S[i].astype(int)) for i in np.where(E < Emin + 1e-7)[0])
        if mins not in ref: ref[mins] = f"N{len(ref)}"; np.save(f"phase_{ref[mins]}_rc{rc:g}_d{d:.4f}.npy", np.array(sorted(mins), float))
        row.append(ref[mins])
    out[rc] = row
    print(f"rc={rc:5.1f}a: " + " ".join(row) + f"   [{time.time()-t:.0f}s]", flush=True)
json.dump({"ds": ds, "phases": {str(k): v for k, v in out.items()}}, open(f"rc_conv_{sys.argv[3]}.json", "w"))
