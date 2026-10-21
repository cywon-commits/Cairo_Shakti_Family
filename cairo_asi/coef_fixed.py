"""Fixed-state linear responses Q = dE/d(eps), dE/d(theta) for the best A, C and columnar (M) states, plus zero-field energies."""
import numpy as np, json, sys
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo3 import CairoLattice3
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from selection_T import FastDimers
from cairo2 import S0
L = CairoLattice3(4, 2, 0.20); F = FastDimers(L, local_needle_coupling(L))
ps = np.array([F.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
cls = {"A": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(pc) < 1e-9), "C": (np.abs(np.abs(pc) - 1) < 1e-9) & (np.abs(pa) < 1e-9),
       "M": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(np.abs(pc) - 1) < 1e-9)}
out = {}
for db in [float(x) for x in sys.argv[1].split(",")]:
    E0 = 0.5*np.einsum('ij,jk,ik->i', S0, hybrid_coupling(CairoLattice3(4, 2, db), 64.0), S0)/40/2*2
    E0 = 0.5*np.einsum('ij,jk,ik->i', S0, hybrid_coupling(CairoLattice3(4, 2, db), 64.0), S0)/80
    best = {k: int(np.flatnonzero(m)[np.argmin(E0[m])]) for k, m in cls.items()}
    rec = {k: {"E0": float(E0[i])} for k, i in best.items()}
    for kw, h in (("eps", 0.002), ("theta", 0.01)):
        Jp = hybrid_coupling(CairoLattice3(4, 2, db, **{kw: h}), 64.0); Jm = hybrid_coupling(CairoLattice3(4, 2, db, **{kw: -h}), 64.0)
        for k, i in best.items():
            s = S0[i]; rec[k]["|Q_" + kw + "|"] = float(abs(0.5*s@Jp@s - 0.5*s@Jm@s)/80/(2*h))
    out[db] = rec
    print(f"d_bar={db:.3f}: " + "; ".join(f"{k}: E0 {v['E0']:.5f}, |Q_eps| {v['|Q_eps|']:.2f}, |Q_theta| {v['|Q_theta|']:.3f}" for k, v in rec.items()), flush=True)
json.dump(out, open(f"coef_fixed_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
