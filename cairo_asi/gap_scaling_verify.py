import numpy as np, json, sys, time
import needle
from cairo3 import CairoLattice3
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from selection_T import FastDimers
from cairo2 import S0
pred = json.load(open("gap_scaling_pred.json"))
L0 = CairoLattice3(4, 2, 0.20); F = FastDimers(L0, local_needle_coupling(L0))
ps = np.array([F.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
cls = {"A": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(pc) < 1e-9), "C": (np.abs(np.abs(pc) - 1) < 1e-9) & (np.abs(pa) < 1e-9),
       "M": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(np.abs(pc) - 1) < 1e-9)}
for arg in sys.argv[1:]:
    g, d = map(float, arg.split(":")); needle.GAP = g; needle.SUBDIV = 8; needle.FIXED_M = True
    es = pred[f"{g:.2f}_{d:.4f}"]["eps_star"]; t0 = time.time(); rows = []
    for f in (0.7, 1.0, 1.3):
        L = CairoLattice3(4, 2, d, eps=f*es); J = hybrid_coupling(L, 64.0)
        E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/L.N; m = {k: E[v].min() for k, v in cls.items()}
        rows.append((f, m["C"] - min(m["A"], m["M"]), min(m, key=m.get)))
    # linear interpolation of the actual crossing
    (f1, x1, _), (f2, x2, _), (f3, x3, _) = rows
    fc = None
    for (fa, xa, _), (fb, xb, _) in zip(rows, rows[1:]):
        if xa < 0 <= xb or xa > 0 >= xb: fc = fa + (fb - fa)*(-xa)/(xb - xa)
    print(f"g={g} d={d}: predicted eps* {es:.4f}; E_C - min(E_A, E_M) at 0.7/1.0/1.3 eps*: " + ", ".join(f"{x:+.5f} ({w})" for f, x, w in rows)
          + (f"; actual crossing at {fc:.2f} x predicted" if fc else "; no crossing in range") + f"  [{time.time()-t0:.0f}s]", flush=True)
