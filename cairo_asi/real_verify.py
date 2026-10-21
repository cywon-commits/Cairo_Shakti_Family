import numpy as np, json, sys, time
from real_islands import RealArray, full_J
from cand_response import STATES, names, NX, NY
exec(open("real_thresholds.py").read().split("rows = []")[0])
for arg in sys.argv[1:]:
    X, d, key = arg.split(":"); d = float(d); rec = r[f"{d:.4f}"]; ks = threshold(rec, X, key); t0 = time.time(); out = []
    for f in (0.7, 1.3):
        E = {}
        for sg in (1, -1):
            A = RealArray(NX, NY, d, **{key: sg*f*ks}); _, J = full_J(A)
            for Y in names: E[Y] = min(E.get(Y, 1e9), 0.5*STATES[Y]@J@STATES[Y]/A.L.N)
        w = min(E, key=E.get); out.append(f"{f} x k*: winner {w} ({X} - best other {E[X] - min(v for Y, v in E.items() if Y != X):+.2f} K)")
    print(f"{X} via {key} at d={d}: predicted k* = {ks:.4f} -> " + " | ".join(out) + f" [{time.time()-t0:.0f}s]", flush=True)
