import sys, json, numpy as np, time
from cairo2 import energies
eps = 0.004; out = {}
for db in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    ep, em = energies(db + eps/2, db - eps/2), energies(db - eps/2, db + eps/2)    # eps = d1 - d2
    Q = {k: (ep[k] - em[k])/eps for k in ("A1", "A2", "C")}
    out[db] = Q
    print(f"d_bar={db:.4f}: Q_A1 = {Q['A1']:+.4f}, Q_A2 = {Q['A2']:+.4f}, Q_C = {Q['C']:+.5f}  (per spin per unit eps; D/a^3)  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"cairo2_lambda_{sys.argv[1].replace(',', '_')}.json", "w"))
