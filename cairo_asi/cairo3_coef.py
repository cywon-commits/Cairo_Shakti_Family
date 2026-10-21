import json, sys, time, numpy as np
from cairo3 import energies
EPS, TH = 0.002, 0.01; out = {}
for db in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    r0, re, rt, rb = energies(db), energies(db, eps=EPS), energies(db, theta=TH), energies(db, eps=EPS, theta=TH)
    A = lambda r: min(r["A1"], r["A2"]); C = lambda r: min(r["C+"], r["C-"])
    dEAC = A(r0) - C(r0)                                   # > 0: C wins (bulk)
    lamA = ((A(r0) - A(re)) - (C(r0) - C(re)))/EPS          # relative gain of A per unit eps
    lamC = ((C(r0) - C(rt)) - (A(r0) - A(rt)))/TH           # relative gain of C per rad
    pred = dEAC - lamA*EPS + lamC*TH; actual = A(rb) - C(rb)
    out[db] = dict(EA_minus_EC=dEAC, lamA=lamA, lamC=lamC, both_pred=pred, both_actual=actual)
    print(f"d_bar={db:.4f}: E_A - E_C = {dEAC:+.5f}; lambda_A = {lamA:.2f}/unit eps; lambda_C = {lamC:.3f}/rad; "
          f"both (eps={EPS}, theta={TH}): predicted {pred:+.5f} vs actual {actual:+.5f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"cairo3_coef_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
