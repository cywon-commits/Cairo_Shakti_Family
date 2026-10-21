import json, numpy as np
from cand_response import CairoLattice4
r = json.load(open("real_islands_scan.json")); names = ["A", "C", "M", "W", "L3"]
def threshold(rec, X, key):
    E, lam = rec["E0"], rec["lam"][key]; ks = [0.0]
    for Y in names:
        if Y == X: continue
        dE, dl = E[X] - E[Y], lam[X] - lam[Y]
        if dE <= 0: continue
        if dl <= 1e-3*max(abs(lam[X]), abs(lam[Y])) + 1e-6: return None
        ks.append(dE/dl)
    return max(ks)
def units(d, key, k):
    if k is None: return "—"
    if k == 0: return "0 (already GS)"
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[q]].sum() == 7 for q in range(ref.N)]); scale = 472/ref.elen[is34].mean()
    if key == "eps": return f"{k:.4f} = {k/d*100:.1f}% of d, z3 shift {k/2*np.sqrt(2)*scale:.1f} nm"
    if key == "theta": return f"{k:.4f} rad = {np.degrees(k):.2f} deg"
    return f"{k*100:.2f}% strain"
rows = []
for dk in sorted(r, key=float):
    rec = r[dk]; d = float(dk)
    if not rec["valid"]: continue
    gs = min(rec["E0"], key=rec["E0"].get)
    print(f"d={d:.3f} (zero-field GS {gs}; E - E_GS: " + ", ".join(f"{X} {rec['E0'][X]-rec['E0'][gs]:+.1f} K" for X in names if X != gs) + ")")
    for X in names:
        if X == gs: continue
        best = [(key, threshold(rec, X, key)) for key in ("eps", "theta", "eta")]
        print(f"     {X:3s}: " + " | ".join(f"{key}: {units(d, key, k)}" for key, k in best))
