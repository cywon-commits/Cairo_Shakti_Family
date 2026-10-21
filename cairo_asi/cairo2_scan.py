import sys, json, numpy as np, time
from cairo2 import energies
db = float(sys.argv[1]); epss = [float(x) for x in sys.argv[2].split(",")]; rows = []
for e in epss:
    t0 = time.time(); r = energies(db + e/2, db - e/2)
    gap = r["C"] - min(r["A1"], r["A2"])           # > 0: A wins
    rows.append((e, r["A1"], r["A2"], r["C"], gap, r["El_C"] - min(r["El_A1"], r["El_A2"])))
    print(f"d_bar={db:.3f} eps={e:+.4f}: E_C - min E_A = {gap:+.5f} (vertex part {rows[-1][5]:+.5f}); winner {'A' if gap > 0 else 'C'}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(rows, open(f"cairo2_scan_{db:.3f}.json", "w"))
