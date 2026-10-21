"""Step 1: exact (non-linear) energies of the candidate crystals at finite deformations on the common 12x4 torus.
Orientation: a 90-degree rotation maps eps -> -eps, eta -> -eta, and the mirror maps theta -> -theta, so for each candidate
the best orientation/variant at +k is min(E(+k), E(-k)) of the stored state."""
import numpy as np, json, sys, time, glob
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from hybrid_coupling import hybrid_coupling
from cand_response import CairoLattice4, STATES, names, NX, NY
R = {}
for f in glob.glob("cand_response12x4_*.json"): R.update({float(k): v for k, v in json.load(open(f)).items()})
D = np.array(sorted(R))
def lin(X, d, key, k): return np.interp(d, D, [R[x][X]["E0"] - R[x][X]["|Q_" + key + "|"]*k for x in D])
out = json.load(open("verify_exact.json")) if glob.glob("verify_exact.json") else {}
VALS = {"eps": [0.002, 0.005, 0.01], "theta": [0.01, 0.02, 0.04], "eta": [0.002, 0.004, 0.008]}
for db in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    for key, vals in VALS.items():
        for k in vals:
            Ep = {}; 
            for sg in (+1, -1):
                L = CairoLattice4(NX, NY, db, **{key: sg*k}); J = hybrid_coupling(L, 64.0)
                for X in names: Ep[(X, sg)] = 0.5*STATES[X]@J@STATES[X]/L.N
            ex = {X: min(Ep[(X, 1)], Ep[(X, -1)]) for X in names}
            li = {X: lin(X, db, key, k) for X in names}
            wex, wli = min(ex, key=ex.get), min(li, key=li.get)
            srt = sorted(ex.values()); margin = srt[1] - srt[0]
            out[f"{db}_{key}_{k}"] = dict(exact=ex, linear=li, winner_exact=wex, winner_linear=wli, margin=margin)
            print(f"d={db:.3f} {key}={k:<6}: exact winner {wex:3s} (margin {margin:.4f}) | linear predicted {wli:3s} {'OK' if wex == wli else 'MISMATCH'}", flush=True)
    json.dump(out, open("verify_exact.json", "w"), indent=1)
    print(f"   [{time.time()-t0:.0f}s]")
