"""psi of the states in the barrier region of P(E) at T*(n) (slab/interface states vs. a third phase), d = 0.145."""
import json, sys, numpy as np
import fss_analyze as fa
d = 0.145; out = {}
for n in [int(x) for x in sys.argv[1:]]:
    N = 10*n*n; R = json.load(open(f"fss_d{d:.4f}_n{n}.json")); runs = [fa.load(d, n, s) for s in "AC"]
    Eall = np.concatenate([r["E"].ravel() for r in runs]); edges = np.arange(Eall.min() - fa.DE_FINE, Eall.max() + 2*fa.DE_FINE, fa.DE_FINE)
    W = fa.solve(runs, lambda r: slice(None), N, edges); T = R["Tstar"]
    eA, eC = R["eA"], R["eC"]; rows = {}
    for f0 in np.arange(0.0, 1.0001, 0.125):                   # windows along the line e_A -> e_C
        e = eA + f0*(eC - eA); w = 0.0625*(eC - eA)
        r = fa.psi_at(runs, W, T, 0.0, ((e - w)*N, (e + w)*N))["highE_peak"]
        rows[f"{f0:.3f}"] = dict(e=e, **r)
        print(f"n={n} frac {f0:.3f} e+39.84={e+39.84:+.4f}: weight {r['weight']:.3f}  rms psiA {r['rmsA']:.3f}  rms psiC {r['rmsC']:.3f}  A-like {r['fracA']:.2f}  C-like {r['fracC']:.2f}", flush=True)
    out[n] = rows
json.dump(out, open(f"fss_middle_d{d:.4f}.json", "w"), indent=1)
