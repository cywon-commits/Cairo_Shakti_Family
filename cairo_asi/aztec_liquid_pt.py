"""Per-cell partner-direction entropy vs beta from parallel tempering (equilibrated), small n."""
import sys, numpy as np, matplotlib, json
n = int(sys.argv[1]); nround = int(sys.argv[2])
sys.argv = ["x", str(n), "0.0", "1.0", "1", "1"]
import aztec_small as az
matplotlib.use("Agg")
import matplotlib.pyplot as plt
betas = np.geomspace(0.1, 8.0, 20); M = len(betas); ncell = int(az.inside.sum()); rng = np.random.default_rng(4)
reps = [az.init() for _ in range(M)]; E = np.array([az.total_E(p, az.valid, az.sgn, 0.0, 1.0) for p in reps])
cnt = np.zeros((M,) + reps[0].shape + (4,)); nmeas = 0; blocks = np.zeros((M, 2))
for rnd in range(nround):
    for m in range(M):
        az.sweep(reps[m], az.valid, az.sgn, 0.0, 1.0, betas[m], 10*ncell, 7*rnd + 1000*m + 1); E[m] = az.total_E(reps[m], az.valid, az.sgn, 0.0, 1.0)
    for m in range(rnd % 2, M - 1, 2):
        x = (betas[m+1] - betas[m])*(E[m+1] - E[m])
        if x >= 0 or rng.random() < np.exp(x): reps[m], reps[m+1] = reps[m+1], reps[m]; E[m], E[m+1] = E[m+1], E[m]
    if rnd >= nround//4:
        nmeas += 1
        for m in range(M):
            for d in range(4): cnt[m, ..., d] += (reps[m] == d)
            blocks[m] += az.counts(reps[m], az.valid, az.sgn)
p = cnt/nmeas
with np.errstate(divide="ignore", invalid="ignore"): H = -np.sum(np.where(p > 0, p*np.log(p), 0.0), axis=3)
H[:, ~az.inside] = np.nan
res = []
for m in range(M):
    res.append(dict(beta=float(betas[m]), liquid=float(np.nanmean(H[m] > 0.3)), Hmean=float(np.nanmean(H[m])),
                    even=float(blocks[m, 0]/nmeas), odd=float(blocks[m, 1]/nmeas)))
json.dump(res, open(f"aztec_liquid_pt_n{n}.json", "w"), indent=1)
np.save(f"aztec_liquid_pt_H_n{n}.npy", H)
for r_ in res[::3] + [res[-1]]:
    print(f"n={n} beta={r_['beta']:.2f}: liquid fraction {r_['liquid']:.2f}, mean entropy {r_['Hmean']:.3f}, <blocks> even {r_['even']:.1f} odd {r_['odd']:.1f}")
