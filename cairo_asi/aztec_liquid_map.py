"""Where does the liquid reappear? Per-cell entropy of the partner direction vs beta, model of aztec_small.py."""
import sys, numpy as np, matplotlib
n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
sys.argv = ["x", str(n), "0.0", "1.0", "1", "1"]
import aztec_small as az
matplotlib.use("Agg")
import matplotlib.pyplot as plt
betas = [0.0, 1.0, 2.0, 3.0, 4.5, 8.0]; ncell = int(az.inside.sum())
fig, axs = plt.subplots(1, len(betas), figsize=(4.0*len(betas), 4.6))
part = az.init()
az.sweep(part, az.valid, az.sgn, 0.0, 1.0, 0.0, 400*ncell*n, 3)             # uniform first
summary = []
for k, b in enumerate(betas):
    # anneal gently from the previous beta
    prev = betas[k-1] if k else 0.0
    for bb in np.linspace(prev, b, 21)[1:]: az.sweep(part, az.valid, az.sgn, 0.0, 1.0, bb, 20*ncell, 11 + int(100*bb))
    az.sweep(part, az.valid, az.sgn, 0.0, 1.0, b, 200*ncell*n//4, 17 + k)
    cnt = np.zeros(part.shape + (4,)); ns = 300
    for t in range(ns):
        az.sweep(part, az.valid, az.sgn, 0.0, 1.0, b, 10*ncell, 1000*k + t)
        for d in range(4): cnt[..., d] += (part == d)
    p = cnt/ns
    with np.errstate(divide="ignore", invalid="ignore"): H = -np.nansum(np.where(p > 0, p*np.log(p), 0.0), axis=2)
    H[~az.inside] = np.nan
    liquid = np.nanmean(H > 0.3)
    ne, no = az.counts(part, az.valid, az.sgn)
    summary.append((b, liquid, ne, no))
    ax = axs[k]; S = part.shape[0]; off = n + 1
    ax.imshow(H.T, origin="lower", extent=[-off, S - off, -off, S - off], cmap="magma", vmin=0, vmax=np.log(2))
    ax.set_title(f"β = {b:g}: liquid fraction {liquid:.2f}\nlast sample blocks (even, odd) = ({ne}, {no})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    print(f"beta={b:g}: fraction of cells with partner entropy > 0.3: {liquid:.2f}; blocks (even, odd) in last sample = ({ne}, {no})", flush=True)
fig.suptitle(f"Aztec diamond n = {n}, δ = 0: per-cell entropy of the domino direction (black = frozen or crystal, bright = liquid)", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(f"aztec_liquid_map_n{n}.png", dpi=110)
