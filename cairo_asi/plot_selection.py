import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sel = json.load(open("selT_n8.json")); wt = json.load(open("wall_tension.json"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cmap = plt.get_cmap("coolwarm")
ds = sorted(sel, key=float)
for d in ds:
    r = sel[d]; T = [x["T"] for x in r]; c = cmap((float(d) - 0.10)/(0.25 - 0.10))
    ax[0].plot(T, [x["C"] for x in r], "-o", ms=3, color=c, label=f"d={float(d):.3f}")
    ax[1].plot(T, [x["psiA"] for x in r], "-", color=c); ax[1].plot(T, [x["psiC"] for x in r], "--", color=c)
for a in ax[:2]: a.set_xscale("log"); a.set_xlabel("T  (D/a³)")
ax[0].set_ylabel("C / N"); ax[0].legend(fontsize=7); ax[0].set_title("specific heat within the dimer manifold (8x8)")
ax[1].set_ylabel("|ψ|"); ax[1].set_title("solid: |ψ_A| (partner orientation), dashed: |ψ_C| (z4 sublattice)")
dw = sorted(wt, key=float)
sw = [np.mean([v for v in (wt[d]["sigma_wide"], wt[d]["sigma_sharp"]) if v is not None]) for d in dw]
ax[2].plot([float(d) for d in dw], sw, "ks-"); ax[2].axhline(0, color="0.5", lw=0.8)
ax[2].axvline(0.1527, color="r", ls=":", label="A→C (d≈0.153)"); ax[2].axvline(0.25, color="b", ls=":", label="Shakti")
ax[2].set_xlabel("d"); ax[2].set_ylabel("σ per wall cell  (D/a³)"); ax[2].set_title("tension of sublattice domain walls in the C region")
ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("selection_strength.png", dpi=120)
for d in ds:
    r = sel[d]; k = int(np.argmax([x["C"] for x in r]))
    print(f"d={float(d):.4f}: T_peak {r[k]['T']:.3f}, C_peak {r[k]['C']:.4f}, low-T |psiA| {r[-1]['psiA']:.2f} |psiC| {r[-1]['psiC']:.2f}")
