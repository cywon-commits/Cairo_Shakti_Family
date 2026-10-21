import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("plot_candidates.py").read().split("# thresholds")[0].replace('plt.savefig("candidates_phase_estimate.png", dpi=105)', ''))
V = json.load(open("verify_exact.json")); col = {"A": "#c06a00", "C": "#1f5fa8", "M": "#2e8b2e", "W": "#6a3fb0", "L3": "#b22222", "L3p": "#b22222"}
axk = {"eps": axs[1], "theta": axs[2], "eta": axs[3]}
for key, v in V.items():
    d, knob, k = key.split("_"); ax = axk[knob]
    ax.plot(float(d), float(k), "o", ms=9, mfc=col[v["winner_exact"]], mec="k", mew=1.2)
for nm, d, knob, k in [("P1", 0.153, "eta", 0.004), ("P2", 0.150, "eta", 0.002), ("P3", 0.137, "theta", 0.02), ("P4", 0.25, "eps", 0.01)]:
    axk[knob].plot(d, k, "*", ms=22, mfc="white", mec="k", mew=1.5); axk[knob].annotate(nm, (d, k), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold")
axs[3].plot([], [], "o", ms=9, mfc="white", mec="k", label="exact energies (12x4), colour = winner"); axs[3].plot([], [], "*", ms=16, mfc="white", mec="k", label="12x12 annealing checks")
axs[3].legend(loc="upper left", fontsize=10)
plt.suptitle("Candidate phase estimates with verification: all 63 exact points agree with the linear estimate; 12x12 annealing finds nothing below the predicted state", fontsize=12)
plt.savefig("candidates_phase_verified_L3p.png", dpi=105)
