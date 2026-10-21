import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
names = ["A", "C", "M", "W", "L3p"]; cols = {"A": "#f3c38b", "C": "#9ec3ea", "M": "#b9dfa7", "W": "#c9b3e6", "L3": "#e8a0a0", "L3p": "#e8a0a0"}
lc = {"A": "#c06a00", "C": "#1f5fa8", "M": "#2e8b2e", "W": "#6a3fb0", "L3": "#b22222", "L3p": "#b22222"}
# needle model (g = 0): fixed-state total responses; realistic: vertex responses (dipolar part ~10%)
N = {}
for f in glob.glob("cand_response12x4_*.json"): N.update({float(k): v for k, v in json.load(open(f)).items() if "L3p" in v})
def needle(X, key, d): return np.interp(d, sorted(N), [N[x][X][key] for x in sorted(N)])
Rr = {float(k): v for k, v in json.load(open("real_islands_scan.json")).items() if v["valid"]}
def real(X, key, d):
    D = sorted(Rr)
    if key == "E0": return np.interp(d, D, [Rr[x]["E0"][X] for x in D])
    k = {"|Q_eps|": "eps", "|Q_theta|": "theta", "|Q_eta|": "eta"}[key]; return np.interp(d, D, [Rr[x]["lam"][k][X] for x in D])
models = [("ideal needle (g = 0), D/a^3", needle, (0.137, 0.25)), ("realistic islands (Saccone-type gaps, width 100 nm), K/island", real, (0.105, 0.25))]
fig, axs = plt.subplots(2, 4, figsize=(24, 11))
cmap = ListedColormap([cols[X] for X in names])
for row, (title, F, (d0, d1)) in enumerate(models):
    dg = np.linspace(d0, d1, 300)
    def E(X, d, e=0, t=0, n=0): return F(X, "E0", d) - F(X, "|Q_eps|", d)*abs(e) - F(X, "|Q_theta|", d)*abs(t) - F(X, "|Q_eta|", d)*abs(n)
    Emin = np.array([min(E(X, d) for X in names) for d in dg])
    ax = axs[row, 0]
    for X in names: ax.plot(dg, np.array([E(X, d) for d in dg]) - Emin, lw=2.5, color=lc[X], label=X.replace("L3p", "L3′"))
    top = 0.04 if row == 0 else 30
    ax.set_ylim(-0.03*top, top); ax.set_xlim(0.10, 0.25); ax.set_xlabel("d̄"); ax.set_ylabel("E − E_min"); ax.set_title(f"zero field: {title}", fontsize=11); ax.legend(fontsize=10)
    for col, (key, kmax, conv, lab) in enumerate([("e", 0.012, 1, "anisotropy ε = d1 − d2"), ("t", np.radians(1.5), np.degrees(1), "twist θ (deg)"), ("n", 0.08, 100, "axial strain η (%)")]):
        ax = axs[row, col + 1]; kg = np.linspace(0, kmax, 220)
        W = np.array([[int(np.argmin([E(X, d, **{key: k}) for X in names])) for d in dg] for k in kg])
        ax.pcolormesh(dg, kg*conv, W, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto"); ax.set_xlim(0.10, 0.25)
        ax.set_xlabel("d̄"); ax.set_ylabel(lab); ax.set_title(("ideal" if row == 0 else "realistic") + f": {lab.split(' (')[0].split(' =')[0]} only", fontsize=11)
        if row == 0: ax.axvspan(0.10, 0.137, color="0.85")
for X in names: axs[0, 1].plot([], [], "s", ms=14, color=cols[X], label=X.replace("L3p", "L3′"))
axs[0, 1].legend(loc="upper left", fontsize=10); axs[0, 0].axvspan(0.10, 0.137, color="0.9"); axs[0, 0].text(0.104, 0.03, "not computed", fontsize=10)
plt.suptitle("Phase diagrams from the same candidates and the same first-order formula: ideal needle vs realistic islands", fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig("model_compare_phase_L3p.png", dpi=100)
