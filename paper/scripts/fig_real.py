import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts")
import numpy as np, json
from figstyle import *
from matplotlib.colors import ListedColormap
Rr = {float(k): v for k, v in json.load(open("/mnt/user-data/outputs/cairo_asi/real_islands_scan.json")).items() if v["valid"] and "L3p" in v["E0"]}
D = np.array(sorted(Rr)); names = ["A", "C", "M", "W", "L3p"]
def g(X, key, d):
    if key == "E0": return np.interp(d, D, [Rr[x]["E0"][X] for x in D])
    return np.interp(d, D, [Rr[x]["lam"][key][X] for x in D])
def E(X, d, eps=0, theta=0, eta=0): return g(X, "E0", d) - g(X, "eps", d)*abs(eps) - g(X, "theta", d)*abs(theta) - g(X, "eta", d)*abs(eta)
def kstar(X, d, key):
    ks = [0.0]
    for Y in names:
        if Y == X: continue
        dE, dl = g(X, "E0", d) - g(Y, "E0", d), g(X, key, d) - g(Y, key, d)
        if dE <= 0: continue
        if dl <= 1e-3*max(abs(g(X, key, d)), abs(g(Y, key, d))) + 1e-6: return None
        ks.append(dE/dl)
    return max(ks)
cmap = ListedColormap([FILL[X] for X in names]); dg = np.linspace(D[0], D[-1], 300)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
spec = [("eps", 0.012, 1, "anisotropy $\\varepsilon=d_1-d_2$"), ("theta", np.radians(1.5), np.degrees(1), "twist $\\theta$ (deg)"), ("eta", 0.08, 100, "axial strain $\\eta$ (%)")]
checks = {"theta": [("M", 0.105)], "eps": [("A", 0.25)], "eta": [("L3p", 0.17)]}
checks["theta"].append(("C", 0.20))
for i, (ax, (key, kmax, conv, lab)) in enumerate(zip(axs, spec)):
    kg = np.linspace(0, kmax, 220); W = np.array([[int(np.argmin([E(X, d, **{key: k}) for X in names])) for d in dg] for k in kg])
    ax.pcolormesh(dg, kg*conv, W, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto", rasterized=True)
    for X, d in checks.get(key, []):
        k = kstar(X, d, key); ax.plot(d, k*conv, "*", ms=10, mfc=COL[X], mec="k", mew=0.8, clip_on=False, zorder=5)
    ax.set_xlabel("$d$"); ax.set_ylabel(lab); ax.set_xlim(D[0], D[-1]); ax.set_ylim(0, kmax*conv)
    ax.set_title(f"({'abc'[i]}) " + lab.split(" $")[0].split(" (")[0] + " only", fontsize=8.5, loc="left", pad=4)
hs = [plt.Rectangle((0, 0), 1, 1, fc=FILL[X], ec="0.4", lw=0.5) for X in names] + [plt.Line2D([], [], marker="*", ls="", mfc="0.7", mec="k", ms=9)]
fig.legend(hs, ["A", "C", "M (columnar)", "W", "L3$'$", "predicted threshold, verified (winner flips between 0.7 and 1.3$\\times$)"], loc="lower center", ncol=6,
           frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.17), handlelength=1.2, columnspacing=0.9)
plt.subplots_adjust(wspace=0.42); save(fig, "fig8_real")
print("ok")
