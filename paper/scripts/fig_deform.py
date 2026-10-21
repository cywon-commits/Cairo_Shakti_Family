import sys, os; from pathlib import Path; _H = Path(__file__).resolve().parent; _CODE = _H.parents[1] / "cairo_asi"
sys.path.insert(0, str(_H)); sys.path.insert(0, str(_CODE)); os.chdir(_CODE)   # data files are read relative to cairo_asi/
import numpy as np, json, glob
from figstyle import *
from matplotlib.colors import ListedColormap
R = {}
for f in glob.glob("cand_response12x4_*.json"): R.update({float(k): v for k, v in json.load(open(f)).items() if "L3p" in v})
D = np.array(sorted(R)); names = ["A", "C", "M", "W", "L3p"]
def g(X, key, d): return np.interp(d, D, [R[x][X][key] for x in D])
def E(X, d, e=0, t=0, n=0): return g(X, "E0", d) - g(X, "|Q_eps|", d)*abs(e) - g(X, "|Q_theta|", d)*abs(t) - g(X, "|Q_eta|", d)*abs(n)
cmap = ListedColormap([FILL[X] for X in names])
V = json.load(open("verify_exact.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
dg = np.linspace(D[0], D[-1], 300)
spec = [("e", 0.012, 1, "anisotropy $\\varepsilon=d_1-d_2$", "eps"), ("t", np.radians(2.3), np.degrees(1), "twist $\\theta$ (deg)", "theta"), ("n", 0.012, 100, "axial strain $\\eta$ (%)", "eta")]
stars = [("P1", 0.153, "eta", 0.004), ("P2", 0.150, "eta", 0.002), ("P3", 0.137, "theta", 0.02), ("P4", 0.25, "eps", 0.01)]
for i, (ax, (key, kmax, conv, lab, vk)) in enumerate(zip(axs, spec)):
    kg = np.linspace(0, kmax, 220); W = np.array([[int(np.argmin([E(X, d, **{key: k}) for X in names])) for d in dg] for k in kg])
    ax.pcolormesh(dg, kg*conv, W, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto", rasterized=True)
    for kk, v in V.items():
        d, knob, k = kk.split("_")
        if knob != vk: continue
        w = v["winner_exact"]; w = "L3p" if w == "L3" else w
        ax.plot(float(d), float(k)*conv, "o", ms=4.2, mfc=COL[w], mec="k", mew=0.6, clip_on=False, zorder=4)
    for nm, d, knob, k in stars:
        if knob != vk: continue
        ax.plot(d, k*conv, "*", ms=11, mfc="white", mec="k", mew=0.9, clip_on=False, zorder=5)
        off = {"P1": (-5, 8), "P2": (-5, -14), "P3": (6, 8), "P4": (-17, 7)}[nm]
        ax.annotate(nm, (d, k*conv), xytext=off, textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("$d$"); ax.set_ylabel(lab); ax.set_xlim(D[0], D[-1]); ax.set_ylim(0, kmax*conv)
    ax.set_title(f"({'abc'[i]}) " + lab.split(" $")[0].split(" (")[0] + " only", fontsize=8.5, loc="left", pad=4)
hs = [plt.Rectangle((0, 0), 1, 1, fc=FILL[X], ec="0.4", lw=0.5) for X in names]
hs += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=5), plt.Line2D([], [], marker="*", ls="", mfc="white", mec="k", ms=9)]
fig.legend(hs, ["A", "C", "M (columnar)", "W", "L3$'$", "exact check (colour = winner)", "12$\\times$12 annealing check"], loc="lower center", ncol=7,
           frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.17), handlelength=1.2, columnspacing=0.9)
plt.subplots_adjust(wspace=0.42); save(fig, "fig5_deform")
print("ok")
