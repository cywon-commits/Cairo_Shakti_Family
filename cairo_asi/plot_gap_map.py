import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cairo_asi import D_EQ
r = json.load(open("gap_scan.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
cols = {"A": "#E24B4A", "C": "#378ADD", "other": "#999999"}
for k, v in r.items():
    g, d = map(float, k.split("_"))
    if v["valid"]: ax[0].scatter(d, g, s=90, c=cols[v["sel"]], edgecolors="k", zorder=3)
    else: ax[0].scatter(d, g, s=90, marker="x", c="k", zorder=3)
gs = sorted({float(k.split("_")[0]) for k in r}); dcs = []
for g in gs:
    pts = sorted((float(k.split("_")[1]), v) for k, v in r.items() if float(k.split("_")[0]) == g and v["valid"])
    for (d1, v1), (d2, v2) in zip(pts, pts[1:]):
        if v1["EA_minus_EC"] < 0 <= v2["EA_minus_EC"]:
            dcs.append((d1 + (d2-d1)*(-v1["EA_minus_EC"])/(v2["EA_minus_EC"]-v1["EA_minus_EC"]), g)); break
ax[0].plot(*zip(*dcs), "k-", lw=2, label="A|C transition d_c(g)")
for x, t in [(D_EQ, "equilateral Cairo"), (0.25, "Shakti")]:
    ax[0].axvline(x, color="0.5", lw=0.8); ax[0].text(x, 0.33, t, ha="center", fontsize=8)
ax[0].scatter([], [], c=cols["A"], edgecolors="k", label="A selected"); ax[0].scatter([], [], c=cols["C"], edgecolors="k", label="C selected")
ax[0].scatter([], [], marker="x", c="k", label="manifold broken (z3 level crossing)")
ax[0].set_xlabel("d"); ax[0].set_ylabel("vertex gap g / a"); ax[0].set_ylim(-0.02, 0.36); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax[0].set_title("selection in the (d, g) plane (4x2 exact, rc = 64a)", fontsize=10)
for d, c in [(0.137, "#E24B4A"), (0.25, "#378ADD")]:
    pts = sorted((float(k.split("_")[0]), v) for k, v in r.items() if abs(float(k.split("_")[1]) - d) < 1e-3 and v["valid"])
    ax[1].plot([g for g, _ in pts], [v["flip_p10"]/v["spread"] for _, v in pts], "o-", color=c, label=f"d = {d}")
ax[1].set_yscale("log"); ax[1].set_xlabel("vertex gap g / a"); ax[1].set_ylabel("single-flip defect cost / manifold energy spread")
ax[1].set_title("separation of vertex (kinetic) and selection energy scales", fontsize=10); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("gap_map.png", dpi=125)
