import sys; sys.path.insert(0, "/mnt/user-data/outputs/paper/scripts")
import numpy as np, json
from figstyle import *
UNIT_K = 1e-7*(790e3*((300 - 100)*100 + np.pi*50**2)*3e-27)**2/(472e-9)**3/1.380649e-23
S = {}
for f in ("stripe_sums_0.20_0.25.json", "stripe_sums_0.153_0.17.json"): S.update(json.load(open("/mnt/user-data/outputs/cairo_asi/" + f)))
cm = plt.get_cmap("viridis"); DC = {"0.153": cm(0.0), "0.17": cm(0.33), "0.2": cm(0.62), "0.25": cm(0.9)}
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.05)); mu = np.linspace(0.5, 3.5, 300)
ax = axs[0]
for d in ["0.153", "0.17", "0.2", "0.25"]:
    r = S[d]["0"]; ax.plot(mu, r["SS"] + r["LS"]*mu + r["LL"]*mu**2, color=DC[d])
ax.axhline(0, color="0.4", lw=0.7)
ax.plot([3.15], [-130.0/UNIT_K], "s", ms=5, mfc="white", mec=DC["0.25"], mew=1.3, zorder=5); ax.plot([2.42], [-55.7/UNIT_K], "s", ms=5, mfc="white", mec=DC["0.2"], mew=1.3, zorder=5)
ax.set_xlabel("moment ratio $\\mu$"); ax.set_ylabel("$\\sigma$ per wall cell ($\\mu_0 m_S^2/4\\pi a^3$)"); ax.set_ylim(-0.85, 0.12)
ax.set_title("(a) C$|$C$'$ walls\n(basketweave family)", fontsize=8.5, loc="left", pad=4)
ax = axs[2]
for d, (a, b, c) in {"0.153": (-0.0285, -0.0681, -0.0115), "0.2": (-0.2913, -0.0508, -0.0098)}.items(): ax.plot(mu, a + b*mu + c*mu**2, color=DC[d])
ax.plot([1.70], [-56.03/UNIT_K], "s", ms=5, mfc="white", mec=DC["0.153"], mew=1.3, zorder=5); ax.plot([2.42], [-144.32/UNIT_K], "s", ms=5, mfc="white", mec=DC["0.2"], mew=1.3, zorder=5)
ax.axhline(0, color="0.4", lw=0.7); ax.set_ylim(-0.85, 0.12); ax.set_xlabel("moment ratio $\\mu$"); ax.set_ylabel("$\\sigma_{\\rm col}$ per wall cell ($\\mu_0 m_S^2/4\\pi a^3$)")
ax.set_title("(c) walls between columnar\nvariants (ladder family)", fontsize=8.5, loc="left", pad=4)
ax = axs[1]
series = [("W, real., $d$=0.25", [-130.0, -127.56, -130.23], DC["0.25"], "s", True), ("W, real., $d$=0.20", [-55.70, -54.31, -55.70], DC["0.2"], "s", True),
          ("L3$'$, ideal, $d$=0.153", [-0.1081, -0.0971, -0.0969], DC["0.153"], "o", False), ("L3$'$, real., $d$=0.153", [-56.03, -50.60], DC["0.153"], "^", True),
          ("L3$'$, real., $d$=0.20", [-144.32, -137.35], DC["0.2"], "^", True)]
for name, vals, c, mk, real in series:
    v = np.array(vals); ax.plot(np.arange(1, len(v) + 1), v/v[0], marker=mk, color=c, ms=4.5, lw=1.0, mfc="white" if real else c, ls="-" if mk == "s" else "--")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["densest", "2nd", "3rd"]); ax.set_xlim(0.7, 3.3); ax.set_ylim(0.85, 1.05)
ax.set_xlabel("wall spacing"); ax.set_ylabel("$\\sigma/\\sigma_{\\rm densest}$")
ax.text(2.0, 0.995, "C$|$C$'$ walls", fontsize=7, ha="center", va="bottom"); ax.text(2.55, 0.875, "columnar walls", fontsize=7, ha="center")
ax.set_title("(b) wall energy vs\nwall spacing", fontsize=8.5, loc="left", pad=4)
hs = [plt.Line2D([], [], color=DC[d], lw=1.6) for d in ["0.153", "0.17", "0.2", "0.25"]]
hs += [plt.Line2D([], [], marker="s", ls="", mfc="white", mec="k", ms=5), plt.Line2D([], [], marker="o", ls="", mfc="k", mec="k", ms=4.5)]
fig.legend(hs, ["$d=0.153$", "$d=0.17$", "$d=0.20$", "$d=0.25$", "realistic islands (open)", "ideal islands (filled)"], loc="lower center", ncol=6, frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.0)
plt.subplots_adjust(wspace=0.62, bottom=0.27); save(fig, "fig6_walls")
print("ok")
