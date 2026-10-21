import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
r = json.load(open("coef_fixed_0.137_0.145_0.153_0.17_0.20_0.25.json"))
D = np.array(sorted(float(k) for k in r)); K = {float(k): v for k, v in r.items()}
def f(state, key, d): return np.interp(d, D, [K[x][state][key] for x in D])
def winner(d, e, t):
    E = [f(X, "E0", d) - f(X, "|Q_eps|", d)*abs(e) - f(X, "|Q_theta|", d)*abs(t) for X in ("A", "C", "M")]
    return int(np.argmin(E))
cmap = ListedColormap(["#f3c38b", "#9ec3ea", "#b9dfa7"]); names = ["A", "C", "M (columnar)"]
fig, axs = plt.subplots(1, 3, figsize=(19, 6.0))
# (a) d-bar vs eps, theta = 0
dg = np.linspace(D[0], D[-1], 300); eg = np.linspace(0, 0.012, 240)
W = np.array([[winner(d, e, 0) for d in dg] for e in eg])
axs[0].pcolormesh(dg, eg, W, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
axs[0].axhline(0.004, color="k", ls=":", lw=1); axs[0].text(0.139, 0.0043, "tested linear range", fontsize=10)
axs[0].set_xlabel("d̄"); axs[0].set_ylabel("anisotropy ε = d1 − d2"); axs[0].set_title("(a) anisotropy only (θ = 0)")
# (b) d-bar vs theta, eps = 0
tg = np.linspace(0, 0.04, 240)
W = np.array([[winner(d, 0, t) for d in dg] for t in tg])
axs[1].pcolormesh(dg, tg, W, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
axs[1].axhline(0.01, color="k", ls=":", lw=1); axs[1].text(0.139, 0.0105, "tested linear range", fontsize=10)
axs[1].set_xlabel("d̄"); axs[1].set_ylabel("twist θ (rad)"); axs[1].set_title("(b) twist only (ε = 0)")
# (c) eps vs theta at d-bar = 0.153 (near d_c)
eg2 = np.linspace(0, 0.004, 200); tg2 = np.linspace(0, 0.04, 200)
W = np.array([[winner(0.153, e, t) for e in eg2] for t in tg2])
axs[2].pcolormesh(eg2, tg2, W, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
axs[2].set_xlabel("anisotropy ε"); axs[2].set_ylabel("twist θ (rad)"); axs[2].set_title("(c) both, at d̄ = 0.153 (near d_c)")
axs[2].axhline(0.01, color="k", ls=":", lw=1)
for ax in axs:
    for i, nm in enumerate(names): ax.plot([], [], "s", ms=14, color=cmap(i), label=nm)
axs[0].legend(loc="upper right", fontsize=11, frameon=True)
plt.suptitle("Estimated phase diagrams from first-order responses (4x2 states A, C, columnar M only; needle g = 0); dotted = end of tested linear range", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig("deform_phase_estimate.png", dpi=110)
# key thresholds for the text
for d in (0.137, 0.153, 0.25):
    eA, eC, eM = (f(X, "E0", d) for X in ("A", "C", "M"))
    print(f"d_bar={d}: E_C-E_A {eC-eA:+.4f}; M-A {eM-eA:+.4f}; M-C {eM-eC:+.4f}; "
          f"theta for M to beat A: {(eM-eA)/f('M','|Q_theta|',d):.4f} rad; eps for M to beat C: {max(eM-eC,0)/f('M','|Q_eps|',d):.5f}")
