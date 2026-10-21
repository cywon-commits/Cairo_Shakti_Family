import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
R = {}
for f in glob.glob("cand_response12x4_*.json"): R.update({float(k): v for k, v in json.load(open(f)).items() if "L3p" in v})
D = np.array(sorted(R)); names = ["A", "C", "M", "W", "L3p"]
cols = {"A": "#f3c38b", "C": "#9ec3ea", "M": "#b9dfa7", "W": "#c9b3e6", "L3": "#e8a0a0", "L3p": "#e8a0a0"}
def g(X, key, d): return np.interp(d, D, [R[x][X][key] for x in D])
def E(X, d, e=0, t=0, n=0): return g(X, "E0", d) - g(X, "|Q_eps|", d)*abs(e) - g(X, "|Q_theta|", d)*abs(t) - g(X, "|Q_eta|", d)*abs(n)
def win(d, **kw): return int(np.argmin([E(X, d, **kw) for X in names]))
fig, axs = plt.subplots(1, 4, figsize=(24, 6.2))
dg = np.linspace(D[0], D[-1], 400)
Emin = np.array([min(E(X, d) for X in names) for d in dg])
for X in names: axs[0].plot(dg, [E(X, d) for d in dg] - Emin, lw=2.5, color=cols[X] if X != "C" else "#2f6db5", label=X.replace("L3p", "L3′"))
axs[0].set_ylim(-0.002, 0.04); axs[0].set_xlabel("d̄"); axs[0].set_ylabel("E − E_min  (per spin)"); axs[0].legend(); axs[0].set_title("(a) zero-field energies (12x4 torus)")
cmap = ListedColormap([cols[X] for X in names])
for ax, key, rng, lab in [(axs[1], "e", 0.012, "anisotropy ε"), (axs[2], "t", 0.04, "twist θ (rad)"), (axs[3], "n", 0.012, "axial strain η")]:
    kg = np.linspace(0, rng, 200)
    W = np.array([[win(d, **{key: k}) for d in dg] for k in kg])
    ax.pcolormesh(dg, kg, W, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto"); ax.set_xlabel("d̄"); ax.set_ylabel(lab)
    ax.set_title(f"({'bcd'[['e','t','n'].index(key)]}) {lab.split(' (')[0]} only")
for X in names: axs[1].plot([], [], "s", ms=14, color=cols[X], label=X.replace("L3p", "L3′"))
axs[1].legend(loc="upper right")
plt.suptitle("Candidate crystals A, C, columnar M, wall crystal W, ladder L3: first-order estimates (needle g = 0, rc = 64a; linear range tested: ε ≤ 0.002, θ ≤ 0.01, η ≤ 0.004)", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig("candidates_phase_estimate.png", dpi=105)
# thresholds: smallest single-knob value that makes X the ground state
print("smallest single deformation that makes each candidate the ground state (None = not within range):")
for d in (0.137, 0.145, 0.150, 0.153, 0.160, 0.20, 0.25):
    row = []
    for X in names:
        best = []
        for key, rng in (("e", 0.03), ("t", 0.1), ("n", 0.03)):
            ks = np.linspace(0, rng, 3001); ok = [k for k in ks if names[win(d, **{key: k})] == X]
            if ok: best.append(f"{'εθη'['etn'.index(key)]}={ok[0]:.4f}")
        row.append(f"{X}: {', '.join(best) if best else '—'}")
    gs = names[win(d)]
    print(f"  d={d:.3f} (GS {gs}): " + " | ".join(row))
