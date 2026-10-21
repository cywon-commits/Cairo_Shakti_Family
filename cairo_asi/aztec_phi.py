"""Uniform Aztec-diamond domino tilings (plaquette-flip MCMC) and the block-parity order parameter
phi = F_p (-1)^{i+j} over 2x2 plaquettes (flippable F_p = 1)."""
import numpy as np, sys, matplotlib
from numba import njit
matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
S = 2*n; off = n                               # cell (i,j), i,j in [-n, n-1] -> array index +n
inside = np.zeros((S, S), np.bool_)
for a in range(S):
    for b in range(S):
        i, j = a - off, b - off
        inside[a, b] = abs(i + 0.5) + abs(j + 0.5) <= n
# partner direction code per cell: 0 right, 1 left, 2 up, 3 down ; start: all horizontal
part = -np.ones((S, S), np.int64)
for b in range(S):
    cols = [a for a in range(S) if inside[a, b]]
    for k in range(0, len(cols), 2):
        part[cols[k], b] = 0; part[cols[k+1], b] = 1

@njit(cache=True)
def sweep(part, inside, nsteps, seed):
    np.random.seed(seed); S = part.shape[0]
    for _ in range(nsteps):
        a = np.random.randint(S - 1); b = np.random.randint(S - 1)
        if not (inside[a, b] and inside[a+1, b] and inside[a, b+1] and inside[a+1, b+1]): continue
        if part[a, b] == 0 and part[a, b+1] == 0:          # two horizontal -> vertical
            part[a, b] = 2; part[a, b+1] = 3; part[a+1, b] = 2; part[a+1, b+1] = 3
        elif part[a, b] == 2 and part[a+1, b] == 2:        # two vertical -> horizontal
            part[a, b] = 0; part[a+1, b] = 1; part[a, b+1] = 0; part[a+1, b+1] = 1

@njit(cache=True)
def plaquettes(part, inside):
    S = part.shape[0]; F = np.zeros((S - 1, S - 1))
    for a in range(S - 1):
        for b in range(S - 1):
            if inside[a, b] and inside[a+1, b] and inside[a, b+1] and inside[a+1, b+1]:
                if (part[a, b] == 0 and part[a, b+1] == 0) or (part[a, b] == 2 and part[a+1, b] == 2): F[a, b] = 1.0
    return F

ncell = inside.sum()
sweep(part, inside, 400*ncell*n, SEED)                        # equilibrate
Fm = np.zeros((S - 1, S - 1)); ns = 300
for t in range(ns):
    sweep(part, inside, 20*ncell, 1000*SEED + t); Fm += plaquettes(part, inside)
Fm /= ns
ii, jj = np.meshgrid(np.arange(S - 1) - off, np.arange(S - 1) - off, indexing="ij")
sgn = (-1.0)**(ii + jj); phi_raw = Fm*sgn
# coarse-grain over 2x2 windows of plaquettes (removes the trivial alternation)
def cg(M, w=2):
    out = np.zeros_like(M)
    for a in range(M.shape[0] - w + 1):
        for b in range(M.shape[1] - w + 1): out[a, b] = M[a:a+w, b:b+w].sum()
    return out
Fc, Pc = cg(Fm), cg(phi_raw)
with np.errstate(invalid="ignore", divide="ignore"): Phi = np.where(Fc > 0.02, Pc/Fc, np.nan)
# plaquette-centre coordinates in units of n
X = (ii + 1)/n; Y = (jj + 1)/n; r = np.abs(X) + np.abs(Y)
def region(mask, name):
    f = Fm[mask].mean(); p = phi_raw[mask].sum()/max(Fm[mask].sum(), 1e-12)
    print(f"  {name:40s}: <F> = {f:.3f}, Phi = {p:+.3f}")
print(f"Aztec diamond n={n}: {ncell} cells, {ns} samples")
inner = ~np.isnan(Phi)
region((r < 0.35), "centre (|x|+|y| < 0.35 n)")
for qx, qy, nm in [(1, 0, "east axis"), (-1, 0, "west axis"), (0, 1, "north axis"), (0, -1, "south axis")]:
    m = (np.abs(X - qx*0.55) < 0.12) & (np.abs(Y - qy*0.55) < 0.12); region(m, f"{nm}, 0.55 n from centre (liquid)")
for qx, qy, nm in [(1, 1, "NE diagonal"), (-1, 1, "NW diagonal"), (-1, -1, "SW diagonal"), (1, -1, "SE diagonal")]:
    m = (np.abs(X - qx*0.3) < 0.1) & (np.abs(Y - qy*0.3) < 0.1); region(m, f"{nm}, (+-0.3n, +-0.3n) (liquid)")
for qx, qy, nm in [(1, 0, "east corner"), (0, 1, "north corner")]:
    m = (np.abs(X - qx*0.9) < 0.06) & (np.abs(Y - qy*0.0) < 0.06) if qx else (np.abs(Y - 0.9) < 0.06) & (np.abs(X) < 0.06)
    region(m, f"{nm} (frozen)")
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
im0 = ax[0].imshow(Fm.T, origin="lower", extent=[-n, n-2, -n, n-2], cmap="viridis"); ax[0].set_title("flippable-plaquette density <F_p>")
plt.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(Phi.T, origin="lower", extent=[-n, n-2, -n, n-2], cmap="bwr", vmin=-1, vmax=1); ax[1].set_title("block-parity Φ (2x2 coarse-grained); white/blank = no blocks")
plt.colorbar(im1, ax=ax[1], shrink=0.8)
t = np.linspace(0, 2*np.pi, 400)
for a in ax: a.plot(n/np.sqrt(2)*np.cos(t)*np.sqrt(2)*0.7071*np.sqrt(2)/np.sqrt(2), n/np.sqrt(2)*np.sin(t), "k--", lw=0.8)
plt.suptitle(f"Uniform Aztec diamond, order n = {n} (dashed: arctic circle)"); plt.tight_layout(); plt.savefig(f"aztec_phi_n{n}_s{SEED}.png", dpi=120); np.savez(f"aztec_phi_n{n}_s{SEED}.npz", Fm=Fm, phi_raw=phi_raw)
