"""Aztec diamond with parity-breaking weight P ~ exp(K sum_p F_p (-1)^{x_p+y_p}) (F_p = flippable plaquette).
K > 0 favours blocks on even plaquettes (C-type), K < 0 on odd plaquettes (A-type). Usage: python aztec_weighted.py n K seed"""
import numpy as np, sys
from numba import njit
n = int(sys.argv[1]); K = float(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
S = 2*n; off = n
inside = np.zeros((S, S), np.bool_)
for a in range(S):
    for b in range(S): inside[a, b] = abs(a - off + 0.5) + abs(b - off + 0.5) <= n
part = -np.ones((S, S), np.int64)
for b in range(S):
    cols = [a for a in range(S) if inside[a, b]]
    for k in range(0, len(cols), 2): part[cols[k], b] = 0; part[cols[k+1], b] = 1
par = np.array([[1.0 if (a + b) % 2 == 0 else -1.0 for b in range(S - 1)] for a in range(S - 1)])   # (-1)^{x+y}, x=a-off

@njit(cache=True)
def Fp(part, inside, a, b):
    S = part.shape[0]
    if a < 0 or b < 0 or a >= S - 1 or b >= S - 1: return 0.0
    if not (inside[a, b] and inside[a+1, b] and inside[a, b+1] and inside[a+1, b+1]): return 0.0
    if (part[a, b] == 0 and part[a, b+1] == 0) or (part[a, b] == 2 and part[a+1, b] == 2): return 1.0
    return 0.0

@njit(cache=True)
def run(part, inside, par, K, nsteps, seed):
    np.random.seed(seed); S = part.shape[0]; acc = 0
    for _ in range(nsteps):
        a = np.random.randint(S - 1); b = np.random.randint(S - 1)
        if Fp(part, inside, a, b) == 0.0: continue
        before = 0.0
        for da in range(-1, 2):
            for db in range(-1, 2):
                if 0 <= a+da < S-1 and 0 <= b+db < S-1: before += par[a+da, b+db]*Fp(part, inside, a+da, b+db)
        horiz = part[a, b] == 0
        if horiz:
            part[a, b] = 2; part[a, b+1] = 3; part[a+1, b] = 2; part[a+1, b+1] = 3
        else:
            part[a, b] = 0; part[a+1, b] = 1; part[a, b+1] = 0; part[a+1, b+1] = 1
        after = 0.0
        for da in range(-1, 2):
            for db in range(-1, 2):
                if 0 <= a+da < S-1 and 0 <= b+db < S-1: after += par[a+da, b+db]*Fp(part, inside, a+da, b+db)
        dS = after - before
        x = K*dS
        if x >= 0 or np.random.random() < np.exp(x): acc += 1
        else:                                               # undo
            if horiz:
                part[a, b] = 0; part[a+1, b] = 1; part[a, b+1] = 0; part[a+1, b+1] = 1
            else:
                part[a, b] = 2; part[a, b+1] = 3; part[a+1, b] = 2; part[a+1, b+1] = 3
    return acc

@njit(cache=True)
def allF(part, inside):
    S = part.shape[0]; F = np.zeros((S - 1, S - 1))
    for a in range(S - 1):
        for b in range(S - 1): F[a, b] = Fp(part, inside, a, b)
    return F

ncell = int(inside.sum())
# equilibrate the uniform measure first, then anneal K slowly, then equilibrate at K
run(part, inside, par, 0.0, 400*ncell*n, SEED*13 + 1)
for f in np.linspace(0, 1, 41)[1:]:
    run(part, inside, par, K*f, 20*ncell*n, SEED*7 + int(1000*f))
run(part, inside, par, K, 400*ncell*n, SEED)
Fm = np.zeros((S - 1, S - 1)); m00 = np.zeros((S - 1, S - 1)); ns = 200
for t in range(ns):
    run(part, inside, par, K, 20*ncell, 1000*SEED + t); F = allF(part, inside); Fm += F
Fm /= ns
np.savez(f"aztecW_n{n}_K{K:+.2f}_s{SEED}.npz", Fm=Fm, part=part)
ii, jj = np.meshgrid(np.arange(S - 1) - off, np.arange(S - 1) - off, indexing="ij")
X, Y = (ii + 1)/n, (jj + 1)/n; r = np.abs(X) + np.abs(Y); geo = r <= 1 - 1.0/n
even = (ii + jj) % 2 == 0; e00 = (ii % 2 == 0) & (jj % 2 == 0); e11 = (ii % 2 == 1) & (jj % 2 == 1)
o01 = (ii % 2 == 0) & (jj % 2 == 1); o10 = (ii % 2 == 1) & (jj % 2 == 0)
def reg(mask):
    fe, fo = Fm[mask & even & geo].mean(), Fm[mask & ~even & geo].mean()
    c = [Fm[mask & q & geo].mean() for q in (e00, e11, o01, o10)]
    return fe, fo, (fe - fo)/(fe + fo) if fe + fo > 1e-9 else float("nan"), c
line = f"n={n} K={K:+.2f}:"
for name, m in [("centre r<0.2", r < 0.2), ("r 0.3-0.5", (r > 0.3) & (r < 0.5)), ("r 0.6-0.8", (r > 0.6) & (r < 0.8)), ("corner r>0.9", r > 0.9)]:
    fe, fo, p, c = reg(m); line += f"\n   {name:13s}: <F> even {fe:.3f} odd {fo:.3f}  Phi {p:+.3f}   by class (00,11,01,10) {np.round(c, 3).tolist()}"
hor = (np.abs(Y) < 0.15) & (np.abs(X) > 0.4) & (np.abs(X) < 0.75); ver = (np.abs(X) < 0.15) & (np.abs(Y) > 0.4) & (np.abs(Y) < 0.75)
line += f"\n   rotation check: <F> in E/W arms {Fm[hor & geo].mean():.3f} vs N/S arms {Fm[ver & geo].mean():.3f}"
print(line, flush=True)
