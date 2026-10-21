"""Small Aztec diamonds (order n) with E = -sum_p F_p (1 + delta s_p) + r sum_<pq> F_p F_q, s_p = (-1)^{x+y};
bulk A|C (odd|even plaquette class) transition exactly at delta = 0. Parallel tempering over beta.
Usage: python aztec_small.py n delta r nround seed"""
import numpy as np, sys, json
from numba import njit
n = int(sys.argv[1]); delta = float(sys.argv[2]); r = float(sys.argv[3]); nround = int(sys.argv[4]); SEED = int(sys.argv[5])
S = 2*n + 2; off = n + 1                                   # padding of one cell
inside = np.zeros((S, S), np.bool_)
for a in range(S):
    for b in range(S): inside[a, b] = abs(a - off + 0.5) + abs(b - off + 0.5) <= n
sgn = np.array([[1.0 if ((a - off) + (b - off)) % 2 == 0 else -1.0 for b in range(S - 1)] for a in range(S - 1)])
valid = np.array([[inside[a, b] and inside[a+1, b] and inside[a, b+1] and inside[a+1, b+1] for b in range(S - 1)] for a in range(S - 1)])
def init():
    part = -np.ones((S, S), np.int64)
    for b in range(S):
        cols = [a for a in range(S) if inside[a, b]]
        for k in range(0, len(cols), 2): part[cols[k], b] = 0; part[cols[k+1], b] = 1
    return part

@njit(cache=True)
def Fp(part, valid, a, b):
    if a < 0 or b < 0 or a >= valid.shape[0] or b >= valid.shape[1] or not valid[a, b]: return 0.0
    return 1.0 if (part[a, b] == 0 and part[a, b+1] == 0) or (part[a, b] == 2 and part[a+1, b] == 2) else 0.0

@njit(cache=True)
def local_E(part, valid, sgn, delta, r, a, b):
    """energy of plaquettes within Chebyshev distance 2 of (a,b) incl. pair terms touching distance <= 1 plaquettes"""
    E = 0.0
    for da in range(-1, 2):
        for db in range(-1, 2):
            p, q = a + da, b + db
            f = Fp(part, valid, p, q)
            if f == 0.0: continue
            E -= (1.0 + delta*sgn[p, q]) if p >= 0 and q >= 0 and p < sgn.shape[0] and q < sgn.shape[1] else 0.0
            # pair terms with right and up neighbours, and with left/down neighbours outside the 3x3 block
            for (u, v) in ((p + 1, q), (p, q + 1), (p - 1, q), (p, q - 1)):
                g = Fp(part, valid, u, v)
                if g == 0.0: continue
                inner = (abs(u - a) <= 1) and (abs(v - b) <= 1)
                E += r*(0.5 if inner else 1.0)
    return E

@njit(cache=True)
def total_E(part, valid, sgn, delta, r):
    E = 0.0
    for p in range(valid.shape[0]):
        for q in range(valid.shape[1]):
            f = Fp(part, valid, p, q)
            if f == 0.0: continue
            E -= 1.0 + delta*sgn[p, q]
            E += r*(Fp(part, valid, p + 1, q) + Fp(part, valid, p, q + 1))
    return E

@njit(cache=True)
def sweep(part, valid, sgn, delta, r, beta, nsteps, seed):
    np.random.seed(seed)
    for _ in range(nsteps):
        a = np.random.randint(valid.shape[0]); b = np.random.randint(valid.shape[1])
        if Fp(part, valid, a, b) == 0.0: continue
        e0 = local_E(part, valid, sgn, delta, r, a, b)
        h = part[a, b] == 0
        if h: part[a, b] = 2; part[a, b+1] = 3; part[a+1, b] = 2; part[a+1, b+1] = 3
        else: part[a, b] = 0; part[a+1, b] = 1; part[a, b+1] = 0; part[a+1, b+1] = 1
        dE = local_E(part, valid, sgn, delta, r, a, b) - e0
        if dE > 0 and np.random.random() >= np.exp(-beta*dE):
            if h: part[a, b] = 0; part[a+1, b] = 1; part[a, b+1] = 0; part[a+1, b+1] = 1
            else: part[a, b] = 2; part[a, b+1] = 3; part[a+1, b] = 2; part[a+1, b+1] = 3

@njit(cache=True)
def counts(part, valid, sgn):
    ne = 0; no = 0
    for p in range(valid.shape[0]):
        for q in range(valid.shape[1]):
            if Fp(part, valid, p, q) == 1.0:
                if sgn[p, q] > 0: ne += 1
                else: no += 1
    return ne, no

if __name__ == "__main__":
    betas = np.geomspace(0.1, 8.0, 20); M = len(betas); ncell = int(inside.sum())
    reps = [init() for _ in range(M)]; rng = np.random.default_rng(SEED)
    E = np.array([total_E(p, valid, sgn, delta, r) for p in reps])
    meas = {m: [] for m in range(M)}; best = (1e9, None)
    for rnd in range(nround):
        for m in range(M):
            sweep(reps[m], valid, sgn, delta, r, betas[m], 10*ncell, SEED*1000003 + rnd*31 + m)
            E[m] = total_E(reps[m], valid, sgn, delta, r)
            if E[m] < best[0] - 1e-9: best = (E[m], counts(reps[m], valid, sgn)); best_part = reps[m].copy()
        for m in range(rnd % 2, M - 1, 2):
            x = (betas[m+1] - betas[m])*(E[m+1] - E[m])
            if x >= 0 or rng.random() < np.exp(x):
                reps[m], reps[m+1] = reps[m+1], reps[m]; E[m], E[m+1] = E[m+1], E[m]
        if rnd >= nround//4:
            for m in range(M):
                ne, no = counts(reps[m], valid, sgn); meas[m].append((ne, no, E[m]))
    out = dict(n=n, delta=delta, r=r, betas=betas.tolist(), best_E=float(best[0]), best_counts=[int(x) for x in best[1]], rows=[])
    for m in range(M):
        a = np.array(meas[m], float); tot = a[:, 0] + a[:, 1]
        phi = np.where(tot > 0, (a[:, 0] - a[:, 1])/np.maximum(tot, 1), 0.0)
        out["rows"].append(dict(beta=float(betas[m]), Phi=float(phi.mean()), Phi_err=float(phi.std()/np.sqrt(len(phi)/20)),
                                Ne=float(a[:, 0].mean()), No=float(a[:, 1].mean()), E=float(a[:, 2].mean())))
    json.dump(out, open(f"aztecS_n{n}_d{delta:+.3f}_r{r}_s{SEED}.json", "w"), indent=1)
    np.save(f"aztecS_gs_n{n}_d{delta:+.3f}_s{SEED}.npy", best_part)
    print(f"n={n} delta={delta:+.3f}: ground state found E={best[0]:.2f} with (N_even, N_odd) = {best[1]}")
    for row in out["rows"][::3] + [out["rows"][-1]]:
        print(f"   beta={row['beta']:.2f}: <Phi> {row['Phi']:+.3f} (+-{row['Phi_err']:.3f}), <N_even> {row['Ne']:.2f}, <N_odd> {row['No']:.2f}")
