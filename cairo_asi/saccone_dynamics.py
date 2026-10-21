"""Saccone geometry in kelvin: (a) T_A from manifold MC, (b) single-island-flip slow cooling.
Usage: python saccone_dynamics.py c n mode(TA|cool) [nseed]"""
import numpy as np, sys, time, json, os
from saccone_geometry import SacconeArray, coupling_matrices, K_PER_EV
from selection_T import FastDimers
from manifold_mc2 import mc_arrays, mc
from cairo_asi import anneal
from plot_phases import tile
c = float(sys.argv[1]); n = int(sys.argv[2]); mode = sys.argv[3]; nseed = int(sys.argv[4]) if len(sys.argv) > 4 else 2
t0 = time.time(); jf = f"Jsac_c{c:.0f}_n{n}.npz"
Arr = SacconeArray(n, n, c, nl=16, nw=3); L = Arr.L
if os.path.exists(jf): z = np.load(jf); Jl, J = z["Jl"], z["J"]
else: Jl, J = coupling_matrices(Arr); Jl *= K_PER_EV; J *= K_PER_EV; np.savez(jf, Jl=Jl, J=J)   # kelvin
F = FastDimers(L, Jl); arr = mc_arrays(L)
import itertools
_v = np.where(L.vtype == 4)[0][0]; _k = [k for k, _ in L.incidence[_v]]
Emin4 = min(sum(Jl[_k[x], _k[y]]*c_[x]*c_[y] for x in range(4) for y in range(x+1, 4)) for c_ in itertools.product([-1, 1], repeat=4))
sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); EA = 0.5*sA@J@sA/L.N; El0 = 0.5*sA@Jl@sA
print(f"c={c:.0f} n={n}: couplings ready [{time.time()-t0:.0f}s]; E_A = {EA:.1f} K/island", flush=True)
def stats(s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    lev = F.lev[np.arange(len(code)), code]; unh = lev == 1; lab = np.zeros(len(code))
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        lab[heal] = np.where(np.isin(F.etype[heal, q], [6, 7, 8, 9]), 1, -1)
    q4 = L.charges(s)[L.vtype == 4]
    v4 = np.where(L.vtype == 4)[0]; typeI = 0
    for v in v4:
        ks = [k for k, _ in L.incidence[v]]; e = sum(Jl[ks[x], ks[y]]*s[ks[x]]*s[ks[y]] for x in range(4) for y in range(x+1, 4))
        typeI += e < Emin4 + 1e-6
    return dict(psiA=float(abs(lab[unh].mean())) if unh.any() else 0.0, frac_B=float(unh.mean()), ice4=float(np.mean(q4 == 0)),
                typeI=float(typeI/len(v4)), dE=float(0.5*s@J@s/L.N - EA), exc=float((0.5*s@Jl@s - El0)/L.N))
if mode == "TA":
    Ts = np.geomspace(2000, 5, 26); s = sA.copy(); rows = []
    for iT, T in enumerate(Ts):
        s, _, _ = mc(s, J, Jl, *arr, T, T, 800, 17*iT + 1); ps = []
        for b in range(60):
            s, _, _ = mc(s, J, Jl, *arr, T, T, 10, 5000 + 100*iT + b); ps.append(stats(s)["psiA"])
        rows.append((float(T), float(np.mean(ps))))
    TA = None
    for (T1, p1), (T2, p2) in zip(rows, rows[1:]):
        if p1 < 0.5 <= p2: TA = float(np.exp(np.log(T1) + (0.5-p1)/(p2-p1)*(np.log(T2)-np.log(T1)))); break
    print("  manifold MC |psi_A|(T):", " ".join(f"{T:.0f}K:{p:.2f}" for T, p in rows[::2]))
    print(f"  => T_A (|psi_A| = 0.5) = {TA} K  [{time.time()-t0:.0f}s]")
    json.dump(dict(c=c, rows=rows, TA=TA), open(f"saccone_TA_c{c:.0f}.json", "w"))
else:
    spd = int(os.environ.get('SPD', 20000)); Tf = float(os.environ.get('TF', 30)); Tg = np.geomspace(3000, Tf, int(np.log10(3000/Tf)*20) + 1); res = []
    for sd in range(nseed):
        s = np.random.default_rng(sd).choice([-1., 1.], L.N); traj = []
        for T in Tg:
            s, _ = anneal(J, s, T, T, spd // 20, 300 + 1000*sd + int(T) % 997)
            if abs(np.log10(T) - round(np.log10(T)*4)/4) < 1e-6 or T == Tg[-1]: traj.append((round(float(T)), stats(s)))
        res.append(traj); f = traj[-1][1]
        print(f"  single-flip cooling 3000->{Tf:.0f} K, seed {sd}: final |psi_A| {f['psiA']:.2f}, frac B {f['frac_B']:.3f}, z4 ice-rule {f['ice4']:.3f}, z4 type I {f['typeI']:.3f}, "
              f"E - E_A {f['dE']:+.1f} K/island, local excess {f['exc']:.1f} K/island  [{time.time()-t0:.0f}s]", flush=True)
        if False: print("     along the way (T: |psi_A|, frac_B):", " ".join(f"{T}:{x['psiA']:.2f},{x['frac_B']:.2f}" for T, x in traj))
    json.dump(res, open(f"saccone_cool_c{c:.0f}_Tf{Tf:.0f}.json", "w"))
