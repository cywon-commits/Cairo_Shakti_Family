"""Realistic annealing: single-island Metropolis flips only, geometric cooling from T_hi to T_lo.
Usage: python cool_single.py d n which model(needle|P1) T_hi T_lo sweeps_per_decade nseed"""
import numpy as np, sys, time, json
from cairo_asi import CairoLattice, anneal
from hybrid_coupling import hybrid_coupling, local_needle_coupling, point_coupling
from selection_T import FastDimers
from plot_phases import tile
import needle, os
needle.GAP = float(os.environ.get('GAP', 0.0))
d = float(sys.argv[1]); n = int(sys.argv[2]); which = sys.argv[3]; model = sys.argv[4]
Thi, Tlo, spd, nseed = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]); t0 = time.time()
L = CairoLattice(n, n, d=d)
if model == "needle": Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0)
else: Jl = point_coupling(L, None); J = point_coupling(L, 64.0)
F = FastDimers(L, Jl)
gs = tile(np.load("scan3_min_d0.1372.npy" if which == "A" else "scan3_min_d0.2500.npy")[0], 4, 2, n)
Egs = 0.5*gs@J@gs/L.N; El0 = 0.5*gs@Jl@gs
setx = [6, 7, 8, 9] if which == "A" else [3, 5, 7, 9]
def metrics(s):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
    lev = F.lev[np.arange(len(code)), code]; unh = lev == 1; lab = np.zeros(len(code))
    for q in range(3):
        heal = unh & (F.lev[np.arange(len(code)), code ^ (1 << (2 - q))] == 0) & F.is34[:, q]
        lab[heal] = np.where(np.isin(F.etype[heal, q], setx), 1, -1)
    return dict(psi=float(abs(lab[unh].mean())), exc_local=float((0.5*s@Jl@s - El0)/L.N), dE=float(0.5*s@J@s/L.N - Egs),
                bad_z3=int(np.sum(lev == 2)), frac_unhappy=float(unh.mean()))
ndec = np.log10(Thi/Tlo); Tgrid = np.geomspace(Thi, Tlo, int(ndec*20) + 1)
out = []
for sd in range(nseed):
    s = np.random.default_rng(sd).choice([-1., 1.], L.N)
    for T in Tgrid:
        s, _ = anneal(J, s, T, T, max(1, spd // 20), 777 + sd*1000 + int(1e3*np.log(T+1e-9)) % 997)
    m = metrics(s); out.append(m); np.save(f"coolstate_{which}_{model}_g{needle.GAP:.2f}_d{d:.4f}_n{n}_spd{spd}_s{sd}.npy", s)
    print(f"{which} {model} d={d} n={n} cooling {Thi}->{Tlo}, {spd} sweeps/decade, seed {sd}: |psi| {m['psi']:.2f}, "
          f"E-E_gs {m['dE']:.4f}/spin, local excess {m['exc_local']:.4f}/spin, z3 in top level {m['bad_z3']}, unhappy frac {m['frac_unhappy']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"cool_{which}_{model}_g{needle.GAP:.2f}_d{d:.4f}_n{n}_spd{spd}.json", "w"))
