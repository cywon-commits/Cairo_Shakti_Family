"""Predict eps*(d, g) = (E_A - E_C)(d, g) / lambda_A(d, g) with lambda_A from single-vertex splittings."""
import numpy as np, itertools, json, sys
import needle
from cairo3 import CairoLattice3
from hybrid_coupling import local_needle_coupling
def z3_gaps(d, eps, b_list=(2, 4)):
    L = CairoLattice3(4, 2, d, eps=eps); Jl = local_needle_coupling(L); out = []
    for b in b_list:
        v = [x for x in range(len(L.vtype)) if x % 6 == b][0]; ks = [k for k, _ in L.incidence[v]]
        t = sorted(set(np.round([sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(3), 2)) for c in itertools.product([-1, 1], repeat=3)], 8)))
        out.append(t[1] - t[0])
    return out
gs = json.load(open("gap_scan.json"))
res = {}
print(" g    | d     | Delta_z3 | lambda_A (vertex) | E_A - E_C (4x2, scan) | predicted eps* for C")
for g in (0.0, 0.05, 0.1, 0.2):
    needle.GAP = g; needle.SUBDIV = 8; needle.FIXED_M = True
    for d in (0.16, 0.20, 0.25):
        D1, D3 = z3_gaps(d, 0.002); D0 = z3_gaps(d, 0.0)[0]
        lam = (D1 - D3)/(10*0.002)
        k = f"{g:.2f}_{d:.4f}"; dE = gs[k]["EA_minus_EC"] if k in gs and gs[k].get("valid") else None
        eps_star = dE/lam if dE is not None and dE > 0 else None
        res[k] = dict(Delta=D0, lamA=lam, dEAC=dE, eps_star=eps_star)
        print(f" {g:.2f} | {d:.3f} | {D0:8.3f} | {lam:8.3f} | {dE if dE is None else f'{dE:+.5f}'} | {eps_star if eps_star is None else f'{eps_star:.4f}'}", flush=True)
json.dump(res, open("gap_scaling_pred.json", "w"), indent=1)
print("\nratio R(g) = eps*(g)/eps*(0):")
for d in (0.16, 0.20, 0.25):
    e0 = res[f"0.00_{d:.4f}"]["eps_star"]
    print(f"  d={d}: " + ", ".join(f"g={g}: {res[f'{g:.2f}_{d:.4f}']['eps_star']/e0:.1f}x" for g in (0.05, 0.1, 0.2) if res[f'{g:.2f}_{d:.4f}']['eps_star'] and e0))
