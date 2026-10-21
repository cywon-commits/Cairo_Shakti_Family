"""Tune the moment ratio mu = area(partner island)/area(z3-z4 island) through the partner gap, at fixed d; compare the realistic
E_W - E_C with the point-dipole quadratic and its root mu_c."""
import numpy as np, json, sys, time
import real_islands as RI
from cand_response import CairoLattice4, STATES, NX, NY
S = json.load(open("lattice_sums_W.json"))
UNIT_K = 1e-7*(RI.MS*((300 - 100)*100 + np.pi*50**2)*3e-27)**2/(472e-9)**3/1.380649e-23   # mu0 m_S^2/(4 pi a^3) in K
def area(L): return (L - RI.WID)*RI.WID + np.pi*(RI.WID/2)**2
for d in [float(x) for x in sys.argv[1].split(",")]:
    s = S[str(d)]; a, b, c = s["LL"], s["LS"], s["SS"]; mu_c = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)])
    sc = 472/ref.elen[is34].mean(); bnm = ref.elen[~is34].mean()*sc; aS = area(472 - 2*RI.G34)
    print(f"d={d}: partner edge {bnm:.0f} nm; point-dipole mu_c = {mu_c:.3f}  (unit {UNIT_K:.0f} K)")
    for mu in (0.75*mu_c, mu_c, 1.3*mu_c, 1.7*mu_c):
        Lp = (mu*aS - np.pi*(RI.WID/2)**2)/RI.WID + RI.WID; gp = (bnm - Lp)/2
        if gp < 5: continue
        RI.GP = gp; t0 = time.time(); A = RI.RealArray(NX, NY, d); _, J = RI.full_J(A)
        e = {X: 0.5*STATES[X]@J@STATES[X]/A.L.N for X in ("W", "C", "A")}
        pred = (c + b*mu + a*mu*mu)*UNIT_K
        print(f"   mu={mu:.2f} (partner island {Lp:.0f} nm, gap {gp:.0f} nm): realistic E_W - E_C = {e['W']-e['C']:+.2f} K (point formula {pred:+.2f} K); "
              f"E_A - E_W = {e['A']-e['W']:+.2f} K  [{time.time()-t0:.0f}s]", flush=True)
    RI.GP = 22.0
