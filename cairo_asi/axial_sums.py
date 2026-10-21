import numpy as np, sys, json, time
from cand_response import CairoLattice4
from stripe_sums import dipole_matrix
from stripes_axial import axial, C_on_xy
out = {}
for d in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time(); rec = {}
    for t in (0, 1, 2):
        s, nx = axial(t); L = CairoLattice4(nx, 2, d); J, is34 = dipole_matrix(L); c = C_on_xy(nx, 2)
        D = 0.5*J*(np.outer(s, s) - np.outer(c, c))/L.N; S = is34.astype(float); G = 1 - S; f = 10*nx/4
        rec[t] = dict(SS=float(S@D@S)*f, LS=float(2*S@D@G)*f, LL=float(G@D@G)*f)
    out[d] = rec
    msg = f"d={d:.3f}:"
    for t in (0, 1, 2):
        r = rec[t]; roots = sorted(x.real for x in np.roots([r["LL"], r["LS"], r["SS"]]) if abs(x.imag) < 1e-12)
        msg += f"\n   t={t}: sigma_ax(mu) = {r['SS']:+.4f} {r['LS']:+.4f} mu {r['LL']:+.4f} mu^2 | sigma_ax(1) {r['SS']+r['LS']+r['LL']:+.4f}, (2) {r['SS']+2*r['LS']+4*r['LL']:+.4f} | zero at mu = {[round(x, 2) for x in roots]}"
    print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
json.dump({str(k): {str(t): v for t, v in r.items()} for k, r in out.items()}, open("axial_sums.json", "w"), indent=1)
