"""Wall tension per wall cell, sigma_k = (e_stripe - e_C) * 10 * P/2, split into SS, LS, LL point-dipole lattice sums
(sigma(mu) = a + b mu + c mu^2), for wall spacings k = 0 (W), 1, 2."""
import numpy as np, json, sys, time
from cand_response import CairoLattice4
from stripes import stripe, C_on
def dipole_matrix(L, rc=30.0):
    is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)]); a = L.elen[is34].mean()
    C = L.epos/a; m = L.edir; box = L.box/a; N = L.N; J = np.zeros((N, N)); mm = m@m.T
    nx, ny = int(np.ceil(rc/box[0])) + 1, int(np.ceil(rc/box[1])) + 1
    for sx in range(-nx, nx + 1):
        for sy in range(-ny, ny + 1):
            r = C[None, :, :] - C[:, None, :] + np.array([sx, sy])*box; d = np.linalg.norm(r, axis=2); ok = (d > 1e-9) & (d < rc)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (mm - 3*np.einsum('id,ijd->ij', m, r)*np.einsum('jd,ijd->ij', m, r)/d**2)/d**3
            J += np.where(ok, v, 0.0)
    return J, is34
if __name__ == '__main__':
    out = {}
    for d in [float(x) for x in sys.argv[1].split(",")]:
        t0 = time.time(); rec = {}
        for k, n in ((0, 12), (1, 8), (2, 12)):
            L = CairoLattice4(n, n, d); J, is34 = dipole_matrix(L); P = 4*k + 4
            s, c = stripe(k, n), C_on(n); D = 0.5*J*(np.outer(s, s) - np.outer(c, c))/L.N
            S = is34.astype(float); G = 1 - S; f = 10*P/2
            rec[k] = dict(SS=float(S@D@S)*f, LS=float(2*S@D@G)*f, LL=float(G@D@G)*f)
        out[d] = rec
        line = f"d={d:.3f}:"
        for k in (0, 1, 2):
            r = rec[k]; roots = np.roots([r["LL"], r["LS"], r["SS"]]); roots = sorted(x.real for x in roots if abs(x.imag) < 1e-12)
            line += f"\n   k={k}: sigma(mu) = {r['SS']:+.4f} {r['LS']:+.4f} mu {r['LL']:+.4f} mu^2 | sigma(2) {r['SS']+2*r['LS']+4*r['LL']:+.4f}, sigma(3) {r['SS']+3*r['LS']+9*r['LL']:+.4f} | sign change at mu = {[round(x, 2) for x in roots]}"
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
    json.dump({str(k): {str(kk): v for kk, v in r.items()} for k, r in out.items()}, open(f"stripe_sums_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
