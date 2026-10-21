"""Point-dipole lattice sums for E_X - E_C split by island types: E = S_SS + mu S_LS + mu^2 S_LL (unit short moment,
units mu0 m_S^2 / (4 pi a^3) per spin, a = z3-z4 edge length). Unit-cell periodic sums over images up to rc."""
import numpy as np, sys, json
from cand_response import CairoLattice4, STATES, NX, NY
def dipole_matrix(L, rc=40.0):
    is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)]); a = L.elen[is34].mean()
    C = L.epos/a; m = L.edir; box = L.box/a; N = L.N; J = np.zeros((N, N))
    nx, ny = int(np.ceil(rc/box[0])) + 1, int(np.ceil(rc/box[1])) + 1
    for sx in range(-nx, nx + 1):
        for sy in range(-ny, ny + 1):
            r = C[None, :, :] - C[:, None, :] + np.array([sx, sy])*box; d = np.linalg.norm(r, axis=2)
            ok = (d > 1e-9) & (d < rc)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = ((m@m.T) - 3*np.einsum('id,ijd->ij', m, r)*np.einsum('jd,ijd->ij', m, r)/d**2)/d**3
            J += np.where(ok, v, 0.0)
    return J, is34
def sums(d, X, rc=40.0):
    L = CairoLattice4(NX, NY, d); J, is34 = dipole_matrix(L, rc)
    sX, sC = STATES[X], STATES["C"]; D = 0.5*J*(np.outer(sX, sX) - np.outer(sC, sC))/L.N
    S = is34.astype(float); Lg = 1 - S
    return dict(SS=float(S@D@S), LS=float(2*S@D@Lg), LL=float(Lg@D@Lg))
if __name__ == "__main__":
    X = sys.argv[1]; import os
    fn = f"lattice_sums_{X}.json"; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for d in [float(x) for x in sys.argv[2].split(",")]:
        s = sums(d, X); out[str(d)] = s
        a, b, c = s["LL"], s["LS"], s["SS"]; disc = b*b - 4*a*c
        roots = sorted([(-b - np.sqrt(disc))/(2*a), (-b + np.sqrt(disc))/(2*a)]) if disc >= 0 else None
        f = lambda mu: c + b*mu + a*mu*mu
        print(f"{X}-C, d={d:.3f}: S_SS {c:+.5f}, S_LS {b:+.5f}, S_LL {a:+.5f} | value at mu=2: {f(2):+.5f}, mu=3: {f(3):+.5f} | roots mu_c: {roots}", flush=True)
        json.dump(out, open(fn, "w"), indent=1)
