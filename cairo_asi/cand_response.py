"""Zero-field energies and fixed-state linear responses of candidate crystals to eps (diagonal anisotropy),
theta (twist) and eta (axial strain: x -> x(1+eta/2), y -> y(1-eta/2))."""
import numpy as np, json, sys, time
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo3 import CairoLattice3
from hybrid_coupling import hybrid_coupling
class CairoLattice4(CairoLattice3):
    def __init__(self, nx, ny, dbar, eps=0.0, theta=0.0, eta=0.0):
        self.eta = eta; super().__init__(nx, ny, dbar, eps, theta)
    def _build_vertices(self):
        super()._build_vertices()
        sc = np.array([1 + self.eta/2, 1 - self.eta/2]); self.vpos = self.vpos*sc; self.box = self.box*sc
info = json.load(open("candidates.json")); names = list(info)
NX, NY = 12, 4                                    # common torus for all candidates
def tile_xy(s0, cx, cy, nx, ny):
    t = np.empty(nx*ny*10)
    for i in range(nx):
        for j in range(ny): t[(i*ny + j)*10:(i*ny + j)*10 + 10] = s0[((i % cx)*cy + (j % cy))*10:((i % cx)*cy + (j % cy))*10 + 10]
    return t
STATES = {k: tile_xy(np.load(f"cand_{k}.npy"), info[k]["px"], info[k]["py"], NX, NY) for k in names}
H = {"eps": 0.002, "theta": 0.01, "eta": 0.004}
if __name__ == '__main__':
    out = {}
    for db in [float(x) for x in sys.argv[1].split(",")]:
        t0 = time.time(); rec = {}; JC = {}
        for k in names:
            s = STATES[k]
            def E(**kw):
                key = tuple(sorted(kw.items()))
                if key not in JC: L = CairoLattice4(NX, NY, db, **kw); JC[key] = (hybrid_coupling(L, 64.0), L.N)
                J, N = JC[key]; return 0.5*s@J@s/N
            r = {"E0": float(E())}
            for key, h in H.items(): r["|Q_" + key + "|"] = float(abs(E(**{key: h}) - E(**{key: -h}))/(2*h))
            rec[k] = r
        out[db] = rec
        print(f"d_bar={db:.3f} [{time.time()-t0:.0f}s]: " + "; ".join(f"{k} E0 {v['E0']:.5f} |Qe| {v['|Q_eps|']:.2f} |Qt| {v['|Q_theta|']:.3f} |Qn| {v['|Q_eta|']:.3f}" for k, v in rec.items()), flush=True)
        json.dump(out, open(f"cand_response12x4_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
