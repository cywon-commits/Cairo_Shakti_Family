"""Selection temperature within the dimer manifold: fixed-T manifold MC on an n x n Cairo(d) torus.
Measures E, specific heat, and dimer order parameters
  psi_A = (#dimers on C3/C4-pair edges - #dimers on C1/C2-pair edges) / #dimers   (partner-bond orientation choice; A: +-1)
  psi_C = (#dimers on z3-B edges - #dimers on z3-A edges) / #dimers              (z4 sublattice choice; C: +-1)"""
import numpy as np, sys, json, time, itertools, os
from selection_T import FastDimers
from cairo_asi import CairoLattice
from hybrid_coupling import hybrid_coupling, local_needle_coupling, point_coupling
from manifold_mc2 import mc_arrays, mc
from plot_phases import tile

class _Unused:
    def __init__(self, L, Jl):
        self.v3 = np.where(L.vtype == 3)[0]
        self.ks = np.array([[k for k, _ in L.incidence[v]] for v in self.v3])        # 3 edges per z3
        self.lev = np.zeros((len(self.v3), 8), int)
        for a, v in enumerate(self.v3):
            ks = self.ks[a]
            E = [sum(Jl[ks[x], ks[y]]*c[x]*c[y] for x, y in itertools.combinations(range(3), 2))
                 for c in itertools.product([-1, 1], repeat=3)]
            lv = sorted(set(np.round(E, 6)))
            self.lev[a] = [lv.index(np.round(e, 6)) for e in E]
        self.is34 = np.array([[L.vtype[L.edges[k]].sum() == 7 for k in row] for row in self.ks])
        self.etype = self.ks % 10
    def dimers(self, s):
        bits = ((self.ks_spin(s) > 0).astype(int))                                    # (n3, 3), +1 -> 1
        code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]
        unh = self.lev[np.arange(len(code)), code] > 0
        out = []
        for t in range(3):
            flip = code ^ (1 << (2 - t))
            heal = unh & (self.lev[np.arange(len(code)), flip] == 0) & self.is34[:, t]
            out.append(self.etype[heal, t])
        return np.concatenate(out)
    def ks_spin(self, s): return s[self.ks]
    def psi(self, s):
        et = self.dimers(s); n = len(et)
        a = (np.isin(et, [6, 7, 8, 9]).sum() - np.isin(et, [2, 3, 4, 5]).sum()) / n
        c = (np.isin(et, [3, 5, 7, 9]).sum() - np.isin(et, [2, 4, 6, 8]).sum()) / n
        return a, c

if __name__ == "__main__":
    n = int(sys.argv[1]); ds = [float(x) for x in sys.argv[2].split(",")]
    Ts = np.geomspace(1.0, 0.001, 24); neq, nmeas, every = 1500, 3000, 10
    fn = f"selT_P1_n{n}.json"; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for d in ds:
        t0 = time.time(); rc = 64.0 if d < 0.34 else 128.0
        L = CairoLattice(n, n, d=d); Jl = point_coupling(L, None); J = point_coupling(L, 64.0); arr = mc_arrays(L)
        F = FastDimers(L, Jl); nd = len(F.dimers(tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)))
        s = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n)
        rows = []
        for iT, T in enumerate(Ts):
            s, Etr, st = mc(s, J, Jl, *arr, T, T, neq, 17*iT + 3)
            Es, pa, pc = [], [], []
            for blk in range(nmeas // every):
                s, Etr, st = mc(s, J, Jl, *arr, T, T, every, 100000 + 1000*iT + blk)
                Es.extend(Etr.tolist()); a, c = F.psi(s); pa.append(abs(a)); pc.append(abs(c))
            Es = np.array(Es)
            rows.append(dict(T=float(T), E=float(Es.mean()/L.N), C=float(Es.var()/(L.N*T*T)),
                             psiA=float(np.mean(pa)), psiC=float(np.mean(pc))))
        out[f"{d:.4f}"] = rows; json.dump(out, open(fn, "w"), indent=1)
        k = int(np.argmax([r["C"] for r in rows]))
        print(f"d={d:.4f}: C peak at T={rows[k]['T']:.4f}; lowest T: E/N {rows[-1]['E']:.5f}, |psiA| {rows[-1]['psiA']:.2f}, |psiC| {rows[-1]['psiC']:.2f}; "
              f"highest T: |psiA| {rows[0]['psiA']:.2f}, |psiC| {rows[0]['psiC']:.2f}  [{time.time()-t0:.0f}s]", flush=True)
