import numpy as np, sys, time
import real_islands as RI
from cand_response import CairoLattice4, STATES, NX, NY, names
exec(open("plot_dmu_phase.py").read().split("dg = np.linspace")[0])
def area(L): return (L - RI.WID)*RI.WID + np.pi*(RI.WID/2)**2
for arg in sys.argv[1:]:
    d, mus = arg.split(":"); d = float(d)
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)]); sc = 472/ref.elen[is34].mean()
    bnm = ref.elen[~is34].mean()*sc; aS = area(472 - 2*RI.G34)
    for mu in [float(x) for x in mus.split(",")]:
        Lp = (mu*aS - np.pi*(RI.WID/2)**2)/RI.WID + RI.WID; gp = (bnm - Lp)/2
        if gp < 5 or Lp < RI.WID: print(f"d={d} mu={mu}: not realisable (gap {gp:.0f} nm, length {Lp:.0f} nm)"); continue
        RI.GP = gp; t0 = time.time(); A = RI.RealArray(NX, NY, d); _, J = RI.full_J(A)
        e = {X: 0.5*STATES[X]@J@STATES[X]/A.L.N for X in names}; w = min(e, key=e.get); m = min(e.values())
        pa = names[int(np.argmin([E(X, d, mu) for X in names]))]
        print(f"d={d:.3f} mu={mu:.2f} (partner {Lp:.0f} nm, gap {gp:.0f} nm): realistic winner {w:3s} | analytic {pa:3s} | "
              + ", ".join(f"{X} {e[X]-m:+.2f}" for X in names) + f" K  [{time.time()-t0:.0f}s]", flush=True)
    RI.GP = 22.0
