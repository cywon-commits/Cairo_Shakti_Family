import numpy as np, sys, time
import real_islands as RI
from cand_response import CairoLattice4, STATES, NX, NY, names
exec(open("plot_dmu_phase.py").read().split("dg = np.linspace")[0])
def area(L, w): return (L - w)*w + np.pi*(w/2)**2
for arg in sys.argv[1:]:
    d, mus = arg.split(":"); d = float(d)
    ref = CairoLattice4(2, 2, d); is34 = np.array([ref.vtype[ref.edges[k]].sum() == 7 for k in range(ref.N)]); sc = 472/ref.elen[is34].mean()
    Lp = ref.elen[~is34].mean()*sc - 2*22; aS = area(472 - 2*86, 100)
    for mu in [float(x) for x in mus.split(",")]:
        target = mu*aS; c2 = 1 - np.pi/4; w = (Lp - np.sqrt(Lp*Lp - 4*c2*target))/(2*c2)      # solve Lp w - (1 - pi/4) w^2 = target
        RI.WID_L = w; t0 = time.time(); A = RI.RealArray(NX, NY, d); _, J = RI.full_J(A)
        e = {X: 0.5*STATES[X]@J@STATES[X]/A.L.N for X in names}; win = min(e, key=e.get); m = min(e.values())
        pa = names[int(np.argmin([E(X, d, mu) for X in names]))]
        print(f"d={d:.3f} mu={mu:.2f} (partner {Lp:.0f} nm x {w:.0f} nm): realistic {win:3s} | analytic {pa:3s} | " + ", ".join(f"{X} {e[X]-m:+.2f}" for X in ("A", "C", "W", "L3")) + f" K [{time.time()-t0:.0f}s]", flush=True)
    RI.WID_L = None
