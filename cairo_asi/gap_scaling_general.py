"""General threshold formula k*(g) = dE_X(g) / lambda_vertex(g), tested for C via twist and M via axial strain (4x2)."""
import numpy as np, time
import needle
from cand_response import CairoLattice4
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from selection_T import FastDimers
from cairo2 import S0
L0 = CairoLattice4(4, 2, 0.20); F = FastDimers(L0, local_needle_coupling(L0))
ps = np.array([F.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
cls = {"A": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(pc) < 1e-9), "C": (np.abs(np.abs(pc) - 1) < 1e-9) & (np.abs(pa) < 1e-9),
       "M": (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(np.abs(pc) - 1) < 1e-9)}
def Eall(d, sym=False, **kw):
    J = hybrid_coupling(CairoLattice4(4, 2, d, **kw), 64.0); E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/80
    if sym:                                    # best orientation: also evaluate the mirrored deformation
        kw2 = {k: -v for k, v in kw.items()}; J2 = hybrid_coupling(CairoLattice4(4, 2, d, **kw2), 64.0)
        E = np.minimum(E, 0.5*np.einsum('ij,jk,ik->i', S0, J2, S0)/80)
    return E
def vertex_lambda(d, X, key, h):
    """|dE_vertex/dk| of the best state of class X (local couplings only)."""
    E0 = Eall(d); i = int(np.flatnonzero(cls[X])[np.argmin(E0[cls[X]])]); s = S0[i]
    Ep = 0.5*s@local_needle_coupling(CairoLattice4(4, 2, d, **{key: h}))@s/80; Em = 0.5*s@local_needle_coupling(CairoLattice4(4, 2, d, **{key: -h}))@s/80
    return abs(Ep - Em)/(2*h), E0
for X, key, d, h in (("C", "theta", 0.153, 0.01), ("M", "eta", 0.20, 0.004)):
    for g in (0.0, 0.1, 0.2):
        needle.GAP = g; needle.SUBDIV = 8; needle.FIXED_M = True; t0 = time.time()
        lam, E0 = vertex_lambda(d, X, key, h)
        others = [Y for Y in cls if Y != X]; dE = E0[cls[X]].min() - min(E0[cls[Y]].min() for Y in others)
        kstar = dE/lam if dE > 0 else 0.0
        # actual crossing from exact energies at 0.6, 1.0, 1.4 x prediction
        rows = []
        for f in (0.6, 1.0, 1.4):
            E = Eall(d, sym=True, **{key: f*kstar}); rows.append((f, E[cls[X]].min() - min(E[cls[Y]].min() for Y in others)))
        fc = next((fa + (fb - fa)*(-xa)/(xb - xa) for (fa, xa), (fb, xb) in zip(rows, rows[1:]) if xa > 0 >= xb), None)
        print(f"{X} via {key}, d={d}, g={g}: zero-field gap {dE:+.5f}, vertex lambda {lam:.3f} -> predicted {key}* = {kstar:.4f}; "
              f"exact E_X - E_rest at 0.6/1.0/1.4: " + ", ".join(f"{x:+.5f}" for f, x in rows) + (f"; actual crossing {fc:.2f} x predicted" if fc else "") + f" [{time.time()-t0:.0f}s]", flush=True)
