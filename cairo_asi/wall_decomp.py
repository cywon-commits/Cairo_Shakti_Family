"""Decompose E_W - E_C into island-pair classes (by island types and centre-to-centre separation) on the 12x4 torus,
for the ideal needle model and the realistic islands. Shows which pair classes carry the wall energy."""
import numpy as np, collections, sys
import needle
from cand_response import CairoLattice4, STATES, NX, NY
from hybrid_coupling import hybrid_coupling
from real_islands import RealArray, full_J
d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
sW, sC = STATES["W"], STATES["C"]
def decomp(J, L, scale):
    is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)])
    diff = 0.5*J*(np.outer(sW, sW) - np.outer(sC, sC))           # per-pair contribution to E_W - E_C (both i<j and j>i halves)
    out = collections.defaultdict(float); cnt = collections.Counter()
    for i in range(L.N):
        for j in range(i + 1, L.N):
            if abs(diff[i, j]) < 1e-12: continue
            r = np.linalg.norm(L.mic(L.epos[j] - L.epos[i]))*scale
            ti = "S" if is34[i] else "L"; tj = "S" if is34[j] else "L"
            key = ("".join(sorted(ti + tj)), round(r, 2))
            out[key] += 2*diff[i, j]/L.N; cnt[key] += 1
    return out, cnt
needle.GAP = 0.0; needle.GAP_P = None; needle.SUBDIV = 8; needle.FIXED_M = True
L = CairoLattice4(NX, NY, d); Jn = hybrid_coupling(L, 64.0)
on, cn = decomp(Jn, L, 1.0/L.elen[np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)])].mean())
A = RealArray(NX, NY, d); _, Jr = full_J(A)
orr, cr = decomp(Jr, A.L, 1.0/A.L.elen[np.array([A.L.vtype[A.L.edges[k]].sum() == 7 for k in range(A.L.N)])].mean())
tn, tr = sum(on.values()), sum(orr.values())
print(f"d={d}: E_W - E_C total: needle {tn:+.6f} D/a^3 per spin, realistic {tr:+.3f} K per island")
print("pair class (S = z3-z4 island, L = partner island; r in units of a) | #pairs changed | needle contribution | realistic contribution")
keys = sorted(set(on) | set(orr), key=lambda k: -abs(orr.get(k, 0)))
for k in keys[:12]:
    print(f"   {k[0]:2s} r={k[1]:.2f} | {cn.get(k, cr.get(k, 0)):4d} | {on.get(k, 0):+.6f} | {orr.get(k, 0):+.3f}")
near = [k for k in keys if k[1] < 1.6]; far = [k for k in keys if k[1] >= 1.6]
print(f"near pairs (r < 1.6a): needle {sum(on.get(k,0) for k in near):+.6f}, realistic {sum(orr.get(k,0) for k in near):+.3f};  "
      f"far pairs: needle {sum(on.get(k,0) for k in far):+.6f}, realistic {sum(orr.get(k,0) for k in far):+.3f}")
