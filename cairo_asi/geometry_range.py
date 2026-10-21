"""Geometry of CairoLattice(d) over an extended d range: edge lengths, interior angles, planarity."""
import numpy as np, itertools
from cairo_asi import CairoLattice
def seg_cross(p1, p2, q1, q2):
    def o(a, b, c): return np.sign((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    return o(p1, p2, q1)*o(p1, p2, q2) < 0 and o(q1, q2, p1)*o(q1, q2, p2) < 0
for d in [0.0001, 0.03, 0.06, 0.1, 0.1371, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]:
    L = CairoLattice(3, 3, d=d)
    is34 = np.array([tuple(sorted(L.vtype[L.edges[k]])) == (3, 4) for k in range(L.N)])
    a = L.elen[is34].mean(); lp = L.elen[~is34].mean()
    # interior angles of one pentagon (signed, from its CCW polygon)
    p = L.face_pts[0]; angs = []
    for t in range(5):
        u = p[t-1]-p[t]; w = p[(t+1) % 5]-p[t]
        ang = np.degrees(np.arctan2(u[0]*w[1]-u[1]*w[0], u@w)) % 360
        angs.append(round(360-ang if False else ang, 1))
    # planarity: any two non-adjacent edges crossing?
    segs = [(L.epos[k]-0.5*L.elen[k]*L.edir[k], L.epos[k]+0.5*L.elen[k]*L.edir[k]) for k in range(L.N)]
    cross = 0
    for i, j in itertools.combinations(range(L.N), 2):
        if set(L.edges[i]) & set(L.edges[j]): continue
        a1, a2 = segs[i]; b1, b2 = segs[j]; sh = L.mic(b1 - a1) - (b1 - a1); b1, b2 = b1+sh, b2+sh
        if seg_cross(a1, a2, b1, b2): cross += 1
    print(f"d={d:.4f}: long/short island = {lp/a:.3f}, pentagon angles {sorted(angs)}, faces {len(L.faces)}, crossing edges {cross}")
