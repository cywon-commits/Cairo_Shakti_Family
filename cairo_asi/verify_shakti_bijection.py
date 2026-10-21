import numpy as np, itertools
from collections import Counter
from shakti import ShaktiLattice
from dimer_map import DimerMap
from compare_shakti import setup_from
S = ShaktiLattice(2, 2); D = DimerMap(S)
J, ze, zc, part, P = setup_from(S)
gs, E0 = [], None
for bits in range(1 << len(ze)):
    s = np.zeros(S.N)
    for t in range(len(ze)): s[ze[t]] = zc[t]*(1 - 2*((bits >> t) & 1))
    opts = []
    for p in part:
        e = []
        for sg in (-1., 1.): s[p] = sg; e.append(s[p]*(J[p] @ s))
        opts.append([-1.] if e[0] < e[1]-1e-9 else [1.] if e[1] < e[0]-1e-9 else [-1., 1.])
    for combo in itertools.product(*opts):
        s[part] = combo; E = 0.5*s@J@s
        if E0 is None or E < E0-1e-7: E0, gs = E, [s.copy()]
        elif abs(E-E0) < 1e-7: gs.append(s.copy())
ok, mats = 0, set()
for s in gs:
    dm = D.dimers(s)
    if D.is_perfect(dm): ok += 1; mats.add(frozenset(dm))
unh = Counter(sum(D.level(v, s) for v in range((len(S.vpos))) if S.vtype[v] == 3) for s in gs[:200])
print(f"Shakti 2x2: GS {len(gs)}, perfect matchings {ok}/{len(gs)}, distinct coverings {len(mats)} (= W/2: {2*len(mats)==len(gs)})")
print("unhappy z3 per GS (of", (S.vtype==3).sum(), "):", dict(unh))
# which angle does the happy z3 favour? check Cairo z3 geometry
from cairo_asi import CairoLattice
C = CairoLattice(2, 2); v = np.where(C.vtype == 3)[0][0]
for k, _ in C.incidence[v]:
    pass
ks = [k for k, _ in C.incidence[v]]
def outvec(L, v, k):
    u = L.edges[k][0] if L.edges[k][1] == v else L.edges[k][1]; o = L.mic(L.vpos[u]-L.vpos[v]); return o/np.linalg.norm(o)
for a, b in itertools.combinations(ks, 2):
    ang = np.degrees(np.arccos(outvec(C, v, a) @ outvec(C, v, b)))
    ta = "partner" if tuple(C.vtype[C.edges[a]]) == (3, 3) else "z3z4"; tb = "partner" if tuple(C.vtype[C.edges[b]]) == (3, 3) else "z3z4"
    print(f"Cairo z3: angle between {ta} and {tb} = {ang:.1f} deg")
