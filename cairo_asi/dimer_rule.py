import numpy as np, itertools
from collections import Counter
from cairo_asi import *
from count_degeneracy import setup

nx = ny = 2
L, ze, zc, part, P = setup(nx, ny); J = L.coupling_matrix(None)
# --- collect all vertex-local ground states (2x2 periodic) by enumeration ---
nz = len(ze); states = []
for bits in range(1 << nz):
    s = np.zeros(L.N)
    for t in range(nz):
        s[ze[t]] = zc[t] * (1 - 2*((bits >> t) & 1))
    opts = []
    for p in part:
        e = []
        for sg in (-1., 1.):
            s[p] = sg; e.append(s[p]*(J[p] @ s))
        opts.append([-1.] if e[0] < e[1]-1e-9 else [1.] if e[1] < e[0]-1e-9 else [-1., 1.])
    for combo in itertools.product(*opts):
        s[part] = combo; states.append(s.copy())
E = np.array([0.5*s@J@s for s in states]); E0 = E.min()
gs = [s for s, e in zip(states, E) if abs(e-E0) < 1e-7]
print("GS count:", len(gs))

# --- pentagon adjacency by edge type ---
e2f = {}
for fi, fe in enumerate(L.face_edges):
    for k, o in fe: e2f.setdefault(k, []).append(fi)
is34 = lambda k: tuple(sorted(L.vtype[L.edges[k]])) == (3, 4)
deg34 = Counter()
for k, fs in e2f.items():
    if is34(k): deg34[fs[0]] += 1; deg34[fs[1]] += 1
print("pentagon neighbours via z3-z4 edges:", set(deg34.values()), "(4 => square-lattice graph)")

# vertex level (0 happy / 1 unhappy) for z3
def level(v, s):
    ks = [k for k, _ in L.incidence[v]]
    tab = {c: sum(J[ks[a], ks[b]]*c[a]*c[b] for a, b in itertools.combinations(range(len(ks)), 2))
           for c in itertools.product([-1, 1], repeat=len(ks))}
    lv = sorted(set(np.round(list(tab.values()), 4)))
    return lv.index(np.round(tab[tuple(int(s[k]) for k in ks)], 4))

def z3end(k): return [v for v in L.edges[k] if L.vtype[v] == 3][0]
def spin_into_z3(k, s):
    u, v = L.edges[k]                      # reference dir u->v; sigma=+1 points into v
    return (v == z3end(k)) == (s[k] > 0)

rules = {
    "z3 end unhappy":             lambda k, s: level(z3end(k), s) == 1,
    "z3 end happy":               lambda k, s: level(z3end(k), s) == 0,
    "spin points into z3":        lambda k, s: spin_into_z3(k, s),
    "spin points into z4":        lambda k, s: not spin_into_z3(k, s),
    "same-sign chirality":        lambda k, s: np.sign(L.chirality(s)[e2f[k][0]]) == np.sign(L.chirality(s)[e2f[k][1]]),
}
for name, rule in rules.items():
    good, matchings = 0, set()
    for s in gs:
        dm = [k for k in e2f if is34(k) and rule(k, s)]
        c = Counter(f for k in dm for f in e2f[k])
        if len(c) == len(L.faces) and all(v == 1 for v in c.values()):
            good += 1; matchings.add(frozenset(dm))
    print(f"rule '{name}': perfect matching in {good}/{len(gs)} GS, distinct matchings {len(matchings)}")

print("\n--- rules tied to the unhappy z3 vertex (2 unhappy z3 per cell = 2 dimers per cell) ---")
def flip_makes_happy(k, s):
    v = z3end(k)
    if level(v, s) != 1: return False
    t = s.copy(); t[k] = -t[k]
    return level(v, t) == 0
def other_flip_makes_happy(k, s):
    v = z3end(k)
    if level(v, s) != 1: return False
    other = [kk for kk, _ in L.incidence[v] if kk != k and is34(kk)][0]
    t = s.copy(); t[other] = -t[other]
    return level(v, t) == 0
for name, rule in {"flip this z3-z4 spin heals unhappy z3": flip_makes_happy,
                   "flip the other z3-z4 spin heals it": other_flip_makes_happy}.items():
    good, matchings = 0, set()
    for s in gs:
        dm = [k for k in e2f if is34(k) and rule(k, s)]
        c = Counter(f for k in dm for f in e2f[k])
        if len(c) == len(L.faces) and all(v == 1 for v in c.values()):
            good += 1; matchings.add(frozenset(dm))
    print(f"rule '{name}': perfect matching in {good}/{len(gs)} GS, distinct matchings {len(matchings)}")
# unhappy z3 anatomy
v3 = np.where(L.vtype == 3)[0]
anat = Counter()
for s in gs[:50]:
    for v in v3:
        if level(v, s) == 1:
            ks = [k for k, _ in L.incidence[v]]
            heal = tuple(sorted(('partner' if not is34(k) else 'z3z4') for k in ks
                         if (lambda t: (t.__setitem__(k, -t[k]), level(v, t) == 0)[1])(s.copy())))
            anat[heal] += 1
print("which single flips heal an unhappy z3:", dict(anat))
