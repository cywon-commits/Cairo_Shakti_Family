import numpy as np
from collections import Counter
from cairo_asi import *

L = CairoLattice(8, 8)
# pentagon adjacency via shared edges
e2f = {}
for fi, fe in enumerate(L.face_edges):
    for k, o in fe:
        e2f.setdefault(k, []).append(fi)
etype = {k: tuple(sorted(L.vtype[L.edges[k]])) for k in range(L.N)}   # (3,3)=partner bond, (3,4)

def analyse(s, name):
    chi = L.chirality(s)
    dark = np.where(np.abs(chi) > 0.9)[0]
    kinds, ok = Counter(), 0
    partner = {}
    for f in dark:
        # neighbours of opposite |chi| class, same sign
        cand = []
        for k, _ in L.face_edges[f]:
            g = [x for x in e2f[k] if x != f][0]
            if abs(chi[g]) < 0.9 and np.sign(chi[g]) == np.sign(chi[f]):
                cand.append((g, etype[k]))
        kinds[tuple(sorted(c[1] for c in cand))] += 1
        if len(cand) == 1:
            ok += 1; partner[f] = cand[0][0]
    light_used = Counter(partner.values())
    print(f"[{name}] dark pentagons {len(dark)}; unique same-sign light neighbour: {ok}; "
          f"each light used once: {all(v == 1 for v in light_used.values()) and len(light_used) == ok}")
    print("   shared-edge type of candidate partners:", dict(kinds))
    return partner

p = analyse(np.load("gs_cell2x2.npy"), "dipolar 2x2 GS")
p = analyse(np.load("sa_best_dip6_8.npy"), "best SA")

print("\n--- relations across each shared edge, 2x2 GS ---")
s = np.load("gs_cell2x2.npy"); chi = L.chirality(s)
rel = Counter()
for k, fs in e2f.items():
    f, g = fs
    a, b = sorted([(round(abs(chi[f]),1), int(np.sign(chi[f]))), (round(abs(chi[g]),1), int(np.sign(chi[g])))])
    same = "same sign" if a[1] == b[1] else "opp sign"
    rel[(etype[k], f"|chi| {a[0]}-{b[0]}", same)] += 1
for k, v in sorted(rel.items()): print(k, v)

# connected same-sign clusters
import itertools
parent = list(range(len(chi)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for k, (f, g) in e2f.items():
    if np.sign(chi[f]) == np.sign(chi[g]): parent[find(f)] = find(g)
sizes = Counter(Counter(find(x) for x in range(len(chi))).values())
print("same-sign cluster sizes (size: count):", dict(sizes))
