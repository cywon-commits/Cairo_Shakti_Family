"""Exact expected Thurston height in uniformly random tilings of AZ(n) from the inverse Kasteleyn matrix
(conventions of Won, 'Parity-dependent sublattice structure...': cell (i,j) = [j,j+1]x[i,i+1], black if i+j even,
K[b,w] = +1 right, -1 left, +i up, -i down). Height: along a unit edge with the black cell on the left,
Delta h = +1 if no domino crosses the edge and -3 if one does => <Delta h> = s (1 - 4 P_cross), s = +-1."""
import numpy as np, sys
from collections import deque
def edge_probs(n):
    cells = [(i, j) for i in range(-n, n) for j in range(-n, n) if abs(i + .5) + abs(j + .5) <= n]
    B = [c for c in cells if (c[0] + c[1]) % 2 == 0]; W = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    bi = {c: k for k, c in enumerate(B)}; wi = {c: k for k, c in enumerate(W)}
    K = np.zeros((len(B), len(W)), complex)
    for (i, j) in B:
        for (di, dj), ph in (((0, 1), 1), ((0, -1), -1), ((1, 0), 1j), ((-1, 0), -1j)):
            w = (i + di, j + dj)
            if w in wi: K[bi[(i, j)], wi[w]] = ph
    Ki = np.linalg.inv(K)
    P = {}
    for (i, j) in B:
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            w = (i + di, j + dj)
            if w in wi: P[frozenset([(i, j), w])] = (K[bi[(i, j)], wi[w]]*Ki[wi[w], bi[(i, j)]]).real
    return set(cells), P
def mean_height(n):
    cells, P = edge_probs(n)
    black = lambda c: (c[0] + c[1]) % 2 == 0
    verts = {(x, y) for (i, j) in cells for (x, y) in ((j, i), (j+1, i), (j, i+1), (j+1, i+1))}
    def step(p, q):
        (x, y), (x2, y2) = p, q
        # cells left/right of the directed edge p->q (cell (i,j) occupies x in [j,j+1], y in [i,i+1])
        if x2 == x + 1: L, R = (y, x), (y - 1, x)
        elif x2 == x - 1: L, R = (y - 1, x - 1), (y, x - 1)
        elif y2 == y + 1: L, R = (y, x - 1), (y, x)
        else: L, R = (y - 1, x), (y - 1, x - 1)
        if L not in cells and R not in cells: return None
        s = 1 if black(L) else -1
        pc = P.get(frozenset([L, R]), 0.0) if (L in cells and R in cells) else 0.0
        return s*(1 - 4*pc)
    start = (-n, 0); h = {start: 0.0}; Q = deque([start]); err = 0.0
    while Q:
        p = Q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + d[0], p[1] + d[1])
            if q not in verts: continue
            dh = step(p, q)
            if dh is None: continue
            if q not in h: h[q] = h[p] + dh; Q.append(q)
            else: err = max(err, abs(h[q] - h[p] - dh))
    return h, err
if __name__ == "__main__":
    print(" n | h(W,E corners) | h(N), h(S) | <h(centre)> | centre residue r = h_c mod 4 (deterministic) | <h_c> - r | (n+1) mod 4 | loop err")
    for n in range(4, 23):
        h, err = mean_height(n)
        hc = h[(0, 0)]; r = round(hc) % 4 if abs(hc - round(hc)) < 1e-9 else None
        print(f"{n:2d} | {h[(-n,0)]:+.3f} {h[(n,0)]:+.3f} | {h[(0,n)]:+.3f} {h[(0,-n)]:+.3f} | {hc:+.6f} | {r} | {'' if r is None else f'{hc - r:+.3f}'} | {(n+1) % 4} | {err:.1e}")
