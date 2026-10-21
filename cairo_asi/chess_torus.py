"""Global chessboard coordinates (mod 2n) for the n x n Cairo torus, and conversions dimers <-> direction maps."""
import numpy as np
from chessboard import chess_coords
def torus_coords(L, D, n):
    pos, conf = chess_coords(L, D); M = 2*n
    for k, (f1, f2) in D.e2f.items():                      # check: every z3-z4 adjacency is a unit step mod M
        if not D.is34[k]: continue
        dx, dy = (pos[f2][0] - pos[f1][0]) % M, (pos[f2][1] - pos[f1][1]) % M
        if (dx, dy) not in ((1, 0), (M - 1, 0), (0, 1), (0, M - 1)): raise ValueError("inconsistent coordinates")
    return {f: (x % M, y % M) for f, (x, y) in pos.items()}
def direction_map(L, D, s, xy, n):
    M = 2*n; g = np.full((M, M), "?", dtype=object)
    for k in D.dimers(s):
        f1, f2 = D.e2f[k]; (x1, y1), (x2, y2) = xy[f1], xy[f2]
        for (a, b, c, d) in ((x1, y1, x2, y2), (x2, y2, x1, y1)):
            dx, dy = (c - a) % M, (d - b) % M
            g[a, b] = {(1, 0): "r", (M - 1, 0): "l", (0, 1): "u", (0, M - 1): "d"}[(dx, dy)]
    return g
def dimers_from_map(D, xy, g, n):
    M = 2*n; at = {v: f for f, v in xy.items()}; out = []
    for k, (f1, f2) in D.e2f.items():
        if not D.is34[k]: continue
        (x1, y1), (x2, y2) = xy[f1], xy[f2]; dx, dy = (x2 - x1) % M, (y2 - y1) % M
        want = {(1, 0): "r", (M - 1, 0): "l", (0, 1): "u", (0, M - 1): "d"}[(dx, dy)]
        if g[x1, y1] == want: out.append(k)
    return out
