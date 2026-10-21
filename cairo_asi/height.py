"""Thurston height function of domino tilings stored as partner codes (0 right, 1 left, 2 up, 3 down) on cells (a,b).
Cell (a,b) occupies [a,a+1]x[b,b+1]; colour black if (a - off + b - off) even. Heights on grid vertices (u,v)."""
import numpy as np
from collections import deque

def heights(part, inside, off, ref=None):
    S = part.shape[0]
    def cell_in(a, b): return 0 <= a < S and 0 <= b < S and inside[a, b]
    def black(a, b): return ((a - off) + (b - off)) % 2 == 0
    verts = set()
    for a in range(S):
        for b in range(S):
            if inside[a, b]: verts |= {(a, b), (a+1, b), (a, b+1), (a+1, b+1)}
    def step(p, q):
        """height change walking p -> q along a unit edge (left-hand-cell rule)"""
        (u, v), (u2, v2) = p, q; du, dv = u2 - u, v2 - v
        if du == 1:  left, right = (u, v), (u, v - 1)            # east: left cell is above the edge
        elif du == -1: left, right = (u - 1, v - 1), (u - 1, v)  # west: left cell below
        elif dv == 1: left, right = (u - 1, v), (u, v)           # north: left cell is to the west
        else: left, right = (u, v - 1), (u - 1, v - 1)           # south: left cell to the east
        if not (cell_in(*left) or cell_in(*right)): return None
        crossed = False
        if cell_in(*left) and cell_in(*right):
            (a1, b1), (a2, b2) = left, right
            if a1 == a2: crossed = (part[a1, min(b1, b2)] == 2)   # vertical domino across a horizontal edge
            else: crossed = (part[min(a1, a2), b1] == 0)           # horizontal domino across a vertical edge
        s = 1 if black(*left) else -1
        return -3*s if crossed else s
    h = {}; start = ref if ref is not None else min(verts)
    h[start] = 0; Q = deque([start]); bad = 0
    while Q:
        p = Q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + d[0], p[1] + d[1])
            if q not in verts: continue
            dh = step(p, q)
            if dh is None: continue
            if q not in h: h[q] = h[p] + dh; Q.append(q)
            elif h[q] != h[p] + dh: bad += 1
    return h, bad
