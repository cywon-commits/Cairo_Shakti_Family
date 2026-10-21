"""Stripe states with C | C' walls at controlled spacing, assembled line by line (lines = i - j) from the C crystal,
the C' crystal and W's two wall lines.  Period P = 4k + 4 lines; k = 0 is W."""
import numpy as np
import needle
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from selection_T import FastDimers
from dimer_map import DimerMap
from cairo2 import S0
_W = np.load("cand_W.npy"); _C = np.load("cand_C.npy")
def cell(s, i, j, py): return s[(i*py + j)*10:(i*py + j)*10 + 10]
X, w1, Xp, w3 = (cell(_W, t, 0, 4) for t in range(4))                    # W lines i - j = 0..3, taken at j = 0
# C's other line content Y: neighbour line of X in C (line parity)
def other_line(cr, py, px, x):
    for i in range(px):
        for j in range(py):
            if np.array_equal(cell(cr, i, j, py), x): return cell(cr, (i + 1) % px, j, py)
    raise ValueError
Y = other_line(_C, 2, 4, X)
_L = CairoLattice(4, 2, d=0.25); _F = FastDimers(_L, local_needle_coupling(_L))
def uniform_along_diag(s, px=4, py=2):
    for i in range(px):
        for j in range(py):
            if not np.array_equal(cell(s, i, j, py), cell(s, (i + 1) % px, (j + 1) % py, py)): return False
    return True
Cp = None
for s in S0:
    if not uniform_along_diag(s): continue
    if any(np.array_equal(cell(s, i, j, 2), Xp) for i in range(4) for j in range(2)): Cp = s; break
assert Cp is not None and uniform_along_diag(_C)
Yp = other_line(Cp, 2, 4, Xp)
def stripe(k, n):
    """n x n torus, P = 4k+4 must divide n"""
    seq = []
    for m in range(2*k + 1): seq.append(X if m % 2 == 0 else Y)
    seq.append(w1)
    for m in range(2*k + 1): seq.append(Xp if m % 2 == 0 else Yp)
    seq.append(w3)
    P = len(seq); assert n % P == 0, (n, P)
    s = np.empty(n*n*10)
    for i in range(n):
        for j in range(n): s[(i*n + j)*10:(i*n + j)*10 + 10] = seq[(i - j) % P]
    return s
def C_on(n):
    s = np.empty(n*n*10)
    for i in range(n):
        for j in range(n): s[(i*n + j)*10:(i*n + j)*10 + 10] = cell(_C, i % 4, j % 2, 2)
    return s
if __name__ == "__main__":
    needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
    for k, n in ((0, 4), (1, 8), (2, 12)):
        L = CairoLattice(n, n, d=0.25); D = DimerMap(L, local_needle_coupling(L)); s = stripe(k, n)
        print(f"k={k} (P={4*k+4}) on {n}x{n}: perfect matching {D.is_perfect(D.dimers(s))}; C on same torus perfect {D.is_perfect(D.dimers(C_on(n)))}; "
              f"W reproduced: {np.array_equal(s, _W) if k == 0 else '-'}")
