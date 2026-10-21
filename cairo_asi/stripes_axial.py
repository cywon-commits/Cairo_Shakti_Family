"""Axial wall crystals generalising L3: C+ and C- domains of width 2 + 4t columns separated by alternating alpha (direct)
and beta (split, through an A-like column pair) walls. t = 0 reproduces L3 (period 12 columns, 2 cells high)."""
import numpy as np, collections
import needle
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from selection_T import FastDimers
from dimer_map import DimerMap
from cairo2 import S0
_L3 = np.load("cand_L3.npy"); _C = np.load("cand_C.npy")
def col(s, i, py, px): return s[((i % px)*py)*10:((i % px)*py)*10 + 10*py].copy()
cols3 = [col(_L3, i, 2, 12) for i in range(12)]
# find, for each 2-column domain of L3, a C crystal (4 x 2) containing it and its column cycle
_Lr = CairoLattice(4, 2, d=0.20); _F = FastDimers(_Lr, local_needle_coupling(_Lr))
pureC = [s for s in S0 if abs(_F.psi(s)[0]) < 1e-9 and abs(abs(_F.psi(s)[1]) - 1) < 1e-9]
def cycle_for(a, b):
    for s in pureC:
        cs = [col(s, i, 2, 4) for i in range(4)]
        for k in range(4):
            if np.array_equal(cs[k], a) and np.array_equal(cs[(k + 1) % 4], b): return [cs[(k + q) % 4] for q in range(4)]
    raise ValueError("no crystal")
domains = {1: cycle_for(cols3[1], cols3[2]), 5: cycle_for(cols3[5], cols3[6]), 7: cycle_for(cols3[7], cols3[8]), 11: cycle_for(cols3[11], cols3[0])}
def axial(t, ny=2):
    """columns: domain(1..2) beta(3,4) domain(5..6) alpha domain(7..8) beta(9,10) domain(11,0) alpha, each domain 2 + 4t columns"""
    seq = []
    for start, inserts in ((1, [3, 4]), (5, []), (7, [9, 10]), (11, [])):
        cyc = domains[start]; seq += [cyc[q % 4] for q in range(2 + 4*t)]; seq += [cols3[i] for i in inserts]
    nx = len(seq); s = np.empty(nx*ny*10)
    for i in range(nx):
        c = seq[i]                                          # column holds 2 cells (j = 0, 1); tile in j
        for j in range(ny): s[(i*ny + j)*10:(i*ny + j)*10 + 10] = c[(j % 2)*10:(j % 2)*10 + 10]
    return s, nx
def C_on_xy(nx, ny):
    s = np.empty(nx*ny*10)
    for i in range(nx):
        for j in range(ny): s[(i*ny + j)*10:(i*ny + j)*10 + 10] = _C[((i % 4)*2 + (j % 2))*10:((i % 4)*2 + (j % 2))*10 + 10]
    return s
if __name__ == "__main__":
    needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
    for t in (0, 1, 2):
        s, nx = axial(t); L = CairoLattice(nx, 2, d=0.20); D = DimerMap(L, local_needle_coupling(L))
        print(f"t={t}: {nx} x 2 cells, perfect matching {D.is_perfect(D.dimers(s))}" + (f", reproduces L3: {np.array_equal(s, _L3)}" if t == 0 else ""))
