"""Candidate crystals in the dimer manifold: minimal periodic cells, count-based order parameters, net moment."""
import numpy as np, json
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from selection_T import FastDimers
from manifold_mc import shift_state
from plot_phases import tile
from cairo2 import S0
def min_period(s, n):
    """smallest (px, py) with s invariant under (px,0) and (0,py), allowing a global spin flip"""
    def inv(dx, dy): return np.array_equal(shift_state(s, n, dx, dy), s)
    px = next(p for p in range(1, n + 1) if n % p == 0 and inv(p, 0)); py = next(p for p in range(1, n + 1) if n % p == 0 and inv(0, p))
    return px, py
def sub(s, n, px, py):
    out = np.empty(px*py*10)
    for i in range(px):
        for j in range(py): out[(i*py + j)*10:(i*py + j)*10 + 10] = s[(i*n + j)*10:(i*n + j)*10 + 10]
    return out
def describe(s, px, py, d=0.20):
    L = CairoLattice(px, py, d=d); Jl = local_needle_coupling(L); F = FastDimers(L, Jl)
    pa, pc = F.psi(s); m = (s[:, None]*L.edir*L.elen[:, None]).sum(0)/L.N
    return float(pa), float(pc), m
# 4x2 states
L42 = CairoLattice(4, 2, d=0.20); F42 = FastDimers(L42, local_needle_coupling(L42))
ps = np.array([F42.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
cands = {}
cands["A"] = (np.load("scan3_min_d0.1372.npy")[0].astype(float), 4, 2)
cands["C"] = (np.load("scan3_min_d0.2500.npy")[0].astype(float), 4, 2)
M_idx = np.flatnonzero((np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(np.abs(pc) - 1) < 1e-9)); cands["M"] = (S0[M_idx[0]].copy(), 4, 2)
W8 = np.load("pt_best_d0.2000_n8_A3.npy").astype(float); pxW, pyW = min_period(W8, 8)
L12 = np.load("fss/lowT_state_d0.1500_n12.npy").astype(float); pxL, pyL = min_period(L12, 12)
print(f"minimal periods: W {pxW}x{pyW} (from 8x8), L3 {pxL}x{pyL} (from 12x12)")
pyW = max(pyW, 2); pyL = max(pyL, 2); pxW = max(pxW, 2); pxL = max(pxL, 2)
cands["W"] = (sub(W8, 8, pxW, pyW), pxW, pyW); cands["L3"] = (sub(L12, 12, pxL, pyL), pxL, pyL)
info = {}
for k, (s, px, py) in cands.items():
    a, c, m = describe(s, px, py); info[k] = dict(px=px, py=py, psiA=a, psiC=c, m=[float(m[0]), float(m[1])])
    print(f"{k:3s}: cell {px}x{py}, psi_A {a:+.3f}, psi_C {c:+.3f}, net moment per island ({m[0]:+.3f}, {m[1]:+.3f}) |m| {np.hypot(*m):.3f}")
    np.save(f"cand_{k}.npy", s)
json.dump(info, open("candidates.json", "w"), indent=1)
