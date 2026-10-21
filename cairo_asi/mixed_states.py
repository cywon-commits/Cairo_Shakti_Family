import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo3 import CairoLattice3
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from selection_T import FastDimers
from dimer_map import DimerMap
from chessboard import draw
from cairo2 import S0
from plot_phases import tile
L = CairoLattice3(4, 2, 0.20); Jl = local_needle_coupling(L); F = FastDimers(L, Jl)
ps = np.array([F.psi(s) for s in S0]); pa, pc = ps[:, 0], ps[:, 1]
pureA = (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(pc) < 1e-9); pureC = (np.abs(np.abs(pc) - 1) < 1e-9) & (np.abs(pa) < 1e-9)
mixed = (np.abs(np.abs(pa) - 1) < 1e-9) & (np.abs(np.abs(pc) - 1) < 1e-9)
print(f"4x2 manifold: pure A (|psiA|=1, psiC=0) {pureA.sum()}, pure C (psiA=0, |psiC|=1) {pureC.sum()}, MIXED (|psiA|=|psiC|=1) {mixed.sum()}")
print(" d_bar | E_A min | E_C min | E_mixed min  (per spin, eps = theta = 0) | mixed - A | mixed - C")
best_mixed = None
for db in (0.137, 0.145, 0.153, 0.16, 0.20, 0.25):
    Ld = CairoLattice3(4, 2, db); J = hybrid_coupling(Ld, 64.0); E = 0.5*np.einsum('ij,jk,ik->i', S0, J, S0)/Ld.N
    eA, eC, eM = E[pureA].min(), E[pureC].min(), E[mixed].min()
    print(f" {db:.3f} | {eA:.5f} | {eC:.5f} | {eM:.5f} | {eM - eA:+.5f} | {eM - eC:+.5f}", flush=True)
    if db == 0.153: best_mixed = S0[mixed][np.argmin(E[mixed])]
# draw the best mixed state (tiled to 8x8) on the chessboard
from cairo_asi import CairoLattice
L8 = CairoLattice(8, 8, d=0.153); D8 = DimerMap(L8, local_needle_coupling(L8))
fig, ax = plt.subplots(figsize=(7.5, 7.8)); draw(ax, L8, D8, tile(best_mixed, 4, 2, 8), "best mixed state (|psi_A| = |psi_C| = 1), d = 0.153")
plt.tight_layout(); plt.savefig("mixed_state_chessboard.png", dpi=110)
