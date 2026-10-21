"""Height-plateau analysis of the d = 0.150 low-T mixed A/C superstructure (n = 12)."""
import numpy as np, collections
import needle; needle.GAP = 0.0; needle.SUBDIV = 8; needle.FIXED_M = True
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling, hybrid_coupling
from dimer_map import DimerMap
from block_order import BlockOrder
from plot_phases import tile
n, d = 12, 0.150
L = CairoLattice(n, n, d=d); Jl = local_needle_coupling(L); J = hybrid_coupling(L, 64.0); D = DimerMap(L, Jl); B = BlockOrder(L, D)
PLAT = {"z4B": 0, "p1": 1, "z4A": 2, "p2": 3}
sA = tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, n); EA = 0.5*sA@J@sA/L.N
centre = {}
for cyc, org, kind, _ in B.plaq: centre[(org, kind)] = np.mean([L.face_center[f] for f in cyc], axis=0)
def analyse(s, name):
    dm = D.dimers(s); perfect = D.is_perfect(dm); e = 0.5*s@J@s/L.N
    fl = B.flippable(s); kinds = collections.Counter(k for _, k in fl)
    plats = collections.Counter(PLAT[k] for _, k in fl)
    print(f"{name}: manifold {perfect}, e - e_A = {e - EA:+.2e}/spin, blocks {len(fl)} by plateau {dict(sorted(plats.items()))}")
    return fl, e
s0 = np.load("fss/lowT_state_d0.1500_n12.npy")
fl0, e0 = analyse(s0, "lowT state")
res = {}
for tag in ("A", "C"):
    z = np.load(f"fss/pt_d0.1500_n12_{tag}_s1.npz", allow_pickle=True)
    T = z["T"]; S = z["S"]; order = np.argsort(T)
    print(f"--- {tag}-start run: T range {T.min():.3f}-{T.max():.3f}; meta: {str(z['meta'])[:90]}")
    for m in order[:4]:
        analyse(S[m], f"   T={T[m]:.4f}")
    res[tag] = S[order[0]]
same = np.array_equal(res["A"], res["C"]) or np.array_equal(res["A"], -res["C"])
print("lowest-T states of A- and C-start identical (up to time reversal):", same)

# ---- staircase test: fit phase phi = pi h / 2 to a plane wave allowed on the torus ----
def staircase(s, name):
    fl = B.flippable(s)
    R = np.array([centre[(o, k)] for o, k in fl]); phi = np.array([PLAT[k] for _, k in fl])*np.pi/2
    best = []
    for m1 in range(-n, n + 1):
        for m2 in range(-n, n + 1):
            kv = 2*np.pi*np.array([m1, m2])/L.box
            amp = abs(np.mean(np.exp(1j*(phi - R@kv))))
            best.append((amp, m1, m2))
    best.sort(reverse=True)
    a0 = abs(np.mean(np.exp(1j*phi)))
    a2 = max(abs(np.mean(np.exp(2j*phi - 1j*R@(2*np.pi*np.array([m1, m2])/L.box)))) for m1 in range(-n, n+1) for m2 in range(-n, n+1))
    print(f"{name}: k = 0 amplitude {a0:.3f}; best plane wave (m1, m2) = {best[0][1:]} amplitude {best[0][0]:.3f}; "
          f"next {best[1][1:]} {best[1][0]:.3f}; best second-harmonic (A/C stripe test) amplitude {a2:.3f}")
    return best[0]
b = staircase(s0, "lowT state")
m1, m2 = b[1], b[2]
# winding: phase advances by 2*pi*(m1, m2) across the torus -> height changes by 4*m per period (one plateau cycle = 4 height units)
print(f"=> height winds by 4 x ({m1}, {m2}) across the {n}x{n} torus: mean slope = 4 x (m1, m2)/n per lattice constant "
      f"= ({4*m1/n:+.3f}, {4*m2/n:+.3f}) height units per Cairo cell; terrace (plateau) width = n / sqrt(m1^2+m2^2) / 4 = {n/np.hypot(m1, m2)/4:.2f} cells")
for tag in ("A", "C"):
    staircase(res[tag], f"{tag}-start lowest-T slot")
staircase(sA, "A crystal (reference)")
