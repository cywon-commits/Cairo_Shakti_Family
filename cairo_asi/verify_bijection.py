import numpy as np, itertools
from cairo_asi import *
from count_degeneracy import setup
from dimer_map import DimerMap
for nx, ny in [(2,2),(3,2),(4,2),(3,3)]:
    L, ze, zc, part, P = setup(nx, ny); D = DimerMap(L); J = D.J
    nz = len(ze); E0 = None; gs = []
    for bits in range(1 << nz):
        s = np.zeros(L.N)
        for t in range(nz): s[ze[t]] = zc[t]*(1 - 2*((bits >> t) & 1))
        opts = []
        for p in part:
            e = []
            for sg in (-1., 1.): s[p] = sg; e.append(s[p]*(J[p] @ s))
            opts.append([-1.] if e[0] < e[1]-1e-9 else [1.] if e[1] < e[0]-1e-9 else [-1., 1.])
        for combo in itertools.product(*opts):
            s[part] = combo; E = 0.5*s@J@s
            if E0 is None or E < E0 - 1e-7: E0, gs = E, [s.copy()]
            elif abs(E - E0) < 1e-7: gs.append(s.copy())
    ok, mats = 0, set()
    for s in gs:
        dm = D.dimers(s)
        if D.is_perfect(dm): ok += 1; mats.add(frozenset(dm))
    print(f"{nx}x{ny}: GS {len(gs)}, perfect matchings {ok}/{len(gs)}, distinct dimer coverings {len(mats)} (= W/2: {len(mats)*2 == len(gs)})", flush=True)
