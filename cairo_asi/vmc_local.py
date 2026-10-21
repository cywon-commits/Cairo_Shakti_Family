"""Start from every candidate (both orientations via the mirrored coupling where needed) and anneal at low T."""
import numpy as np, json, sys, time
exec(open("vmc_run.py").read().split("fn = \"verify_mc.json\"")[0].replace("name, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(\",\")]", "name, seeds = sys.argv[1], [0]"))
print(f"{name}: best candidate {best} {cand[best]:.5f}")
for X in names:
    for sd in range(2):
        s = ST[X].copy()
        for T in np.geomspace(0.05, 0.002, 25): s, _, _ = mc(s, J, Jl0, *arr, T, T, 100, int(1e5*T) + 13*sd)
        e = 0.5*s@J@s/L.N; pa, pc = F.psi(s)
        print(f"   start {X:3s} seed {sd}: end E - best = {e - cand[best]:+.5f} (start {0.5*ST[X]@J@ST[X]/L.N - cand[best]:+.5f}); psi_A {pa:+.2f} psi_C {pc:+.2f}", flush=True)
