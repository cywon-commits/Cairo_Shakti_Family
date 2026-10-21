import numpy as np, sys, time
import real_islands as RI
from stripes import stripe, C_on
for d in [float(x) for x in sys.argv[1].split(",")]:
    t0 = time.time(); out = []
    for k, n in ((0, 12), (1, 8), (2, 12)):
        A = RI.RealArray(n, n, d); _, J = RI.full_J(A); s, c = stripe(k, n), C_on(n)
        de = (0.5*s@J@s - 0.5*c@J@c)/A.L.N; out.append((k, de*10*(4*k + 4)/2))
    print(f"realistic islands d={d}: sigma per wall cell (K): " + ", ".join(f"k={k}: {v:+.2f}" for k, v in out) + f"  [{time.time()-t0:.0f}s]", flush=True)
