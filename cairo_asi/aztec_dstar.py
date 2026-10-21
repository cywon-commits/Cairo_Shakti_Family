"""Ground-state class vs delta for small Aztec diamonds: find delta*(n) where the ground state switches class."""
import numpy as np, sys, subprocess, json
n = int(sys.argv[1]); sign = -1 if n % 2 == 0 else +1          # preferred class at delta=0: odd for even n, even for odd n
deltas = [sign*(-x) for x in (0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3)]   # push toward the disfavoured class
res = []
for dl in deltas:
    out = subprocess.run(["python", "aztec_small.py", str(n), f"{dl}", "1.0", "700", "3"], capture_output=True, text=True).stdout.splitlines()[0]
    ne, no = [int(x) for x in out.split("(N_even, N_odd) = (")[1].rstrip(")").split(",")]
    phi = (ne - no)/(ne + no); res.append((dl, ne, no, phi)); print(f"n={n} delta={dl:+.2f}: GS (N_even, N_odd) = ({ne}, {no}), Phi_GS {phi:+.2f}", flush=True)
json.dump(res, open(f"aztec_dstar_n{n}.json", "w"))
