"""Wait for the n=20/24 ladders, then launch their A- and C-start production runs (detached)."""
import os, time, subprocess, sys
todo = {20: 600000, 24: 400000}
while todo:
    for n in list(todo):
        if os.path.exists(f"fss/ladder_d0.1450_n{n}.json"):
            for st in "AC":
                subprocess.run([sys.executable, "fss_launch.py", f"fss/run_n{n}_{st}.log", "fss_pt.py", "run", "--d", "0.145", "--n", str(n),
                                "--start", st, "--seed", "1", "--rounds", str(todo[n]), "--every", "20", "--ladder", f"fss/ladder_d0.1450_n{n}.json"])
            print(time.ctime(), "launched n =", n, flush=True); del todo[n]
    time.sleep(30)
