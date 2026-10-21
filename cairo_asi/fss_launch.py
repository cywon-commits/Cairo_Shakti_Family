"""Start a fss_pt.py job detached from the terminal/app (new session, caffeinate -i keeps the Mac from idle-sleeping).
   python fss_launch.py LOGFILE fss_pt.py run --n 16 ...      (re-launching a `run` resumes from its checkpoint)"""
import sys, os, subprocess
env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
log = open(sys.argv[1], "a")
p = subprocess.Popen(["caffeinate", "-i", sys.executable, "-u"] + sys.argv[2:], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, env=env)
print(p.pid)
