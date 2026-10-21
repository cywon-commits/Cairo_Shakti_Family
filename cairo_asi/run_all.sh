#!/bin/zsh
# Runs / resumes the whole d = 0.145 FSS campaign (safe to re-run: every job resumes from its checkpoint).
# Run it from macOS Terminal (outside the Claude app) so that quitting the app does not stop it:   ./run_all.sh
[ -z "$FSS_CAFF" ] && exec env FSS_CAFF=1 caffeinate -i /bin/zsh "$0" "$@"   # keep the Mac awake for the whole campaign
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { local n=$1 st=$2 R=$3
  python3 -u fss_pt.py run --d 0.145 --n $n --start $st --seed 1 --rounds $R --every 20 --ladder fss/ladder_d0.1450_n$n.json >> fss/run_n${n}_$st.log 2>&1; }
tune() { local n=$1 M=$2     # n >= 20: ladder 0.08-0.25 only (the 0.25-0.4 replicas were the expensive part and are not needed for tunnelling)
  [ -f fss/ladder_d0.1450_n$n.json ] || python3 -u fss_pt.py tune --d 0.145 --n $n --M $M --Tmax 0.25 --frac_dense 0.85 --rounds 30000 --iters 3 >> fss/tune_n$n.log 2>&1; }
waitfor() { while [ ! -f fss/ladder_d0.1450_n$1.json ]; do sleep 30; done; }
run 8 A 1000000 &
run 8 C 1000000 &
run 12 A 600000 &
run 12 C 600000 &
run 16 A 500000 &
run 16 C 500000 &
{ tune 20 24; run 20 A 300000 } &
{ waitfor 20; run 20 C 300000 } &
{ tune 24 28; run 24 A 200000 } &
{ waitfor 24; run 24 C 200000 } &
wait
python3 fss_analyze.py 0.145 8 12 16 20 24 > fss/analysis.log 2>&1
python3 fss_plot.py 0.145 8 12 16 20 24 >> fss/analysis.log 2>&1
echo "all done" >> fss/analysis.log
