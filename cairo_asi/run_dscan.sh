#!/bin/zsh
# A|? line in the (d, T) plane at n = 12 (task 4): ladder tuning + A- and C-start PT for each d. Resume-safe; run from Terminal.
[ -z "$FSS_CAFF" ] && exec env FSS_CAFF=1 caffeinate -i /bin/zsh "$0" "$@"
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
one() { local d=$1 Tlo=$2 Tdense=$3 st=$4; local D=$(printf "%.4f" $d)
  if [ $st = A ]; then
    [ -f fss/ladder_d${D}_n12.json ] || python3 -u fss_pt.py tune --d $d --n 12 --M 24 --Tlo $Tlo --Tdense $Tdense --Tmax 0.4 --rounds 24000 --iters 3 >> fss/tune_d${D}_n12.log 2>&1
  else
    while [ ! -f fss/ladder_d${D}_n12.json ]; do sleep 20; done
  fi
  python3 -u fss_pt.py run --d $d --n 12 --start $st --seed 1 --rounds 400000 --every 20 --ladder fss/ladder_d${D}_n12.json >> fss/run_d${D}_n12_$st.log 2>&1; }
for st in A C; do
  one 0.140  0.06 0.20 $st &
  one 0.1475 0.04 0.18 $st &
  one 0.150  0.03 0.16 $st &
  one 0.1527 0.02 0.15 $st &
done
wait
echo "dscan done" >> fss/dscan.log
