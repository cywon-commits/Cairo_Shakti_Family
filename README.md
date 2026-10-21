# Cairo–Shakti family of artificial spin ice

Ising dipoles on the edges of the pentagonal lattice family Cairo(d), which runs from square ice
through the equilateral Cairo lattice (d = (√7−1)/12) to the Shakti lattice (d = 1/4).

## Layout

| Path | Contents |
|---|---|
| `cairo_asi/` | Simulation code (lattice builder, dimer mapping, dipolar couplings, manifold MC, parallel tempering, finite-size scaling) and its results (`.json`, `.npy`, `.log`, `.png`). See `cairo_asi/README.md` for a section-by-section log of results and `cairo_asi/HANDOFF.md` (Korean) for the FSS work brief. |
| `cairo_asi/fss/` | FSS campaign at d = 0.145 and the n = 12 (d, T) scan: tuned PT ladders, run/tune logs, tests. |
| `paper/` | Manuscript *One dimer manifold, many crystals* (`cairo_shakti.tex` / `.pdf`), figures (`figs/`) and figure scripts (`scripts/`). |

Requirements: `numpy`, `scipy`, `numba`, `matplotlib`.

## Not included

- Raw PT checkpoints `cairo_asi/fss/pt_d*_n*_{A,C}_s1.npz` and the coupling cache `cairo_asi/couplings/*.npz`
  (regenerable with `cairo_asi/run_all.sh` / `fss_mc.py`; the scripts `fss_analyze.py` and `fss_local.py` need them).
- Most of the paper's inputs. The figure scripts in `paper/scripts/` import modules that are not in this
  repository (`chessboard`, `cand_response`, `cairo2`) and read data files that are also missing
  (`cand_response12x4_*.json`, `L3alt_d0.153_n12.npy`, `verify_exact.json`, `lattice_sums_*.json`,
  `real_islands_scan.json`, `stripe_sums_*.json`, `aztecS_gs_n*_s5.npy`). They also use absolute paths under
  `/mnt/user-data/outputs/`. Only `fig_height.py` (Fig. 3) is self-contained, and it still needs that output path changed.
  The rendered figures are in `paper/figs/`.
