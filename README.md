# Cairo–Shakti family of artificial spin ice

Code, data and manuscript for *One dimer manifold, many crystals: height plateaus, wall crystals and design
rules in Cairo–Shakti artificial spin ice* (C. Won, Kyung Hee University).

```
cairo_asi/          simulation code, small result files (json/npy), analysis notes (README.md)
cairo_asi/fss/      finite-size-scaling logs, ladder files, low-T state at d = 0.150
paper/              cairo_shakti.tex / .pdf, figs/ (vector PDF + preview PNG), scripts/ (one script per figure)
requirements.txt    numpy, scipy, numba, matplotlib
```

## Regenerating the paper figures

```bash
pip install -r requirements.txt
for f in fig_lattice fig_height fig_shapes fig_deform fig_walls fig_dmu fig_real fig_expt_aztec; do
    python paper/scripts/$f.py
done
cd paper && pdflatex cairo_shakti.tex && pdflatex cairo_shakti.tex
```

The scripts can be run from any directory: each one adds `cairo_asi/` to the import path, reads its data
relative to `cairo_asi/`, and writes into `paper/figs/`. All of them run from the files in this repository
alone (checked on a fresh copy; total about 2 minutes). The first run of `fig_shapes.py` uses the cached
manifold `cairo_asi/manifold_4x2_S0.npy`; if that file is deleted, `cairo2.py` re-enumerates it.

| Paper | File | Script | Data read (in `cairo_asi/`) |
|---|---|---|---|
| Fig. 1 | fig1_family | fig_lattice.py | – |
| Fig. 2 | fig2_ACW | fig_lattice.py | scan3_min_d0.1372.npy, scan3_min_d0.2500.npy, pt_best_d0.2000_n8_A3.npy |
| Fig. 3 | fig3_height | fig_height.py | – |
| Fig. 4 | fig4_ladder | fig_shapes.py | manifold_4x2_S0.npy, L3alt_d0.153_n12.npy, cand_response12x4_*.json |
| Fig. 5 | fig5_deformations | fig_shapes.py | – (lattice geometry only) |
| Fig. 6 | fig5_deform | fig_deform.py | cand_response12x4_*.json, verify_exact.json |
| Fig. 7 | fig6_walls | fig_walls.py | stripe_sums_0.20_0.25.json, stripe_sums_0.153_0.17.json |
| Fig. 8 | fig7_dmu | fig_dmu.py | lattice_sums_{A,M,W,L3p}.json |
| Fig. 9 | fig8_real | fig_real.py | real_islands_scan.json |
| Fig. 10 | fig9_saccone | fig_expt_aztec.py | – (values from manifold MC / vertex-population fits, see cairo_asi/README.md) |
| Fig. 11 | fig10_aztec | fig_expt_aztec.py | aztecS_gs_n{6,7,8,9}_d+0.000_s5.npy |

(File names follow an earlier numbering; the "Paper" column is the order in the manuscript.)

## How the data files were produced

| Result | Scripts | Output |
|---|---|---|
| Dimer manifold, counting | count_degeneracy.py, verify_bijection.py, dimer_map.py | degeneracy.txt |
| Candidate crystals A, C, M, W, L3, L3′ | candidates.py, columnar_stripes.py (L3′) | cand_*.npy, candidates.json, L3alt_*.npy |
| Energies and linear responses (ideal islands) | cand_response.py | cand_response12x4_*.json |
| Exact checks at finite deformation | verify_exact.py, vmc_run.py | verify_exact.json |
| Realistic islands | real_islands.py, real_islands_scan.py, real_verify.py | real_islands_scan.json |
| Point-dipole lattice sums, μ diagram | wall_lattice_sums.py, plot_dmu_phase*.py, dmu_verify*.py | lattice_sums_*.json |
| Wall tensions | stripes.py, stripe_sums.py, stripe_real.py, columnar_stripes.py, columnar_real.py | stripe_sums_*.json, axial_sums.json |
| Aztec diamonds | aztec_small.py, aztec_phi.py, aztec_weighted.py | aztecS_gs_*.npy |
| Finite-size scaling, d = 0.145 | fss_pt.py, fss_analyze.py, fss_plot.py, run_all.sh | fss_*.json, fss/*.log |

`cairo_asi/README.md` is the running research log, in chronological order. Statements that later
results corrected are marked; in particular the early "no intermediate phase between A and C" (from 4×2
cells) is superseded by the ladder crystal L3′ for 0.1495 ≲ d ≲ 0.157 found on 12×4 and 12×12 tori.

## Not included (regenerable)

- Coupling-matrix caches `cairo_asi/J_d*.npz`, `vmcJ_*.npz`, `Jsac_*.npz`, `couplings/*.npz` (≈280 MB):
  written automatically on first use.
- Parallel-tempering checkpoints `cairo_asi/fss/pt_*.npz` (≈60 MB each): rerun `cairo_asi/run_all.sh`.
  `fss_analyze.py` and `fss_local.py` need them; the analysed results are in `fss_*.json`.
  `superstructure*.py` reads `fss/pt_d0.1500_n12_{A,C}_s1.npz` from the same campaign.
