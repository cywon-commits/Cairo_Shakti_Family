# Cairo pentagonal artificial spin ice — Ising dipoles on edges

Files
- `cairo_asi.py`   : lattice builder (p4g coordinates, parameter d), couplings, numba MC, exact enumeration
- `run_tests.py`   : exact ground states of 1x1 / 2x1 periodic cells (brute force, N = 10, 20)
- `run_sa.py n R`  : R simulated-annealing runs on n x n cells, both interaction models; compares with tiled exact-cell states
- `cell_search.py` : searches low-energy periodic states in 2x2, 4x2, 4x4, 8x8 cells (dipolar)
- `plot_states.py` : figure with spins, vertex charges and pentagon chirality (red = CW, blue = CCW)

Geometry: A=(0,0), B=(1/2,1/2) are z=4; pairs C1,C2 = (1/2±d, ±d), C3,C4 = (∓d, 1/2±d) are z=3.
d = (sqrt7 - 1)/12 gives the equilateral Cairo tiling (angles 90°, 114.3°, 131.4°).
Energy in units of D/a^3, a = edge length. Models: vertex-local (pairs sharing a vertex) or full dipolar sum over periodic images, |r| < rc.

## Dimer (domino) mapping of the vertex-local ground-state manifold
- `vertex_frustration.py` : vertex energy levels; GS has all z=4 happy, exactly half of z=3 unhappy
- `count_degeneracy.py`   : exact GS degeneracy W(nx,ny) (numba enumeration)
- `torus_sectors.py`      : square-lattice torus dimer counts resolved by winding-parity sector
- `dimer_map.py`          : spin -> dimer rule: each unhappy z3 has exactly one z3-z4 spin whose flip heals it;
                            the two pentagons sharing that edge form a dimer
- `verify_bijection.py`   : checks that every GS gives a perfect matching and #matchings = W/2
- `dimer_dipolar.py`      : applies the map to the dipolar GS candidate and SA state -> `cairo_dimers.png`

Result: W = 2 x Z_dimer(2nx x 2ny torus, winding sector fixed by parities of nx, ny), checked for
2x2, 3x2, 4x2, 3x3 (count) and 4x3 (count); bijection verified explicitly for 2x2, 3x2, 4x2.

## Cairo <-> Shakti as one family: CairoLattice(d)
- d = (sqrt7-1)/12 : equilateral Cairo;  d = 1/4 : Shakti (rotated 45 deg; z3 at bond midpoints, straight long islands)
- `shakti.py` : explicit Shakti builder (used for the first combinatorial comparison)
- `needle.py` : islands as chains of sub-dipoles (moment ~ length). FIXED_M=True + SUBDIV=8 gives energies smooth in d
- `manifold_select.py d` : enumerates the full vertex-local GS manifold on periodic cells, exact dipolar (rc=6a) minimum
- `compare_ends.py` : same manifold at both ends (39952 states, 4x2), but dipolar ordering nearly uncorrelated
- `d_scan3.py d1 d2 ...` : selected minimiser set vs d (smooth needle model); log in scan3.log
- `plot_phases.py` : phases A (Cairo side), B (0.151 < d < 0.155), C (d > 0.156, up to Shakti) -> phases_ABC.png

## Cutoff convergence (hybrid_coupling.py, rc_convergence.py)
Island-level cutoff; needle near field (centre distance < 6a), point dipoles (moment ~ length) beyond.
- A -> C transition: d = 0.1525-0.1530, converged for rc >= 16a; NO intermediate phase (earlier "B" was a cutoff artifact).
- C is stable from d = 0.153 to 0.3467 for all rc (includes Shakti).
- Concave side needs rc ~ 96a: C -> narrow window (N10, D; width ~0.002) -> A (re-entrant, 0.349-0.415) -> N9 (0.42-0.47) -> E (>= 0.475).
- rc = 96a and 128a give identical labels. Figure: phase_line_converged.png

## Manifold-restricted MC beyond 4x2 periods (manifold_mc2.py, run_mmc2.py, mmc_batch.py)
Moves: flip all 4 spins of one z4 (or of two z4 sharing a z3) + reset affected partner spins to the local optimum;
single partner flips. Accepted only if the vertex-local energy is unchanged; Metropolis on full dipolar energy.
(Directed spin-ice loops never preserve the manifold: a loop through a z4 breaks its Type-I state.)
- Cairo side (d = 0.137, 0.145): all runs reach exactly tiled A on 8x8 -> A confirmed beyond 4x2 periods.
- d = 0.155, 0.16, 0.23-0.25: runs reach exactly tiled C.
- d = 0.17-0.21: non-4x2 states (C', uses all 8 edge types, no sublattice selection) lie 1e-5 - 5e-5 per spin
  below C; sign depends on rc at rc=16a, stable at rc >= 64a -> C and C' are nearly degenerate.
- d = 0.26-0.32: MC does not reach C (stays 2-5e-4 above) -> sampling not converged there.
- d = 0.34: non-4x2 state 5e-4 below C -> C -> concave-side boundary likely shifts below 0.3467.

## Selection strength (selection_T.py, wall_tension.py, plot_selection.py -> selection_strength.png)
Fixed-T manifold MC on 8x8, rc = 64a. Order parameters from dimer edge types (psi_A orientation, psi_C sublattice).
- A side (d <= 0.145): single specific-heat peak (T ~ 0.1-0.17), full order |psi_A| = 1.
- C side: specific-heat peak (T ~ 0.2-0.6) forms local C structure, but low-T states are diagonal stripes of
  opposite sublattice choice separated by walls (|psi_C| = 0, 0.5, 1 all nearly degenerate).
- Wall tension sigma(d) per wall cell: ~0 at 0.16, negative at 0.18-0.20 (walls favoured -> wall-rich C' states),
  +0.0019 at 0.22, +0.0048 at 0.25 (Shakti), +0.0095 at 0.28.
- Ordering temperatures (|psi|=0.5 crossing): T_A = 1.80 (d=0.10), 0.95 (0.12), 0.39 (0.137), 0.115 (0.145);
  local C order onset T_C ~ 0.51-0.58 (d=0.145-0.16), 0.32 (0.25).
- d = 0.145: re-entrance -- C-like order at intermediate T (|psi_C| up to 0.6), A at low T (entropic stabilisation of
  the wall-rich C family). Below T ~ 0.2 the C region is NOT equilibrated: walls are system-spanning lines that
  local manifold moves cannot remove (low-T |psi_C| = 0.5 is frozen, not equilibrium).

## Parallel tempering within the manifold (pt.py) -- equilibrated low-T states, 8x8, rc = 64a
Runs started from A and from C converge to the same state (equilibrium check).
- d = 0.25 (Shakti): pure C (no walls). Stacking order sets in at T ~ 0.02-0.04 (walls 2 -> 0, |psi_C| 0.3 -> 1).
- d = 0.20: wall crystal W, stripes (+1, wall, -1, wall) with period 4 diagonal lines (densest walls); both starts
  give the identical state. Stacking still fluctuating at T = 0.004 (walls 2.6-2.8 of 4) -> T_stack < ~0.004.
- d = 0.18: W, wide-wall (+1 0 -1 0) and sharp-wall (+1 +1 -1 -1) forms degenerate within 1e-5; T_stack < 0.003.
- Two temperature scales on the C side: local C structure at T ~ 0.3-0.6, stacking order 10-100x lower.

## 12x12 checks (pt_chunk.py, checkpointed PT, 30 replicas 0.5 -> 0.003)
- d = 0.20: W (6 walls on 12 diagonal lines, period 4) is the lowest state; E(W) - E(C) = -4.9e-5 per spin.
  PT at T = 0.003 still sits in a 4-wall state 5e-6 per spin above W -> stacking not yet ordered (T_stack < ~0.003).
- d = 0.25 (Shakti): PT from A reaches exactly tiled C (difference 0).
- Draft (d,T) phase diagram: phase_diagram_draft.png

## Wall rigidity (kink_energy.py, wall_dynamics.py), d = 0.25, 8x8
- Every elementary manifold move on a straight-wall state costs >= 1.82 (bulk and near wall alike),
  vs wall tension 0.005 per cell -> bending a wall costs ~ local-C energy scale.
- Fixed-T MC from a 2-wall state: walls never bend or move for T <= 0.2 (3000 sweeps each); a few bent snapshots
  only at T = 0.3, where local C order itself melts (T_C ~ 0.32).
- => walls are rigid straight lines. Stacking is effectively 1D: wall-pair cost 2 sigma L grows with L, so the
  "stacking order temperature" is a finite-size crossover (~2 sigma L / ln L), while wall motion is kinetically
  blocked below the local-C scale -> stacking disorder is expected to be frozen in experiments.

## Domain-wall mobility: A vs C (coarsening.py), 12x12, quench from T=1 then hold 3000 sweeps
- A (d = 0.137): full coarsening to a single domain (|psi_A| -> 1, wall density -> 0) within a few hundred sweeps,
  both at T = 0.15 (~0.4 T_A) and T = 0.08 (~0.2 T_A). 5/5 seeds.
- C (d = 0.25): |psi_C| freezes at 0.67, wall density 0.17-0.25 after ~300 sweeps, unchanged to 3000 sweeps,
  at T = 0.12 (~0.4 T_C) 3/3 seeds; at T = 0.2 (~0.6 T_C) 1 of 2 seeds frozen, the other removed its walls early in
  the quench (before local C order was complete).

## Island-model dependence (hybrid_coupling.point_coupling, model_check.py, coarsening_P1.py, selection_T_P1.py)
P1 = point dipoles at island centres, moment ~ length; P2 = point dipoles, equal moments (as in Kwon 2022, Makarova 2021).
- Manifold validity (z3 levels 2/4/2 and identical 39952-state manifold on 4x2):
  needle d ~ 0.07-0.48; P1 d <= 0.22 (z3 level crossing at d ~ 0.23); P2 d <= 0.155 (crossing at ~0.16).
  Point models lose the Shakti end because the long island's dipole sits far from the z3 vertex.
- Selection (4x2 exact, rc = 64a): A on the Cairo side in all models; A -> C at d = 0.163-0.165 (P1) vs 0.153 (needle);
  P2 loses the manifold before any C appears.
- P1 ordering temperatures (8x8): T_A(0.137) = 0.52, T_C(0.20) = 0.165 (P1 energy units ~10x smaller than needle).
- P1 coarsening at ~0.3 T_order: A 3/3 single domain; C 2/3 frozen with walls (|psi_C| 0.33, 0.67), 1/3 ordered.
  At 0.29 T_A one A run stalled; at 0.6 T_C 1/3 C runs frozen.

## Single-island-flip kinetics (kinetic.py, cool_single.py, short_range.py), needle model, 8x8
- Single-flip cost from the ground state: A (Cairo) min 0.46 (a few cheap partner flips), 10th pct 90;
  C (Shakti) min 86. P1: A min 0.50, C min 9.0.
- From multi-domain manifold states, single flips at T = 0.4-2.0 change nothing in 20000 sweeps (A as well as C).
- Slow geometric cooling 300 -> 0.05 (2000 and 20000 sweeps/decade): neither A nor C order forms (|psi| 0.03-0.38).
  Shakti reaches the manifold more often (2/4 exactly, excess <= 0.11) than Cairo (0/4, excess 0.06-0.43 per spin).
- => A/C selection is an equilibrium property; the "mobile A walls vs frozen C walls" contrast holds only for
  manifold (collective) moves and is NOT a prediction for single-flip annealing.
- Short-range correlations of frozen states (pairs in same/neighbouring cells):
  Cairo: g_A 0.055 (liquid 0.023 +- 0.039), g_C 0.089 (liquid 0.094) -> essentially dimer liquid.
  Shakti: g_C 0.308 (0.20-0.47 in 4/4 states; liquid 0.075 +- 0.066) -> clear short-range sublattice (C) correlations.

## Vertex-gap parameter g (needle.GAP; gap_scan.py -> gap_scan.json, plot_gap_map.py -> gap_map.png)
Islands shortened by g at each end (moment ~ remaining length); g = 0 is the needle model used elsewhere.
- A|C transition d_c(g): 0.153 (g=0), 0.155 (0.05), 0.157 (0.1), 0.161 (0.2), 0.172 (0.3). P1 (0.164) lies inside.
- Manifold lower validity edge rises with g (z3 level crossing): ~0.07 (g=0), 0.12 (0.05), between 0.12 and 0.137 (g>=0.1).
  Equilateral Cairo is valid for all g scanned. At g = 0.3 C narrows (A and C nearly degenerate at Shakti, A at d=0.30).
- Defect cost / selection energy ratio falls from ~400-600 (g=0) to ~40-50 (g=0.3).
- Single-flip slow cooling at g = 0.2 (30 -> 0.003, 20000 sweeps/decade): still no A or C order (|psi| 0.06-0.19).
  Cairo freezes with excess unhappy z3 (unhappy fraction 0.60-0.63 > 1/2). Short-range g_C in Shakti frozen states
  0.15 (0.07-0.24; liquid 0.08 +- 0.06) -- weaker than at g = 0 (0.31); statistics still small (3 states).

## Saccone et al. PRMaterials 3, 104402 (2019) samples (saccone_geometry.py, saccone_selection.py, saccone_dynamics.py)
Geometry from Fig. 1(a): b = 344 nm = z3-z3 partner edge, a = 472 nm = z3-z4 edge -> d = 0.1053.
c = distance between collinear islands across a z4 vertex (z3-z4 islands shifted along their edge);
islands 300 x 100 x 3 nm stadia, Ms = 790 kA/m, sub-dipole grids (16 x 3). Energies in K.
- Pair couplings vs their MuMax3 Fig. 3 (read by eye): z4 pairs within 15-35%; z3 pairs ~2x high at small c,
  matched at c = 600; J1 = J3 equalisation at c = 600 reproduced.
- Vertex levels: c = 376-500 z3 gap 1200-1800 K, z4 gap 2600-8900 K, manifold valid, ground state A
  (E_A - E_C = -36, -51, -63 K per island); c = 600 z3 A-B gap 21 K -> manifold broken (exp. A:B = 40:60 ~ degeneracy 1:2).
- T_A (manifold MC, 8x8): 322 K (c=376), 432 K (450), 562 K (500).
- Single-flip cooling frozen at 150 K: c=376 no A, excess B; c=450 dimer liquid (B = 50%), no A; c=500 A forms.
- Effective temperature from their z4 populations (Fig. 4d, by eye) with our z4 levels:
  ~1900-2300 K (376), ~1330 K (450), ~1360 K (500) -> T_A / T_eff = 0.16, 0.32, 0.41.
  => experiments froze well above T_A; c ~ 500 nm is closest to showing A order.

## A-C competition at finite T (pt_joint.py), 8x8, needle g = 0, rc = 64a
Joint (psi_A, psi_C) samples from parallel tempering.
- d = 0.1527 (T=0 boundary): C dominates for 0.1 < T < 0.4 (>90% C-like samples) -> entropy favours C.
  At T <= 0.06 a few A-like samples appear. Mixed samples (both |psi| > 0.3) <= 3%.
- d = 0.145: equilibrium re-entrance A (T <= 0.085) -> C-like (T >= 0.2) -> disorder.
  At T = 0.117 the energy histogram is bimodal (A branch -39.840, C/mixed branch -39.832 per spin):
  first-order A|C transition at T ~ 0.12 with coexistence; mixed samples sit on the C branch (no mixed phase).
- => the A|C line tilts toward smaller d with increasing T (bicritical-type direct competition, no mixed phase).
  Finite-size scaling of the double-peak not yet done.

## Finite-size scaling of the low-T transition at d = 0.145 (fss_pt.py, fss_analyze.py, fss_plot.py, run_all.sh)
Needle g = 0, rc = 64a, manifold MC. n = 8, 12, 16, 20, 24 (N = 10 n^2 up to 5760).
- Code: `fss_mc.py` = `mc_keep`, the manifold_mc2 move set with h = J s kept between calls and Jl used only as CSR
  (bit-identical to `manifold_mc2.mc` for the same seed, fss/test_fast.py; 10-100x faster), vectorised
  `hybrid_coupling_fast` (identical to 1e-16), couplings cached in `couplings/*.npz`.
  `fss_pt.py tune` iterates the ladder to uniform -ln(acc); `fss_pt.py run` = PT, 1 sweep per round, all replicas
  start from A (or all from C), E every round, signed (psi_A, psi_C) every 20 rounds, replica labels / round trips,
  npz checkpoint every 5 min and on SIGTERM (resume by re-running). `run_all.sh` runs the whole campaign.
- Runs (each for A-start and C-start): n=8 1e6 rounds (M=20, T 0.08-0.40), 12: 6e5 (22), 16: 5e5 (26),
  20: 3e5 (24, T 0.08-0.25), 24: 2e5 (28, 0.08-0.25). Min swap acceptance 0.70 / 0.56 / 0.40 / 0.09 / 0.12 (one pair,
  exactly at T*); round trips 1400 / 180 / 47 / 9-18 / 5-9. tau_int(E) at T*: 110 / 620 / 2000-3000 / 3500-4300 /
  2900-4000 sweeps -> samples/tau >= 7000 / 760 / 130 / 56 / 40. Energy drift <= 1.2e-7 per run.
- Analysis: binned WHAM over all slots of both runs (bin 0.02 D/a^3), jackknife (10 time blocks); errors below also
  include half the A-start/C-start difference. P(E) at T* is smoothed with a Gaussian of width 1.0 D/a^3 in total
  energy: the A branch is a comb of discrete levels (spacing 0.5 D/a^3: ground state + single excitations), so
  unsmoothed peak heights depend on the bin width (the earlier n=8 "Delta F = 2.6" was this artefact).

| n | T*(n) equal height | Delta F = ln(Pmax/Pmin) | e_C - e_A at T* | C_max / N | 2/3 - V_min |
|---|---|---|---|---|---|
| 8  | 0.11619(7)  | 1.022(5)  | 0.008579 | 1.33e-3 | 1.51e-8 |
| 12 | 0.11543(8)  | 1.93(8)   | 0.008597 | 1.21e-3 | 1.35e-8 |
| 16 | 0.11504(9)  | 2.31(9)   | 0.008516 | 1.15e-3 | 1.28e-8 |
| 20 | 0.1152(3)   | 2.03(14)  | 0.008385 | 1.04e-3 | 1.16e-8 |
| 24 | 0.1151(3)   | 2.29(19)  | 0.008318 | 1.01e-3 | 1.12e-8 |

(equal-weight T* agrees within 2e-4; doubling the smoothing width lowers Delta F by 0.2-0.6 but keeps the trend.)
- First-order signatures that hold up to N = 5760: P(E) is bimodal at T* for every n with the minimum 2 or more units
  of ln P below the peaks; the latent heat stays finite, 0.0083-0.0086 per spin (extrapolations 0.00835(6) in 1/N,
  0.0082(1) in 1/n); C_max grows as N^0.89(2); 2/3 - V_min stays at 1.1-1.5e-8, close to the two-delta-peak value
  (e_C^2 - e_A^2)^2 / (12 e_A^2 e_C^2) = 1.45-1.55e-8, instead of decaying like 1/N.
- What does NOT follow the first-order form: Delta F stops growing above n = 16 (1.0, 1.9, 2.3, 2.0, 2.3); a linear fit
  over n >= 12 gives slope 0.028(28) per cell (chi2 8.2 / 2 dof) -> interface tension
  sigma = slope T*/2 = 0.0016(16) D/a^3 per cell length = 0.0006(6) per a, i.e. consistent with zero,
  sigma < ~0.005 per cell length (2 sigma bound). Consistently, C_max/N lies BELOW the two-delta-peak limit
  Delta e^2 / (4 T*^2) = 1.32e-3 and keeps falling (a first-order transition approaches it from above): the states between
  the peaks are not exponentially suppressed at these sizes. At n = 20, 24 the barrier region is a flat plateau with a
  weak middle hump (slab/interface states: along the barrier rms psi_A falls smoothly from 1 to 0.43 while psi_C stays
  <= 0.1, fss_middle.py).
- T*(n) is flat within errors for n >= 16 (0.1150-0.1152); a 1/n^2 fit over n >= 12 gives T*(inf) = 0.1147(2) (chi2 2.0/2).
  Adopted: T*(inf) = 0.1149 +- 0.0003.
- The high-energy side is NOT the C phase (fss_psi_d0.1450.png, fss_local.py). At T*, the low-E branch is pure A
  (rms psi_A 0.93-0.98, psi_C < 0.06). In the high-E peak rms psi_C = 0.37, 0.30, 0.25, 0.22, 0.19 for n = 8..24
  (~ n^-0.8, heading for zero) and only 0-12% of its samples are C-like, while rms psi_A = 0.50 -> 0.43 decays only
  slowly (~ n^-0.13). Block-local (4x4 cells) order in the n = 24 snapshots just above T* is |psi_A| ~ 0.45-0.55 and
  |psi_C| ~ 0.35-0.45 at the same time, with the global order switching between A-like and C-like from sample to sample:
  a strongly fluctuating state with competing local A and C correlations. Global C order (rms psi_C 0.54-0.57,
  n-independent for n >= 16 at T = 0.3; still drifting down slowly at T = 0.2) is established only at T >~ 0.2.
- Verdict for d = 0.145: the transition at T* = 0.1149(3) is discontinuous in energy (latent heat ~0.0083 D/a^3 per spin,
  entropy jump Delta e / T* ~ 0.072 per spin), and at the reachable sizes (L <= 24 cells) it is a
  first-order transition with an anomalously small interface tension, sigma ~ 0.002 +- 0.002 per cell length. A clean
  2 sigma L growth of Delta F is not seen; this is compatible with (i) a weak first-order transition whose
  interface cost is below our resolution, or (ii) a pseudo-first-order crossover (e.g. near a critical end point) that would
  lose its latent heat at larger L. The drift of e_A (up by 2.4e-4 from n=12 to 24) makes the latent heat decrease slowly,
  so (ii) cannot be excluded. It is an A -> (fluctuating A/C-correlated state) transition, not a direct A|C coexistence;
  the earlier "A|C first-order at T ~ 0.12" (8x8) should be read that way.
- Limits: n = 20, 24 have only 5-18 round trips and 40-70 tau per run, and A-start/C-start T* differ by 4-5e-4 there
  (errors above include this). The swap acceptance at the T* pair (0.09-0.21) is the PT bottleneck; a multicanonical
  (or Wang-Landau) run in E seeded by the WHAM ln g(E) would be the next step to pin down Delta F(n) at n >= 20.
  The psi estimators are global and blind to walls; block-local values come from single final snapshots only.
- Files: fss_d0.1450_n{n}.json (per-T samples summary, tau, swap, round trips, f_up(T), A-vs-C z, reweighted C, V, e,
  P(E) at T*, psi at T*), fss_summary_d0.1450.json (fits), fss_d0.1450.png (P(E) at T*(n), Delta F(n), C_max(N),
  T*(n) vs 1/n^2, latent heat, Binder), fss_diag_d0.1450.png (PT diagnostics), fss_psi_d0.1450.png, fss_middle_d0.1450.json,
  fss_local_d0.1450_n24.json; raw PT data fss/pt_d0.1450_n{n}_{A,C}_s1.npz (E stored as float32 offset by meta.Eref).

## (d, T) plane near the A boundary at n = 12 (run_dscan.sh, fss_phase_dT.py -> fss_dT_phase.png, fss_dscan_n12.json)
PT at n = 12, 4e5 rounds per start, ladders tuned per d (fss/ladder_d*_n12.json). Only a single size, so these are
positions, not FSS verdicts.
- d = 0.140: A up to T ~ 0.15 and a single specific-heat peak at T = 0.153 (C/N = 0.30); P(E) not bimodal at n = 12;
  |psi_C| <= 0.1 up to T = 0.2 (no C region). A- and C-start agree (max |z| 2.0, 1700 round trips).
- d = 0.1475: A below T ~ 0.085, C-like above (|psi_C| 0.3 at 0.094 -> 0.69 at 0.19). A-start run: bimodal P(E),
  T* = 0.086, Delta F = 0.30 at n = 12. The C-start run is NOT equilibrated below T = 0.088 (its lowest slots hold a
  wall state 1.7e-3 per spin above A, swap acceptance 0.01 across the gap 0.040 -> 0.0765 of the tuned ladder) -> T* tentative.
- d = 0.150: A never appears. Below T ~ 0.075 both starts reach a mixed A/C superstructure (diagonal bands; 2x2-cell
  block |psi_A| 0.57 and |psi_C| 0.34 simultaneously, global psi 0-0.17) whose energy is 4.4e-4 per spin BELOW the A
  state (fss_lowT_d0.1500_n12.png). C (|psi_C| 0.56-0.82) above T ~ 0.083, C_max at T = 0.0795. Glassy: tau_E ~ 2e4,
  A/C starts differ (max |z| 52).
- d = 0.1527: C / C' (walls; global |psi_C| = 2/3 in the lowest states) at all T, energy 1.8e-4 below A; no A.
- => The A region ends between d = 0.1475 and 0.150 at n = 12, i.e. below the 8x8 T = 0 estimate d_c = 0.153: the
  n = 12 torus admits a mixed A/C superstructure that 8x8 (and the 4x2-period searches) cannot host. The low-T
  boundary line T*(d) falls steeply: ~0.15 (0.140, continuous-looking at n = 12) -> 0.115 (0.145, first order)
  -> 0.086 (0.1475). The d = 0.150 superstructure must be checked on other commensurate sizes (n = 6, 18, 24) before
  it is called a phase.
