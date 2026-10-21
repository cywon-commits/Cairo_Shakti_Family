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
  [SUPERSEDED on 12x4 / 12x12 tori: the 4x2 cell cannot hold the 3-cell ladder crystal. A ladder crystal L3' is the ground
  state for 0.1495 <~ d <~ 0.157 -- see "L3' replaces L3" and "Columnar-family walls" below and paper Sec. IV B.]
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

## A, C, W as domino tilings of the pentagon chessboard (chessboard.py -> chessboard_ACW.png)
Chessboard coordinates by BFS with the CCW rotation system of the pentagon (z3-z4 adjacency) graph; no conflicts.
- A and C are BOTH basketweave tilings (2x2 blocks of two parallel dominoes, H and V blocks alternating).
- They differ in where the block grid sits: A blocks have origin parity x+y odd ((0,1) or (1,0) over all A variants),
  C blocks x+y even ((0,0) or (1,1)). The 4-fold offset degeneracy of basketweave on a uniform square lattice is split
  by the Cairo sublattice structure into two pairs = A and C. A->C is a one-unit shift of the basketweave.
- Colouring: C blocks are monochrome (H blocks CCW, V blocks CW); A blocks contain one CW and one CCW domino;
  all closed pentagons of A lie on one chessboard colour.
- W = stripes of the two C variants ((0,0) and (1,1) blocks) separated by walls.

## Aztec diamond: block-parity order parameter Phi (aztec_phi.py, aztec_weighted.py, plot_aztec_weighted.py)
Uniform measure (plaquette-flip MCMC, n = 24, 40, 60):
- frozen corners: no flippable plaquettes (staggered bricks) -> Phi undefined (unnormalised density 0).
- liquid: Phi -> 0 as ~1/n (centre Phi*n ~ 1.1-1.4, sign set by the boundary colouring). Symmetry argument:
  odd translation (colour swap, slope -> -slope) + 180-degree rotation about a vertex (slope -> -slope) => Phi = 0
  for every slope in infinite-volume Gibbs measures. Use class-averaged estimator (window sums are biased).
Weighted measure P ~ exp(K sum_p F_p (-1)^{x+y}) (minimal parity-breaking model; K ~ (E_A - E_C)/T), n = 32:
- equilibration: uniform first (400n sweeps), K annealed in 40 steps, 400n sweeps at K; checked by 90-degree symmetry.
- |K| = 3: central square basketweave crystal (class density 1.00) directly bordered by frozen regions.
  K = +3 -> C crystal, class (0,0) (boundary selects the C variant deterministically).
  K = -3 -> A crystal, class (0,1) in 2/2 seeds, slightly rectangular (A variants are rotation-related: spontaneous choice).
- K = +1: biased liquid (Phi ~ +0.3), no crystal; crystallisation threshold between K ~ 1.5 and 3.
- Earlier K = +-1.5 runs with a short schedule were not equilibrated (horizontal-band artefact) and were discarded.

## z3-coupling uncertainty (saccone_z3scale.py): scale only pairs sharing a z3 vertex (J1, J3) by lam
- lam = 0.5 halves the z3 gap (c=376: 1197 -> 598 K) but leaves the manifold identical and E_A - E_C, manifold spread
  unchanged to 0.01 K (c=376: -36.34, 90.95 K/island). Vertex-local energy is constant on the manifold, so selection
  energies and T_A (manifold MC) are exactly independent of vertex-pair couplings as long as the level order holds.
- Single-flip cooling to 150 K with lam = 0.5: c=376 still no A (|psi_A| 0.03-0.04, B 58-60%); c=500 still A (0.91-0.97).
=> the ~2x overestimate of J1, J3 by our dipolar model does not affect T_A or the kinetic conclusions.

## Boundary selection in small Aztec diamonds (aztec_small.py, aztec_dstar.py), n = 6-9
Model: E = -sum_p F_p (1 + delta s_p) + r sum_<pq> F_p F_q (r = 1 > 1/2 makes basketweave the ground state; bulk A|C
transition exactly at delta = 0). Parallel tempering in beta (0.1-8).
- delta = 0: preferred class alternates with n: n = 6, 8 -> odd class (12, 20 blocks); n = 7, 9 -> even class = class of
  the central plaquette (16, 25 blocks). Maximum = floor((n+1)^2/4). Low-T <Phi> = -0.96, +0.98, -0.97, +0.96.
- The disfavoured class cannot form a pure crystal in the diamond: forcing it gives mixed states.
- Ground-state switching vs delta (lower envelope over all states found; competitors are upper bounds, so true
  switching may be earlier): pure preferred crystal survives to |delta| = 0.14 (n=6,7), 0.11 (n=8,9);
  Phi_GS changes sign at |delta| = 0.25, 0.20, 0.17, 0.17 -> delta*(n) ~ 1.4/n with sign alternating with n parity.
- Cairo reading: if the patch is centred on a z4 vertex (central plaquette = C class), odd n favours C and even n favours A;
  centred on a partner bond, the reverse.

## Height-function view (height.py): A and C are height plateaus of a Z4 clock
Thurston height (black-left rule: +1 along a domino boundary, -3 across a domino). Heights at a vertex are fixed mod 4
by position, so plateaus are compared mod 4.
- Basketweave is flat: block-centre heights take two values differing by 4 (H/V corrugation).
- Bulk: the four block origins sit on the four plateaus mod 4: (0,0) -> 0 [C], (1,0) -> 1 [A], (1,1) -> 2 [C], (0,1) -> 3 [A];
  H/V swap leaves the plateau unchanged. => psi_C = cos(pi h/2), psi_A = sin(pi h/2), Phi = cos(pi h) (= psi_C^2 - psi_A^2).
- Aztec (n = 6-9 ground states): facet plateau relative to the centre vertex = +-(n+1) mod 4 (sign from the 90-degree
  rotation, h -> -h). Odd n -> even plateau (C), variant fixed: n = 3 mod 4 -> (0,0), n = 1 mod 4 -> (1,1).
  Even n -> plateau +-1 (A); the two A variants are exactly degenerate (rotated ground state has the same energy).
- Coarse-grained reading: locking potential ~ -u cos(2 pi h) (basketweave) - delta cos(pi h) (A vs C; d_c where the
  coefficient changes sign); cos(pi h/2) (variant selection) is forbidden in the bulk and supplied only by boundaries.

## Anisotropic family Cairo(d1, d2) (cairo2.py, cairo2_lambda.py, cairo2_scan.py), 4x2 exact, needle g = 0, rc = 64a
C1-C2 pairs spaced by d1, C3-C4 pairs by d2; d_bar = (d1+d2)/2, eps = d1 - d2. Same topology and same dimer manifold.
- d1 = d2: A1 and A2 degenerate to 1e-13 (check).
- d1 != d2: the vertex energy is NOT constant on the manifold. Unhappy z3 can sit on either pair set; the two sets now
  have different z3 gaps. Exact identity on the manifold: E_vertex = E_0 - lambda_v eps psi_A (residual 3e-13), because
  psi_A counts the imbalance of unhappy vertices (dimers) between the two sets. lambda_v ~ 12.3 per spin per unit eps
  at d_bar = 0.25. Pure-A sectors (psi_A = +-1): 512 states each of 39952.
- Symmetry: the diagonal mirror (x,y) -> (1/2-y, 1/2-x) survives and swaps z4 A <-> B, so C and C' stay degenerate;
  the 90-degree rotation (A1 <-> A2) is broken. Q_C = 0 by symmetry.
- E_C - min E_A is linear in eps (total slope ~12 at d_bar = 0.25, ~15 at 0.20, dominated by the vertex term):
  C is destroyed at eps* ~ 0.008 (d_bar = 0.25, ~3% relative) and ~0.0046 (d_bar = 0.20, ~2%).
  => A|C boundary in the (d1, d2) plane is V-shaped around the diagonal: eps*(d_bar) = [E_A - E_C](d_bar) / lambda(d_bar).
- Caveat: needle g = 0 overstates vertex energies; with vertex gaps (g > 0) lambda_v shrinks faster than the dipolar
  selection energy, so eps* should be several times larger for realistic islands (not computed).

## Symmetry-breaking deformations: twist and the columnar state (cairo3.py, coef_fixed.py, mixed_states.py, plot_deform_phase.py)
Twist theta: each z3 pair rotated about its centre by theta (same sense for both sets): keeps the 90-degree rotation,
breaks the mirrors (p4g -> p4). Checks at d_bar = 0.20: A1 = A2 to 1e-13; C+ - C- = -0.029 at theta = 0.01;
exact identity on the manifold E_vertex = E_0 - lambda_C theta psi_C (residual 2e-13), no psi_A term.
Fixed-state linear responses (needle g = 0, rc = 64a, 4x2; per spin): A responds only to eps (|Q_eps| 18.6 -> 12.0
from d = 0.137 to 0.25), C only to theta (|Q_theta| 1.39, 1.04, 0.32, 1.45, 6.0 at d = 0.137, 0.153, 0.17, 0.20, 0.25;
vertex lambda_C changes sign near d ~ 0.17). Beyond-vertex parts are ~10% (an earlier class-minimum estimate that gave
lambda_A ~ 0 at Cairo was an artefact: the |psi_C| = 1 class minimum was a state with |psi_A| = 1 as well).
Columnar state M (|psi_A| = |psi_C| = 1, 8 states in the 4x2 manifold; all rows of parallel dominoes stacked): responds
to BOTH (|Q_eps| ~ A's, |Q_theta| ~ C's). Zero-field: M - A = +0.014 to +0.022; M - C = -0.007 (0.137), +0.017 (0.153),
+0.114 (0.25). Correction: the columnar state couples LINEARLY to both deformations (count-defined psi), not via eps*theta.
Linear phase estimates (deform_phase_estimate.png): anisotropy alone -> A; twist alone -> C (and M on the Cairo side,
theta ~ 0.01 rad at d = 0.137); both near d_c (d = 0.153): M needs theta > ~0.02 rad and eps > ~0.001 (extrapolated beyond
the tested theta range). Not included: W, the 3-cell ladder crystal, longer periods; g > 0 not done.

## Candidate crystals and their stabilization (candidates.py, cand_response.py, plot_candidates.py)
Candidates: A, C, columnar M (4x2), wall crystal W (4x4, from the d = 0.20 PT state), ladder L3 (12x2, from the
d = 0.150 FSS low-T state). All are manifold states. IMPORTANT: energies per spin carry a torus-size-dependent,
state-independent offset (self-image terms; 4x2 vs 8x8 differ by 0.0067), so all comparisons use one common torus, 12x4.
Count-based order parameters: |psi_A|, |psi_C| = A (1, 0), C (0, 1), M (1, 1), W (0, 0), L3 (0, 0).
Linear responses (|dE/dk| per spin; vertex identity explains eps and theta): eps -> A, M only; theta -> C, M only;
axial strain eta (x(1+eta/2), y(1-eta/2)) -> M (9.3-14.8) and L3 (3.1-5.0) only; W responds to nothing.
Zero-field ground state (12x4): A (d = 0.137, 0.145), L3 (0.150, 0.153), C ~ W (0.160-0.25; W - C within 4e-5).
Check: L3 - A = -0.00044 at d = 0.150, identical to the FSS 12x12 PT value.
Smallest single knob making each candidate the ground state (linear extrapolation): M via eta = 0.0015-0.008 at all d
(or theta ~ 0.01 rad on the Cairo side); L3 zero-field window d ~ 0.149-0.156, widened by eta ~ 0.0014-0.0019;
C via theta near d_c; A via eps; W: no knob (stabilized only isotropically, degenerate with C).
Figure: candidates_phase_estimate.png. Caveats: needle g = 0; linear ranges tested only to eps 0.002, theta 0.01,
eta 0.004; candidate list is not exhaustive.

## Verification of the candidate estimates (verify_exact.py, vmc_J.py, vmc_run.py, vmc_local*.py)
Sizes: candidate comparison on 12x4 (smallest common torus of all candidates); orientation handled by symmetry
(min over +-deformation). Unbiased search on 12x12 (smallest torus hosting every candidate in every orientation).
- Exact energies at finite deformations (eps 0.002-0.01, theta 0.01-0.04, eta 0.002-0.008; d = 0.137-0.25): the winner
  agrees with the linear estimate at all 63 points. Non-linear corrections grow quadratically (max 0.056 per spin at
  eps = 0.01) but shift all candidates similarly. Weak calls: winner margins ~1e-4 (W vs C at d = 0.20; C vs A at 0.25).
- Manifold MC on deformed lattices must use the UNDEFORMED vertex couplings for the manifold constraint (the deformed
  ones are not degenerate on the manifold, so every psi-changing move would be rejected).
- 12x12 annealing (6 seeds each): P1 (0.153, eta 0.004) and P3 (0.137, theta 0.02) reach exactly M in every run.
  P2 (0.150, eta 0.002): runs stop at M (+0.0031 above L3); starting from the favourable L3 it stays (0.0000); no state
  below L3 from any start. P4 (0.25, eps 0.01): runs stop in the A sector with defects (+0.0024); A itself is stable.
  => no state below the predicted ground state was found at any test point (candidates_phase_verified.png).

## Island-shape (vertex gap g) dependence of the deformation thresholds: formula and test
(vertex_formula.py, gap_scaling.py, gap_scaling_verify.py, gap_scaling_general.py; 4x2, rc = 64a)
Formula: k*_X(d, g) = Delta E_X(d, g) / lambda^(v)_{k,X}(d, g), numerator = zero-field (dipolar) selection energy,
denominator = vertex-only linear response. For anisotropy, lambda_A = (Delta_set1 - Delta_set2)/(10 eps) from single
z3 vertices of the deformed lattice (exact match with the manifold fit, 19.1 at d = 0.153). NOTE: the isotropic
derivative Delta'(d)/10 is wrong (the needle moment normalisation by the mean edge length adds an overall-scale term).
Results (actual crossing / predicted): eps* for C at g = 0.1, 0.2 and d = 0.20, 0.25: 1.08, 1.02, 0.97, 0.93.
theta* for C at d = 0.153: g = 0.1: 1.10, g = 0.2: 1.01. eta* for columnar M at d = 0.20: g = 0: 1.00, g = 0.1: 0.92,
g = 0.2: 0.74 (eta* = 8%: non-linear; the formula overestimates, i.e. is conservative).
Enlargement relative to g = 0: eps (destroying C): 1.4-2.9x (lambda_A drops 4-9x but Delta E_AC also drops 2-5x);
eta (stabilising M): ~7-8x (M's vertex eta-response collapses 12-30x, Delta E only 1.6-2.9x).

## Realistic islands across the family (real_islands.py, real_islands_scan.py, real_thresholds.py, real_verify.py)
Stadium islands 100 nm wide, 3 nm thick, Ms = 790 kA/m; z3-z4 edge 472 nm (undeformed); fixed vertex gaps 86 nm (z3-z4
islands) and 22 nm (partner islands) -> Saccone's islands at d = 0.105; long partner islands toward Shakti. Energies in K.
Sub-dipole counts fixed per island type from the undeformed lattice (rounding them per deformed edge made derivatives jump).
- The width matters: a zero-width needle with the same gaps INVERTS the z3 levels ([4,2,2]); the realistic model gives the
  normal [2,4,2] with z3 gaps 1670 K (d = 0.105) to 4160 K (Shakti).
- Zero-field ground state among the candidates (12x4): A for d <= 0.17; W for d >= 0.18 (C 2-6 K, A 2-8 K above W).
  Candidates are the needle-model crystals; other wall spacings could do better for real islands (not searched).
- Thresholds (formula, verified at four points: winner flips between 0.7x and 1.3x of the prediction):
  M (columnar) via twist 0.7-1.0 deg on the Cairo side (d = 0.105-0.145); via axial strain 3.7-6.8% everywhere.
  L3 via axial strain 3.1-4.4% near d = 0.16-0.18 (6-13% elsewhere). C via twist 0.14-0.36 deg on the Shakti side,
  0.6-1.0 deg near d = 0.153-0.16. A via anisotropy 1-3% of d (z3 shift 1.5-6 nm) on the Shakti side.

## Ideal vs realistic islands: phase diagrams (plot_model_compare.py -> model_compare_phase.png)
Same candidates, same first-order formula. Zero field: ideal A -> L3 (0.149-0.156) -> C ~ W; realistic A (up to ~0.175)
-> W (C never wins; W lower by 2-6 K/island; L3 window gone). Anisotropy: realistic A takes over W at eps ~0.002-0.007,
so A covers almost the whole plane. Twist: realistic M region much larger on the Cairo side (theta > 0.7-1 deg), C needs
0.15-0.35 deg to beat W on the Shakti side, A survives up to ~1 deg near d ~ 0.15. Axial strain: ideal 0.15-0.8% -> M;
realistic 3.7-6.8% (L3 pocket at d ~ 0.16-0.175, 3-6%). Twist and anisotropy remain strong knobs; axial strain becomes weak.

## Analytic criterion for the wall crystal W: moment ratio mu (wall_decomp.py, wall_lattice_sums.py, plot_dmu_phase*.py,
## wall_mu_verify.py, dmu_verify.py, dmu_verify_width.py)
- E_W - E_C comes entirely from distant island pairs (r >= 2.7a; near pairs cancel identically), as a delicate sum of
  large alternating terms. Hence the point-dipole limit is accurate and the island shape enters only through the moment
  ratio mu = m_long/m_short:  E_X - E_C = S_SS(d) + mu S_LS(d) + mu^2 S_LL(d)  [units mu0 m_S^2/(4 pi a^3) ~ 300 K].
  W: S_LL < 0 (long-long pairs favour walls), S_LS > 0. A: S_LS = 0 exactly, S_LL < 0. M: only S_SS (= A's), so M never
  beats A at zero deformation. Ideal needle has mu = 2; realistic fixed-gap islands mu = 1.45 (d = 0.137) -> 3.1 (Shakti).
- (d, mu) diagram (dmu_phase_verified.png): W for mu_c(d) < mu < mu_A(d); mu_c = 1.21 (0.153), 1.38 (0.175), 1.57 (0.20),
  1.77 (0.25); mu_A ~ 1.9-2.1 (0.175), 2.8 (0.20), 3.3 (0.25). Narrow C band below mu_c, small-mu W (mu < 0.9, C within
  0.1-0.4 K), L3 pocket near d = 0.15-0.165 at small mu.
- Realistic checks: tuning mu via the partner gap, E_W - E_C matches the point formula within ~0.1 K and changes sign at
  mu_c (d = 0.20, 0.25). Upper W|A boundary (mu raised via wider long islands): realistic switch between 1.9-2.3 (0.175),
  2.6-3.0 (0.20), 3.2-3.6 (0.25), consistent with the analytic values (near-field shifts A up by ~2 K at d = 0.175).
  With 100 nm width the fixed-gap realistic curve runs just below mu_A near Shakti (A only ~5 K above W there).
- L3 pocket check (realistic islands, partner gap tuned): 12 of 14 points agree with the analytic diagram. L3 confirmed at
  (0.150, 0.6), (0.155, 0.6), (0.155, 0.9), (0.160, 1.2); L3 -> A above mu ~ 0.9-1.2. The two misses lie on the pocket's
  right edge with margins 0.04-0.9 K (C wins at (0.160, 0.9) and (0.165, 1.3); C, W, L3 within 0.2 K at (0.160, 0.9)).
  Mechanism: near d_c the short-island terms of A and C cancel, A's gain is ~mu^2, L3 has a linear S_LS < 0 term,
  so weak long islands (mu ~ 1, i.e. long islands about as strong as short ones) favour L3.

## Wall tension from controlled stripe states (stripes.py, stripe_sums.py, stripe_real.py)
W = lines (i - j) [X | w1 | X' | w3]; C = [X | Y] alternating, C' = [X' | Y']. Stripes X Y ... X (2k+1) w1 X' Y' ... X' (2k+1) w3,
period P = 4k+4, are manifold states (k = 0 reproduces W). sigma_k = (e_stripe - e_C) * 10 * P/2 per wall cell, same torus.
- Walls are sharp and nearly non-interacting: sigma_k for k = 0, 1, 2 agree within ~1% (point sums) and ~2% (realistic:
  d = 0.25: -130.0, -127.6, -130.2 K; d = 0.20: -55.7, -54.3, -55.7 K). => when sigma < 0 the densest wall crystal (W) wins.
- sigma(mu, d) = a + b mu + c mu^2 with a, c < 0, b > 0; sigma = 0 at mu ~ 0.9 and mu_c = 1.20 (0.153), 1.36 (0.17), 1.54 (0.20),
  1.81 (0.25) = the W|C boundary. Point formula vs realistic: -125 vs -130 K (0.25), -58 vs -56 K (0.20).
- Height picture: a C|C' wall (dh = 2) passes through the odd plateau (A). At mu = 2, sigma tracks E_A - E_C nearly 1:1
  (both change by ~+0.12-0.13 from d = 0.153 to 0.25): walls get cheaper near d_c. Sharp-wall limit: sigma ~ sigma_dip(mu, d)
  + w (E_A - E_C), linear rather than the continuum square-root form.

## L3 reinterpreted: wall crystal of the columnar family (stripes_axial.py, axial_sums.py)
- Column-pair decomposition: L3 = 2-column units separated by alternating direct (alpha) and split (beta) interfaces, but the
  units are NOT the C crystal: they belong to higher-energy pure-C-sector states (+0.003 to +0.06 per spin above C), so an
  axial "C|C' wall" tension grows with domain width (artefact of that reading). Discarded.
- Evidence for "walls between columnar variants (h = 0.5 and 2.5)": local block plateaus {2,3} and {0,1} (half-integer means),
  chessboard bands of vertical-domino ladders, response to axial strain only.
- Lattice sums: L3 - M has all three coefficients negative for d >= 0.153 (walls always favourable in the columnar family);
  at d = 0.137 negative for mu > 1.27. So uniform M is never the zero-field ground state; L3 replaces it.
- Unified zero-field picture: E_GS = min(E_A, min(E_C, E_W), E_L3); W wins when sigma_C(mu, d) < 0 (mu > mu_c), L3 near d_c at small mu.
- Response table explained by symmetry: a wall crystal keeps only what its two domain variants share. W (C, C') -> nothing;
  L3 (M at h = 0.5 and 2.5: psi_A and psi_C flip, ladder axis shared) -> axial strain only.
- Not done: direct construction of variable-spacing columnar stripes (needs rotated columnar pieces) to test spacing independence.

## Columnar-family walls verified by direct construction (dimer_inverse.py, chess_torus.py, columnar_stripes.py)
- Inverse map dimers -> spins: given each vertex's status, the allowed configurations form one time-reversal pair, so one spin
  fixes all; round trip exact for A, C, M, W, L3. Global chessboard coordinates mod 2n on the n x n torus (even n only:
  odd n is a different winding sector).
- L3 on the chessboard = [one layer of vertical dominoes (2 rows)] + [one horizontal row], period 3 rows, with irregular
  horizontal-row phases. Generalised: m stacked vertical layers + one horizontal row (period 2m+1), m = 1, 2, 3.
- Wall energy per horizontal-row cell relative to uniform columnar (point sums): all of a, b, c negative at d = 0.153 and 0.20
  (walls favourable for every mu); m = 2 and 3 equal (-0.0979, -0.0974 at mu = 1), m = 1 6-10% lower (weak attraction).
  => the densest columnar wall crystal (m = 1) wins: L3 is the columnar analogue of W.
- New: the regular alternating-phase m = 1 crystal (L3') is lower than PT's L3 in the full needle model:
  relative to A, d = 0.150: L3 -0.00044, L3' -0.00094; d = 0.153: L3 -0.00320, L3' -0.00368 (per spin). Files L3alt_d*.npy.

## Final theory wrap-up: L3' replaces L3; realistic columnar walls (columnar_real.py)
- L3' (regular alternating horizontal-row phases) has a 3 x 2 minimal cell, psi_A = psi_C = 0, and is registered as candidate
  "L3p" (cand_L3p.npy). Needle model (12x4): L3' below L3 by ~5e-4 per spin at every d (same responses: eps 0, theta 0,
  eta 3.1-3.3). Realistic islands: L3' below L3 by 0.1-0.4 K but never the zero-field ground state on the fixed-gap curve.
- Point sums L3' - C: S_LS, S_LL < 0 everywhere; S_SS changes sign near d = 0.155.
- Realistic columnar walls (alternating phases): per wall cell -56.0 K (m=1) / -50.6 K (m=2) at d = 0.153 (mu = 1.70),
  -144.3 / -137.4 K at d = 0.20 (mu = 2.42); point formula -53.2 / -47.7 and -141.7 / -135.8 K. Same picture as the needle model.
- Redrawn with L3': candidates_phase_verified_L3p.png, model_compare_phase_L3p.png, dmu_phase_verified_L3p.png
  (qualitatively unchanged; the ladder window widens slightly).

<!-- merged from the Claude Code finite-size-scaling campaign -->
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
