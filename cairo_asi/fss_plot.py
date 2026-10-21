"""Figures + extrapolations from fss_d{d}_n{n}.json.   python fss_plot.py 0.145 8 12 16 20
 -> fss_d{d}.png (P(e) at T*(n), DeltaF(n), C_max(n), T*(n) vs 1/n^2, latent heat, Binder), fss_diag_d{d}.png (PT diagnostics),
    fss_summary_d{d}.json (fits)."""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cairo_asi import CairoLattice


def wfit(x, y, s, deg=1):
    """weighted polynomial fit, returns coeffs (highest first) and their errors (scaled by sqrt(chi2/dof) if > 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.where(np.asarray(s, float) > 0, s, np.nanmedian(np.asarray(s, float)[np.asarray(s, float) > 0]) if np.any(np.asarray(s, float) > 0) else 1.0)
    A = np.vander(x, deg + 1); W = 1/s**2
    cov = np.linalg.inv(A.T @ (A*W[:, None])); p = cov @ (A.T @ (W*y))
    dof = len(x) - deg - 1; chi2 = float(np.sum(W*(y - A @ p)**2))
    sc = max(1.0, chi2/dof) if dof > 0 else 1.0
    return p, np.sqrt(np.diag(cov)*sc), chi2, dof


d = float(sys.argv[1]); ns = [int(x) for x in sys.argv[2:]]
R = {n: json.load(open(f"fss_d{d:.4f}_n{n}.json")) for n in ns}
ok = [n for n in ns if R[n]["Tstar"] is not None]
g = lambda k: np.array([R[n][k] for n in ok], float)
def ge(k):
    """jackknife error combined in quadrature with half the A-start/C-start spread (the larger n are not fully equilibrated)"""
    out = []
    for n in ok:
        e = R[n]["err"].get(k) or 0.0
        v = [R[n]["runs"][s_]["reweighted"].get(k) for s_ in ("A", "C") if s_ in R[n]["runs"]]
        spread = 0.5*abs(v[0] - v[1]) if len(v) == 2 and None not in v else 0.0
        out.append(np.hypot(e, spread))
    return np.array(out, float)
nn = np.array(ok, float); NN = 10*nn**2
L = CairoLattice(4, 4, d=d); is34 = np.array([L.vtype[L.edges[k]].sum() == 7 for k in range(L.N)]); a_over_L = L.elen[is34].mean()

S = dict(d=d, n=ok, a_over_cell=float(a_over_L))
fit_n = [n for n in ok if n >= 12] if len([n for n in ok if n >= 12]) >= 2 else ok
sel = np.isin(ok, fit_n)
# T*(n) = T*(inf) + b / n^2
p, pe, chi2, dof = wfit(1/nn[sel]**2, g("Tstar")[sel], ge("Tstar")[sel])
S["Tstar_inf"] = [float(p[1]), float(pe[1])]; S["Tstar_slope_1_over_n2"] = [float(p[0]), float(pe[0])]; S["Tstar_fit"] = dict(n=fit_n, chi2=chi2, dof=dof)
# DeltaF(n) = 2 sigma n / T* + c
p2, pe2, chi2b, dofb = wfit(nn[sel], g("dF")[sel], ge("dF")[sel])
Tinf = S["Tstar_inf"][0]
S["dF_slope"] = [float(p2[0]), float(pe2[0])]; S["dF_fit"] = dict(n=fit_n, chi2=chi2b, dof=dofb, intercept=float(p2[1]))
sig_L = p2[0]*Tinf/2; sig_L_e = pe2[0]*Tinf/2
S["sigma_per_cell_length"] = [float(sig_L), float(sig_L_e)]
S["sigma_per_a"] = [float(sig_L*a_over_L), float(sig_L_e*a_over_L)]
# latent heat per spin: Delta e(n) = Delta e(inf) + c/N
p3, pe3, _, _ = wfit(1/NN[sel], g("latent")[sel], ge("latent")[sel])
S["latent_inf_per_spin"] = [float(p3[1]), float(pe3[1])]
p3b, pe3b, _, _ = wfit(1/nn[sel], g("latent")[sel], ge("latent")[sel])
S["latent_inf_per_spin_1_over_n_fit"] = [float(p3b[1]), float(pe3b[1])]
S["dF_values"] = g("dF").tolist(); S["dF_errors"] = ge("dF").tolist(); S["Tstar_values"] = g("Tstar").tolist(); S["Tstar_errors"] = ge("Tstar").tolist()
S["Cmax_over_N"] = (g("Cmax")/NN).tolist(); S["latent_values"] = g("latent").tolist(); S["dV_values"] = g("dVmax").tolist()
# C_max (per spin) ~ N: slope of log C_max vs log N
p4, pe4, _, _ = wfit(np.log(NN[sel]), np.log(g("Cmax")[sel]), ge("Cmax")[sel]/g("Cmax")[sel])
S["Cmax_exponent_vs_N"] = [float(p4[0]), float(pe4[0])]
# first-order prediction C_max/N -> (Delta e)^2 / (4 T*^2)
S["Cmax_over_N_pred_inf"] = float(S["latent_inf_per_spin"][0]**2/(4*Tinf**2))
p5, pe5, _, _ = wfit(1/NN[sel], g("Cmax")[sel]/NN[sel], ge("Cmax")[sel]/NN[sel])
S["Cmax_over_N_inf"] = [float(p5[1]), float(pe5[1])]
# Binder: 2/3 - V_min, N (2/3 - V_min) should be const for 2nd order-like (-> 0 as 1/N); first order -> const
p6, pe6, _, _ = wfit(1/NN[sel], g("dVmax")[sel], ge("dVmax")[sel])
S["two_thirds_minus_Vmin_inf"] = [float(p6[1]), float(pe6[1])]
eA, eC = g("eA"), g("eC")
S["dV_pred_from_peaks"] = (((eC**2 - eA**2)**2)/(12*eA**2*eC**2)).tolist()
S["table"] = {str(n): {k: R[n][k] for k in ["Tstar", "dF", "latent", "eA", "eC", "Cmax", "T_Cmax", "dVmax", "A_vs_C_maxabsz"] if k in R[n]}
              | {"err": R[n]["err"], "round_trips": {s: R[n]["runs"][s]["round_trips"] for s in R[n]["runs"]},
                 "min_swap": {s: min(R[n]["runs"][s]["swap_acc"]) for s in R[n]["runs"]},
                 "max_tau_E": {s: max(R[n]["runs"][s]["tau_E"]) for s in R[n]["runs"]},
                 "min_n_over_tau": {s: min(R[n]["runs"][s]["n_over_tau"]) for s in R[n]["runs"]},
                 "Tstar_by_start": {s: R[n]["runs"][s]["reweighted"]["Tstar"] for s in R[n]["runs"]},
                 "dF_by_start": {s: R[n]["runs"][s]["reweighted"]["dF"] for s in R[n]["runs"]}} for n in ok}
json.dump(S, open(f"fss_summary_d{d:.4f}.json", "w"), indent=1)

# ---------------- main figure
cm = plt.get_cmap("viridis"); col = {n: cm(i/max(len(ns) - 1, 1)) for i, n in enumerate(ns)}
fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
a = ax[0, 0]
for n in ok:
    P = R[n]["PE_at_Tstar"]; e = np.array(P["e"]); lp = np.array(P["lnP"])
    a.plot(e + 39.84, lp - lp.max(), color=col[n], label=f"n={n}, T*={R[n]['Tstar']:.4f}")
a.set_xlabel("E/N + 39.84"); a.set_ylabel("ln P(E) at T*(n)  (equal heights, smoothed)"); a.set_ylim(-12, 0.5); a.legend(fontsize=8)
a.set_xlim(min(R[n]["eA"] for n in ok) + 39.84 - 0.003, max(R[n]["eC"] for n in ok) + 39.84 + 0.005)
a = ax[0, 1]
a.errorbar(nn, g("dF"), ge("dF"), fmt="o", color="k"); x = np.linspace(0, nn.max()*1.1, 10)
a.plot(x, np.polyval(p2, x), "r--", label=f"fit n≥{min(fit_n)}: slope {p2[0]:.3f}±{pe2[0]:.3f}")
a.set_xlabel("n (cells per side)"); a.set_ylabel(r"$\Delta F = \ln(P_{max}/P_{min})$"); a.legend(fontsize=8); a.set_xlim(0)
a = ax[0, 2]
a.errorbar(NN, g("Cmax"), ge("Cmax"), fmt="o", color="k", label="C_max per spin")
x = np.geomspace(NN.min()*0.8, NN.max()*1.25, 20); a.plot(x, g("Cmax")[-1]*x/NN[-1], "r--", label="∝ N")
a.plot(x, S["Cmax_over_N_pred_inf"]*x, ":", color="0.5", label=r"$N\,\Delta e_\infty^2/4T_\infty^{*2}$")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("N = 10 n²"); a.set_ylabel("C_max / N"); a.legend(fontsize=8)
a.set_title(f"exponent vs N: {p4[0]:.2f}±{pe4[0]:.2f}", fontsize=10)
a = ax[1, 0]
a.errorbar(1/nn**2, g("Tstar"), ge("Tstar"), fmt="o", color="k", label="T*(n) equal height")
a.plot(1/nn**2, g("T_Cmax"), "s", mfc="none", color="C1", label="T of C_max")
x = np.linspace(0, 1/nn.min()**2*1.05, 10); a.plot(x, np.polyval(p, x), "r--", label=f"T*(∞) = {p[1]:.5f}±{pe[1]:.5f}")
a.set_xlabel("1/n²"); a.set_ylabel("T"); a.legend(fontsize=8); a.set_xlim(0)
a = ax[1, 1]
a.errorbar(1/NN, g("latent"), ge("latent"), fmt="o", color="k"); x = np.linspace(0, 1/NN.min()*1.05, 10)
a.plot(x, np.polyval(p3, x), "r--", label=f"Δe(∞) = {p3[1]:.5f}±{pe3[1]:.5f}")
a.set_xlabel("1/N"); a.set_ylabel("latent heat per spin  e_C − e_A at T*"); a.legend(fontsize=8); a.set_xlim(0)
a = ax[1, 2]
a.errorbar(1/NN, g("dVmax"), ge("dVmax"), fmt="o", color="k", label=r"$2/3 - V_{min}$")
a.plot(1/NN, S["dV_pred_from_peaks"], "x", color="C3", label="two-δ-peak value from e_A, e_C")
a.plot(x, np.polyval(p6, x), "r--", label=f"→ {p6[1]:.2e}±{pe6[1]:.1e}")
a.set_xlabel("1/N"); a.set_ylabel(r"$2/3 - V_{min}$"); a.legend(fontsize=8); a.set_xlim(0)
fig.suptitle(f"A|C transition, d = {d}, needle g = 0, rc = 64a: finite-size scaling (PT + WHAM)")
plt.tight_layout(); plt.savefig(f"fss_d{d:.4f}.png", dpi=120)

# ---------------- diagnostics
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for n in ns:
    for st, ls in [("A", "-"), ("C", "--")]:
        if st not in R[n]["runs"]: continue
        r = R[n]["runs"][st]; T = np.array(r["T"])
        ax[0, 0].plot(0.5*(T[1:] + T[:-1]), r["swap_acc"], ls, color=col[n], label=f"n={n} {st}")
        ax[0, 1].plot(T, r["f_up"], ls, color=col[n])
        ax[1, 0].semilogy(T, r["tau_E"], ls, color=col[n])
        ax[1, 1].plot(T, r["e_mean"], ls, color=col[n])
    ax[1, 1].plot(R[n]["T_rw"], R[n]["e_rw"], ":", color=col[n], lw=0.8)
ax[0, 0].axhline(0.2, color="0.6", lw=0.8); ax[0, 0].set_ylabel("swap acceptance"); ax[0, 0].legend(fontsize=7, ncol=2)
ax[0, 1].set_ylabel("f_up(T) (fraction last at lowest T)"); ax[1, 0].set_ylabel("τ_int(E) [rounds = sweeps]")
ax[1, 1].set_ylabel("⟨E⟩/N (solid A-start, dashed C-start, dotted WHAM)")
for a in ax.ravel(): a.set_xlabel("T")
plt.tight_layout(); plt.savefig(f"fss_diag_d{d:.4f}.png", dpi=110)
print(json.dumps({k: v for k, v in S.items() if k != "table"}, indent=1))

# ---------------- order parameters vs size: is the high-T side really C (long-range) and what coexists with A at T*?
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
Tk = list(R[ok[0]]["psi_rw"].keys()); S["psi_scaling"] = {}
for i, T in enumerate(Tk):
    yC = np.array([R[n]["psi_rw"][T]["rmsC"] for n in ok]); yA = np.array([R[n]["psi_rw"][T]["rmsA"] for n in ok])
    c_ = plt.get_cmap("plasma")(i/len(Tk))
    ax[0].loglog(nn, yC, "o-", color=c_, label=f"ψ_C, T={T}"); ax[0].loglog(nn, yA, "s--", color=c_, mfc="none", lw=0.8)
    s_ = sel & (nn >= 12)
    kC = np.polyfit(np.log(nn[s_]), np.log(yC[s_]), 1)[0] if s_.sum() >= 2 else np.nan
    kA = np.polyfit(np.log(nn[s_]), np.log(yA[s_]), 1)[0] if s_.sum() >= 2 else np.nan
    S["psi_scaling"][T] = dict(rmsC=yC.tolist(), rmsA=yA.tolist(), slope_C=float(kC), slope_A=float(kA))
ax[0].loglog(nn, 0.5*(nn/nn[0])**-1.0, ":", color="0.4", label="∝ 1/n (short-range)")
ax[0].set_xlabel("n"); ax[0].set_ylabel("rms ψ (circles ψ_C, squares ψ_A), reweighted"); ax[0].legend(fontsize=7)
for k_, mk in [("lowE_A_branch", "o"), ("highE_peak", "s")]:
    ax[1].loglog(nn, [R[n]["psi_at_Tstar"][k_]["rmsA"] for n in ok], mk + "-", color="C0", label=f"ψ_A, {k_}")
    ax[1].loglog(nn, [R[n]["psi_at_Tstar"][k_]["rmsC"] for n in ok], mk + "--", color="C3", label=f"ψ_C, {k_}")
S["psi_at_Tstar"] = {str(n): R[n]["psi_at_Tstar"] for n in ok}
ax[1].set_xlabel("n"); ax[1].set_ylabel("rms ψ at T*(n)"); ax[1].legend(fontsize=7); ax[1].set_title("branches of P(E) at T*", fontsize=10)
plt.tight_layout(); plt.savefig(f"fss_psi_d{d:.4f}.png", dpi=110)
json.dump(S, open(f"fss_summary_d{d:.4f}.json", "w"), indent=1)
print(json.dumps(S["psi_scaling"], indent=1))
