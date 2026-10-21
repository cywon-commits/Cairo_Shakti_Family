"""Analysis of fss_pt.py runs: autocorrelation, replica round trips, A-start vs C-start consistency, binned multiple-histogram
reweighting (WHAM) of all temperatures of both runs, and the first-order diagnostics at the equal-height temperature T*(n):
  Delta F = ln(P_max/P_min), latent heat (peak separation), C_max, Binder V_E = 1 - <E^4>/(3<E^2>^2).
  python fss_analyze.py 0.145 8 12 16 20      -> fss_d0.1450_n{n}.json for each n
Energies: units D/a^3, per spin unless noted (N = 10 n^2 spins)."""
import sys, os, json, glob
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d

BURN = 0.2          # fraction of rounds discarded
NBLK = 10           # jackknife blocks (per run, same time blocks for both runs)
DE_FINE = 0.02      # WHAM bin width, total energy
KS = 1.0            # Gaussian smoothing of P(E) for peak heights, total-energy units (2x the A-branch level spacing 0.5)


def tau_int(x, c=6.0):
    """Integrated autocorrelation time (Sokal automatic window), in rounds."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if n < 10 or x.var() == 0: return 0.5
    f = np.fft.rfft(x, 2*n); ac = np.fft.irfft(f*np.conj(f))[:n]; ac /= ac[0]
    t = 0.5
    for w in range(1, n):
        t += ac[w]
        if w >= c*t: break
    return max(t, 0.5)


def load(d, n, start, seed=1):
    fn = f"fss/pt_d{d:.4f}_n{n}_{start}_s{seed}.npz"
    if not os.path.exists(fn): return None
    z = np.load(fn); meta = json.loads(str(z["meta"]))
    R = int(z["round"]); r0 = int(BURN*R)
    E = z["E"][r0:R].astype(float) + meta["Eref"]          # total energy, (rounds, M), temperature-slot order
    pr = z["psi_round"]; keep = pr >= r0
    return dict(fn=fn, T=z["T"], E=E, psi=z["psi"][keep], psi_round=pr[keep], rep=z["rep"][r0:R], rounds=R, r0=r0, meta=meta,
                acc=z["acc"]/np.maximum(z["att"], 1), trips=z["trips"], nup=z["nup"], ndn=z["ndn"])


def wham(Tlist, Hs, edges, iters=200000, tol=1e-11):
    """Binned WHAM. Tlist: temperature of each ensemble; Hs: (K, nb) histograms.
    Returns dict: Eb (occupied bin centres), lng (ln density of states, max 0), f, B, lnNk, E0 (for sample reweighting)."""
    Eb = 0.5*(edges[1:] + edges[:-1]); B = 1/np.asarray(Tlist); Nk = Hs.sum(1)
    Ht = Hs.sum(0); occ = Ht > 0; Eb = Eb[occ]; lnH = np.log(Ht[occ])
    E0 = Eb.min(); x = Eb - E0
    f = np.zeros(len(B)); lnNk = np.log(Nk)
    for it in range(iters):
        lng = lnH - logsumexp(lnNk[:, None] + f[:, None] - B[:, None]*x[None, :], axis=0)
        fn = -logsumexp(lng[None, :] - B[:, None]*x[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol: f = fn; break
        f = fn
    return dict(Eb=Eb, lng=lng - lng.max(), f=f, B=B, lnNk=lnNk, E0=E0)


def canon(W, T):
    lw = W["lng"] - (W["Eb"] - W["E0"])/T; lw -= logsumexp(lw); return np.exp(lw)


def moments(W, T, N):
    Eb = W["Eb"]; p = canon(W, T); m = np.sum(p*Eb); x = Eb - m
    var = np.sum(p*x**2)
    # V - 2/3 = -Var(E^2) / (3 <E^2>^2); Var(E^2) computed stably from E^2 - <E^2> = 2 m x + x^2 - <x^2>
    vE2 = np.sum(p*(2*m*x + x**2 - var)**2)
    V = 2/3 - vE2/(3*np.sum(p*Eb**2)**2)
    return dict(e=m/N, C=var/(N*T*T), V=V, dV=2/3 - V)


def smoothP(W, T, ks):
    """canonical P(E) on a regular grid (step DE_FINE, total energy), Gaussian-smoothed with width ks (total-energy units):
    the A branch is a comb of discrete excitation levels (spacing ~1-2 D/a^3), so peak heights need a physical smoothing scale."""
    p = canon(W, T); k = np.rint((W["Eb"] - W["E0"])/DE_FINE).astype(int)
    P = np.zeros(k.max() + 1); np.add.at(P, k, p)
    Ps = gaussian_filter1d(P, ks/DE_FINE, mode="constant") if ks > 0 else P
    st = max(1, int(ks/(2*DE_FINE))); Eg = W["E0"] + DE_FINE*np.arange(len(P))
    return Eg[::st], Ps[::st], Eg, P


def peak_analysis(W, T, N, ks=KS):
    """two most prominent maxima of the smoothed ln P(E) (low E = A branch, high E = C/high-T branch) and the minimum between;
    also the weights of the two branches (split at the minimum), from the unsmoothed P."""
    Eg, Ps, Ef, Pf = smoothP(W, T, ks)
    ok = Ps > 1e-300; Eg, Ps = Eg[ok], Ps[ok]; lps = np.log(Ps)
    mx = [i for i in range(1, len(lps) - 1) if lps[i] >= lps[i - 1] and lps[i] >= lps[i + 1]]
    if len(lps) > 1 and lps[0] > lps[1]: mx = [0] + mx
    if len(lps) > 1 and lps[-1] > lps[-2]: mx = mx + [len(lps) - 1]
    if len(mx) < 2: return None
    best = None
    for a_ in range(len(mx)):
        for b_ in range(a_ + 1, len(mx)):
            i, j = mx[a_], mx[b_]; k = i + np.argmin(lps[i:j + 1])
            depth = min(lps[i], lps[j]) - lps[k]
            if best is None or depth > best[0]: best = (depth, i, j, k)
    depth, i, j, k = best
    if depth < 0.05: return None
    def para(q):                      # parabolic refinement of an extremum at grid point q
        if q == 0 or q == len(lps) - 1: return Eg[q], lps[q]
        y0, y1, y2 = lps[q - 1], lps[q], lps[q + 1]; den = y0 - 2*y1 + y2
        if den == 0: return Eg[q], y1
        t = np.clip(0.5*(y0 - y2)/den, -0.5, 0.5); return Eg[q] + t*(Eg[1] - Eg[0]), y1 - 0.25*(y0 - y2)*t
    (EA, lA), (EC, lC), (Em, lm) = para(i), para(j), para(k)
    wA = Pf[Ef < Em].sum(); wC = Pf[Ef >= Em].sum()
    return dict(eA=EA/N, eC=EC/N, emin=Em/N, lpA=lA, lpC=lC, lpmin=lm, lnwA=np.log(wA), lnwC=np.log(wC), e=Eg/N, lnP=lps)


def find_T(W, N, Tlo, Thi, crit="height", ks=KS):
    def h(T):
        r = peak_analysis(W, T, N, ks)
        if r is None: return np.nan
        return r["lpA"] - r["lpC"] if crit == "height" else r["lnwA"] - r["lnwC"]
    Tg = np.linspace(Tlo, Thi, 121); hg = np.array([h(T) for T in Tg])
    ok = np.isfinite(hg)
    for a_ in range(len(Tg) - 1):
        if ok[a_] and ok[a_ + 1] and np.sign(hg[a_]) != np.sign(hg[a_ + 1]):
            return brentq(h, Tg[a_], Tg[a_ + 1], xtol=1e-7)
    return None


def solve(runs, sl, N, edges):
    Tl, Hs = [], []
    for R in runs:
        E = R["E"][sl(R)]
        for m, T in enumerate(R["T"]):
            Tl.append(T); Hs.append(np.histogram(E[:, m], edges)[0])
    return wham(Tl, np.array(Hs, float), edges)


def observables(W, N, Tgrid, Tlo, Thi):
    mom = [moments(W, T, N) for T in Tgrid]
    C = np.array([m["C"] for m in mom]); V = np.array([m["V"] for m in mom])
    k = int(np.argmax(C)); kv = int(np.argmin(V))
    out = dict(Cmax=float(C[k]), T_Cmax=float(Tgrid[k]), Vmin=float(V[kv]), dVmax=float(2/3 - V[kv]), T_Vmin=float(Tgrid[kv]),
               Tstar=None, dF=None, eA=None, eC=None, latent=None, Tstar_w=None, dF_w=None, dF_ks2x=None, Tstar_ks2x=None)
    Ts = find_T(W, N, Tlo, Thi, "height")
    if Ts is not None:
        pa = peak_analysis(W, Ts, N)
        out.update(Tstar=float(Ts), dF=float(0.5*(pa["lpA"] + pa["lpC"]) - pa["lpmin"]), eA=float(pa["eA"]), eC=float(pa["eC"]),
                   latent=float(pa["eC"] - pa["eA"]), emin=float(pa["emin"]))
    Tw = find_T(W, N, Tlo, Thi, "weight")
    if Tw is not None:
        pw = peak_analysis(W, Tw, N)
        out.update(Tstar_w=float(Tw), dF_w=float(0.5*(pw["lpA"] + pw["lpC"]) - pw["lpmin"]), latent_w=float(pw["eC"] - pw["eA"]))
    T2 = find_T(W, N, Tlo, Thi, "height", ks=2*KS)            # sensitivity to the smoothing scale
    if T2 is not None:
        p2 = peak_analysis(W, T2, N, ks=2*KS)
        out.update(Tstar_ks2x=float(T2), dF_ks2x=float(0.5*(p2["lpA"] + p2["lpC"]) - p2["lpmin"]))
    return out, C, V


def psi_at(runs, W, T, Esplit, Cwin=None):
    """reweight the (psi_A, psi_C) samples of all slots/runs to temperature T (WHAM denominators); split at total energy Esplit."""
    Es, Ps = [], []
    for R in runs:
        idx = R["psi_round"] - R["r0"]; ok = (idx >= 0) & (idx < R["E"].shape[0])
        Es.append(R["E"][idx[ok]].ravel()); Ps.append(R["psi"][ok].reshape(-1, 2))
    E = np.concatenate(Es); P = np.abs(np.concatenate(Ps).astype(float)); x = E - W["E0"]
    lden = np.concatenate([logsumexp(W["lnNk"][:, None] + W["f"][:, None] - W["B"][:, None]*xc[None, :], axis=0)
                           for xc in np.array_split(x, max(1, len(x)//200000))])
    lw = -x/T - lden; w = np.exp(lw - lw.max())
    out = {}
    sels = [("all", np.ones(len(E), bool)), ("lowE_A_branch", E < Esplit), ("highE_branch", E >= Esplit)]
    if Cwin is not None: sels.append(("highE_peak", (E >= Cwin[0]) & (E <= Cwin[1])))     # around the high-E maximum, no interface states
    for name, sel in sels:
        ww = w*sel; ww = ww/ww.sum()
        out[name] = dict(weight=float((w*sel).sum()/w.sum()), absA=float(ww @ P[:, 0]), absC=float(ww @ P[:, 1]),
                         rmsA=float(np.sqrt(ww @ P[:, 0]**2)), rmsC=float(np.sqrt(ww @ P[:, 1]**2)),
                         fracC=float(ww @ ((P[:, 1] > 0.5) & (P[:, 0] < 0.3))), fracA=float(ww @ ((P[:, 0] > 0.5) & (P[:, 1] < 0.3))))
    return out


def analyze(d, n):
    N = 10*n*n
    runs = [r for r in (load(d, n, "A"), load(d, n, "C")) if r is not None]
    if not runs: print("no runs for", n); return None
    res = dict(d=d, n=n, N=N, runs={})
    # --- per-run diagnostics
    for R in runs:
        st = R["meta"]["start"]; E = R["E"]; M = len(R["T"]); nr = E.shape[0]
        taus = np.array([tau_int(E[:, m]) for m in range(M)])
        err = np.sqrt(E.var(0)*2*taus/nr)/N
        ps = R["psi"]; pa, pc = np.abs(ps[..., 0]), np.abs(ps[..., 1])
        fA = ((pa > 0.5) & (pc < 0.3)).mean(0); fC = ((pc > 0.5) & (pa < 0.3)).mean(0)
        fup = R["nup"]/np.maximum(R["nup"] + R["ndn"], 1)
        res["runs"][st] = dict(file=R["fn"], rounds=R["rounds"], burn=R["r0"], T=R["T"].tolist(), swap_acc=R["acc"].tolist(),
                               round_trips=int(R["trips"].sum()), trips_per_replica=R["trips"].tolist(), f_up=fup.tolist(),
                               tau_E=taus.tolist(), n_over_tau=(nr/taus).tolist(), e_mean=(E.mean(0)/N).tolist(), e_err=err.tolist(),
                               C_slot=(E.var(0)/(N*R["T"]**2)).tolist(), frac_A=fA.tolist(), frac_C=fC.tolist(),
                               psiA_abs=pa.mean(0).tolist(), psiC_abs=pc.mean(0).tolist())
    if len(runs) == 2 and np.allclose(runs[0]["T"], runs[1]["T"]):
        a, c = res["runs"]["A"], res["runs"]["C"]
        z = (np.array(a["e_mean"]) - np.array(c["e_mean"]))/np.sqrt(np.array(a["e_err"])**2 + np.array(c["e_err"])**2)
        res["A_vs_C_z"] = z.tolist(); res["A_vs_C_maxabsz"] = float(np.abs(z).max())
    # --- WHAM on everything
    Eall = np.concatenate([R["E"].ravel() for R in runs])
    edges = np.arange(Eall.min() - DE_FINE, Eall.max() + 2*DE_FINE, DE_FINE)
    Tmin = min(R["T"].min() for R in runs); Tmax = max(R["T"].max() for R in runs)
    Tlo, Thi = TRANGE if TRANGE else (max(Tmin, 0.085), min(Tmax, 0.2))       # search window for T*
    Tgrid = np.linspace(Tlo, TRANGE[1] if TRANGE else min(Tmax, 0.25), 661)
    W = solve(runs, lambda R: slice(None), N, edges)
    obs, C, V = observables(W, N, Tgrid, Tlo, Thi)
    res.update(obs); res["T_rw"] = Tgrid.tolist(); res["C_rw"] = C.tolist(); res["V_rw"] = V.tolist()
    res["e_rw"] = [moments(W, T, N)["e"] for T in Tgrid]
    if obs["Tstar"] is not None:
        pa = peak_analysis(W, obs["Tstar"], N); res["PE_at_Tstar"] = dict(e=pa["e"].tolist(), lnP=pa["lnP"].tolist())
        Eg, Ps, Ef, Pf = smoothP(W, obs["Tstar"], 0.0)                 # unsmoothed, per-spin bins of DE_FINE/N
        res["PE_raw_at_Tstar"] = dict(e=(Ef[Pf > 0]/N).tolist(), lnP=np.log(Pf[Pf > 0]).tolist())
        eC, em = obs["eC"]*N, obs["emin"]*N
        res["psi_at_Tstar"] = psi_at(runs, W, obs["Tstar"], em, (eC - 0.5*(eC - em), eC + 0.5*(eC - em)))
        res["psi_rw"] = {f"{T:.3f}": psi_at(runs, W, T, obs["emin"]*N)["all"] for T in [0.10, 0.13, 0.15, 0.2, 0.25, 0.3]
                         if Tmin <= T <= Tmax}
    # --- per-start estimates (consistency) and jackknife errors
    for R in runs:
        o, _, _ = observables(solve([R], lambda R_: slice(None), N, edges), N, Tgrid, Tlo, Thi)
        res["runs"][R["meta"]["start"]]["reweighted"] = o
    jk = []
    for b in range(NBLK):
        def sl(R, b=b):
            nr = R["E"].shape[0]; idx = np.ones(nr, bool); idx[b*nr//NBLK:(b + 1)*nr//NBLK] = False; return idx
        o, _, _ = observables(solve(runs, sl, N, edges), N, Tgrid, Tlo, Thi); jk.append(o)
    res["err"] = {}
    for k in ["Tstar", "dF", "eA", "eC", "latent", "Cmax", "T_Cmax", "dVmax", "Tstar_w", "dF_w", "dF_ks2x"]:
        v = np.array([o[k] for o in jk if o[k] is not None], float)
        res["err"][k] = float(np.sqrt((len(v) - 1)/len(v)*np.sum((v - v.mean())**2))) if len(v) > 2 else None
    fn = f"fss_d{d:.4f}_n{n}.json"; json.dump(res, open(fn, "w"), indent=1)
    e = res["err"]
    fmt = lambda k: "None" if res[k] is None else f"{res[k]:.5g} ± {e[k] if e[k] is not None else float('nan'):.2g}"
    print(f"n={n}: T*={fmt('Tstar')}  dF={fmt('dF')}  latent/N={fmt('latent')}  Cmax={fmt('Cmax')} at T={res['T_Cmax']:.4f}  "
          f"2/3-Vmin={res['dVmax']:.3g}  A-vs-C max|z|={res.get('A_vs_C_maxabsz', float('nan')):.1f}", flush=True)
    for st, r in res["runs"].items():
        o = r["reweighted"]
        print(f"   {st}-start: rounds {r['rounds']}, trips {r['round_trips']}, min swap {min(r['swap_acc']):.2f}, "
              f"max tau_E {max(r['tau_E']):.0f} (min n/tau {min(r['n_over_tau']):.0f}); T*={o['Tstar']}, dF={o['dF']}, T*_w={o['Tstar_w']}", flush=True)
    print(f"   T*_w={fmt('Tstar_w')} dF_w={fmt('dF_w')} dF(ks x2)={fmt('dF_ks2x')}")
    if "psi_at_Tstar" in res:
        for k, v in res["psi_at_Tstar"].items(): print(f"   psi at T*, {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items()))
    return res


TRANGE = None
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--Trange" in args:                                   # e.g. --Trange 0.03 0.3 (default: 0.085-0.2 for T*, 0.085-0.25 grid)
        i = args.index("--Trange"); TRANGE = (float(args[i + 1]), float(args[i + 2])); del args[i:i + 3]
    d = float(args[0])
    for n in [int(x) for x in args[1:]]: analyze(d, n)
