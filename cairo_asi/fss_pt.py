"""Parallel tempering in the dimer manifold for finite-size scaling of the A|C transition (generalises pt_joint.py).
Needle model, g = 0, rc = 64a. All replicas start from the same state (A or C) so that A- and C-start runs are independent.
Per round: 1 sweep (fss_mc.mc_keep, fields kept between rounds) for every replica, then a swap sweep over
alternating even/odd neighbour pairs. Recorded per temperature slot: E every round, signed (psi_A, psi_C) every
`every` rounds, and which replica sits there (replica paths / round trips). Checkpoint = one npz, restart-safe.

  python fss_pt.py tune  --d 0.145 --n 16 --M 28 --rounds 20000             -> fss/ladder_d0.1450_n16.json
  python fss_pt.py run   --d 0.145 --n 16 --start A --seed 1 --rounds 200000 -> fss/pt_d0.1450_n16_A_s1.npz
  (re-running the same `run` command resumes from the checkpoint; --rounds may be increased to extend)"""
import os, sys, json, time, argparse, signal
import numpy as np
from scipy import sparse
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from manifold_mc2 import mc_arrays
from selection_T import FastDimers
from plot_phases import tile
from fss_mc import mc_keep, csr, couplings

OUT = "fss"
REF = {"A": "scan3_min_d0.1372.npy", "C": "scan3_min_d0.2500.npy"}


def default_ladder(M, Tlo=0.08, Thi_dense=0.20, Tmax=0.40, frac_dense=0.7):
    """Geometric, with frac_dense of the replicas inside [Tlo, Thi_dense] and the rest up to Tmax (route around the barrier)."""
    Md = int(round(M*frac_dense))
    lo = np.geomspace(Tlo, Thi_dense, Md)
    hi = np.geomspace(Thi_dense, Tmax, M - Md + 1)[1:]
    return np.r_[lo, hi]


class System:
    def __init__(self, d, n):
        assert n % 4 == 0, "n must be a multiple of 4 (A and C periods)"
        self.d, self.n = d, n
        self.L = L = CairoLattice(n, n, d=d)
        self.J, self.Jl = couplings(L, d, n)
        self.arr = mc_arrays(L); self.sp = csr(self.Jl)
        self.Jl = sparse.csr_matrix(self.Jl)            # keep only the sparse local couplings (memory)
        self.F = FastDimers(L, self.Jl); self.N = L.N

    def start(self, which):
        return tile(np.load(REF[which])[0], 4, 2, self.n)

    def energy(self, s): return 0.5*s @ self.J @ s


def pt_loop(sy, Ts, S, rounds, r0, seed, every, rec, ckpt, ckpt_every=300.0, log_every=2000, tag=""):
    """Run PT rounds r0..rounds-1. S: (M, N) states indexed by *replica*; perm[m] = replica at temperature slot m."""
    M = len(Ts); N = sy.N; J, Jl = sy.J, sy.Jl; lp, li, lv = sy.sp
    perm = rec["perm"]; H = S @ J; HL = np.ascontiguousarray((Jl @ S.T).T); E = np.array([0.5*S[r] @ H[r] for r in range(M)])
    rng = np.random.default_rng([seed, r0])
    Etr = np.empty(1); t_last = time.time(); t0 = time.time()
    first = True
    stop = {"flag": False}
    def handler(sig, frm): stop["flag"] = True
    signal.signal(signal.SIGTERM, handler); signal.signal(signal.SIGINT, handler)
    for rnd in range(r0, rounds):
        for m in range(M):
            r = perm[m]
            E[r], na = mc_keep(S[r], H[r], HL[r], E[r], J, lp, li, lv, *sy.arr, Ts[m], 1,
                               (seed*7919 + r0*31 + 1) if first else -1, Etr)
            first = False
            rec["nacc"][m] += na
        for m in range(rnd % 2, M - 1, 2):
            a, b = perm[m], perm[m + 1]
            rec["att"][m] += 1; x = (1/Ts[m] - 1/Ts[m + 1])*(E[a] - E[b])
            if x >= 0 or rng.random() < np.exp(x):
                perm[m], perm[m + 1] = b, a; rec["acc"][m] += 1
        # replica direction labels: +1 = last visited the lowest T, -1 = the highest T
        lab = rec["label"]
        if lab[perm[0]] == -1: rec["trips"][perm[0]] += 1
        lab[perm[0]] = 1; lab[perm[-1]] = -1
        rec["nup"] += (lab[perm] == 1); rec["ndn"] += (lab[perm] == -1)
        i = rnd - rec["rbase"]
        rec["E"][i] = E[perm] - rec["Eref"]; rec["rep"][i] = perm
        if rnd % every == 0:
            ps = np.array([sy.F.psi(S[perm[m]]) for m in range(M)])
            rec["psi"].append(ps.astype(np.float32)); rec["psi_round"].append(rnd)
        if rnd % log_every == 0:
            acc = rec["acc"]/np.maximum(rec["att"], 1)
            print(f"[{tag}] round {rnd}/{rounds} {time.time()-t0:.0f}s  min swap {acc.min():.2f}  trips {int(rec['trips'].sum())}", flush=True)
        if ckpt is None and stop["flag"]: print(f"[{tag}] stopped by signal", flush=True); sys.exit(0)
        if ckpt is not None and (time.time() - t_last > ckpt_every or rnd == rounds - 1 or stop["flag"]):
            ckpt(S, rnd + 1); t_last = time.time()
            if stop["flag"]: print(f"[{tag}] stopped by signal at round {rnd+1}", flush=True); sys.exit(0)
    # energy drift check
    Echk = np.array([sy.energy(S[r]) for r in range(M)])
    return float(np.abs(Echk - E).max())


def new_rec(M, nrounds, rbase=0):
    return dict(perm=np.arange(M), acc=np.zeros(M - 1), att=np.zeros(M - 1), nacc=np.zeros(M),
                label=np.zeros(M, int), trips=np.zeros(M, int), nup=np.zeros(M), ndn=np.zeros(M),
                E=np.zeros((nrounds, M), np.float32), rep=np.zeros((nrounds, M), np.int16), psi=[], psi_round=[], rbase=rbase)


def cmd_tune(a):
    """Short PT from alternating A/C starts; iteratively respace the ladder in beta so that -ln(acc) is uniform."""
    sy = System(a.d, a.n); M = a.M
    Ts = default_ladder(M, a.Tlo, a.Tdense, a.Tmax, a.frac_dense)
    sA, sC = sy.start("A"), sy.start("C")
    S = np.array([(sA if m % 2 else sC).copy() for m in range(M)])
    per = a.rounds // a.iters; it0 = 0
    ck = f"{OUT}/tuneckpt_d{a.d:.4f}_n{a.n}.npz"            # per-iteration checkpoint (restart-safe tuning)
    if os.path.exists(ck):
        z = np.load(ck); Ts, S, it0 = z["T"], z["S"].copy(), int(z["it"]); print(f"resume tuning at iteration {it0}", flush=True)
    for it in range(it0, a.iters):
        rec = new_rec(M, per); rec["Eref"] = 0.0
        pt_loop(sy, Ts, S, per, 0, a.seed + it, 10**9, rec, None, tag=f"tune n={a.n} it{it}")
        # put states back in temperature order so that the next iteration starts from roughly equilibrated slots
        S = S[rec["perm"]].copy()
        acc = np.clip(rec["acc"]/np.maximum(rec["att"], 1), 0.02, 0.98)
        print(f"  it{it} acc:", np.round(acc, 2).tolist(), flush=True)
        if it == a.iters - 1: break
        B = 1/Ts; dist = np.r_[0, np.cumsum(-np.log(acc))]
        tgt = np.linspace(0, dist[-1], M)
        Bn = np.interp(tgt, dist, B)
        Tn = 1/Bn; Tn[0], Tn[-1] = Ts[0], Ts[-1]
        Ts = 0.5*Ts + 0.5*np.sort(Tn)            # damped update
        np.savez(ck + ".tmp.npz", T=Ts, S=S, it=it + 1); os.replace(ck + ".tmp.npz", ck)
    fn = f"{OUT}/ladder_d{a.d:.4f}_n{a.n}.json"
    json.dump(dict(d=a.d, n=a.n, T=Ts.tolist(), acc=acc.tolist()), open(fn, "w"), indent=1)
    np.save(f"{OUT}/tunestate_d{a.d:.4f}_n{a.n}.npy", S)
    print("ladder ->", fn, np.round(Ts, 4).tolist())


def cmd_run(a):
    sy = System(a.d, a.n)
    fn = f"{OUT}/pt_d{a.d:.4f}_n{a.n}_{a.start}_s{a.seed}.npz"
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        Ts = z["T"]; S = z["S"].copy(); r0 = int(z["round"]); M = len(Ts)
        rec = new_rec(M, a.rounds - r0, rbase=r0)
        for k in ["perm", "acc", "att", "nacc", "label", "trips", "nup", "ndn"]: rec[k] = z[k].copy()
        old = dict(E=z["E"], rep=z["rep"], psi=z["psi"], psi_round=z["psi_round"])
        print(f"resume {fn} at round {r0}", flush=True)
    else:
        lad = json.load(open(a.ladder)) if a.ladder else None
        Ts = np.array(lad["T"]) if lad else default_ladder(a.M, a.Tlo, a.Tdense, a.Tmax, a.frac_dense)
        M = len(Ts); S = np.array([sy.start(a.start) for _ in range(M)]); r0 = 0
        rec = new_rec(M, a.rounds); old = None
    rec["Eref"] = float(np.floor(-39.9*sy.N))   # stored E = E_total - Eref (float32)
    if r0 >= a.rounds: print("already done"); return
    meta = dict(d=a.d, n=a.n, N=sy.N, start=a.start, seed=a.seed, every=a.every, rc=64.0, gap=0.0, sweeps_per_round=1, Eref=rec["Eref"])

    def ckpt(S, rdone):
        k = rdone - rec["rbase"]
        E = rec["E"][:k]; rep = rec["rep"][:k]
        psi = np.array(rec["psi"], np.float32).reshape(-1, M, 2); pr = np.array(rec["psi_round"], np.int64)
        if old is not None:
            E = np.concatenate([old["E"], E]); rep = np.concatenate([old["rep"], rep])
            psi = np.concatenate([old["psi"], psi]); pr = np.concatenate([old["psi_round"], pr])
        tmp = fn + ".tmp.npz"
        np.savez(tmp, T=Ts, S=S, round=rdone, E=E, rep=rep, psi=psi, psi_round=pr,
                 **{k2: rec[k2] for k2 in ["perm", "acc", "att", "nacc", "label", "trips", "nup", "ndn"]},
                 meta=json.dumps(meta))
        os.replace(tmp, fn)

    drift = pt_loop(sy, Ts, S, a.rounds, r0, a.seed, a.every, rec, ckpt, tag=f"n={a.n} {a.start} s{a.seed}")
    acc = rec["acc"]/np.maximum(rec["att"], 1)
    print(f"done {fn}: energy drift {drift:.2e}, swap acc min {acc.min():.2f}, round trips {int(rec['trips'].sum())}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["tune", "run"])
    p.add_argument("--d", type=float, default=0.145); p.add_argument("--n", type=int, default=8)
    p.add_argument("--start", default="A", choices=["A", "C"]); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rounds", type=int, default=100000); p.add_argument("--every", type=int, default=10)
    p.add_argument("--M", type=int, default=24); p.add_argument("--ladder", default=None)
    p.add_argument("--Tlo", type=float, default=0.08); p.add_argument("--Tdense", type=float, default=0.20)
    p.add_argument("--Tmax", type=float, default=0.40); p.add_argument("--frac_dense", type=float, default=0.7)
    p.add_argument("--iters", type=int, default=4)
    a = p.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"tune": cmd_tune, "run": cmd_run}[a.cmd](a)
