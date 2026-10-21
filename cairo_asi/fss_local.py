"""Block-local order parameters from the final PT configurations (one per temperature slot and run):
|psi_A|, |psi_C| evaluated on b x b-cell blocks, so that wall-rich C' states (locally C, globally |psi_C| ~ 0) are recognised.
   python fss_local.py 0.145 24  -> fss_local_d{d}_n{n}.json"""
import sys, json, numpy as np
from fss_pt import System

def block_psi(F, s, n, b):
    bits = (s[F.ks] > 0).astype(int); code = bits[:, 0]*4 + bits[:, 1]*2 + bits[:, 2]; r = np.arange(len(code))
    unh = F.lev[r, code] > 0; ks = []
    for t in range(3):
        heal = unh & (F.lev[r, code ^ (1 << (2 - t))] == 0) & F.is34[:, t]; ks.append(F.ks[heal, t])
    ks = np.concatenate(ks); et = ks % 10; cell = ks // 10; ci, cj = cell // n, cell % n
    blk = (ci // b)*(n // b) + cj // b; nb = (n // b)**2
    a = np.isin(et, [6, 7, 8, 9]).astype(float) - np.isin(et, [2, 3, 4, 5]); c = np.isin(et, [3, 5, 7, 9]).astype(float) - np.isin(et, [2, 4, 6, 8])
    cnt = np.bincount(blk, minlength=nb)
    pa = np.bincount(blk, a, nb)/np.maximum(cnt, 1); pc = np.bincount(blk, c, nb)/np.maximum(cnt, 1)
    return np.abs(pa).mean(), np.abs(pc).mean(), abs(a.mean()), abs(c.mean())

if __name__ == "__main__":
    d = float(sys.argv[1]); n = int(sys.argv[2]); b = 4
    sy = System(d, n); out = dict(d=d, n=n, block=b, ref={})
    for name in "AC":
        out["ref"][name] = block_psi(sy.F, sy.start(name), n, b)
    for st in "AC":
        z = np.load(f"fss/pt_d{d:.4f}_n{n}_{st}_s1.npz"); T = z["T"]; S = z["S"]; perm = z["perm"]
        rows = [dict(T=float(T[m]), **dict(zip(["blockA", "blockC", "globalA", "globalC"], map(float, block_psi(sy.F, S[perm[m]], n, b)))))
                for m in range(len(T))]
        out[st] = rows
        for r in rows: print(f"{st} T={r['T']:.4f}  block|psiA| {r['blockA']:.2f} block|psiC| {r['blockC']:.2f}   global |psiA| {r['globalA']:.2f} |psiC| {r['globalC']:.2f}")
    print("reference states (blockA, blockC, globalA, globalC):", out["ref"])
    json.dump(out, open(f"fss_local_d{d:.4f}_n{n}.json", "w"), indent=1)
