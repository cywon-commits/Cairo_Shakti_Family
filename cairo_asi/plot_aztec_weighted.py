import numpy as np, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = 32; Ks = [(-3.0, 1), (-3.0, 2), (0.0, 1), (1.0, 1), (3.0, 1)]
fig, axs = plt.subplots(2, 5, figsize=(22, 9))
for col, (K, sd) in enumerate(Ks):
    z = np.load(f"aztecW_n{n}_K{K:+.2f}_s{sd}.npz"); Fm = z["Fm"]; S = Fm.shape[0] + 1; off = n
    ii, jj = np.meshgrid(np.arange(S - 1) - off, np.arange(S - 1) - off, indexing="ij")
    # class densities coarse-grained on 2x2 cells of plaquettes: each 2x2 cell holds one plaquette of each class
    C = np.full((S//2 - 1, S//2 - 1, 4), np.nan); Phi = np.full((S//2 - 1, S//2 - 1), np.nan); Ft = np.full_like(Phi, np.nan)
    for A in range(S//2 - 1):
        for B in range(S//2 - 1):
            a, b = 2*A, 2*B
            blk = Fm[a:a+2, b:b+2]                          # classes by (ii%2, jj%2): with off even, (a,b) -> (0,0)
            cls = [blk[0, 0], blk[1, 1], blk[0, 1], blk[1, 0]]   # 00, 11, 01, 10
            xc, yc = (ii[a, b] + 1.5)/n, (jj[a, b] + 1.5)/n
            if abs(xc) + abs(yc) > 1 - 1.5/n: continue
            C[A, B] = cls; tot = sum(cls); Ft[A, B] = tot/4
            Phi[A, B] = (cls[0] + cls[1] - cls[2] - cls[3])/tot if tot > 0.02 else np.nan
    ext = [-1, 1, -1, 1]
    im = axs[0, col].imshow(Phi.T, origin="lower", extent=ext, cmap="bwr", vmin=-1, vmax=1)
    axs[0, col].set_title(f"K = {K:+.1f} (seed {sd}): Φ (local)", fontsize=11)
    dom = np.nanargmax(np.nan_to_num(C, nan=-1), axis=2).astype(float); dom[np.isnan(Phi)] = np.nan
    strength = np.nanmax(C, axis=2); dom[strength < 0.5] = np.nan               # only where one class is clearly crystalline
    cmap4 = matplotlib.colors.ListedColormap(["#1D9E75", "#0C447C", "#EF9F27", "#A32D2D"])
    axs[1, col].imshow(Ft.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=0.3)
    axs[1, col].imshow(dom.T, origin="lower", extent=ext, cmap=cmap4, vmin=-0.5, vmax=3.5, alpha=0.85)
    axs[1, col].set_title("grey: block density; colour: crystalline class\n(00 green, 11 navy = C; 01 orange, 10 red = A)", fontsize=9)
    t = np.linspace(0, 2*np.pi, 300)
    for r_ in range(2):
        axs[r_, col].plot(np.cos(t)/np.sqrt(2), np.sin(t)/np.sqrt(2), "k--", lw=0.7)
        axs[r_, col].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], "k-", lw=0.8); axs[r_, col].set_xticks([]); axs[r_, col].set_yticks([])
plt.colorbar(im, ax=axs[0, :], shrink=0.7, label="Φ")
plt.suptitle(f"Aztec diamond n = {n} with parity-breaking weight exp(K Σ F_p (-1)^(x+y)); dashed = arctic circle of the uniform case", fontsize=12)
plt.savefig("aztec_weighted_maps.png", dpi=110, bbox_inches="tight")
