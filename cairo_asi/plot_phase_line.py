import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cairo_asi import D_EQ
seg = [(0.00, 0.03, "leaves\ndimer\nmanifold", "0.6"), (0.03, 0.07, "vertex\nlevel\ncrossing", "0.8"),
       (0.07, 0.1527, "A", "#E24B4A"), (0.1527, 0.3468, "C", "#378ADD"), (0.3468, 0.3477, "", "#999999"),
       (0.3477, 0.3492, "", "#BBBBBB"), (0.3492, 0.4175, "A", "#E24B4A"), (0.4175, 0.4725, "N9", "#1D9E75"),
       (0.4725, 0.5, "E", "#EF9F27")]
fig, ax = plt.subplots(figsize=(12, 3.0))
for a, b, lab, c in seg:
    ax.axvspan(a, b, color=c, alpha=0.85)
    if lab: ax.text((a+b)/2, 0.5, lab, ha="center", va="center", fontsize=10, color="k" if c.startswith("0") else "w")
for x, t in [(D_EQ, "equilateral\nCairo"), (0.25, "Shakti")]:
    ax.axvline(x, color="k", lw=1.5); ax.text(x, 1.03, t, ha="center", va="bottom", fontsize=9, transform=ax.get_xaxis_transform())
ax.axvline(0.25, color="k"); ax.text(0.3472, -0.35, "narrow window\nN10, D (Δd≈0.002)", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
ax.text(0.0, -0.35, "square ice", fontsize=8, transform=ax.get_xaxis_transform())
ax.text(0.5, -0.35, "edges cross", fontsize=8, ha="right", transform=ax.get_xaxis_transform())
ax.set_xlim(0, 0.5); ax.set_yticks([]); ax.set_xlabel("d")
fig.suptitle("Dipolar selection within the dimer manifold, 4x2-commensurate periods, needle model, rc = 128a (converged)", fontsize=10, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig("phase_line_converged.png", dpi=130)
