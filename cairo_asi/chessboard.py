"""Map dimer-manifold states onto the square lattice (chessboard) of pentagons and draw the domino tilings."""
import numpy as np, matplotlib, collections
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import needle; needle.GAP = 0.0
from cairo_asi import CairoLattice
from hybrid_coupling import local_needle_coupling
from dimer_map import DimerMap
from plot_phases import tile

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]           # E N W S (CCW)
def chess_coords(L, D):
    nb = collections.defaultdict(list)
    for k, fs in D.e2f.items():
        if D.is34[k]: nb[fs[0]].append((fs[1], k)); nb[fs[1]].append((fs[0], k))
    order = {}
    for f, lst in nb.items():
        c = L.face_center[f]
        ang = [np.arctan2(*L.mic(L.face_center[g] - c)[::-1]) for g, _ in lst]
        order[f] = [lst[i] for i in np.argsort(ang)]
    coord, dirof = {0: (0, 0)}, {}
    # direction index of each neighbour slot at face 0
    dirof[0] = {g: i for i, (g, _) in enumerate(order[0])}
    queue = [0]; conflicts = 0
    while queue:
        f = queue.pop(0)
        for (g, k) in order[f]:
            di = dirof[f][g]; dx, dy = DIRS[di]
            if g not in coord:
                coord[g] = (coord[f][0] + dx, coord[f][1] + dy)
                idx = [h for h, _ in order[g]].index(f); back = (di + 2) % 4
                dirof[g] = {order[g][(idx + t) % 4][0]: (back + t) % 4 for t in range(4)}
                queue.append(g)
            else:
                idx = [h for h, _ in order[g]].index(f)
                if dirof[g][f] != (di + 2) % 4: conflicts += 1
    return coord, conflicts

def draw(ax, L, D, s, title):
    coord, conf = chess_coords(L, D)
    chi = L.chirality(s); dm = D.dimers(s)
    xs = [c[0] for c in coord.values()]; ys = [c[1] for c in coord.values()]
    # show a window near the origin
    win = [(f, c) for f, c in coord.items() if -6 <= c[0] <= 5 and -6 <= c[1] <= 5]
    wf = {f for f, _ in win}
    for f, (x, y) in win:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="#f2f2f2" if (x + y) % 2 else "#bdbdbd", ec="none"))
    orient_col = {}
    for k in dm:
        f1, f2 = D.e2f[k]
        if f1 not in wf or f2 not in wf: continue
        (x1, y1), (x2, y2) = coord[f1], coord[f2]
        if abs(x1 - x2) + abs(y1 - y2) != 1: continue
        horiz = y1 == y2
        c1 = "#E24B4A" if np.sign(chi[f1]) < 0 else "#378ADD"          # red = CW, blue = CCW (same sign in a dimer)
        x0, y0 = min(x1, x2), min(y1, y2)
        w, h = (2, 1) if horiz else (1, 2)
        ax.add_patch(Rectangle((x0 - .5 + .08, y0 - .5 + .08), w - .16, h - .16, fc=c1, ec="k", lw=1.2, alpha=0.75))
        for f, (x, y) in ((f1, (x1, y1)), (f2, (x2, y2))):
            if abs(abs(chi[f]) - 1) < 1e-9: ax.plot(x, y, "o", ms=5, color="k")   # closed pentagon
    dset = {}
    for k in dm:
        f1, f2 = D.e2f[k]
        if f1 in coord and f2 in coord:
            c1, c2 = coord[f1], coord[f2]
            if abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) == 1: dset[frozenset([c1, c2])] = 1
    for (x, y) in [c for _, c in win]:
        h2 = frozenset([(x, y), (x+1, y)]) in dset and frozenset([(x, y+1), (x+1, y+1)]) in dset
        v2 = frozenset([(x, y), (x, y+1)]) in dset and frozenset([(x+1, y), (x+1, y+1)]) in dset
        if (h2 or v2) and x + 1 <= 5 and y + 1 <= 5:
            ax.add_patch(Rectangle((x - .5, y - .5), 2, 2, fc="none", ec="#1D9E75" if (x + y) % 2 == 0 else "#EF9F27", lw=2.6))
    ax.set_xlim(-6.5, 5.5); ax.set_ylim(-6.5, 5.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    return conf

if __name__ == "__main__":
    out = []
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2))
    cases = [("A (Cairo side, d=0.137)", 0.1372, tile(np.load("scan3_min_d0.1372.npy")[0], 4, 2, 8)),
             ("C (Shakti side, d=0.25)", 0.25, tile(np.load("scan3_min_d0.2500.npy")[0], 4, 2, 8)),
             ("W (wall crystal, d=0.20)", 0.20, np.load("pt_best_d0.2000_n8_A3.npy"))]
    for ax, (name, d, s) in zip(axs, cases):
        L = CairoLattice(8, 8, d=d); D = DimerMap(L, local_needle_coupling(L))
        conf = draw(ax, L, D, s, name); print(name, "coordinate conflicts:", conf)
    fig.suptitle("Dimer-manifold states drawn as domino tilings of the pentagon chessboard\n"
                 "domino colour = circulation (red CW, blue CCW); dot = closed pentagon; 2x2 block outline: orange = block origin x+y odd (A), green = even (C)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig("chessboard_ACW.png", dpi=120)
