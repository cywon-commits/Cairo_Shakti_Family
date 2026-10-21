"""Search low-energy periodic states in commensurate cells (dipolar rc=6a), then tile to 8x8."""
import numpy as np, sys
from cairo_asi import *
rc = 6.0
Lbig = CairoLattice(8, 8); Jbig = Lbig.coupling_matrix(rc)
def tile(s0, cx, cy, n=8):
    s = np.empty(n*n*10)
    for i in range(n):
        for j in range(n):
            for b in range(10):
                s[(i*n+j)*10+b] = s0[((i%cx)*cy+(j%cy))*10+b]
    return s
res = {}
for cx, cy in [(2,2),(4,2),(4,4),(8,8)]:
    Ls = CairoLattice(cx, cy); Js = Ls.coupling_matrix(rc)
    nr = {2:400,4:150,8:40}[max(cx,cy)]
    best = None
    for sd in range(nr):
        s1,_ = anneal(Js, np.random.choice([-1.,1.],Ls.N), 5.0, 0.01, 2000 if cx<8 else 6000, 1000+sd); s1 = quench(Js,s1)
        e1 = 0.5*s1@Js@s1/Ls.N
        if best is None or e1 < best[0]-1e-9: best = (e1, s1.copy(), 1)
        elif abs(e1-best[0])<1e-9: best = (best[0], best[1], best[2]+1)
    sb = tile(best[1],cx,cy) if cx<8 else best[1]
    eb = 0.5*sb@Jbig@sb/Lbig.N
    np.save(f"gs_cell{cx}x{cy}.npy", sb)
    print(f"cell {cx}x{cy}: best E/N (own cell) {best[0]:.5f}, hits {best[2]}/{nr}, tiled to 8x8: {eb:.5f}")
