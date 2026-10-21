import numpy as np
def basketweave(L, x0, y0, phase):
    """L x L torus of cells; 2x2 blocks with lower-left corners at (x0,y0)+2Z^2; orientation alternates in a checkerboard of blocks."""
    part = -np.ones((L, L), int)
    for bx in range(L//2):
        for by in range(L//2):
            X, Y = (x0 + 2*bx) % L, (y0 + 2*by) % L
            horiz = ((bx + by + phase) % 2 == 0)
            c = [(X, Y), ((X+1) % L, Y), (X, (Y+1) % L), ((X+1) % L, (Y+1) % L)]
            if horiz: part[c[0]] = 0; part[c[1]] = 1; part[c[2]] = 0; part[c[3]] = 1
            else:     part[c[0]] = 2; part[c[2]] = 3; part[c[1]] = 2; part[c[3]] = 3
    return part
def torus_heights(part):
    L = part.shape[0]
    def partner(a, b): return {0: ((a+1) % L, b), 1: ((a-1) % L, b), 2: (a, (b+1) % L), 3: (a, (b-1) % L)}[part[a % L, b % L]]
    black = lambda a, b: (a + b) % 2 == 0
    # unwrapped heights on the (L+1)x(L+1) vertex patch; check periodicity (slope)
    h = np.full((L + 1, L + 1), np.nan); h[0, 0] = 0
    for A in range(L + 1):
        for B in range(L + 1):
            if A == 0 and B == 0: continue
            if A > 0:     # step from (A-1,B) to (A,B): moving +x, left = cell (A-1, B), right = (A-1, B-1)
                l, r = ((A-1) % L, B % L), ((A-1) % L, (B-1) % L)
                s = 1 if black(*l) else -1; h[A, B] = h[A-1, B] + s*(-3 if partner(*l) == r else 1)
            else:         # step from (A,B-1) to (A,B): moving +y, left = cell (A-1, B-1), right = (A, B-1)
                l, r = ((A-1) % L, (B-1) % L), (A % L, (B-1) % L)
                s = 1 if black(*l) else -1; h[A, B] = h[A, B-1] + s*(-3 if partner(*l) == r else 1)
    return h
L = 8
print("class (x0 mod 2, y0 mod 2), phase: mean height over one period (start vertex h=0), height range, slope check (h(L,0), h(0,L))")
for (x0, y0) in [(0, 0), (1, 1), (0, 1), (1, 0)]:
    for ph in (0, 1):
        h = torus_heights(basketweave(L, x0, y0, ph))
        m = h[:L, :L].mean()
        print(f"  ({x0},{y0}) {'even' if (x0+y0)%2==0 else 'odd '} phase {ph}: mean {m:+.3f}, range [{h.min():.0f},{h.max():.0f}], slope check {h[L,0]:.0f},{h[0,L]:.0f}")
