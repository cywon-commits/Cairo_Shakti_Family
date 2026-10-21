"""Count dimer coverings (domino tilings) of the square lattice on an m x n torus by brute-force transfer (column profile DP)."""
import sys
from functools import lru_cache
def torus_dimers(m, n):
    # vertices (i,j), i in rows 0..m-1, j in cols 0..n-1, periodic both ways.
    # DP over cells in row-major order with a broken profile; periodicity in the row direction
    # handled by fixing which vertical bonds cross the seam (first row's 'from above' mask).
    total = 0
    for top in range(1 << n):                       # vertical dimers crossing row m-1 -> row 0
        @lru_cache(None)
        def go(pos, mask, rowstart):
            # mask bit j: cell (row, j) already covered from the row above
            if pos == m * n:
                return 1 if mask == top else 0
            i, j = divmod(pos, n)
            if j == 0:
                rowstart = 0                          # bit0 of rowstart: cell (i,0) covered by horizontal wrap
            res = 0
            covered = (mask >> j) & 1
            if j == n - 1 and (rowstart & 1):
                covered_by_wrap = 1
            else:
                covered_by_wrap = 0
            if covered or covered_by_wrap:
                if covered and covered_by_wrap:
                    return 0
                return go(pos + 1, mask & ~(1 << j), rowstart)
            nm = mask
            # vertical down (to row i+1, or row 0 via seam at last row)
            if i < m - 1:
                res += go(pos + 1, mask | (1 << j), rowstart)
            else:
                if (top >> j) & 1:
                    res += go(pos + 1, mask | (1 << j), rowstart)
            # horizontal right
            if j < n - 1:
                if not (mask >> (j + 1)) & 1:
                    res += go(pos + 1, mask | (1 << (j + 1)), rowstart)
            # horizontal wrap: (i,n-1)-(i,0) handled from j==0 side
            if j == 0 and n > 2:
                res += go(pos + 1, mask, rowstart | 1)
            return res
        # first row: cells in top-mask are covered from 'above' (seam)
        total += go(0, top, 0)
        go.cache_clear()
    return total
if __name__ == "__main__":
    for m, n in [(4,4),(2,4),(4,2),(6,4),(4,6),(8,4),(6,6),(8,6)]:
        print(m, n, torus_dimers(m, n), flush=True)
