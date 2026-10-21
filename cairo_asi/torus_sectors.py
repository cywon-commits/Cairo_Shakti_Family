"""Dimer coverings of the m x n square-lattice torus, resolved by winding parities
(px = parity of # dimers crossing the horizontal seam, py = parity across the vertical seam).
Plain DFS over edges; fine for m*n <= 48."""
import sys
sys.setrecursionlimit(10000)
def sectors(m, n):
    V = m*n; idx = lambda i, j: (i % m)*n + (j % n)
    edges = []                                  # (u, v, crosses_row_seam, crosses_col_seam)
    for i in range(m):
        for j in range(n):
            edges.append((idx(i,j), idx(i+1,j), i == m-1, False))
            edges.append((idx(i,j), idx(i,j+1), False, j == n-1))
    adj = [[] for _ in range(V)]
    for e in edges:
        adj[e[0]].append(e); adj[e[1]].append(e)
    used = [False]*V; cnt = {}
    def dfs(px, py):
        try: u = used.index(False)
        except ValueError:
            cnt[(px, py)] = cnt.get((px, py), 0) + 1; return
        used[u] = True
        for a, b, cr, cc in adj[u]:
            w = b if a == u else a
            if not used[w]:
                used[w] = True; dfs(px ^ cr, py ^ cc); used[w] = False
        used[u] = False
    dfs(0, 0)
    return dict(sorted(cnt.items()))
for m, n in [(4,4),(6,4),(8,4),(6,6)]:
    c = sectors(m, n); print(f"{m}x{n} torus: total {sum(c.values())}, sectors {c}", flush=True)
