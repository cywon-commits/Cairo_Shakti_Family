"""
Cairo pentagonal artificial spin ice (Ising dipoles on the edges).

Geometry (p4g, lattice constant L = 1):
  z=4 vertices : A = (0,0),  B = (1/2,1/2)
  z=3 vertices : pair at (1/2,0):  C1 = (1/2+d,  d),  C2 = (1/2-d, -d)
                 pair at (0,1/2):  C3 = (-d, 1/2+d),  C4 = ( d, 1/2-d)
  Each z=3 vertex connects to its partner, one A and one B.
  d_eq = (sqrt(7)-1)/12 ~ 0.1371 gives the equilateral Cairo tiling.
  Per unit cell: 6 vertices, 10 edges (= spins), 4 pentagons.

Energy (point dipoles at edge midpoints, unit moments along edges):
  E = D * sum_{i<j} [ m_i.m_j - 3 (m_i.r)(m_j.r) ] / r^3 ,   (r in units of L)
  Same convention as Kwon et al. Sci. Rep. 12, 15026 (2022), Eq. (1).
"""
import numpy as np
from numba import njit

D_EQ = (np.sqrt(7) - 1) / 12


# ----------------------------------------------------------------------------
# 1. Lattice
# ----------------------------------------------------------------------------
class CairoLattice:
    def __init__(self, nx, ny, d=D_EQ):
        self.nx, self.ny, self.d = nx, ny, d
        self.box = np.array([nx, ny], float)
        self._build_vertices()
        self._build_edges()
        self._build_faces()

    # --- vertices ---------------------------------------------------------
    def _build_vertices(self):
        d = self.d
        basis = np.array([[0, 0], [0.5, 0.5],                  # A, B  (z=4)
                          [0.5 + d, d], [0.5 - d, -d],          # C1, C2 (z=3)
                          [-d, 0.5 + d], [d, 0.5 - d]])         # C3, C4 (z=3)
        self.vtype = []
        pos, idx = [], {}
        for i in range(self.nx):
            for j in range(self.ny):
                for b, p in enumerate(basis):
                    idx[(i, j, b)] = len(pos)
                    pos.append(p + [i, j])
                    self.vtype.append(4 if b < 2 else 3)
        self.vpos = np.array(pos)
        self.vtype = np.array(self.vtype)
        self._vidx = idx

    def _v(self, i, j, b):
        return self._vidx[(i % self.nx, j % self.ny, b)]

    def mic(self, dr):
        """minimum-image displacement"""
        return dr - self.box * np.round(dr / self.box)

    # --- edges (Ising spins) -------------------------------------------------
    def _build_edges(self):
        # (b1, di, dj, b2): edge from basis b1 in cell (i,j) to b2 in (i+di, j+dj)
        # z3 -> z4 bonds oriented from z3 to z4 (reference direction, sigma=+1)
        rules = [
            (2, 0, 0, 3),                    # C1-C2 partner bond
            (4, 0, 0, 5),                    # C3-C4 partner bond
            (2, 1, 0, 0), (2, 0, 0, 1),      # C1 -> A(1,0), B(1/2,1/2)
            (3, 0, 0, 0), (3, 0, -1, 1),     # C2 -> A(0,0), B(1/2,-1/2)
            (4, 0, 1, 0), (4, -1, 0, 1),     # C3 -> A(0,1), B(-1/2,1/2)
            (5, 0, 0, 0), (5, 0, 0, 1),      # C4 -> A(0,0), B(1/2,1/2)
        ]
        e = []
        for i in range(self.nx):
            for j in range(self.ny):
                for b1, di, dj, b2 in rules:
                    e.append((self._v(i, j, b1), self._v(i + di, j + dj, b2)))
        self.edges = np.array(e)
        u, v = self.edges[:, 0], self.edges[:, 1]
        vec = self.mic(self.vpos[v] - self.vpos[u])
        self.elen = np.linalg.norm(vec, axis=1)
        self.edir = vec / self.elen[:, None]                   # unit moment for sigma=+1
        self.epos = self.vpos[u] + 0.5 * vec                   # dipole position
        self.N = len(e)
        # vertex -> incident edges with sign (+1 if edge points INTO vertex for sigma=+1)
        inc = [[] for _ in range(len(self.vpos))]
        for k, (a, b) in enumerate(self.edges):
            inc[a].append((k, -1))
            inc[b].append((k, +1))
        self.incidence = inc

    # --- faces (pentagons) via planar face traversal ------------------------
    def _build_faces(self):
        nv = len(self.vpos)
        nbrs = [[] for _ in range(nv)]
        for k, (a, b) in enumerate(self.edges):
            nbrs[a].append((b, k))
            nbrs[b].append((a, k))
        # sort neighbours counter-clockwise
        order = []
        for a in range(nv):
            ang = [np.arctan2(*self.mic(self.vpos[b] - self.vpos[a])[::-1]) for b, _ in nbrs[a]]
            order.append([nbrs[a][t] for t in np.argsort(ang)])
        seen, faces = set(), []
        for a in range(nv):
            for b, _ in order[a]:
                if (a, b) in seen:
                    continue
                face, u, w = [], a, b
                while (u, w) not in seen:
                    seen.add((u, w))
                    face.append(u)
                    lst = [x for x, _ in order[w]]
                    p = lst.index(u)
                    u, w = w, lst[(p - 1) % len(lst)]    # next edge clockwise from arrival -> CCW face
                faces.append(face)
        # keep bounded faces with positive (CCW) signed area
        good = []
        for f in faces:
            pts = [self.vpos[f[0]]]
            for t in range(1, len(f)):
                pts.append(pts[-1] + self.mic(self.vpos[f[t]] - self.vpos[f[t - 1]]))
            pts = np.array(pts)
            area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            if area > 0:
                good.append((f, pts))
        self.faces = [f for f, _ in good]
        self.face_pts = [p for _, p in good]
        # CCW orientation of each face edge: +1 if reference edge dir follows CCW
        emap = {}
        for k, (a, b) in enumerate(self.edges):
            emap[(a, b)] = (k, +1)
            emap[(b, a)] = (k, -1)
        self.face_edges = []
        for f in self.faces:
            self.face_edges.append([emap[(f[t], f[(t + 1) % len(f)])] for t in range(len(f))])
        self.face_center = np.array([p.mean(0) for p in self.face_pts])

    # --- interactions ----------------------------------------------------------
    def coupling_matrix(self, rc=None):
        """J_ij such that E = sum_{i<j} J_ij s_i s_j, in units of D/a^3 (a = edge length).
        rc=None  -> vertex-local: only edges sharing a vertex (scheme of Kwon et al. 2022).
        rc=float -> full dipolar sum over all periodic images with |r| < rc (in units of a).
        Self-image terms are constants (s_i^2 = 1) and are dropped."""
        N, m = self.N, self.edir
        a = self.elen.mean()
        J = np.zeros((N, N))
        if rc is None:
            for inc in self.incidence:
                ks = [k for k, _ in inc]
                for x in range(len(ks)):
                    for y in range(x + 1, len(ks)):
                        i, k = ks[x], ks[y]
                        r = self.mic(self.epos[k] - self.epos[i]) / a
                        dd = np.linalg.norm(r)
                        v = (m[i] @ m[k] - 3 * (m[i] @ r) * (m[k] @ r) / dd**2) / dd**3
                        J[i, k] = J[k, i] = v
            return J
        R = rc * a
        nimx, nimy = int(np.ceil(R / self.nx)) + 1, int(np.ceil(R / self.ny)) + 1
        dpos = self.epos[None, :, :] - self.epos[:, None, :]          # (N,N,2)
        mm = m @ m.T
        for ix in range(-nimx, nimx + 1):
            for iy in range(-nimy, nimy + 1):
                r = (dpos + np.array([ix * self.nx, iy * self.ny])) / a
                dd = np.linalg.norm(r, axis=2)
                mask = (dd < rc) & (dd > 1e-9)
                if not mask.any():
                    continue
                mr_i = np.einsum('id,ijd->ij', m, r)
                mr_j = np.einsum('jd,ijd->ij', m, r)
                with np.errstate(divide='ignore', invalid='ignore'):
                    v = (mm - 3 * mr_i * mr_j / dd**2) / dd**3
                J += np.where(mask, v, 0.0)
        np.fill_diagonal(J, 0.0)
        return J

    # --- observables --------------------------------------------------------------
    def charges(self, s):
        """q = (#in - #out) at each vertex"""
        return np.array([sum(sg * s[k] for k, sg in inc) for inc in self.incidence])

    def chirality(self, s):
        """circulation around each pentagon, normalised to [-1,1]; +1 = CCW"""
        return np.array([sum(o * s[k] for k, o in fe) / len(fe) for fe in self.face_edges])

    def vertex_stats(self, s):
        q = self.charges(s)
        out = {}
        for z in (3, 4):
            qq = q[self.vtype == z]
            vals, cnt = np.unique(qq, return_counts=True)
            out[z] = {int(v): c / len(qq) for v, c in zip(vals, cnt)}
        return out


# ----------------------------------------------------------------------------
# 2. Monte Carlo (numba)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _energy(J, s):
    return 0.5 * s @ (J @ s)


@njit(cache=True)
def anneal(J, s, T0, T1, nsweep, seed):
    """Metropolis simulated annealing with linear T schedule. Returns final spins and energy trace."""
    np.random.seed(seed)
    N = s.size
    h = J @ s
    trace = np.empty(nsweep)
    for t in range(nsweep):
        T = T0 + (T1 - T0) * t / max(nsweep - 1, 1)
        beta = 1.0 / max(T, 1e-9)
        for _ in range(N):
            i = np.random.randint(N)
            dE = -2.0 * s[i] * h[i]
            if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                s[i] = -s[i]
                h += 2.0 * s[i] * J[:, i]
        trace[t] = 0.5 * s @ h
    return s, trace


@njit(cache=True)
def quench(J, s):
    """zero-T greedy relaxation to a local minimum"""
    h = J @ s
    changed = True
    while changed:
        changed = False
        for i in range(s.size):
            if s[i] * h[i] > 1e-12:          # flipping lowers E
                s[i] = -s[i]
                h += 2.0 * s[i] * J[:, i]
                changed = True
    return s


def exact_ground_state(J):
    """Brute force for small N (<= ~22)."""
    N = J.shape[0]
    best, bs = np.inf, None
    for c in range(1 << (N - 1)):              # fix s_0 (time reversal)
        s = np.array([1.0 - 2.0 * ((c >> k) & 1) for k in range(N)])
        e = 0.5 * s @ J @ s
        if e < best - 1e-12:
            best, bs = e, s.copy()
    return best, bs


@njit(cache=True)
def exact_ground_state_fast(J):
    """Numba brute force, Gray-code enumeration, N <= ~30."""
    N = J.shape[0]
    s = np.ones(N)
    h = J @ s
    E = 0.5 * s @ h
    best, bs = E, s.copy()
    for c in range(1, 1 << (N - 1)):
        i = 0
        while not (c >> i) & 1:
            i += 1
        i += 1                       # never flip spin 0 (time reversal)
        E += -2.0 * s[i] * h[i]
        s[i] = -s[i]
        h += 2.0 * s[i] * J[:, i]
        if E < best - 1e-12:
            best, bs = E, s.copy()
    return best, bs
