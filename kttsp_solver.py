"""
Keplerian Tomato TSP - zgjidhës i plotë (greedy me rrjet kohor + përjashtime).
NUK është testuar këtu (pa pykep). Ekzekuto lokalisht:

    python kttsp_solver.py small.kttsp small
    python kttsp_solver.py medium.kttsp medium
    python kttsp_solver.py large.kttsp large

Duhen: util.py (TomatoProblem nga SpOC4 GitHub), pykep 2.x, numpy.
"""
import sys, json
import numpy as np
try:
    from util_user import TomatoProblem
except ImportError:
    from util import TomatoProblem


# ---------------------------------------------------------------- leximi
def read_instance(path):
    P, rows = None, []
    for line in open(path):
        s = line.strip()
        if not s or s[0] == "c":
            continue
        if s[0] == "p":
            p = s.split()
            P = dict(t0=float(p[2]), dtmin=float(p[3]), tmax=float(p[4]),
                     dv=float(p[5]), dvex=float(p[6]), E=int(float(p[7])))
        else:
            rows.append([float(v) for v in s.split()])
    return P, np.array(rows)


# ---------------------------------------------------------------- ndihmës
def neighbour_order(orb, K):
    """Për çdo tomate, K kandidatët më 'të ngjashëm' (heuristikë: a, i, Omega).
    Vetëm për të shpejtuar kërkimin; nuk është kufi i saktë i ΔV."""
    a = orb[:, 0] / 1e6
    i, Om = orb[:, 2], orb[:, 3]
    n = np.stack([np.sin(i) * np.sin(Om), -np.sin(i) * np.cos(Om), np.cos(i)], 1)
    ang = np.arccos(np.clip(n @ n.T, -1, 1))
    d = np.abs(np.log(a[:, None] / a[None, :])) * 2.0 + ang
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :K]


class Solver:
    def __init__(self, path):
        self.P, self.orb = read_instance(path)
        self.N = len(self.orb)
        self.util = TomatoProblem(path)
        self.cache = {}
        self.ncalls = 0

    def dv(self, i, j, t, tof):
        key = (i, j, round(t, 6), round(tof, 6))
        if key in self.cache:
            return self.cache[key]
        self.ncalls += 1
        try:
            v = float(self.util.compute_transfer(i, j, t, tof))
            if not np.isfinite(v):
                v = np.inf
        except Exception:
            v = np.inf
        self.cache[key] = v
        return v

    def best_move(self, cur, t_now, cands, limit, waits, tofs):
        """Transferi me arritje më të hershme drejt ndonjë kandidati, ΔV <= limit."""
        best = None  # (arrival, j, t_dep, tof, dv)
        tmax = self.P["tmax"]
        for j in cands:
            for w in waits:
                t = t_now + w
                if best is not None and t >= best[0]:
                    break
                for tof in tofs:
                    arr = t + tof
                    if arr > tmax or (best is not None and arr >= best[0]):
                        break  # tof rritet -> arritja vetëm vjen e rritet
                    d = self.dv(cur, j, t, tof)
                    if d <= limit:
                        best = (arr, j, t, tof, d)
                        break
        return best

    # ------------------------------------------------------------ greedy
    def greedy(self, start, K=40, verbose=True):
        P, N = self.P, self.N
        near = neighbour_order(self.orb, min(K, N - 1))
        tofs = np.geomspace(max(P["dtmin"], 0.005), 3.0, 14)
        wait_levels = [np.linspace(0, 0.5, 11), np.linspace(0, 3, 25),
                       np.linspace(0, 15, 61)]
        unvis = set(range(N)) - {start}
        path, ts, tof_list = [start], [], []
        t_now, ex_left, ex_used = P["t0"], P["E"], []

        while unvis:
            cur = path[-1]
            move, used_ex = None, False
            # 1) kandidatë të afërt, ΔV normal; pastaj zgjero kërkimin
            for pool in ("near", "all"):
                cands = [j for j in near[cur] if j in unvis] if pool == "near" \
                    else list(unvis)
                for waits in wait_levels:
                    move = self.best_move(cur, t_now, cands, P["dv"], waits, tofs)
                    if move:
                        break
                if move:
                    break
            # 2) përjashtim nëse ngecim
            if move is None and ex_left > 0:
                cands = list(unvis)
                for waits in wait_levels:
                    move = self.best_move(cur, t_now, cands, P["dvex"], waits, tofs)
                    if move:
                        break
                if move:
                    ex_left -= 1
                    used_ex = True
            if move is None:
                if verbose:
                    print(f"  ngecur te {cur}, vizituar {len(path)}/{N}")
                return None
            arr, j, td, tof, d = move
            ts.append(td); tof_list.append(tof); path.append(j)
            unvis.discard(j); t_now = arr
            if used_ex:
                ex_used.append(len(ts))
            if verbose and len(path) % 10 == 0:
                print(f"  {len(path)}/{N}  t={t_now:.3f}  ex_left={ex_left}  "
                      f"thirrje={self.ncalls}")
        return ts + tof_list + path, t_now

    # ------------------------------------------------------------ verifikim
    def verify(self, x):
        P, N = self.P, self.N
        ts, tofs = x[:N - 1], x[N - 1:2 * N - 2]
        path = [int(v) for v in x[2 * N - 2:]]
        ok = sorted(path) == list(range(N)) and len(x) == 3 * N - 2
        ex = 0
        for k in range(N - 1):
            d = self.util.compute_transfer(path[k], path[k + 1], ts[k], tofs[k])
            if d > P["dvex"] + 1e-9:
                ok = False
            elif d > P["dv"] + 1e-9:
                ex += 1
            if tofs[k] < P["dtmin"] - 1e-12 or ts[k] < P["t0"] - 1e-12:
                ok = False
            if k < N - 2 and ts[k] + tofs[k] > ts[k + 1] + 1e-12:
                ok = False
        ok = ok and ex <= P["E"] and ts[-1] + tofs[-1] <= P["tmax"]
        return ok, ex, ts[-1] + tofs[-1]


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "small.kttsp"
    name = sys.argv[2] if len(sys.argv) > 2 else "small"
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}")

    # provo disa pika nisjeje; mbaj zgjidhjen më të mirë
    rng = np.random.default_rng(0)
    starts = rng.choice(S.N, size=min(5, S.N), replace=False)
    best = None
    for s in starts:
        print(f"nisje nga tomata {s}")
        r = S.greedy(int(s))
        if r and (best is None or r[1] < best[1]):
            best = r
            print(f"  -> f = {r[1]:.3f}")
    if best is None:
        print("Asnjë nisje nuk mbaroi: rrit K, wait_levels ose tof-et.")
        sys.exit(1)

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")
    if ok:
        try:
            from submisson_helper import create_submission
            create_submission("spoc-4-keplerian-tomato-traveling-salesperson",
                              name, x, f"submission_{name}.json",
                              f"greedy_{name}", "greedy + time grid")
        except Exception as e:
            print("create_submission nuk u ekzekutua:", e)
