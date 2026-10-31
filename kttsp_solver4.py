
import sys, time, json
import numpy as np
from kttsp_solver import Solver

BIG = 1e7
ANG_NORMAL = 0.13
ANG_SKIP = 0.50        
ANG_SIZE = 0.30        
TOF_GRID = np.geomspace(0.02, 3.0, 30)


def geometry(orb):
    a, i, Om = orb[:, 0], orb[:, 2], orb[:, 3]
    n = np.stack([np.sin(i) * np.sin(Om), -np.sin(i) * np.cos(Om), np.cos(i)], 1)
    ang = np.arccos(np.clip(n @ n.T, -1, 1))
    np.fill_diagonal(ang, np.inf)
    same = (a > BIG)[:, None] == (a > BIG)[None, :]
    normal_ok = same & (ang <= ANG_NORMAL)
    skip_ok = same & (ang > ANG_NORMAL) & (ang <= ANG_SKIP)
    size_ok = (~same) & (ang <= ANG_SIZE)
    return normal_ok, skip_ok, size_ok


def leg(S, cur, j, t_now, limit):
    
    P = S.P
    for W in (2.0, 6.0):
        t_grid = t_now + np.arange(0.0, W + 1e-9, 0.05)
        D = np.full((len(t_grid), len(TOF_GRID)), np.inf)
        for a, t in enumerate(t_grid):
            for b, tof in enumerate(TOF_GRID):
                if t + tof <= P["tmax"]:
                    D[a, b] = S.dv(cur, j, float(t), float(tof))
        idx = np.argwhere(D <= limit * 1.6)
        if len(idx) == 0:
            continue
        cells = sorted(idx.tolist(), key=lambda ab: t_grid[ab[0]] + TOF_GRID[ab[1]])[:8]
        low = np.argsort(D, axis=None)[:4]
        cells += [list(np.unravel_index(k, D.shape)) for k in low]
        best = None
        for a, b in cells:
            t0, tof0 = t_grid[a], TOF_GRID[b]
            for dt in np.arange(-0.05, 0.0501, 0.01):
                for f in np.linspace(0.85, 1.15, 11):
                    t, tof = t0 + dt, tof0 * f
                    if t < t_now or tof < P["dtmin"] or t + tof > P["tmax"]:
                        continue
                    d = S.dv(cur, j, float(t), float(tof))
                    if d <= limit and (best is None or t + tof < best[0]):
                        best = (t + tof, j, float(t), float(tof), d)
        if best is not None:
            return best
    return None


def dfs(S, start, normal_ok, skip_ok, size_ok, time_limit, B=3, BX=2):
    P, N = S.P, S.N
    t_end = time.time() + time_limit
    sys.setrecursionlimit(10000)
    path, ts, tl, visited = [start], [], [], {start}
    st = {"best": 1, "res": None, "bestpath": [start]}

    def moves(cur, t, ex_left):
        normal = []
        for j in np.where(normal_ok[cur])[0]:
            j = int(j)
            if j in visited:
                continue
            m = leg(S, cur, j, t, P["dv"])
            if m:
                onward = sum(1 for k in np.where(normal_ok[j])[0]
                             if int(k) not in visited and int(k) != j)
                remaining = N - len(visited) - 1
                dead = 1 if (onward == 0 and remaining > 0) else 0
                normal.append(((dead, onward, m[0]), m))
        normal.sort(key=lambda x: x[0])
        out = [m for _, m in normal[:B]]
        if ex_left > 0:
            pools = [size_ok[cur]]
            if not normal:            
                pools.append(skip_ok[cur])
            exc = []
            for pool in pools:
                for j in np.where(pool)[0]:
                    j = int(j)
                    if j in visited:
                        continue
                    m = leg(S, cur, j, t, P["dvex"])
                    if m:
                        exc.append((m[0], m))
            exc.sort(key=lambda x: x[0])
            out += [m for _, m in exc[:BX]]
        return out

    def rec(cur, t, ex_left):
        if len(path) > st["best"]:
            st["best"] = len(path)
            st["bestpath"] = list(path)
            print(f"    thellësia: {len(path)}/{N}  t={t:.2f}  ex_left={ex_left}")
        if len(path) == N:
            st["res"] = (list(ts) + list(tl) + list(path), t)
            return True
        if time.time() > t_end:
            return False
        for m in moves(cur, t, ex_left):
            arr, j, td, tof, d = m
            used = 1 if d > P["dv"] else 0
            path.append(j); visited.add(j); ts.append(td); tl.append(tof)
            if rec(j, arr, ex_left - used):
                return True
            path.pop(); visited.discard(j); ts.pop(); tl.pop()
            if time.time() > t_end:
                return False
        return False

    rec(start, P["t0"], P["E"])
    return st["res"], st["best"], st["bestpath"]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "easy.kttsp"
    name = sys.argv[2] if len(sys.argv) > 2 else "small"
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}")
    normal_ok, skip_ok, size_ok = geometry(S.orb)

    inc = S.orb[:, 2]
    starts = list(dict.fromkeys([int(k) for k in np.argsort(inc)[:3]] +
                                [int(k) for k in np.argsort(-inc)[:3]]))
    print("nisjet që do provohen (skajet e shkallës):", starts)

    best, deepest, deepest_path = None, 0, []
    for s in starts:
        print(f"nisje nga tomata {s}")
        r, depth, bp = dfs(S, s, normal_ok, skip_ok, size_ok, time_limit=240)
        if depth > deepest:
            deepest, deepest_path = depth, bp
        if r:
            best = r
            print(f"  -> zgjidhje e plotë, f = {r[1]:.3f}")
            break
    if best is None:
        print(f"Asnjë nisje nuk mbaroi (thellësia maksimale {deepest}/{S.N}).")
        print("rruga më e thellë:", deepest_path)
        sys.exit(1)

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")