
import sys, time, json
import numpy as np
from kttsp_solver import Solver

BIG = 1e7              
ANG_NORMAL = 0.13      
ANG_EXC = 0.50         
ANG_EXC_SIZE = 0.30  


def geometry(orb):
    a, i, Om = orb[:, 0], orb[:, 2], orb[:, 3]
    n = np.stack([np.sin(i) * np.sin(Om), -np.sin(i) * np.cos(Om), np.cos(i)], 1)
    ang = np.arccos(np.clip(n @ n.T, -1, 1))
    np.fill_diagonal(ang, np.inf)
    big = a > BIG
    same = big[:, None] == big[None, :]
    normal_ok = same & (ang <= ANG_NORMAL)
    exc_ok = (same & (ang > ANG_NORMAL) & (ang <= ANG_EXC)) | \
             (~same & (ang <= ANG_EXC_SIZE))
    return normal_ok, exc_ok


def dfs(S, start, normal_ok, exc_ok, time_limit, B=3, BX=3):
    P, N = S.P, S.N
    tofs = np.geomspace(max(P["dtmin"], 0.005), 3.0, 16)
    waits = [np.linspace(0, 2, 101), np.linspace(0, 8, 81)]
    t_end = time.time() + time_limit
    sys.setrecursionlimit(10000)
    path, ts, tl, visited = [start], [], [], {start}
    st = {"best": 1, "res": None}

    def find(cur, t, j, limit):
        for w in waits:
            m = S.best_move(cur, t, [j], limit, w, tofs)
            if m:
                return m
        return None

    def moves(cur, t, ex_left):
        normal = []
        for j in np.where(normal_ok[cur])[0]:
            j = int(j)
            if j in visited:
                continue
            m = find(cur, t, j, P["dv"])
            if m:
                onward = sum(1 for k in np.where(normal_ok[j])[0]
                             if int(k) not in visited and int(k) != j)
                remaining = N - len(visited) - 1
                dead = 1 if (onward == 0 and remaining > 0) else 0
                normal.append(((dead, onward, m[0]), m))
        normal.sort(key=lambda x: x[0])
        out = [m for _, m in normal[:B]]
        if ex_left > 0:
            exc = []
            for j in np.where(exc_ok[cur])[0]:
                j = int(j)
                if j in visited:
                    continue
                m = find(cur, t, j, P["dvex"])
                if m:
                    exc.append((m[0], m))
            exc.sort(key=lambda x: x[0])
            out += [m for _, m in exc[:BX]]
        return out

    def rec(cur, t, ex_left):
        if len(path) > st["best"]:
            st["best"] = len(path)
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
    return st["res"], st["best"]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "easy.kttsp"
    name = sys.argv[2] if len(sys.argv) > 2 else "small"
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}")

    normal_ok, exc_ok = geometry(S.orb)
    deg = normal_ok.sum(axis=1)
    print("fqinjë normalë për tomate (0..N-1):", [int(d) for d in deg])
    print("tomate pa asnjë fqinjë normal:", [int(i) for i in np.where(deg == 0)[0]])

    # nisje: nga skajet e grafit (pak fqinjë), pastaj të tjerat
    starts = [int(i) for i in np.argsort(deg, kind="stable") if deg[i] > 0][:8]
    best, deepest = None, 0
    for s in starts:
        print(f"nisje nga tomata {s}")
        r, depth = dfs(S, s, normal_ok, exc_ok, time_limit=180)
        deepest = max(deepest, depth)
        if r:
            best = r
            print(f"  -> zgjidhje e plotë, f = {r[1]:.3f}")
            break
    if best is None:
        print(f"Asnjë nisje nuk mbaroi (thellësia maksimale {deepest}/{S.N}).")
        sys.exit(1)

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")