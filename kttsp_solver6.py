
import sys, time, json
import numpy as np
from kttsp_solver import Solver
from kttsp_solver5 import geometry, leg, BIG


def small_clusters(orb, normal_ok):
    """Grupet e orbitave të vogla (komponente të lidhura sipas planit)."""
    cid, c = {}, 0
    for s in np.where(orb[:, 0] <= BIG)[0]:
        s = int(s)
        if s in cid:
            continue
        cid[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for v in np.where(normal_ok[u])[0]:
                v = int(v)
                if v not in cid and orb[v, 0] <= BIG:
                    cid[v] = c
                    stack.append(v)
        c += 1
    return cid


def dfs(S, start, normal_ok, skip_ok, size_ok, cid, time_limit, B=2, BX=2):
    P, N = S.P, S.N
    is_small = S.orb[:, 0] <= BIG
    large_list = [k for k in range(N) if not is_small[k]]
    t_end = time.time() + time_limit
    sys.setrecursionlimit(10000)
    path, ts, tl, visited = [start], [], [], {start}
    st = {"best": 1, "res": None, "bestpath": [start]}

    def can_dip(ex_left, c):
        others = {cid[k] for k in cid if k not in visited and cid[k] != c}
        unv_large = any(k not in visited for k in large_list)
        leave = 1 if (unv_large or others) else 0
        return ex_left >= 1 + leave + len(others)

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
        normal = [m for _, m in normal[:B]]

        dips, exits, skips = [], [], []
        if ex_left > 0:
            for j in np.where(size_ok[cur])[0]:
                j = int(j)
                if j in visited:
                    continue
                if not is_small[cur] and not can_dip(ex_left, cid[j]):
                    continue
                m = leg(S, cur, j, t, P["dvex"])
                if m:
                    (exits if is_small[cur] else dips).append((m[0], m))
            if not normal and not dips:
                for j in np.where(skip_ok[cur])[0]:
                    j = int(j)
                    if j in visited:
                        continue
                    m = leg(S, cur, j, t, P["dvex"])
                    if m:
                        skips.append((m[0], m))
        for lst in (dips, exits, skips):
            lst.sort(key=lambda x: x[0])
        dips = [m for _, m in dips[:BX]]
        exits = [m for _, m in exits[:BX]]
        skips = [m for _, m in skips[:BX]]
        return dips + normal + exits + skips      

    def rec(cur, t, ex_left):
        if len(path) > st["best"]:
            st["best"] = len(path)
            st["bestpath"] = list(path)
            print(f"    thellësia: {len(path)}/{N}  t={t:.2f}  ex_left={ex_left}",
                  flush=True)
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
    print(f"{S.N} tomate, parametra: {S.P}", flush=True)
    normal_ok, skip_ok, size_ok = geometry(S.orb)
    cid = small_clusters(S.orb, normal_ok)
    groups = {}
    for k, c in cid.items():
        groups.setdefault(c, []).append(k)
    print("grupet e orbitave të vogla:", [sorted(v) for v in groups.values()], flush=True)

    inc = S.orb[:, 2]
    order_i = np.argsort(inc)
    starts = list(dict.fromkeys([int(order_i[-1]), int(order_i[0]),
                                 int(order_i[-2]), int(order_i[1])]))
    print("nisjet:", starts, flush=True)

    best, deepest, deepest_path = None, 0, []
    for s in starts:
        print(f"nisje nga tomata {s}", flush=True)
        r, depth, bp = dfs(S, s, normal_ok, skip_ok, size_ok, cid, time_limit=1500)
        if depth > deepest:
            deepest, deepest_path = depth, bp
        if r:
            best = r
            print(f"  -> zgjidhje e plotë, f = {r[1]:.3f}", flush=True)
            break
    if best is None:
        print(f"Asnjë nisje nuk mbaroi (thellësia maksimale {deepest}/{S.N}).")
        print("rruga më e thellë:", deepest_path)
        sys.exit(1)

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    print("rendi:", [int(v) for v in x[-S.N:]])
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")