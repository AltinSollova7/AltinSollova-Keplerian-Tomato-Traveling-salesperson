
import sys, os, time, json
import numpy as np
from kttsp_solver import Solver

SLACK_NORMAL = 150.0     
SLACK_EXC = 700.0       



def static_matrix(S):
    
    N = S.N
    fn = f"mindv_{N}.npy"
    if os.path.exists(fn):
        return np.load(fn)
    M = np.full((N, N), np.inf)
    ts = np.arange(0.0, 12.01, 0.5)
    tofs = np.geomspace(0.01, 2.0, 6)
    t0 = time.time()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            m = np.inf
            for t in ts:
                for tof in tofs:
                    d = S.dv(i, j, float(t), float(tof))
                    if d < m:
                        m = d
            M[i, j] = m
        print(f"  graf statik: {i + 1}/{N}  ({time.time() - t0:.0f} s)")
    np.save(fn, M)
    return M



def dfs_from(S, M, start, time_limit, B=3):
    P, N = S.P, S.N
    tofs = np.geomspace(max(P["dtmin"], 0.005), 3.0, 14)
    waits = [np.linspace(0, 1, 11), np.linspace(0, 6, 25)]
    ok_deg = M <= SLACK_NORMAL
    t_end = time.time() + time_limit
    sys.setrecursionlimit(10000)

    path, ts, tl, visited = [start], [], [], {start}
    state = {"best_len": 1, "result": None, "nodes": 0}

    def find(cur, t, j, limit):
        for w in waits:
            m = S.best_move(cur, t, [j], limit, w, tofs)
            if m:
                return m
        return None

    def moves(cur, t, ex_left):
        normal, exc = [], []
        for j in range(N):
            if j in visited:
                continue
            if M[cur, j] <= SLACK_NORMAL:
                m = find(cur, t, j, P["dv"])
                if m:
                   
                    onward = sum(1 for k in range(N)
                                 if k not in visited and k != j and ok_deg[j, k])
                    normal.append((onward, m[0], m, False))
            elif ex_left > 0 and M[cur, j] <= SLACK_EXC:
                m = find(cur, t, j, P["dvex"])
                if m:
                    exc.append((0, m[0], m, True))
        normal.sort(key=lambda x: (x[0], x[1]))
        exc.sort(key=lambda x: x[1])
        return normal[:B] + exc[:2]

    def rec(cur, t, ex_left):
        state["nodes"] += 1
        if len(path) > state["best_len"]:
            state["best_len"] = len(path)
            print(f"    thellësia më e mirë: {len(path)}/{N}  t={t:.2f}  "
                  f"ex_left={ex_left}")
        if len(path) == N:
            state["result"] = (list(ts) + list(tl) + list(path), t)
            return True
        if time.time() > t_end:
            return False
        for _, _, m, is_ex in moves(cur, t, ex_left):
            arr, j, td, tof, _ = m
            path.append(j); visited.add(j); ts.append(td); tl.append(tof)
            if rec(j, arr, ex_left - (1 if is_ex else 0)):
                return True
            path.pop(); visited.discard(j); ts.pop(); tl.pop()
            if time.time() > t_end:
                return False
        return False

    rec(start, S.P["t0"], P["E"])
    return state["result"]



if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "easy.kttsp"
    name = sys.argv[2] if len(sys.argv) > 2 else "small"
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}")

    print("Po ndërtoj grafin statik (një herë; ruhet në mindv_N.npy)...")
    M = static_matrix(S)
    print("çifte ΔV<=100:", int((M <= 100).sum()),
          "| <=600:", int((M <= 600).sum()))

    indeg = (M <= SLACK_NORMAL).sum(axis=0)
    outdeg = (M <= SLACK_NORMAL).sum(axis=1)
    print("tomate pa asnjë hyrje të lirë:", [int(i) for i in np.where(indeg == 0)[0]])
    print("tomate pa asnjë dalje të lirë:", [int(i) for i in np.where(outdeg == 0)[0]])

   
    starts = [int(i) for i in np.argsort(indeg)][:10]
    best = None
    for s in starts:
        print(f"nisje nga tomata {s}")
        r = dfs_from(S, M, s, time_limit=90)
        if r and (best is None or r[1] < best[1]):
            best = r
            print(f"  -> zgjidhje e plotë, f = {r[1]:.3f}")
            break  
    if best is None:
        print("Asnjë nisje nuk mbaroi. Ma ngjit daljen këtu.")
        sys.exit(1)

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")