import sys, time, json
import numpy as np
from kttsp_solver import Solver
from kttsp_solver5 import leg, BIG

HALF = np.pi / 2


def build_blocks(orb):
    a, inc = orb[:, 0], orb[:, 2]
    small = a <= BIG
    eq = np.abs(inc) <= 0.045
    po = np.abs(inc - HALF) <= 0.035
    re = np.abs(inc - np.pi) <= 0.035
    idx = np.arange(len(a))

    def pick(mask):
        return [int(k) for k in idx[mask]]

    g = {"eq_small": pick(eq & small), "eq_large": pick(eq & ~small),
         "po_small": pick(po & small), "po_large": pick(po & ~small),
         "re_small": pick(re & small), "re_large": pick(re & ~small)}
    lad = sorted(pick(~(eq | po | re) & ~small), key=lambda k: inc[k])
    ladder1 = [k for k in lad if inc[k] < HALF]
    ladder2 = [k for k in lad if inc[k] > HALF]
    leftover_small = pick(small & ~(eq | po | re))
    return g, ladder1, ladder2, leftover_small


def make_plans(g, l1, l2):
    plan_a = [("free", g["eq_small"]), ("free", g["eq_large"]), ("ordered", l1),
              ("free", g["po_large"]), ("free", g["po_small"]), ("ordered", l2),
              ("free", g["re_large"]), ("free", g["re_small"])]
    plan_b = [("free", g["re_small"]), ("free", g["re_large"]), ("ordered", l2[::-1]),
              ("free", g["po_large"]), ("free", g["po_small"]), ("ordered", l1[::-1]),
              ("free", g["eq_large"]), ("free", g["eq_small"])]
    return {"A (nis nga grupi ekuatorial i vogël)": plan_a,
            "B (nis nga grupi retrograd i vogël)": plan_b}


def dfs(S, plan, start, time_limit, B=3, mode="earliest", seed=0):
    P, N = S.P, S.N
    small = S.orb[:, 0] <= BIG
    rng = np.random.default_rng(seed)
    t_end = time.time() + time_limit
    sys.setrecursionlimit(10000)
    path, ts, tl, visited = [start], [], [], {start}
    st = {"best": 1, "res": None, "bestpath": [start]}

    def candidates():
        for kind, nodes in plan:
            rem = [k for k in nodes if k not in visited]
            if rem:
                return rem if kind == "free" else [rem[0]]
        return []

    def moves(cur, t, ex_left):
        cands = candidates()
        if mode == "random":
            rng.shuffle(cands)
        out = []
        for j in cands:
            cross = bool(small[cur] != small[j])
            if cross and ex_left <= 0:
                continue
            m = leg(S, cur, j, t, P["dvex"] if cross else P["dv"])
            if m:
                out.append(m)
                if mode == "random" and len(out) >= B:
                    break
        if mode != "random":
            out.sort(key=lambda m: m[0])
        return out[:B]

    def rec(cur, t, ex_left):
        if len(path) > st["best"]:
            st["best"] = len(path)
            st["bestpath"] = list(path)
            if len(path) % 5 == 0 or len(path) == N:
                print(f"    thellësia: {len(path)}/{N}  t={t:.1f}  ex_left={ex_left}",
                      flush=True)
        if len(path) == N:
            st["res"] = (list(ts) + list(tl) + list(path), t)
            return True
        if time.time() > t_end:
            return False
        for arr, j, td, tof, d in moves(cur, t, ex_left):
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


def run(path, name, mode="earliest", B=3, time_limit=1800):
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}", flush=True)
    print(f"algoritmi: mode={mode}, B={B}, kufi kohe={time_limit}s për nisje", flush=True)

    g, l1, l2, leftover = build_blocks(S.orb)
    print("blloqet:", {k: len(v) for k, v in g.items()},
          "| shkalla 1:", len(l1), "| shkalla 2:", len(l2), flush=True)
    total = sum(len(v) for v in g.values()) + len(l1) + len(l2)
    if leftover or total != S.N:
        print("PARALAJMËRIM: disa domate nuk hynë në asnjë bllok:",
              leftover, f"({total} nga {S.N})", flush=True)

    plans = make_plans(g, l1, l2)
    best, deepest, deepest_path, deepest_plan = None, 0, [], ""
    for pname, plan in plans.items():
        for start in plan[0][1][:2]:
            print(f"plani {pname}, nisje nga tomata {start}", flush=True)
            r, depth, bp = dfs(S, plan, start, time_limit, B=B, mode=mode)
            if depth > deepest:
                deepest, deepest_path, deepest_plan = depth, bp, pname
            if r:
                best = r
                print(f"  -> zgjidhje e plotë, f = {r[1]:.3f}", flush=True)
                break
        if best:
            break

    if best is None:
        print(f"Asnjë plan nuk mbaroi (thellësia maksimale {deepest}/{S.N}, plani {deepest_plan}).")
        print("rruga më e thellë:", deepest_path)
        return None

    x, f = best
    ok, ex, f2 = S.verify(x)
    print(f"VERIFIKIM: valid={ok}, përjashtime={ex}, f={f2:.3f}")
    print("rendi:", [int(v) for v in x[-S.N:]])
    json.dump({"x": [float(v) for v in x[:-S.N]] + [int(v) for v in x[-S.N:]],
               "f": f2}, open(f"x_{name}.json", "w"))
    print(f"ruajtur x_{name}.json")
    return f2


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "medium.kttsp"
    name = sys.argv[2] if len(sys.argv) > 2 else "medium"
    run(path, name, mode="earliest", B=3, time_limit=1800)