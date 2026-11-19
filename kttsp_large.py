"""
Zgjidhës me blloqe për instancat si Medium: 3 grupe (ekuatorial, polar, retrograd),
secili me domate të mëdha dhe të vogla në të njëjtin plan, të lidhura nga 2 shkallë
inklinimesh. Rendi i blloqeve caktohet nga struktura; brenda një grupi kërkimi është
i lirë me kthim prapa. NUK është testuar nga unë.

Parametrat e algoritmit (funksioni run):
  mode = "earliest"  -> brenda grupit zgjedh domaten që arrihet më herët
  mode = "random"    -> brenda grupit zgjedh domaten në rend të rastësishëm
  B                  -> sa kandidatë provohen në çdo hap (B=1: pa kthim prapa)

Duhet të jetë në dosje me kttsp_solver.py, kttsp_solver5.py dhe util_user.py.

    python kttsp_block.py easy.kttsp small_blocks
    python kttsp_block.py medium.kttsp medium

VERSIONI PER INSTANCA TE MADHE (kttsp_large.py): brenda nje blloku "free" me qindra domate
nuk provohen te gjitha kandidatet ne cdo hap (do te zgjaste dite). Nga kandidatet zgjidhen
vetem K me te afert ne hapesire ne momentin t (llogaritje me numpy, pa pykep); vetem nese
asnje prej tyre nuk ka kalim te lejuar provohen te tjeret.
"""
import sys, time, json
import numpy as np
from kttsp_solver import Solver
from kttsp_solver5 import leg, BIG

HALF = np.pi / 2
MU_MOON = 4.9048695e12


def positions(orb, idx, t, t0):
    """Pozicionet (n,3) te domateve idx ne kohen t [dite]. Elementet: a,e,i,Omega,omega,M0
    (pykep.planet.keplerian merr anomalite mesatare); Kepler i zgjidhur me iteracion."""
    a, e, inc, W, w, M0 = (orb[idx, k] for k in range(6))
    M = M0 + np.sqrt(MU_MOON / a ** 3) * (t - t0) * 86400.0
    E = M.copy()
    for _ in range(8):
        E = M + e * np.sin(E)
    xo, yo = a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E)
    cW, sW, cw, sw, ci, si = np.cos(W), np.sin(W), np.cos(w), np.sin(w), np.cos(inc), np.sin(inc)
    x = (cW * cw - sW * sw * ci) * xo + (-cW * sw - sW * cw * ci) * yo
    y = (sW * cw + cW * sw * ci) * xo + (-sW * sw + cW * cw * ci) * yo
    z = (sw * si) * xo + (cw * si) * yo
    return np.stack([x, y, z], axis=1)


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


def dfs(S, plan, start, time_limit, B=3, mode="earliest", seed=0, K=6):
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

    def try_cands(cands, cur, t, ex_left):
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
        return out

    def moves(cur, t, ex_left):
        cands = candidates()
        if mode == "random":
            rng.shuffle(cands)
        rest = []
        if K and len(cands) > K:
            idx = np.array(cands)
            p_cur = positions(S.orb, np.array([cur]), t, P["t0"])[0]
            d = np.linalg.norm(positions(S.orb, idx, t, P["t0"]) - p_cur, axis=1)
            order = np.argsort(d)
            cands = [int(idx[k]) for k in order[:K]]
            rest = [int(idx[k]) for k in order[K:]]
        out = try_cands(cands, cur, t, ex_left)
        while not out and rest:          # asnje nga K me te afertit nuk shkon -> provo te tjeret
            cands, rest = rest[:K], rest[K:]
            out = try_cands(cands, cur, t, ex_left)
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


def run(path, name, mode="earliest", B=3, time_limit=1800, K=6):
    S = Solver(path)
    print(f"{S.N} tomate, parametra: {S.P}", flush=True)
    print(f"algoritmi: mode={mode}, B={B}, K={K}, kufi kohe={time_limit}s për nisje", flush=True)

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
            r, depth, bp = dfs(S, plan, start, time_limit, B=B, mode=mode, K=K)
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
    import os

    # python kttsp_large.py large.kttsp large [K] [ore]
    if len(sys.argv) < 2:
        print("Perdorimi: python kttsp_large.py <instanca.kttsp> [emri] [K] [ore_per_nisje]")
        sys.exit(1)
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(path))[0] + "_blocks"
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    hours = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
    if not os.path.exists(path):
        print(f"Skedari {path} nuk ekziston në dosjen {os.getcwd()}")
        sys.exit(1)
    print(f"\n===== {path} =====", flush=True)
    run(path, name, mode="earliest", B=1, time_limit=int(hours * 3600), K=K)