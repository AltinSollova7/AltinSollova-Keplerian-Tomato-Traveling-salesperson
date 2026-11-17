
import sys, json, time
import numpy as np


def dp_segment(legfn, cur0, t_in, seq, nxt, exit_fn, window=2, log=None):
    """
    legfn(cur, j, t) -> (arr, j, t_dep, tof, dv) ose None
    exit_fn(cur, t) -> arritja te nxt ose None (nëse nxt është None nuk thirret)
    Kthen (rendi i ri, koha e mbërritjes te nxt ose te domatja e fundit, kalimet).
    """
    pos = {n: i for i, n in enumerate(seq)}
    layers = [{(frozenset(), cur0): (t_in, None, None)}]
    for step in range(len(seq)):
        new = {}
        for (vis, cur), (arr, _, _) in layers[-1].items():
            low = min(pos[n] for n in seq if n not in vis)
            for j in seq:
                if j in vis or pos[j] > low + window:
                    continue
                m = legfn(cur, j, arr)
                if m is None:
                    continue
                key = (vis | {j}, j)
                if key not in new or m[0] < new[key][0]:
                    new[key] = (m[0], (vis, cur), m)
        if not new:
            return None
        layers.append(new)
        if log:
            log(step + 1, len(seq), len(new), min(v[0] for v in new.values()))

    best = None
    for key, (arr, _, _) in layers[-1].items():
        if nxt is None:
            total = arr
        else:
            total = exit_fn(key[1], arr)
            if total is None:
                continue
        if best is None or total < best[0]:
            best = (total, key)
    if best is None:
        return None

    path, legs, key = [], [], best[1]
    for L in range(len(seq), 0, -1):
        arr, parent, m = layers[L][key]
        path.append(key[1])
        legs.append(m)
        key = parent
    path.reverse()
    legs.reverse()
    return path, best[0], legs


# ---------------------------------------------------------------- pjesa me pykep
TOF_FINE = np.geomspace(0.1, 16.0, 100)
STEP = 0.02


def fine_leg(S, cur, j, t_now, limit, hint):
    P = S.P
    best = None
    t_h, tof_h = hint
    if t_h >= t_now - 1e-9 and t_h + tof_h <= P["tmax"]:
        d = S.dv(cur, j, float(t_h), float(tof_h))
        if d <= limit:
            best = (t_h + tof_h, float(t_h), float(tof_h), d)
    for W in (2.0, 4.0):
        t_grid = t_now + np.arange(0.0, W + 1e-9, STEP)
        D = np.full((len(t_grid), len(TOF_FINE)), np.inf)
        for a, t in enumerate(t_grid):
            for b, tof in enumerate(TOF_FINE):
                if t + tof <= P["tmax"] and (best is None or t + tof < best[0]):
                    D[a, b] = S.dv(cur, j, float(t), float(tof))
        cells = []
        idx = np.argwhere(D <= limit * 1.2)
        if len(idx):
            cells = sorted(idx.tolist(),
                           key=lambda ab: t_grid[ab[0]] + TOF_FINE[ab[1]])[:12]
        for k in np.argsort(D, axis=None)[:6]:
            if np.isfinite(D.flat[k]):
                cells.append(list(np.unravel_index(k, D.shape)))
        for a, b in cells:
            t0, tof0 = t_grid[a], TOF_FINE[b]
            for dt in np.arange(-STEP, STEP + 1e-9, 0.005):
                for f in np.linspace(0.95, 1.05, 11):
                    t, tof = t0 + dt, tof0 * f
                    if t < t_now or tof < P["dtmin"] or t + tof > P["tmax"]:
                        continue
                    if best is not None and t + tof >= best[0]:
                        continue
                    d = S.dv(cur, j, float(t), float(tof))
                    if d <= limit:
                        best = (t + tof, float(t), float(tof), d)
    return best


def schedule(S, order, hints):
    """Orari me arritje më të hershme për një rend të dhënë."""
    P = S.P
    small = S.orb[:, 0] <= 1e7
    t_now, legs = P["t0"], []
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        limit = P["dvex"] if small[a] != small[b] else P["dv"]
        h = hints.get((a, b), (1e9, 1.0))
        r = fine_leg(S, a, b, t_now, limit, h)
        if r is None:
            raise RuntimeError(f"nuk gjendet kalim {a}->{b} pas t={t_now:.2f}")
        legs.append((r[1], r[2]))
        t_now = r[0]
    return legs, t_now


if __name__ == "__main__":
    from kttsp_solver import Solver
    from kttsp_solver5 import leg, BIG
    from kttsp_block import build_blocks

    inst = sys.argv[1] if len(sys.argv) > 1 else "medium.kttsp"
    solfile = sys.argv[2] if len(sys.argv) > 2 else "x_medium.json"
    outfile = sys.argv[3] if len(sys.argv) > 3 else "x_medium_ladder.json"

    S = Solver(inst)
    N, P = S.N, S.P
    small = S.orb[:, 0] <= BIG
    x = json.load(open(solfile))["x"]
    ts, tofs = x[:N - 1], x[N - 1:2 * N - 2]
    order = [int(v) for v in x[2 * N - 2:]]
    f_old = ts[-1] + tofs[-1]
    print(f"f fillestar = {f_old:.3f} d", flush=True)

    _, l1, l2, _ = build_blocks(S.orb)
    hints = {(order[k], order[k + 1]): (ts[k], tofs[k]) for k in range(N - 1)}
    arrivals = [P["t0"]] + [ts[k] + tofs[k] for k in range(N - 1)]  # arritja te order[k]

    only = sys.argv[4].split(",") if len(sys.argv) > 4 else ["shkalla1", "shkalla2"]
    for lname, ladder in (("shkalla1", l1), ("shkalla2", l2)):
        if lname not in only:
            continue
        L = set(ladder)
        positions = [i for i, n in enumerate(order) if n in L]
        if not positions or positions[-1] - positions[0] + 1 != len(L) or positions[0] == 0:
            print(f"{lname}: nuk është segment i vazhdueshëm, e anashkaloj", flush=True)
            continue
        p0, p1 = positions[0], positions[-1]
        seq = order[p0:p1 + 1]
        cur0, t_in = order[p0 - 1], arrivals[p0 - 1]
        nxt = order[p1 + 1] if p1 + 1 < N else None
        print(f"\n{lname}: {len(seq)} domate, hyrja nga {cur0} në t={t_in:.2f}", flush=True)

        def legfn(cur, j, t):
            lim = P["dvex"] if small[cur] != small[j] else P["dv"]
            return leg(S, cur, j, t, lim)

        def exit_fn(cur, t):
            lim = P["dvex"] if small[cur] != small[nxt] else P["dv"]
            m = leg(S, cur, nxt, t, lim)
            return None if m is None else m[0]

        def log(step, total, nstates, best_t):
            print(f"  shtresa {step}/{total}: {nstates} gjendje, koha më e mirë {best_t:.2f}",
                  flush=True)

        res = dp_segment(legfn, cur0, t_in, seq, nxt, exit_fn, window=2, log=log)
        if res is None:
            print(f"{lname}: DP nuk gjeti rrugë, e lë siç është", flush=True)
            continue
        path, total, dp_legs = res
        # kalimet që DP gjeti i japim orarit si "hint", që orari të mos dalë më i ngadaltë se DP
        chain = [cur0] + path
        for k, m in enumerate(dp_legs):
            hints[(chain[k], chain[k + 1])] = (m[2], m[3])
        if nxt is not None:
            m_exit = leg(S, path[-1], nxt, dp_legs[-1][0],
                         P["dvex"] if small[path[-1]] != small[nxt] else P["dv"])
            if m_exit is not None:
                hints[(path[-1], nxt)] = (m_exit[2], m_exit[3])
        print(f"{lname}: rendi i ri {path}\n  (i vjetër: {seq})", flush=True)
        new_order = order[:p0] + path + order[p1 + 1:]
        t0 = time.time()
        legs, f_new = schedule(S, new_order, hints)
        print(f"  pas rillogaritjes së orarit: f = {f_new:.3f} d ({time.time() - t0:.0f} s)",
              flush=True)
        if f_new < f_old - 1e-9:
            order, f_old = new_order, f_new
            ts = [t for t, _ in legs]
            tofs = [tof for _, tof in legs]
            hints = {(order[k], order[k + 1]): (ts[k], tofs[k]) for k in range(N - 1)}
            arrivals = [P["t0"]] + [ts[k] + tofs[k] for k in range(N - 1)]
            print(f"  pranuar, f tani = {f_old:.3f} d", flush=True)
        else:
            print("  nuk ka përmirësim, e mbaj rendin e mëparshëm", flush=True)

    x_new = ts + tofs + order
    ok, ex, f_final = S.verify(x_new)
    print(f"\nf final = {f_final:.3f} d, valid={ok}, përjashtime={ex}")
    if ok:
        json.dump({"x": [float(v) for v in ts + tofs] + order, "f": f_final},
                  open(outfile, "w"))
        print(f"ruajtur {outfile}")