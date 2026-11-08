import sys, json, time
import numpy as np
from kttsp_solver import Solver

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


if __name__ == "__main__":
    inst = sys.argv[1] if len(sys.argv) > 1 else "easy.kttsp"
    solfile = sys.argv[2] if len(sys.argv) > 2 else "x_small.json"
    outfile = sys.argv[3] if len(sys.argv) > 3 else "x_small_retimed.json"

    S = Solver(inst)
    N, P = S.N, S.P
    x = json.load(open(solfile))["x"]
    ts, tofs = x[:N - 1], x[N - 1:2 * N - 2]
    order = [int(v) for v in x[2 * N - 2:]]
    f_old = ts[-1] + tofs[-1]
    print(f"f e vjetër = {f_old:.3f} d", flush=True)

    t_now, new_ts, new_tofs = P["t0"], [], []
    t_start = time.time()
    for k in range(N - 1):
        cur, nxt = order[k], order[k + 1]
        d0 = S.dv(cur, nxt, float(ts[k]), float(tofs[k]))
        limit = P["dvex"] if d0 > P["dv"] else P["dv"]
        arr, t, tof, d = fine_leg(S, cur, nxt, t_now, limit, (ts[k], tofs[k]))
        new_ts.append(t)
        new_tofs.append(tof)
        print(f"kalimi {k + 1:2d}/{N - 1}: {cur:3d}->{nxt:3d}  "
              f"arritja e vjetër {ts[k] + tofs[k]:7.2f} -> e re {arr:7.2f}  "
              f"({time.time() - t_start:.0f} s)", flush=True)
        t_now = arr

    x_new = new_ts + new_tofs + order
    ok, ex, f_new = S.verify(x_new)
    print(f"\nf i ri = {f_new:.3f} d (nga {f_old:.3f}), valid={ok}, përjashtime={ex}")
    if ok and f_new < f_old - 1e-9:
        json.dump({"x": [float(v) for v in new_ts + new_tofs] + order, "f": f_new},
                  open(outfile, "w"))
        print(f"ruajtur {outfile}")
    else:
        print("nuk u ruajt skedar i ri (nuk ka përmirësim ose nuk është i vlefshëm)")