"""
Raport: ku shkon koha në një zgjidhje. Për çdo bllok (grupe dhe shkallë) tregon sa
kalime ka, sa ditë zgjat gjithsej dhe mesatarisht, dhe ΔV mesatar. Tregon edhe
10 kalimet më të gjata.

    python leg_report.py medium.kttsp x_medium.json
    python leg_report.py easy.kttsp x_small_blocks.json
"""
import sys, json
from kttsp_solver import Solver
from kttsp_block import build_blocks


def label_map(orb):
    g, l1, l2, _ = build_blocks(orb)
    lab = {}
    for name, nodes in g.items():
        for n in nodes:
            lab[n] = name
    for n in l1:
        lab[n] = "shkalla1"
    for n in l2:
        lab[n] = "shkalla2"
    return lab


if __name__ == "__main__":
    inst = sys.argv[1] if len(sys.argv) > 1 else "medium.kttsp"
    solfile = sys.argv[2] if len(sys.argv) > 2 else "x_medium.json"
    S = Solver(inst)
    N, P = S.N, S.P
    x = json.load(open(solfile))["x"]
    ts, tofs = x[:N - 1], x[N - 1:2 * N - 2]
    order = [int(v) for v in x[2 * N - 2:]]
    lab = label_map(S.orb)

    rows, prev = [], P["t0"]
    for k in range(N - 1):
        a, b = order[k], order[k + 1]
        arr = ts[k] + tofs[k]
        dv = S.dv(a, b, float(ts[k]), float(tofs[k]))
        rows.append(dict(k=k + 1, a=a, b=b, wait=ts[k] - prev, tof=tofs[k],
                         step=arr - prev, dv=dv, lab=lab.get(b, "?")))
        prev = arr
    total = prev

    groups = {}
    for r in rows:
        g = groups.setdefault(r["lab"], dict(n=0, time=0.0, dv=0.0, wait=0.0, tof=0.0))
        g["n"] += 1
        g["time"] += r["step"]
        g["dv"] += r["dv"]
        g["wait"] += r["wait"]
        g["tof"] += r["tof"]

    print(f"f = {total:.3f} d, kalime: {N - 1}\n")
    print(f"{'blloku (ku mbërrin)':20s} {'kalime':>6s} {'ditë':>8s} {'% e totalit':>12s} "
          f"{'ditë/kalim':>11s} {'ΔV mes.':>8s} {'pritje':>8s} {'fluturim':>9s}")
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["time"]):
        print(f"{name:20s} {g['n']:6d} {g['time']:8.1f} {100 * g['time'] / total:11.1f}% "
              f"{g['time'] / g['n']:11.2f} {g['dv'] / g['n']:8.1f} {g['wait']:8.1f} {g['tof']:9.1f}")

    print("\n10 kalimet më të gjata:")
    for r in sorted(rows, key=lambda r: -r["step"])[:10]:
        print(f"  kalimi {r['k']:3d}: {r['a']:3d}->{r['b']:3d} ({r['lab']:9s}) "
              f"{r['step']:6.2f} d  (pritje {r['wait']:5.2f}, fluturim {r['tof']:5.2f}, ΔV {r['dv']:6.1f})")