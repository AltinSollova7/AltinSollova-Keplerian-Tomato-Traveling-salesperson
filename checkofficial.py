"""
Kontroll i pavarur i x_small.json, sipas kufizimeve të verifikuesit zyrtar.
Ekzekutim:   python check_official.py medium.kttsp x_medium_ladder.json
"""
import json, sys
try:
    from util_user import TomatoProblem
except ImportError:
    from util import TomatoProblem

PATH = sys.argv[1] if len(sys.argv) > 1 else "medium.kttsp"
SOL = sys.argv[2] if len(sys.argv) > 2 else "x_medium_ladder.json"

# parametrat nga rreshti 'p' dhe numri i tomateve
params, N = None, 0
for line in open(PATH):
    s = line.strip()
    if not s or s[0] == "c":
        continue
    if s[0] == "p":
        p = s.split()
        params = dict(min_tof=float(p[3]), max_time=float(p[4]),
                      dv=float(p[5]), dvex=float(p[6]), E=int(float(p[7])))
    else:
        N += 1

u = TomatoProblem(PATH)
x = json.load(open(SOL))["x"]
times, tofs = x[:N - 1], x[N - 1:2 * N - 2]
order = [int(round(v)) for v in x[2 * N - 2:]]

ok = True
def check(name, cond):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name)
    ok = ok and cond

check(f"gjatësia e x = 3N-2 = {3 * N - 2}", len(x) == 3 * N - 2)
check("rendi është permutacion i 0..N-1", sorted(order) == list(range(N)))
check("kohët e nisjes brenda [0, max_time]",
      all(0 <= t <= params["max_time"] for t in times))
check("fluturimet >= min_tof dhe <= max_time",
      all(params["min_tof"] <= f <= params["max_time"] for f in tofs))
check("t_i + tof_i <= t_(i+1) (toleranca 1e-6)",
      all(times[i] + tofs[i] <= times[i + 1] + 1e-6 for i in range(N - 2)))

n_exc, worst, bad = 0, 0.0, 0
for i in range(N - 1):
    d = u.compute_transfer(order[i], order[i + 1], times[i], tofs[i])
    worst = max(worst, d)
    if d > params["dvex"] + 1e-6:
        bad += 1
    elif d > params["dv"]:
        n_exc += 1
check(f"asnjë kalim mbi {params['dvex']} m/s (të këqij: {bad}, ΔV maksimal {worst:.1f})", bad == 0)
check(f"përjashtime {n_exc} <= {params['E']}", n_exc <= params["E"])

f = times[-1] + tofs[-1]
check(f"f = {f:.3f} d <= max_time {params['max_time']}", f <= params["max_time"])
print()
print("REZULTATI:", "TË GJITHA KALOJNË" if ok else "KA DËSHTIME")