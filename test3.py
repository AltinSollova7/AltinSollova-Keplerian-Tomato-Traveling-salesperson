import time
from utils_users import TomatoProblem

util = TomatoProblem("easy.kttsp")
N = len(util.tomato_orbital_parameters)
print("Numri i domateve:", N)

starts = list(range(0, 60, 6))              # nisja: ditët 0,6,...,54
tofs = [1, 3, 6, 10, 15, 20, 30]            # kohëzgjatjet e fluturimit

best = {}
t0 = time.time()
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        m = 1e18
        for t in starts:
            for tof in tofs:
                try:
                    dv = util.compute_transfer(i, j, float(t), float(tof))
                except Exception:
                    continue
                if dv < m:
                    m = dv
        best[(i, j)] = m
    if i % 5 == 0:
        print("domatja", i, "përpunuar,", round(time.time() - t0), "sekonda")

total = len(best)
ok100 = sum(1 for v in best.values() if v <= 100)
ok600 = sum(1 for v in best.values() if v <= 600)
print("Çifte gjithsej:", total)
print("Çifte me DV <= 100 diku në rrjetë:", ok100)
print("Çifte me DV <= 600 diku në rrjetë:", ok600)
print("DV më i vogël kudo:", min(best.values()))