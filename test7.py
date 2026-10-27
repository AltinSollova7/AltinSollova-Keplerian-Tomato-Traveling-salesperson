from util_user import TomatoProblem

util = TomatoProblem("easy.kttsp")
N = len(util.tomato_orbital_parameters)
START = 0
t_now = 0.0

results = []
for B in range(N):
    if B == START:
        continue
    best = None
    for wait in range(0, 21):
        for tof in range(1, 21):
            t = t_now + wait
            dv = util.compute_transfer(START, B, float(t), float(tof))
            if dv <= 100:
                arrive = t + tof
                if best is None or arrive < best[0]:
                    best = (arrive, t, tof, dv)
    results.append((B, best))

reachable = [(B, r) for B, r in results if r is not None]
reachable.sort(key=lambda z: z[1][0])

print("Domate te arritshme me DV<=100:", len(reachable), "nga", N - 1)
print("Top 5 me te shpejtat:")
for B, (arrive, t, tof, dv) in reachable[:5]:
    print("domatja", B, "| nisje dita", t, "| fluturim", tof, "| mberrin dita", arrive, "| DV", round(dv, 1))