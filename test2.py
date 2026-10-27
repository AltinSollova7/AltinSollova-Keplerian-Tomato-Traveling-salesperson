from util_user import TomatoProblem

util = TomatoProblem("easy.kttsp")

best = (1e18, None, None)
count_ok = 0
total = 0

for t in range(0, 21):                 # nisja: ditët 0..20
    for k in range(1, 61):             # fluturimi: 0.5..30 ditë
        tof = k * 0.5
        try:
            dv = util.compute_transfer(0, 1, float(t), tof)
        except Exception:
            continue
        total += 1
        if dv <= 100:
            count_ok += 1
        if dv < best[0]:
            best = (dv, t, tof)

print("Kombinime të provuara:", total)
print("Kombinime me DV <= 100 m/s:", count_ok)
print("Më i miri: DV =", best[0], "m/s, nisja ditën", best[1], ", fluturimi", best[2], "ditë")