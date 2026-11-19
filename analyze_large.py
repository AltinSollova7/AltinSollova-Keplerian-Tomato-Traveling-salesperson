
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "large.kttsp"
P, rows = None, []
for line in open(path):
    s = line.strip()
    if not s or s[0] == "c":
        continue
    if s[0] == "p":
        P = s.split()
    else:
        rows.append([float(v) for v in s.split()])
orb = np.array(rows)
N = len(orb)
print("parametrat:", P)
print("domate:", N)

BIG, HALF = 1e7, np.pi / 2          
a, inc = orb[:, 0], orb[:, 2]
small = a <= BIG
eq = np.abs(inc) <= 0.045
po = np.abs(inc - HALF) <= 0.035
re = np.abs(inc - np.pi) <= 0.035
lad = ~(eq | po | re) & ~small
print(f"te vogla (a<=1e7): {small.sum()}   te medha: {(~small).sum()}")
for name, m in [("eq_small", eq & small), ("eq_large", eq & ~small),
                ("po_small", po & small), ("po_large", po & ~small),
                ("re_small", re & small), ("re_large", re & ~small)]:
    print(f"{name:10s} {m.sum():5d}")
l1 = lad & (inc < HALF)
l2 = lad & (inc > HALF)
print(f"shkalla1   {l1.sum():5d}   shkalla2   {l2.sum():5d}")
print(f"te vogla jashte blloqeve (leftover_small): {(small & ~(eq | po | re)).sum()}")
tot = (eq | po | re).sum() + lad.sum() + (small & ~(eq | po | re)).sum()
print(f"kontroll: {tot} nga {N}")