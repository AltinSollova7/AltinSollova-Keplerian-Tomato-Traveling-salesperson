from util_user import TomatoProblem

util = TomatoProblem("easy.kttsp")

# Sa karburant duhet për të shkuar nga domatja 0 te domatja 1,
# duke u nisur në ditën 0, për kohëzgjatje të ndryshme fluturimi?
for tof in [0.5, 1, 2, 5, 10]:
    dv = util.compute_transfer(0, 1, 0.0, tof)
    print("tof =", tof, "ditë  ->  DV =", dv, "m/s")