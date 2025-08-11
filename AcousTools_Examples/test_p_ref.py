from acoustools.Utilities import create_points, BOTTOM_BOARD, propagate
from acoustools.Solvers import wgs

import torch

board = BOTTOM_BOARD
p = create_points(1,1,0,0,0)



x = wgs(p, board=board)

alphas = [0.17, 0.181, 0.2176]
voltages = [i/10 for i in range(0,210, 10)]

results = {

}

for a in alphas:
    results[a] = []
    for v in voltages:
        p_ref = v * a

        pressure = torch.abs(propagate(x, p, p_ref=p_ref, board=board))
        results[a].append(pressure.item())

import matplotlib.pyplot as plt

for a in alphas:
    plt.plot(results[a], label = a)

plt.xlabel("Voltage (V)")
plt.xticks(voltages)

plt.ylabel("Pressure (Pa)")

plt.legend()
plt.show()



