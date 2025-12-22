from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs
from acoustools.Solvers import wgs

from acoustools.Constants import c_0, pi

p = create_points(1,1,0,0,0)
board = TRANSDUCERS

pressures = []
freqs = []

for freq in range(1000, 1000000, 10000):
    print(freq, end='\r')
    wavelength = c_0 / freq
    k = 2*pi / wavelength

    x = wgs(p, board=board, k=k)

    pressure = propagate_abs(x, p, board=board, k=k)


    pressures.append(pressure.item())
    freqs.append(freq)


import matplotlib.pyplot as plt

plt.yscale('log')
plt.xscale('log')
plt.plot(freqs, pressures)


plt.show()