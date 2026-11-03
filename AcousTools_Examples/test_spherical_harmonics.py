from acoustools.SphericalHarmonics.SphericalHarmonics import spherical_harmonic_Y, transducer_expansion_coef_Snm
from acoustools.SphericalHarmonics.Bessel import spherical_bessel_h1
from acoustools.SphericalHarmonics.Propagators import spherical_harmonics_propagate

from acoustools.Utilities import BOTTOM_BOARD, create_points, propagate
from acoustools.Solvers import kd_solver

import torch, scipy

theta = torch.rand(1,1,4) * 3.1415
phi = torch.rand(1,1,4) * 3.1415 * 2 - 2.13145

n = 5
m = 5

Y = spherical_harmonic_Y(theta, phi, n,m)
print(Y)

print(scipy.special.sph_harm_y(n,m,theta, phi))

h1 = spherical_bessel_h1(10, 1, 20)
h1grad = spherical_bessel_h1(10, 1, 20)

board = BOTTOM_BOARD

# com = create_points(1,1, 0.01, 0.02, 0.04)
com = create_points(1,1,0,0,0)

p2 = create_points(1,1,-0.004,0,0.)

x = kd_solver(com, board)



pressure = spherical_harmonics_propagate(x, p2, board, com, max_n=15).abs()
pressure_pm = propagate(x, p2, board).abs()

print(pressure_pm , pressure, pressure_pm / pressure)