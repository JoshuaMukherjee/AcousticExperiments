from acoustools.SphericalHarmonics.SphericalHarmonics import spherical_harmonic_Y, transducer_expansion_coef_Snm
from acoustools.SphericalHarmonics.Bessel import spherical_bessel_h1
from acoustools.SphericalHarmonics.Propagators import spherical_harmonics_propagate, spherical_harmonics_propagate_scat, spherical_harmonics_propagate_inc
from acoustools.Mesh import load_multiple_scatterers, centre_scatterer, scale_to_diameter
from acoustools.Utilities import BOTTOM_BOARD, create_points, propagate
from acoustools.Solvers import kd_solver
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import propagate_BEM_pressure, propagate_BEM
from acoustools.Constants import R, wavelength

import torch, scipy
from line_profiler import profile


board = BOTTOM_BOARD

path = '../BEMMedia'
paths = [path+"/Sphere-lam2.stl"]
scatterer = load_multiple_scatterers(paths)
centre_scatterer(scatterer)


D = 0.01
D = wavelength

scale_to_diameter(scatterer, diameter=D)
print(scatterer.bounds())

# com = create_points(1,1, 0.01, 0.02, 0.04)
com = create_points(1,1,0,0,0)



origin = create_points(1,1,0.01, 0, 0)
x = kd_solver(origin, board)


def scat_fied(activations, points, board=board, **kwargs):
    # inc = propagate(activations, points, board)
    inc = 0
    # total = 2 * inc
    total = propagate_BEM(activations, points, board=board, path=path, scatterer=scatterer).squeeze(2)
    # print(inc.shape, total.shape)
    scat = (total - inc)
    return (scat).abs()


M = 30

Visualise(*ABC(4*D, plane='xz'), x, 
          colour_functions=[spherical_harmonics_propagate, spherical_harmonics_propagate_inc, spherical_harmonics_propagate_scat, scat_fied], 
          depth=0,
        #   colour_functions=[spherical_harmonics_propagate], 
          colour_function_args=[{'board':board,'centre_of_mass':com, 'max_n':M, 'particle_radius':D/2},
                                {'board':board,'centre_of_mass':com, 'max_n':M},
                                {'board':board,'centre_of_mass':com, 'max_n':M, 'particle_radius':D/2},
                                {'board':board},
                                ],
          link_ax=[0,1,2,3],
          vmax=[9000,9000,9000,9000],
          res = (200,200),
          arrangement=(2,2),
          call_abs=True, show=True)



