from acoustools.Utilities import TRANSDUCERS, create_points
from acoustools.Solvers import kd_solver
from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points
from acoustools.Constants import wavelength
from acoustools.BEM import compute_E, propagate_BEM_pressure

from acoustools.Visualiser import Visualise, ABC

path = "../BEMMedia"


sphere_paths = path+"/Sphere-lam2.stl"
scatterer = load_scatterer(sphere_paths)
centre_scatterer(scatterer)
print(scatterer.bounds())
d = wavelength*2


scale_to_diameter(scatterer,d)

board = TRANSDUCERS
p = create_points(1,1,0,0,0)

x = kd_solver(p, board=board)

internal_points  = get_CHIEF_points(scatterer, P = -1, start='centre', method='uniform', scale = 0.3, scale_mode='diameter-scale')
E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, return_components=True, internal_points=internal_points)


Eold,Fold,Gold,Hold = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, return_components=True)


r =300
Visualise(*ABC(0.04), x, res = (r,r), vmax=8000, arangement=(2,2), link_ax=[0,1],
           colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure, '-', '/'],
           colour_function_args=[{'board':board, "H":H, 'scatterer':scatterer}, {'board':board, "H":Hold, 'scatterer':scatterer}]
           )