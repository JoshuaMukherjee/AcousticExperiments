from acoustools.Mesh import load_scatterer, scale_to_diameter
from acoustools.Utilities import create_points, create_board
from acoustools.Solvers import naive
from acoustools.BEM import compute_E, propagate_BEM_pressure

from acoustools.Visualiser import Visualise, ABC

path = '../BEMMedia'

reflector = load_scatterer(path + '/flat-lam4.stl', dz=0.06, roty=180)
scale_to_diameter(reflector, 0.15)

CHIEF_pts = create_points(5,1, min_pos=-0.02, max_pos=0.02)
CHIEF_pts[:,:2] *= 3
CHIEF_pts[:,2] += 0.1

board = create_board(N=17, z=-0.06)

p = create_points(1,1,0,0,0)

E,F,G,H = compute_E(reflector, p , board=board, return_components=True, path=path, internal_points=CHIEF_pts, use_cache_H=False )

x = naive(p, board=board, A=E)

Visualise(*ABC(0.12, origin=p), x, res=(300,300), points=CHIEF_pts,
        colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, '-'],
        colour_function_args=[{'board':board, 'path':path, 'H':H, 'scatterer':reflector, "internal_points":CHIEF_pts, "use_cache_H":False}, 
                              {'board':board, 'path':path, 'scatterer':reflector, "use_cache_H":False},
                              {}])