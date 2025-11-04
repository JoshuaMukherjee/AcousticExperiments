from acoustools.Utilities import transducers, create_points, propagate_abs
from acoustools.Solvers import kd_solver

from acoustools.Mesh import load_scatterer, get_CHIEF_points, centre_scatterer, get_edge_data, scale_to_diameter, get_diameter
from acoustools.BEM import propagate_BEM_pressure

from acoustools.Visualiser import Visualise, ABC

import vedo

board = transducers(z=-0.1)
p = create_points(1,1,0,0,0)

voltage = 18
p_ref = 0.179 * voltage



x = kd_solver(p, board=board)


path = "../BEMMedia/"
discs = ["disc_12mm_lam4.stl", "disc_15mm_lam4.stl", "disc_18mm_lam4.stl"]
ds = [0.012, 0.015, 0.018]

for disc_path, d in zip(discs, ds):

    disc = load_scatterer(path + disc_path)
    scale_to_diameter(disc, d)

    centre_scatterer(disc)
    get_edge_data(disc)
    print(get_diameter(disc))

    chief_pts = get_CHIEF_points(disc, P=10, method='uniform', scale = 1e-3, scale_mode='abs', start='surface')



    r = 200
    Visualise(*ABC(0.01, plane='xz'), x, res=(r,r),
            # points=chief_pts,
            colour_functions=[propagate_BEM_pressure,], 
            colour_function_args=[{'board':board,'p_ref':p_ref, 'path':path, 'scatterer':disc, 'internal_points':chief_pts, 'use_cache_H':False}])
    
    exit()