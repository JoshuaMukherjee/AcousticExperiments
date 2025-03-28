from acoustools.Solvers import gradient_descent_solver
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Gorkov import gorkov_analytical
from acoustools.Visualiser import ABC, Visualise, ABC_2_tiles

from acoustools.Optimise.Objectives import propagate_abs_sum_objective_BEM
from acoustools.Optimise.Constraints import constrain_phase_only


import torch



torch.manual_seed(1)

board = TRANSDUCERS

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)

x_pos = 0

p = create_points(1,1, y=0,x=x_pos,z=0)

H = get_cache_or_compute_H(walls,board)
E = compute_E(walls, p, board=board, H=H)


x = gradient_descent_solver(p,propagate_abs_sum_objective_BEM, 
                            maximise=True, constrains=constrain_phase_only, log=True, lr=1e-1,
                            objective_params = {'scatterer':walls,'E':E}, iters=500)




# x_wgs = wgs(p, board=board, iter=200)
# x_wgs = add_lev_sig(x_wgs)

A,B,C= ABC(0.2, origin=(x_pos, 0,0.0))
tiles = ABC_2_tiles(A,B,C)
res = (600,600)

# diff = lambda *params: BEM_gorkov_analytical(*params, **{'scatterer':walls,'board':board,'H':H,'path':None}) -gorkov_analytical(*params)

# Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure, propagate_abs, BEM_gorkov_analytical, gorkov_analytical ], 
#           colour_function_args=[{'scatterer':walls,'board':board,'H':H,'path':None},{}, {'scatterer':walls,'board':board,'H':H,'path':None},{}], 
#           res=res,link_ax=[[0,1],[2,3]], arangement=(2,2), clr_labels=['Pressure (Pa)', 'Pressure (Pa)', '$U$','$U$'], titles=['BEM', 'PM',None,None ])


Visualise(A,B,C,x,colour_functions=[ propagate_BEM_pressure, propagate_abs ] ,depth=2, res=res, colour_function_args=[{'scatterer':walls,'board':board,'H':H,'path':None},{}])
