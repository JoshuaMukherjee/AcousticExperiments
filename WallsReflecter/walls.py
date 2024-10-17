from acoustools.Solvers import wgs, gspat
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs, add_lev_sig, BOTTOM_BOARD
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force
from acoustools.Visualiser import Visualise_single, ABC, Visualise, Visualise_mesh, ABC_2_tiles, combine_tiles, Visualise_single_blocks

import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


torch.manual_seed(1)

board = TRANSDUCERS

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)



H = get_cache_or_compute_H(walls,board)


p = create_points(1,1, y=0,x=0,z=0)

E = compute_E(walls, p, board=board, H=H)
x_bem = wgs(p, board=board, iter=200, A=E)

A,B,C= ABC(0.2)
tiles = ABC_2_tiles(A,B,C)
res = (800,800)

Visualise(A,B,C,x_bem,colour_functions=[propagate_BEM_pressure, propagate_abs], colour_function_args=[{'scatterer':walls,'board':board,'H':H},{}], res=res)

