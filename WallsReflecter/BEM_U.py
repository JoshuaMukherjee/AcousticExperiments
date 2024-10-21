from acoustools.Solvers import wgs, gspat
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs, add_lev_sig, BOTTOM_BOARD
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, BEM_gorkov_analytical,BEM_forward_model_grad, compute_G
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

x_pos = 0

p = create_points(1,1, y=0,x=x_pos,z=0)

E = compute_E(walls, p, board=board, H=H)
x_wgs = wgs(p, board=board, iter=200)
# x_wgs = add_lev_sig(x_wgs)

A,B,C= ABC(0.2, origin=(x_pos, 0,0.0))
tiles = ABC_2_tiles(A,B,C)
res = (600,600)

# diff = lambda *params: BEM_gorkov_analytical(*params, **{'scatterer':walls,'board':board,'H':H,'path':None}) -gorkov_analytical(*params)

def pGxHxR(activations, **params):

    Ex,Ey,Ez,Fx, Fy, Fz, Gx, Gy, Gz, _ = BEM_forward_model_grad(**params, **{'scatterer':walls,'transducers':board,'H':H,'path':None}, return_components=True)
    return ((Gx@H)@activations).real


def pGxHxI(activations, **params):

    Ex,Ey,Ez,Fx, Fy, Fz, Gx, Gy, Gz, _ = BEM_forward_model_grad(**params, **{'scatterer':walls,'transducers':board,'H':H,'path':None}, return_components=True)
    return ((Gx@H)@activations).imag

def pGxHx(activations, **params):

    Ex,Ey,Ez,Fx, Fy, Fz, Gx, Gy, Gz, _ = BEM_forward_model_grad(**params, **{'scatterer':walls,'transducers':board,'H':H,'path':None}, return_components=True)
    return torch.abs((Gx@H)@activations)

def pFx(activations, **params):

    Ex,Ey,Ez,Fx, Fy, Fz, Gx, Gy, Gz, _  = BEM_forward_model_grad(**params, **{'scatterer':walls,'transducers':board,'H':H,'path':None}, return_components=True)
    return torch.abs(Fx@activations)

def px(activations, **params):

    Ex,Ey,Ez,Fx, Fy, Fz, Gx, Gy, Gz, _  = BEM_forward_model_grad(**params, **{'scatterer':walls,'transducers':board,'H':H,'path':None}, return_components=True)
    return torch.abs(Ex@activations)

def GH(activations, **params):
    G = compute_G(params['points'], walls).to(torch.complex64)
    return torch.abs((G@H)@activations)

Visualise(A,B,C,x_wgs,colour_functions=[ GH, propagate_abs ] ,depth=2, res=res, titles=['GH Contribution', 'F Contribution'])
