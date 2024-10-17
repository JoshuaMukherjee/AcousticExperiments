from acoustools.Solvers import wgs, gspat
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs, add_lev_sig
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force
from acoustools.Visualiser import Visualise_single, ABC

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


p = create_points(1,1, y=0)

x_wgs = wgs(p, board=board, iter=200)

x_gspat = gspat(p,board=board,iterations=200)

E = compute_E(walls, p, board=board, H=H)
x_bem = wgs(p, board=board, iter=200, A=E)


A,B,C = ABC(0.1)
res = (200,200)
wgs_vis = Visualise_single(A,B,C,x_wgs, res=res).cpu().detach()
print('wgs')
gspat_viss = Visualise_single(A,B,C,x_gspat, res=res).cpu().detach()
print('gspat')
bem_vis = Visualise_single(A,B,C,x_bem, colour_function=propagate_BEM_pressure, colour_function_args={'scatterer':walls,'board':board,'H':H}, res=res).cpu().detach()
print('bem')

norm = mcolors.Normalize(vmin=0, vmax=6000)

fig = plt.figure()

ax1 = plt.subplot(1,3,1)
im1= ax1.matshow(wgs_vis, cmap='hot', norm=norm)

ax2 = plt.subplot(1,3,2)
im2=ax2.matshow(gspat_viss, cmap='hot', norm=norm)

ax3 = plt.subplot(1,3,3)
im3=ax3.matshow(bem_vis, cmap='hot', norm=norm)

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = fig.colorbar(im1, cax=cax)



plt.show()