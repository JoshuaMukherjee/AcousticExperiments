from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points, device, DTYPE, create_board, BOARD_POSITIONS
from acoustools.Solvers import wgs
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical, propagate_BEM, get_cache_or_compute_H
from acoustools.Visualiser import Visualise
import acoustools.Constants as c

import vedo, torch


board = create_board(17, BOARD_POSITIONS)

path ='../BEMMedia'
reflector = load_scatterer(path+'/flat-lam2.stl',dz=-0.05)

p = create_points(1,1,x=0,y=0,z=-0.0475)

# 

H = get_cache_or_compute_H(reflector, board,path=path)

E = compute_E(reflector,points=p,board=board,path=path,H=H)
x = wgs(p,A=E)

A = torch.tensor((-0.09,0, 0.09)).to(device)
B = torch.tensor((0.09,0, 0.09)).to(device)
C = torch.tensor((-0.09,0, -0.09)).to(device)
normal = (0,1,0)
origin = (0,0,0)

# Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":reflector,"board":TOP_BOARD,"path":r"C:\Users\joshu\Documents\BEMMedia"}],vmax=5000, show=True,res=[256,256],points=p)

trap_up = p
trap_up[:,2] += c.wavelength/4
print(trap_up)
print(propagate_BEM_pressure(x,trap_up,reflector,board,path=path,H=H))
print(BEM_gorkov_analytical(x,trap_up,reflector,board,path=path,H=H))

Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":reflector,"board":board,"path":path,'H':H}],vmax=5000, show=True,res=[256,256],points=trap_up)