from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.Solvers import wgs
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical, propagate_BEM, get_cache_or_compute_H
from acoustools.Visualiser import Visualise
import acoustools.Constants as c

import vedo, torch


path = r"C:\Users\joshu\Documents\BEMMedia\flat-lam2.stl"

reflector = load_scatterer(path,dz=-0.05)

p = create_points(1,1,x=0,y=0,z=0)

# 

H = get_cache_or_compute_H(reflector, TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia")

E = compute_E(reflector,points=p,board=TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia",H=H)
x = wgs(p,A=E)

A = torch.tensor((-0.09,0, 0.09))
B = torch.tensor((0.09,0, 0.09))
C = torch.tensor((-0.09,0, -0.09))
normal = (0,1,0)
origin = (0,0,0)

# Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":reflector,"board":TOP_BOARD,"path":r"C:\Users\joshu\Documents\BEMMedia"}],vmax=5000, show=True,res=[256,256],points=p)

trap_up = p
trap_up[:,2] += c.wavelength/4
print(trap_up)
print(propagate_BEM_pressure(x,trap_up,reflector,TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia",H=H))
print(BEM_gorkov_analytical(x,trap_up,reflector,TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia",H=H))

Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":reflector,"board":TOP_BOARD,"path":r"C:\Users\joshu\Documents\BEMMedia",'H':H}],vmax=5000, show=True,res=[256,256],points=trap_up)

