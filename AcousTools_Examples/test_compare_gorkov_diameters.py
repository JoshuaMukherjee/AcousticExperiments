from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, propagate_abs
from acoustools.Force import force_mesh, force_fin_diff
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise
from acoustools.Gorkov import gorkov_analytical

import torch, vedo

path = "../BEMMedia"

board = TRANSDUCERS



def GH_Grad(points, scatterer, transducers=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H=None,
                           path:str="Media"):
    
    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(points=points, scatterer=scatterer, transducers=transducers, use_cache_H=use_cache_H, H=H, path=path, return_components=True)
    return Gx@H, Gy@H, Gz@H


sphere_pth =  path+"/Sphere-lam2.stl"

Us = []
BEM_Us = []

diameters = []

N = 8
for i in range(N):
    print(i, end='\r')
    diameter = (c.wavelength/10)/N * i + 0.0004
    diameters.append(diameter)
    

    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,diameter)
    centre_scatterer(sphere)

    points = get_centres_as_points(sphere)
    com = get_centre_of_mass_as_points(sphere)


    E,F,G,H = compute_E(sphere, points, board, path=path, return_components=True)
    x = wgs(points,board=board,A=E)

    # Visualise(*ABC(0.1), x, colour_functions=[propagate_abs, propagate_BEM_pressure], 
    #           colour_function_args=[{},{'scatterer':sphere,'H':H, 'board':board}])
    # exit()

    U = gorkov_analytical(x, com, board).cpu().detach()
    U_BEM = BEM_gorkov_analytical(x, com, sphere,board, H=H, path=path).cpu().detach()


    Us.append(U)
    BEM_Us.append(U_BEM)

    

import matplotlib.pyplot as plt
import plotext as plx

# plt.plot(diameters,forces_x,color='red')
# plt.plot(diameters,forces_y,color='green')
# plt.plot(diameters,forces_z,color='blue')

SCALE = 100000
plt.scatter(Us, BEM_Us, s=[SCALE*d for d in diameters])
# x0,x1 = plt.ylim()
# plt.xlim(x0,x1)
plt.show()