from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, propagate_abs, add_lev_sig
from acoustools.Force import force_mesh, force_fin_diff, compute_force
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise

import torch, vedo

path = "../BEMMedia"

board = TRANSDUCERS



def GH_Grad(points, scatterer, transducers=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H=None,
                           path:str="Media"):
    
    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(points=points, scatterer=scatterer, transducers=transducers, use_cache_H=use_cache_H, H=H, path=path, return_components=True)
    return Gx@H, Gy@H, Gz@H


sphere_pth =  path+"/Sphere-lam2.stl"

forces_x= []
forces_y= []
forces_z= []

forcesU_x= []
forcesU_y= []
forcesU_z= []

diameters = []

N = 8
for i in range(N):
    print(i, end='\r')
    diameter = (c.wavelength/10)/N * i + 0.0004
    diameters.append(diameter)
    print(diameter)
    

    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,diameter)
    centre_scatterer(sphere)

    points = get_centres_as_points(sphere)
    com = get_centre_of_mass_as_points(sphere)


    E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
    x = wgs(points,board=board,A=E)

    surface = load_scatterer(sphere_pth)
    scale_to_diameter(surface,diameter + 0.03, reset=False, origin=False)
    centre_scatterer(surface)

    points_surface = get_centres_as_points(surface)
    norms_surface = get_normals_as_points(surface)
    areas_surface = get_areas(surface)

    E,F,G,H = compute_E(sphere, points_surface, board,path=path, H=H, return_components=True)
    GH = G@H

    # force = force_mesh(x, points_surface,norms_surface,areas_surface,board=board,F=GH, use_momentum=True,
    #                 grad_function=GH_Grad, grad_function_args={'scatterer':sphere,
    #                                                                             'H':H,
    #                                                                             'path':path})

    force = force_mesh(x, points_surface,norms_surface,areas_surface,board=board,F=E, use_momentum=True,
                    grad_function=BEM_forward_model_grad, grad_function_args={'scatterer':sphere,
                                                                                'H':H,
                                                                                'path':path})

    print(torch.sum(force, dim=2))
    
    forces_x.append(torch.sum(force[:,0]).detach().cpu())
    forces_y.append(torch.sum(force[:,1]).detach().cpu())
    forces_z.append(torch.sum(force[:,2]).detach().cpu())

    V = 4/3 * 3.1415 * (diameter/2)**3
    # U_force = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, V=V,
    #                          U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()

    U_force = compute_force(x, com, board)
    print(U_force)

    print( torch.sum(force, dim=2) / U_force)

    print()

    # print(U_force)
    # U_force = force_fin_diff(x,com,V=V).detach().cpu()
    # print(U_force)
    # exit()


    forcesU_x.append((U_force[0]).detach().cpu())
    forcesU_y.append((U_force[1]).detach().cpu())
    forcesU_z.append((U_force[2]).detach().cpu())

exit()

import matplotlib.pyplot as plt
import plotext as plx

plt.plot(diameters,forces_x,color='red')
plt.plot(diameters,forces_y,color='green')
plt.plot(diameters,forces_z,color='blue')

plt.plot(diameters,forcesU_x,color='red',linestyle=':')
plt.plot(diameters,forcesU_y,color='green',linestyle=':')
plt.plot(diameters,forcesU_z,color='blue',linestyle=':')

plt.show()