from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig
from acoustools.Force import force_mesh, force_fin_diff
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise

import torch, vedo

path = "../../BEMMedia"

def bounds_to_diameters(bounds):
    x1,x2,y1,y2,z1,z2 = bounds
    print(bounds)
    print(x2-x1,  y2-y1, z2-z1)
    print(x2+x1,  y2+y1, z2+z1)
    print()

def GH_Grad(points, scatterer, transducers=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H=None,
                           path:str="Media"):
    
    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(points=points, scatterer=scatterer, transducers=transducers, use_cache_H=use_cache_H, H=H, path=path, return_components=True)
    return Gx@H, Gy@H, Gz@H
    

USE_CACHE = True
board = TRANSDUCERS



forces_x= []
forces_y= []
forces_z= []

forces_x_U= []
forces_y_U= []
forces_z_U= []


diameters = []

As = []


for i in range(1,3):
    print(i, end='\r')
    diameter = 0.001 * i

    sphere_pth =  path+"/Sphere-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,diameter)
    centre_scatterer(sphere)
    # bounds_to_diameters(sphere.bounds())

# vedo.show(sphere, axes=1)

    com = get_centre_of_mass_as_points(sphere)


    points = get_centres_as_points(sphere)
    # norms = get_normals_as_points(sphere)
    # areas = get_areas(sphere)


    E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
    x = wgs(points,board=board,A=E)


    U  = BEM_gorkov_analytical(x, com, scatterer=sphere, board=board,H=H, path=path).detach().cpu()
    U_force = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()

# Visualise(*ABC(0.1), x, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path}])
# exit()

    diameter_surface = diameter + 0.02
    # surface  = sphere.copy()
    surface = load_scatterer(sphere_pth)
    # print(surface)
    scale_to_diameter(surface,diameter_surface, reset=False, origin=False)

    centre_scatterer(surface)


    surface_points = get_centres_as_points(surface)
    surface_norms = get_normals_as_points(surface)
    surface_areas = get_areas(surface)

    As.append(torch.mean(surface_areas))

    E,F,G,H = compute_E(sphere, surface_points, board,path=path, H=H, return_components=True)
    GH = G@H


    force = force_mesh(x, surface_points,surface_norms,surface_areas,board=board,F=GH, use_momentum=True,
                    grad_function=GH_Grad, grad_function_args={'scatterer':sphere,
                                                                                'H':H,
                                                                                'path':path})
    
    # print(diameter, torch.sum(force))
    forces_x.append(torch.sum(force[:,0]).detach().cpu().item())
    forces_y.append(torch.sum(force[:,1]).detach().cpu().item())
    forces_z.append(torch.sum(force[:,2]).detach().cpu().item())

    forces_x_U.append(torch.sum(U_force[:,0]).detach().cpu().item())
    forces_y_U.append(torch.sum(U_force[:,1]).detach().cpu().item())
    forces_z_U.append(torch.sum(U_force[:,2]).detach().cpu().item())

    diameters.append(diameter)


print(forces_x[0], end=' ')
print(forces_y[0], end=' ')
print(forces_z[0])


print(forces_x[-1], end=' ')
print(forces_y[-1], end=' ')
print(forces_z[-1])


import matplotlib.pyplot as plt
import plotext as plx


plt.plot(diameters, forces_x, color='red')
plt.plot(diameters, forces_y, color='green')
plt.plot(diameters, forces_z, color='blue')

plt.xlabel('Diameter (m)')
plt.ylabel('Force (N)')

PLT = False
if PLT:
    plt.show()
else:
    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    plx.from_matplotlib(fig)
    plx.axes_color('white')
    plx.canvas_color('white')
    plx.show()


plt.close()


# plt.plot(diameters, As, color='blue')
plt.plot(diameters, forces_x_U, color='red')
plt.plot(diameters, forces_y_U, color='green')
plt.plot(diameters, forces_z_U, color='blue')

plt.xlabel('Diameter (m)')
plt.ylabel('Force (N)')


if PLT:
    plt.show()
else:
    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    plx.from_matplotlib(fig)
    plx.axes_color('white')
    plx.canvas_color('white')
    plx.show()

