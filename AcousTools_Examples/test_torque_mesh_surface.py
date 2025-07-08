from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer, get_edge_data
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, create_points, propagate_abs, propagate_phase
from acoustools.Force import force_mesh, force_fin_diff
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure, torque_mesh_surface, propagate_BEM_phase
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise

import torch, vedo
import matplotlib.pyplot as plt

path = "../BEMMedia"




USE_CACHE = True
board = TRANSDUCERS



Tx = []
Ty = []
Tz = []

diameters = []

As = []

diameter = 0.015
# sphere_pth =  path+"/Cube_Angle_lam4.stl"
sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
scale_to_diameter(sphere,diameter)
centre_scatterer(sphere)
com = get_centre_of_mass_as_points(sphere)
get_edge_data(sphere)

scatterer_points = get_centres_as_points(sphere)


x = wgs(com-create_points(1,1,0,0,1.5*c.wavelength), board=board)


Visualise(*ABC(5*c.wavelength), x, colour_functions=[propagate_BEM_pressure,propagate_BEM_phase], 
          colour_function_args=[{'board':board,'scatterer':sphere,'path':path, 'use_cache_H':USE_CACHE},{'board':board,'scatterer':sphere,'path':path, 'use_cache_H':USE_CACHE}], 
          block=False, cmaps=['hot','hsv'], link_ax=None)
plt.figure()
Visualise(*ABC(5*c.wavelength, plane='xy'), x, colour_functions=[propagate_BEM_pressure,propagate_BEM_phase], 
          colour_function_args=[{'board':board,'scatterer':sphere,'path':path, 'use_cache_H':USE_CACHE},{'board':board,'scatterer':sphere,'path':path, 'use_cache_H':USE_CACHE}], 
          block=False, cmaps=['hot','hsv'], link_ax=None)


for i in range(1,32):
    print(i, end='\r')
    d = 1.5*diameter + c.wavelength/6 * i
    torque = torque_mesh_surface(x, sphere, board, diameter = d, path=path,use_cache_H=USE_CACHE)
    
    Tx.append(torque[:,0].item())
    Ty.append(torque[:,1].item())
    Tz.append(torque[:,2].item())
    diameters.append(d)


plt.figure()
plt.subplot(1,2,1)
plt.plot(diameters, Tx, c='red')
plt.plot(diameters, Ty, c='green')
plt.subplot(1,2,2)
plt.plot(diameters, Tz, c='blue')

plt.show()