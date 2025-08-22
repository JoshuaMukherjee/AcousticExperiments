from acoustools.Utilities import create_points, TRANSDUCERS, DTYPE, device, BOTTOM_BOARD, propagate_abs
from acoustools.Constants import pi,wavelength, P_ref as cPref
from acoustools.Export.Holo import load_holograms
from acoustools.Force import compute_force

from acoustools.BEM import get_cache_or_compute_H, compute_E, BEM_forward_model_grad, propagate_BEM_pressure, force_mesh_surface, torque_mesh_surface, propagate_BEM_phase
from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, get_normals_as_points, centre_scatterer, get_centre_of_mass_as_points, translate, get_edge_data

from acoustools.Visualiser import Visualise, ABC

import torch

board = TRANSDUCERS
M = board.shape[0]


voltage = 18
p_ref = torch.ones((1,M,1), device=device, dtype=DTYPE)
p_ref[:,:256] = 0.181 * voltage
p_ref[:,256:] = 0.2176 * voltage
p_ref = cPref


use_cache = False

#Load Scatterer
root = '../BEMMedia'
sphere_path = root + '/Sphere-solidworks-lam2.stl'





# h_path = 'Optimsed_holo_1755697974095288200.holo'
h_path = 'AcousTools_Examples/data/holos/Optimsed_holo_1755780306180471000.holo'
x = load_holograms(h_path)[0]

d = 0.0001
surface_d = wavelength 
ds = []
max_d = wavelength*3


N = 20
step = max_d / N

Fxs = []
Fys = []
Fzs = []

v = 4/3 * pi * (d/2)**3
sphere = load_scatterer(sphere_path) #Make mesh at 0,0,0
scale_to_diameter(sphere,d)
centre_scatterer(sphere)
com = get_centre_of_mass_as_points(sphere)

Visualise(*ABC(0.1), x, points=com, show=False, colour_functions=[propagate_abs], colour_function_args=[{'board':board}])


Fx,Fy,Fz = compute_force(x, com, board=board, return_components=True, V=v)

E,F,G,H = compute_E(sphere, com, board,path=root, return_components=True, use_cache_H=use_cache)


for i in range(N):
    print(i,end='\r')
    # d = start_d + (max_d/N)*i * 1.01
    # d = max_d * 1.01


    surface_d += step 
    surface = load_scatterer(sphere_path) #Make mesh at 0,0,0
    scale_to_diameter(surface,surface_d)
    centre_scatterer(surface)
    surface_com = get_centre_of_mass_as_points(surface)
    surface_points= get_centres_as_points(surface)
    ds.append(surface_d/wavelength)


    E_step= compute_E(sphere, surface_points, board,H=H, path=root,
                            use_cache_H=use_cache, p_ref=p_ref)
    # print(E_step) #bounds        : x=(-0.0143, 5.70e-3), y=(-0.0100, 0.0100), z=(-0.0100, 0.0100)

    Ax_step, Ay_step, Az_step =BEM_forward_model_grad(surface_points, sphere, board,H=H, path=root, use_cache_H=use_cache, p_ref=p_ref)

    f = force_mesh_surface(x, sphere, board, H=H, path=sphere_path,
                                       surface=surface, E=E_step, Ex=Ax_step, Ey=Ay_step, Ez=Az_step).real.squeeze().cpu().detach()
    
    Fxs.append(f[0])
    Fys.append(f[1])
    Fzs.append(f[2])



import matplotlib.pyplot as plt
plt.figure()

plt.plot(ds,Fxs,c='r', label='$F_x$')
plt.plot(ds,Fys,c='g', label='$F_y$')
plt.plot(ds,Fzs,c='b', label='$F_z$')

plt.hlines(Fx, ds[0], ds[-1],colors='r')
plt.hlines(Fy, ds[0], ds[-1],colors='g')
plt.hlines(Fz, ds[0], ds[-1],colors='b')

plt.show()