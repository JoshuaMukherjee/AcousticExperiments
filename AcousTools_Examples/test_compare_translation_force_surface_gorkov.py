from acoustools.Utilities import create_points, TRANSDUCERS, DTYPE, device, propagate_abs
from acoustools.Constants import wavelength, P_ref as cPref
from acoustools.Export.Holo import load_holograms
from acoustools.Force import compute_force
from acoustools.Solvers import naive
from acoustools.BEM import get_cache_or_compute_H, compute_E, BEM_forward_model_grad, propagate_BEM_pressure, force_mesh_surface, torque_mesh_surface, propagate_BEM_phase
from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, get_normals_as_points, centre_scatterer, get_centre_of_mass_as_points, translate, get_edge_data

from acoustools.Visualiser import ABC, Visualise

import torch

board = TRANSDUCERS
M = board.shape[0]


voltage = 18
p_ref = torch.ones((1,M,1), device=device, dtype=DTYPE)
p_ref[:,:256] = 0.181 * voltage
p_ref[:,256:] = 0.2176 * voltage
# p_ref = cPref


use_cache = False

#Load Scatterer
root = '../BEMMedia'
sphere_path = root + '/Sphere-solidworks-lam2.stl'
# path = root + '/Sphere-lam2.stl'
diameter=0.002
volume = 4/3 * 3.1415 * (diameter/2)**3
m=1e-10
weight = -1* (m) * 9.81

surface_diameter = diameter + 0.1 * wavelength



# h_path = 'Optimsed_holo_1755697974095288200.holo'
h_path = 'AcousTools_Examples/data/holos/Optimsed_holo_1755780306180471000.holo'

# x = load_holograms(h_path)[0]
p = create_points(1,1,0,0,0)
x = naive(p, board=board, p_ref=p_ref)

start_z = 15*wavelength
start = create_points(1,1,0,0,start_z/2)


N = 180

Fxs = []
Fys = []
Fzs = []

ARFxs = []
ARFys = []
ARFzs = []

pARFxs = []
pARFys = []
pARFzs = []

ratio_zs = []

for i in range(N+1):
    print(i, end='\r')
    point = start - create_points(1,1,0,0,start_z/N * i)
    Fx,Fy,Fz = compute_force(x, point, board=board, return_components=True, V=volume, p_ref=p_ref)

    Fxs.append(Fx.cpu().detach().squeeze())
    Fys.append(Fy.cpu().detach().squeeze())
    Fzs.append(Fz.cpu().detach().squeeze())
    

    sphere = load_scatterer(sphere_path)
    scale_to_diameter(sphere, diameter)
    centre_scatterer(sphere)    
    
    translate(sphere, dx=point[:,0].item(), dy=point[:,1].item(),dz=point[:,2].item())

    surface = load_scatterer(sphere_path)
    scale_to_diameter(surface, surface_diameter)
    centre_scatterer(surface)
    translate(surface, dx=point[:,0].item(), dy=point[:,1].item(),dz=point[:,2].item())
    surface_points= get_centres_as_points(surface)

    get_edge_data(surface)
 
    H_step = get_cache_or_compute_H(sphere, board, path=root, use_cache_H=use_cache, p_ref=p_ref)

    E_step= compute_E(sphere, surface_points, board,H=H_step, path=root,
                            use_cache_H=use_cache, p_ref=p_ref)
    # print(E_step) #bounds        : x=(-0.0143, 5.70e-3), y=(-0.0100, 0.0100), z=(-0.0100, 0.0100)

    Ax_step, Ay_step, Az_step =BEM_forward_model_grad(surface_points, sphere, board,H=H_step, path=root, use_cache_H=use_cache, p_ref=p_ref)

    f, momentum = force_mesh_surface(x, sphere, board, H=H_step, path=sphere_path,
                                       surface=surface, E=E_step, Ex=Ax_step, Ey=Ay_step, Ez=Az_step, sum_elements=True, return_momentum=True, use_momentum=True,
                                       diameter=surface_diameter)
    f = f.real.squeeze()
    momentum = torch.sum(momentum, dim=2).real.squeeze()
    # f = f - momentum
    
    
    ARFxs.append(f[0].cpu().detach().squeeze())
    ARFys.append(f[1].cpu().detach().squeeze())
    ARFzs.append(f[2].cpu().detach().squeeze())

    pARFxs.append(momentum[0].cpu().detach().squeeze())
    pARFys.append(momentum[1].cpu().detach().squeeze())
    pARFzs.append(momentum[2].cpu().detach().squeeze())


    ratio_zs.append(f[2].cpu().detach().squeeze() / Fz.cpu().detach().squeeze())

Visualise(*ABC(0.1), x, colour_functions=[propagate_abs, propagate_BEM_pressure], 
          colour_function_args=[{'board':board},{"board":board,"scatterer":sphere, "use_cache_H":use_cache}], show=False)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(Fxs,c='r', label='$F_x$')
plt.plot(Fys,c='g', label='$F_y$')
plt.plot(Fzs,c='b', label='$F_z$')

plt.plot(ARFxs,c='r', label='$F_x$',linestyle=':')
plt.plot(ARFys,c='g', label='$F_y$',linestyle=':')
plt.plot(ARFzs,c='b', label='$F_z$',linestyle=':')

plt.plot([i * max(ARFzs) for i in ratio_zs])

print([a/b for a,b in zip(Fzs, ARFzs)])

# plt.plot(pARFzs,c='b', label='$F_z$',linestyle='--')


plt.legend()
plt.show()