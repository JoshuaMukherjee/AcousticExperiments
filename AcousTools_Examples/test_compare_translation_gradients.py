from acoustools.Utilities import create_points, TRANSDUCERS, DTYPE, device
from acoustools.Constants import wavelength, P_ref as cPref
from acoustools.Export.Holo import load_holograms

from acoustools.BEM import get_cache_or_compute_H, compute_E, BEM_forward_model_grad
from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, centre_scatterer, translate

from acoustools.Utilities.Piston_model_gradients import forward_model_grad

import torch
import random

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
# path = root + '/Sphere-lam2.stl'
diameter=0.0001
volume = 4/3 * 3.1415 * (diameter/2)**3
m=1e-10
weight = -1* (m) * 9.81

surface_diameter = diameter + 1 * wavelength



# h_path = 'Optimsed_holo_1755697974095288200.holo'
h_path = 'AcousTools_Examples/data/holos/Optimsed_holo_1755780306180471000.holo'

x = load_holograms(h_path)[0]

start_z = 10*wavelength
start = create_points(1,1,0,0,start_z/2)

N = 20

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
    

    sphere = load_scatterer(sphere_path)
    scale_to_diameter(sphere, diameter)
    centre_scatterer(sphere)    
    
    translate(sphere, dx=point[:,0].item(), dy=point[:,1].item(),dz=point[:,2].item())

    surface = load_scatterer(sphere_path)
    scale_to_diameter(surface, surface_diameter)
    centre_scatterer(surface)
    translate(surface, dx=point[:,0].item(), dy=point[:,1].item(),dz=point[:,2].item())
    surface_points= get_centres_as_points(surface)
    
    n = random.randint(0,surface_points.shape[2])
    p=surface_points[:,:,n].unsqueeze_(2)

    H_step = get_cache_or_compute_H(sphere, board, path=root, use_cache_H=use_cache, p_ref=p_ref)

    E_step= compute_E(sphere, p, board,H=H_step, path=root,
                            use_cache_H=use_cache, p_ref=p_ref)
    # print(E_step) #bounds        : x=(-0.0143, 5.70e-3), y=(-0.0100, 0.0100), z=(-0.0100, 0.0100)

    Ax, Ay, Az =BEM_forward_model_grad(p, sphere, board,H=H_step, path=root, use_cache_H=use_cache, p_ref=p_ref)
    Fx, Fy, Fz = forward_model_grad(p, board, p_ref=p_ref)


    print((Az@x))
    print(Fz@x)
    print((Az@x) / (Fz@x))
    print()