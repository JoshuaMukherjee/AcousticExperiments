from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, transducers
from acoustools.Force import compute_force
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength, pi
from acoustools.Mesh import load_scatterer, get_centres_as_points, translate, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface_curl
from acoustools.Visualiser import ABC, Visualise


import torch, random, math
import matplotlib.pyplot as plt

# torch.random.manual_seed(1)
torch.set_printoptions(precision=8)

M = 16
board = transducers(M)
# p = create_points(1,1,x=wavelength/8, y=wavelength/8, z=wavelength/8)
p = create_points(1,1, max_pos=wavelength/3, min_pos=-wavelength/3)

x = wgs(p, board=board)
x = add_lev_sig(x, board=board, board_size=M**2)

# Visualise(*ABC(0.1), x,points=[p.real] , colour_functions=[propagate_abs], colour_function_args=[{'board':board}])

path = "../BEMMedia"

cache = True

start_d = wavelength/32
max_d = wavelength * 2
N = 32

curls = []

ds = []

diameters = torch.logspace(math.log10(start_d), math.log10(max_d), steps=N)

for i in range(N):
    print(i,end='\r')
    # d = start_d + (max_d/N)*i * 1.01
    # d = max_d * 1.01
    d = diameters[i]
    v = 4/3 * pi * (d/2)**3

    sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    com = get_centre_of_mass_as_points(sphere)

    E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache)


    ds.append(d)

    surface = load_scatterer(sphere_pth)
    scale_to_diameter(surface,(wavelength*4), reset=False, origin=True)
    centre_scatterer(surface)
    object_com = get_centre_of_mass_as_points(sphere)
    translate(surface, dx = object_com[:,0].item(), dy=object_com[:,1].item(), dz = object_com[:,2].item())

    curl = force_mesh_surface_curl(x, sphere,board, H=H, path=path, surface=surface, magnitude=True)
    print(d, curl)


   
    curls.append(curl.cpu().detach())


# plt.plot(ds, divs)
# plt.show()