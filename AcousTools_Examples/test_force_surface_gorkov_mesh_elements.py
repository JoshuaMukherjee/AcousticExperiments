from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, transducers
from acoustools.Force import compute_force
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength, pi
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface
from acoustools.Visualiser import ABC, Visualise


import torch, random
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

start_d = wavelength * 3
max_d = wavelength * 5
N = 32

U_forces_x = []
U_forces_y = []
U_forces_z = []

U_forces_BEM_x = []
U_forces_BEM_y = []
U_forces_BEM_z = []

A_forces_x = []
A_forces_y = []
A_forces_z = []

ds = []

INDEX = random.randint(1,1000)
print(INDEX)

diameters = torch.linspace(start_d, max_d, steps=N)



sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
scale_to_diameter(sphere,wavelength)
centre_scatterer(sphere)
com = get_centre_of_mass_as_points(sphere)

E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache)

for i in range(N):
    print(i,end='\r')
    # d = start_d + (max_d/N)*i * 1.01
    # d = max_d * 1.01
    d = diameters[i].item()
    ds.append(d)

    A_force= force_mesh_surface(x, sphere, board, return_components=False,H=H,path=path,
                                                        diameter=d, use_cache_H=cache, sum_elements=False).squeeze().detach()
    
    print(A_force[0,INDEX])

    A_forces_x.append(A_force[0,INDEX])
    A_forces_y.append(A_force[1,INDEX])
    A_forces_z.append(A_force[2,INDEX])

 

plt.plot(ds, A_forces_x, color='r', label='$F_x$')
plt.plot(ds, A_forces_y, color='g', label='$F_y$')
plt.plot(ds, A_forces_z, color='b', label='$F_z$')


plt.legend()

plt.ylabel('Force (N)')
plt.xlabel('Particle Diameter (m)')

plt.show()





