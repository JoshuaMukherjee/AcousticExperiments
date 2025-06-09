from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs
from acoustools.Force import compute_force
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength, pi
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface
from acoustools.Visualiser import ABC, Visualise


import torch
import matplotlib.pyplot as plt

torch.random.manual_seed(1)
torch.set_printoptions(precision=8)


board = TRANSDUCERS
p = create_points(1,1,x=wavelength/8, y=wavelength/8, z=wavelength/8)

x = wgs(p, board=board)
x = add_lev_sig(x)



path = "../BEMMedia"

cache = True

start_d = wavelength/64
max_d = wavelength
N = 16

U_forces_x = []
U_forces_y = []
U_forces_z = []

U_forces_BEM_x = []
U_forces_BEM_y = []
U_forces_BEM_z = []

A_forces_x = []
A_forces_y = []
A_forces_z = []

momentum_x = []
momentum_y = []
momentum_z = []



ds = []

for i in range(N):
    print(i,end='\r')
    d = start_d + (max_d/N)*i * 1.01
    # d = max_d * 1.01
    v = 4/3 * pi * (d/2)**3

    sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    com = get_centre_of_mass_as_points(sphere)

    E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache)


    ds.append(d)



# V = c.V

    
    U_force = compute_force(x, com, board, V=v).squeeze().detach()
    # U_force_BEM = BEM_compute_force(x,com, board, scatterer=sphere, path=path, H=H, V=v).squeeze().detach()

    U_forces_x.append(U_force[0])
    U_forces_y.append(U_force[1])
    U_forces_z.append(U_force[2])

    res= force_mesh_surface(x, sphere, board, return_momentum=True,H=H,path=path,
                                                        diameter=wavelength*2, use_cache_H=cache)
    
    A_force, momentum = res

    A_force = A_force.squeeze().detach()
    momentum = torch.sum(momentum.squeeze().detach(), dim=1)

    # print(A_force.shape, momentum.shape)
    
    A_forces_x.append(A_force[0] - momentum[0])
    A_forces_y.append(A_force[1] - momentum[1])
    A_forces_z.append(A_force[2] - momentum[2])

    momentum_x.append(momentum[0])
    momentum_y.append(momentum[1])
    momentum_z.append(momentum[2])

    # U_forces_BEM_x.append(U_force_BEM[0])
    # U_forces_BEM_y.append(U_force_BEM[1])
    # U_forces_BEM_z.append(U_force_BEM[2])
    print(d, v)
    print('U',U_force)
    print('A',A_force - momentum)
    print('P',momentum)
    print('F', A_force)
    print(U_force/A_force, (U_force/A_force)[1] / (U_force/A_force)[2])
    print()

    # Visualise(*ABC(0.05), x,points=[p.real,com.real] , colour_functions=[propagate_BEM_pressure, propagate_abs], 
            #   colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path},{}])

    # exit()



plt.plot(ds, U_forces_x, color='r', linestyle=':')
plt.plot(ds, U_forces_y, color='g', linestyle=':')
plt.plot(ds, U_forces_z, color='b', linestyle=':')

plt.plot(ds, momentum_x, color='r', linestyle='--')
plt.plot(ds, momentum_y, color='g', linestyle='--')
plt.plot(ds, momentum_z, color='b', linestyle='--')


plt.plot(ds, A_forces_x, color='r')
plt.plot(ds, A_forces_y, color='g')
plt.plot(ds, A_forces_z, color='b')

# plt.plot(ds, U_forces_BEM_x, color='r', linestyle=':')
# plt.plot(ds, U_forces_BEM_y, color='g', linestyle=':')
# plt.plot(ds, U_forces_BEM_z, color='b', linestyle=':')

plt.show()





