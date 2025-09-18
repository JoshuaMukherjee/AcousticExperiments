from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, transducers, BOTTOM_BOARD
from acoustools.Force import compute_force
from acoustools.Solvers import wgs, translate_hologram
from acoustools.Constants import wavelength, pi, P_ref as PREF
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer, insert_parasite
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface
from acoustools.Visualiser import ABC, Visualise,force_quiver_3d


import torch, random, math
import matplotlib.pyplot as plt

# torch.random.manual_seed(1)
torch.set_printoptions(precision=8)

M = 16
board = transducers(M)
# board=BOTTOM_BOARD
# p = create_points(1,1,x=wavelength/8, y=wavelength/8, z=wavelength/8)
p = create_points(1,1, 0,0,0)

x = wgs(p, board=board)
x = add_lev_sig(x, board=board, board_size=M**2)
x =translate_hologram(x, dz=0.001)

p_ref = 12* (2.214 / 10) 
# p_ref = PREF

# Visualise(*ABC(0.1), x,points=[p.real] , colour_functions=[propagate_abs], colour_function_args=[{'board':board, 'p_ref':p_ref}])

path = "../BEMMedia"


cache = False

# start_d =0.00001
# max_d = wavelength

U_forces_x = []
U_forces_y = []
U_forces_z = []

U_forces_BEM_x = []
U_forces_BEM_y = []
U_forces_BEM_z = []

A_forces_x = []
A_forces_y = []
A_forces_z = []

infected_A_forces_x = []
infected_A_forces_y = []
infected_A_forces_z = []



N = 100
# diameters = torch.logspace(math.log10(start_d), math.log10(max_d), steps=N)
diameters = torch.linspace(0.005, 3*wavelength, N)
# diameters = torch.linspace(wavelength*1.2, wavelength*1.8, N) * 2

distances = [i+1 for i in range(5)]
distances = [wavelength/d for d in distances]

distances = [0.87, 0.8725, 0.875, 0.8775]

# parasite_path = '/Octohedron_lam2.stl'
parasite_path='/Sphere-lam2.stl'

for delta in distances:
    infected_A_forces_z_d = []
    ds = []
    U_forces_z = []
    A_forces_z = []


    for i in range(N):
        print(delta, i,end='\r')
        # d = start_d + (max_d/N)*i * 1.010
        # d = max_d * 1.01
        d = diameters[i]
        v = 4/3 * pi * (d/2)**3

        sphere_pth =  path+"/Sphere-lam2.stl"
        sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
        scale_to_diameter(sphere,d)
        centre_scatterer(sphere)
        com = get_centre_of_mass_as_points(sphere)

        # infected_sphere = insert_parasite(sphere, parasite_size=(d-delta) / math.sqrt(2), parasite_path=parasite_path)
        infected_sphere = insert_parasite(sphere, parasite_size=d*delta, parasite_path=parasite_path)

        E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache, p_ref=p_ref)
        infected_E,infected_F,infected_G,infected_H = compute_E(infected_sphere, com, board,path=path, return_components=True, use_cache_H=cache, p_ref=p_ref)
        
        # Visualise(*ABC(0.02),x,colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board,'scatterer':infected_sphere,'H':infected_H, 'use_cache_H':cache, 'p_ref':p_ref}], res=(100,100), vmax=8000)
        # norms = get_normals_as_points(sphere)
        # centres = get_centres_as_points(sphere)
        # force_quiver_3d(centres, norms[:,0], norms[:,1], norms[:,2], scale=0.001)
        # exit()

        ds.append(d)
        # V = c.V

        
        U_force = compute_force(x, com, board, V=v, p_ref=p_ref).squeeze().detach()
        # U_force_BEM = BEM_compute_force(x,com, board, scatterer=sphere, path=path, H=H, V=v).squeeze().detach()

        # U_forces_x.append(U_force[0].cpu().detach())
        # U_forces_y.append(U_force[1].cpu().detach())
        U_forces_z.append(U_force[2].cpu().detach())

        dim = 3*wavelength + d.item()
        A_force= force_mesh_surface(x, sphere, board, return_components=False,H=H,path=path,
                                                            diameter=dim, use_cache_H=cache, p_ref=p_ref).squeeze().detach()
        
        # A_forces_x.append(A_force[0].cpu().detach())
        # A_forces_y.append(A_force[1].cpu().detach())
        A_forces_z.append(A_force[2].cpu().detach())

        infected_A_force= force_mesh_surface(x, infected_sphere, board, return_components=False,H=infected_H,path=path,
                                                            diameter=dim, use_cache_H=cache, p_ref=p_ref).squeeze().detach()
        
        # infected_A_forces_x.append(infected_A_force[0].cpu().detach())
        # infected_A_forces_y.append(infected_A_force[1].cpu().detach())
        infected_A_forces_z_d.append(infected_A_force[2].cpu().detach())

        # U_forces_BEM_x.append(U_force_BEM[0])
        # U_forces_BEM_y.append(U_force_BEM[1])
        # U_forces_BEM_z.append(U_force_BEM[2])
        print(d, v, dim)
        print('U',U_force)
        print('A',A_force)
        print(U_force/A_force, (U_force/A_force)[1] / (U_force/A_force)[2])
        print()

        # Visualise(*ABC(0.05), x,points=[p.real,com.real] , colour_functions=[propagate_BEM_pressure, propagate_abs], 
                #   colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path},{}])

        # exit()
    infected_A_forces_z.append(infected_A_forces_z_d)

rs = [d/(2*wavelength) for d in ds]


# plt.plot(rs, U_forces_x, color='r', linestyle=':', label=r'${-\nabla_x U}$')
# plt.plot(rs, U_forces_y, color='g', linestyle=':', label=r'${-\nabla_y U}$')
plt.plot(rs, U_forces_z, color='b', linestyle=':', label=r'${-\nabla_z U}$')
plt.plot(rs, A_forces_z, label='$F_z$', linestyle="-.")


for i in range(len(distances)):
    # plt.plot(rs, A_forces_x, color='r', label='$F_x$', linestyle="-.")
    # plt.plot(rs, A_forces_y, color='g', label='$F_y$', linestyle="-.")
    

    # plt.plot(rs, infected_A_forces_x, color='r', label='$F_x$')
    # plt.plot(rs, infected_A_forces_y, color='g', label='$F_y$')
    plt.plot(rs, infected_A_forces_z[i], label=f'Parasite diameter = d-{distances[i]} $F_z$')

    # plt.scatter(ds, A_forces_x, color='r')
    # plt.scatter(ds, A_forces_y, color='g')
    # plt.scatter(ds, A_forces_z, color='b')

    # plt.plot(ds, U_forces_BEM_x, color='r', linestyle=':')
    # plt.plot(ds, U_forces_BEM_y, color='g', linestyle=':')
    # plt.plot(ds, U_forces_BEM_z, color='b', linestyle=':')

plt.legend()

plt.ylabel('Force (N)')
plt.xlabel('Particle Radius ($\lambda$)')

plt.ylim(-5e-3, 5e-3)

plt.show()





