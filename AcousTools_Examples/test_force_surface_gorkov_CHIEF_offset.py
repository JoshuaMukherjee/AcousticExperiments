from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, transducers, BOTTOM_BOARD, BOARD_POSITIONS
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Solvers import wgs, translate_hologram, iterative_backpropagation
from acoustools.Constants import wavelength, pi, P_ref as PREF
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer,get_CHIEF_points, translate
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface
from acoustools.Visualiser import ABC, Visualise,force_quiver_3d


import torch, random, math
import matplotlib.pyplot as plt

torch.random.manual_seed(1)
torch.set_printoptions(precision=8)

board=transducers(z=0.243/2)
# board=BOTTOM_BOARD
# p = create_points(1,1,x=wavelength/8, y=wavelength/8, z=wavelength/8)
dz = 0.04
p = create_points(1,1, 0, 0, 0.001)
print(p)
print(dz)

x = iterative_backpropagation(p)
x = add_lev_sig(x)
# x =translate_hologram(x, dz=0.001) 
# x = torch.rand_like(x)

p_ref = (2.214 / 10) * 12
# p_ref = PREF


spherical_arf = [
-5.46E-11,
-1.49E-06,
-1.04E-05,
-2.91E-05,
-5.19E-05,
-6.63E-05,
-6.10E-05,
-3.12E-05,
2.18E-05,
9.10E-05,
1.62E-04,
2.18E-04,
2.48E-04,
2.52E-04,
2.40E-04,
2.25E-04,
2.20E-04,
2.31E-04,
2.62E-04,
3.13E-04,
3.77E-04,
4.48E-04,
5.16E-04,
5.74E-04,
6.12E-04,
6.28E-04,
6.24E-04,
6.08E-04,
5.93E-04,
5.91E-04
]

radius = [
    0.00001,
0.000303621,
0.000597241,
0.000890862,
0.001184483,
0.001478103,
0.001771724,
0.002065345,
0.002358966,
0.002652586,
0.002946207,
0.003239828,
0.003533448,
0.003827069,
0.00412069,
0.00441431,
0.004707931,
0.005001552,
0.005295172,
0.005588793,
0.005882414,
0.006176034,
0.006469655,
0.006763276,
0.007056897,
0.007350517,
0.007644138,
0.007937759,
0.008231379,
0.008525
]

pt = create_points(1,1, 0,0,dz)
Visualise(*ABC(0.1), x, points=pt , colour_functions=[propagate_abs], colour_function_args=[{'board':board, 'p_ref':p_ref}], show=False)
plt.figure()

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

fd_force_x = []
fd_force_y = []
fd_force_z = []

p2 = create_points(1,1,0,0,-0.002)
pressures = []

ds = []

# N = 20
# diameters = torch.logspace(math.log10(start_d), math.log10(max_d), steps=N)
# diameters = torch.linspace(0.0001, 0.5*wavelength, N)


# for i in range(N):
for i in range(len(radius)):
    print(i,end='\r')
    # d = start_d + (max_d/N)*i * 1.010
    # d = max_d * 1.01
    # d = diameters[i]
    d = torch.Tensor([2 * radius[i]])
    v = 4/3 * pi * (d/2)**3

    sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    translate(sphere, dz=dz)
    com = get_centre_of_mass_as_points(sphere)

    # internal_points  = get_CHIEF_points(sphere, P = 1, start='centre')
    # internal_points  = get_CHIEF_points(sphere, P = 50, start='centre', method='uniform', scale=0.002)
    internal_points  = get_CHIEF_points(sphere, P = 50, start='centre', method='uniform', scale = 0.1, scale_mode='diameter-scale')


    # E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache, p_ref=p_ref,internal_points=internal_points)
    E,F,G,H = compute_E(sphere, com ,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method="LU", return_components=True, internal_points=internal_points)
    # Visualise(*ABC(0.02),x, points=internal_points,colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board,'scatterer':sphere,'H':H, 'use_cache_H':cache, 'p_ref':p_ref, "internal_points":internal_points}], res=(100,100))
    # norms = get_normals_as_points(sphere)
    # centres = get_centres_as_points(sphere)
    # force_quiver_3d(centres, norms[:,0], norms[:,1], norms[:,2], scale=0.001)
    # exit()

    ds.append(d)
    # V = c.V

    
    U_force = compute_force(x, com, board, V=v, p_ref=p_ref).squeeze().detach()
    # U_force_BEM = BEM_compute_force(x,com, board, scatterer=sphere, path=path, H=H, V=v).squeeze().detach()

    U_forces_x.append(U_force[0].cpu().detach())
    U_forces_y.append(U_force[1].cpu().detach())
    U_forces_z.append(U_force[2].cpu().detach())

    fd_force = force_fin_diff(x, com, board=board, V=v, p_ref=p_ref)
    fdx = fd_force[:,:,0]
    fdy = fd_force[:,:,1]
    fdz = fd_force[:,:,2]
    fd_force_x.append(fdx.item())
    fd_force_y.append(fdy.item())
    fd_force_z.append(fdz.item())


    dim = 2*wavelength + d.item()
    A_force= force_mesh_surface(x, sphere, board, return_components=False,H=H,path=path,
                                                        diameter=dim, use_cache_H=cache, p_ref=p_ref,internal_points=internal_points).squeeze().detach()
    
    pressure = propagate_BEM_pressure(x, p2, sphere, board=board, H=H, path=path, p_ref=p_ref, internal_points=internal_points)
    pressures.append(pressure.item())
    
    A_forces_x.append(A_force[0].cpu().detach())
    A_forces_y.append(A_force[1].cpu().detach())
    A_forces_z.append(A_force[2].cpu().detach())

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

rs = [d/(2*wavelength) for d in ds]

# fz0 = A_forces_z[0]
# A_forces_z = [x - fz0 for x in A_forces_z]

# plt.subplot(2,1,1)

# plt.plot(rs, U_forces_x, color='r', linestyle=':', label=r'${-\nabla_x U}$')
# plt.plot(rs, U_forces_y, color='g', linestyle=':', label=r'${-\nabla_y U}$')
plt.plot(rs, U_forces_z, color='b', linestyle=':', label=r'${-\nabla_z U}$')

plt.plot(rs,spherical_arf)

# plt.plot(rs, fd_force_x, color='r', linestyle='--', label=r'${-\nabla_x U}$')
# plt.plot(rs, fd_force_y, color='g', linestyle='--', label=r'${-\nabla_y U}$')
# plt.plot(rs, fd_force_z, color='b', linestyle='--', label=r'${-\nabla_z U}$')


plt.plot(rs, A_forces_x, color='r', label='$F_x$')
plt.plot(rs, A_forces_y, color='g', label='$F_y$')
plt.plot(rs, A_forces_z, color='b', label='$F_z$')

# plt.scatter(ds, A_forces_x, color='r')
# plt.scatter(ds, A_forces_y, color='g')
# plt.scatter(ds, A_forces_z, color='b')

# plt.plot(ds, U_forces_BEM_x, color='r', linestyle=':')
# plt.plot(ds, U_forces_BEM_y, color='g', linestyle=':')
# plt.plot(ds, U_forces_BEM_z, color='b', linestyle=':')

plt.legend()

plt.ylabel('Force (N)')
plt.xlabel('Particle Radius ($\lambda$)')

# # plt.ylim(-5e-3, 5e-3)

# plt.subplot(2,1,2)

# plt.plot(rs, pressures)
# plt.ylabel('Internal Pressure (Pa)')
# plt.xlabel('Particle Radius ($\lambda$)')


plt.show()





