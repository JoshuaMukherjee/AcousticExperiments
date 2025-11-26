from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, transducers, BOTTOM_BOARD, create_board
from acoustools.Force import compute_force
from acoustools.Solvers import wgs, translate_hologram, iterative_backpropagation
from acoustools.Constants import wavelength, pi, P_ref as PREF
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer,get_CHIEF_points, merge_scatterers, translate
from acoustools.BEM import BEM_compute_force, compute_E, propagate_BEM_pressure, force_mesh_surface
from acoustools.Visualiser import ABC, Visualise,force_quiver_3d


import torch, random, math
import matplotlib.pyplot as plt

torch.random.manual_seed(1)
torch.set_printoptions(precision=8)

# board=create_board(N=17, z=-0.06)
board=BOTTOM_BOARD
# p = create_points(1,1,x=wavelength/8, y=wavelength/8, z=wavelength/8)
p = create_points(1,1, 0,0,-0.06)

# x = add_lev_sig(x)
# x =translate_hologram(x, dz=0.001)

p_ref = 12* (2.214 / 10) 
# p_ref = PREF

# Visualise(*ABC(0.1), x,points=[p.real] , colour_functions=[propagate_abs], colour_function_args=[{'board':board, 'p_ref':p_ref}])

path = "../BEMMedia"
cache = False


reflector = load_scatterer(path+'/flat-lam1.stl', roty=180)
CHIEF_pts = create_points(5,1, min_pos=-0.02, max_pos=0.02)
CHIEF_pts[:,:2] *= 3
CHIEF_pts[:,2] += 0.03


E = compute_E(reflector, p, board, p_ref=p_ref, path=path)
x = iterative_backpropagation(p, board=board, A=E)


# Visualise(*ABC(0.12),x ,colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board,'scatterer':reflector, 'use_cache_H':cache, 'p_ref':p_ref}], res=(200,200), vmax=6000)


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

p2 = create_points(1,1,0,0,-0.062)
pressures = []

ds = []

N = 250
# diameters = torch.logspace(math.log10(start_d), math.log10(max_d), steps=N)
diameters = torch.linspace(1*wavelength, 4*wavelength, N)
# diameters = torch.linspace(0.88 * wavelength * 2, 0.89 * 2 * wavelength , N)

for i in range(N):
    print(i)
    # d = start_d + (max_d/N)*i * 1.010
    # d = max_d * 1.01
    d = diameters[i]
    v = 4/3 * pi * (d/2)**3

    sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    translate(sphere, dz=-0.06)
    com = get_centre_of_mass_as_points(sphere)

    scatterer = merge_scatterers(reflector, sphere)

    # internal_points  = get_CHIEF_points(sphere, P = 1, start='centre')
    # internal_points  = get_CHIEF_points(sphere, P = 50, start='centre', method='uniform', scale=0.002)
    P = int(i/5) + 20
    internal_points  = get_CHIEF_points(sphere, P = P, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
    s_pts = internal_points.clone()
    # internal_points = torch.cat([internal_points, CHIEF_pts], dim=2)
    print(internal_points.shape)


    # E,F,G,H = compute_E(sphere, com, board,path=path, return_components=True, use_cache_H=cache, p_ref=p_ref,internal_points=internal_points)
    E,F,G,H = compute_E(scatterer, com ,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method="LU", return_components=True, internal_points=internal_points)

    ds.append(d)
    # V = c.V

    
    U_force = compute_force(x, com, board, V=v, p_ref=p_ref).squeeze().detach()
    # U_force_BEM = BEM_compute_force(x,com, board, scatterer=sphere, path=path, H=H, V=v).squeeze().detach()

    U_forces_x.append(U_force[0].cpu().detach())
    U_forces_y.append(U_force[1].cpu().detach())
    U_forces_z.append(U_force[2].cpu().detach())

    dim = 4*wavelength + d.item()
    A_force= force_mesh_surface(x, scatterer, board, return_components=False,H=H,path=path,
                                                        diameter=dim, use_cache_H=cache, p_ref=p_ref,internal_points=internal_points).squeeze().detach()
    
    pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref, internal_points=internal_points)
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

    # Visualise(*ABC(0.12),x, points=internal_points,colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board,'scatterer':scatterer,'H':H, 'use_cache_H':cache, 'p_ref':p_ref, "internal_points":internal_points}], res=(200,200), vmax=6000)
    # norms = get_normals_as_points(sphere)
    # centres = get_centres_as_points(sphere)
    # force_quiver_3d(centres, norms[:,0], norms[:,1], norms[:,2], scale=0.001)
    # exit()


    # Visualise(*ABC(0.05), x,points=[p.real,com.real] , colour_functions=[propagate_BEM_pressure, propagate_abs], 
            #   colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path},{}])

    # exit()

rs = [d/(2*wavelength) for d in ds]

plt.subplot(2,1,1)

plt.plot(rs, U_forces_x, color='r', linestyle=':', label=r'${-\nabla_x U}$')
plt.plot(rs, U_forces_y, color='g', linestyle=':', label=r'${-\nabla_y U}$')
plt.plot(rs, U_forces_z, color='b', linestyle=':', label=r'${-\nabla_z U}$')


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

# plt.ylim(-5e-3, 5e-3)

plt.subplot(2,1,2)

plt.plot(rs, pressures)
plt.ylabel('Internal Pressure (Pa)')
plt.xlabel('Particle Radius ($\lambda$)')


plt.show()





