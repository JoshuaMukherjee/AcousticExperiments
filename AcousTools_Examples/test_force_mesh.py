from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, propagate_abs, transducers, create_points
from acoustools.Force import force_mesh, force_fin_diff, compute_force
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure, BEM_compute_force, force_mesh_surface
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise

import torch, vedo, random

# torch.random.manual_seed(1)

path = "../BEMMedia"

def bounds_to_diameters(bounds):
    x1,x2,y1,y2,z1,z2 = bounds
    print(bounds)
    print(x2-x1,  y2-y1, z2-z1)
    print(x2+x1,  y2+y1, z2+z1)
    print()

def GH_Grad(points, scatterer, transducers=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H=None,
                           path:str="Media"):
    
    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(points=points, scatterer=scatterer, transducers=transducers, use_cache_H=use_cache_H, H=H, path=path, return_components=True)
    return Gx@H, Gy@H, Gz@H




USE_CACHE = True
M = 16
board = transducers(M)

sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
d = c.wavelength/10
scale_to_diameter(sphere,d)
centre_scatterer(sphere)
bounds_to_diameters(sphere.bounds())
V = 4/3 * c.pi * (d/2)**3
# V = c.V

# vedo.show(sphere, axes=1)

com = get_centre_of_mass_as_points(sphere)

print('com',com)

points = get_centres_as_points(sphere) * 1.05
norms = get_normals_as_points(sphere)
areas = get_areas(sphere)


E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
# p = create_points(1,1, x=c.wavelength/7, y=c.wavelength/7, z=c.wavelength/7)
p = create_points(1,1, max_pos=c.wavelength/3, min_pos=-c.wavelength/3)
print('p',p)
Ep,Fp,Gp,H = compute_E(sphere, p, board,path=path, return_components=True, H=H)
x = wgs(p,board=board,A=Ep)
# x = add_lev_sig(x)


forces_x= []
forces_y= []
forces_z= []


diameters = []

As = []

# U  = BEM_gorkov_analytical(x, com, scatterer=sphere, board=board,H=H, path=path).detach().cpu()
# U_force_BEM = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()
U_force = compute_force(x, com, board, V=V).unsqueeze(0)
U_force_fd = force_fin_diff(x,com,board=board, V=V)
U_force_BEM_fd = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()
U_force_BEM = BEM_compute_force(x,com, board, scatterer=sphere, path=path, H=H, V=V)
print('FU PM \t\t',U_force.squeeze())
print("FU PM FD\t",U_force_fd.squeeze())
print("FU BEM\t\t",U_force_BEM.squeeze())
print("FU BEM FD\t",U_force_BEM_fd.squeeze())

# print(U_force_BEM)

r = 20
# Visualise(*ABC(0.1), x,points=[p,com] , colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path}])
# Visualise(*ABC(0.1), x, colour_functions=[propagate_GH])
# Visualise(*ABC(0.01, origin=com), x,points=[p.real,com.real] , colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path}], res=(r,r))

# exit()


N = 16
max_pos = 3*c.wavelength
start = d + c.wavelength
for i in range(N):
    print(i, end = '\r')

    diameter = start + (max_pos/N) * i
    # diameter = (random.random()+1) * c.wavelength

    force= force_mesh_surface(x, sphere, board,H=H,path=path,
                                        diameter=diameter, use_cache_H=USE_CACHE).squeeze().detach()
    
    

    # print(diameter, torch.sum(force))
    forces_x.append(torch.sum(force[0]).detach().cpu())
    forces_y.append(torch.sum(force[1]).detach().cpu())
    forces_z.append(torch.sum(force[2]).detach().cpu())

    diameters.append(diameter)


print(forces_x[0], end=' ')
print(forces_y[0], end=' ')
print(forces_z[0])


print(forces_x[-1], end=' ')
print(forces_y[-1], end=' ')
print(forces_z[-1])


import matplotlib.pyplot as plt
import plotext as plx

plt.plot(diameters, forces_x, color='red')
plt.plot(diameters, forces_y, color='green')
plt.plot(diameters, forces_z, color='blue')
# plt.plot(diameters, As, color='blue')



plt.xlabel('Diameter (m)')
plt.ylabel('Force (N)')

PLT = True
plt_U = False
if PLT:
    if plt_U:
        plt.hlines(U_force[:,0], color='red', linestyles=':', xmin=start, xmax=start+max_pos)
        plt.hlines(U_force[:,1], color='green', linestyles=':', xmin=start, xmax=start+max_pos)
        plt.hlines(U_force[:,2], color='blue', linestyles=':', xmin=start, xmax=start+max_pos)

    plt.show()
else:
    if plt_U:
        plt.plot([start,start+max_pos],[U_force[:,0],U_force[:,0]], color='red')
        plt.plot([start,start+max_pos],[U_force[:,1],U_force[:,1]], color='green')
        plt.plot([start,start+max_pos],[U_force[:,2],U_force[:,2]], color='blue')

    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    plx.from_matplotlib(fig)
    plx.axes_color('white')
    plx.canvas_color('white')
    plx.show()
