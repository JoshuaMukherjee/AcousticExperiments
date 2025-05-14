from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, propagate_abs, transducers, create_points
from acoustools.Force import force_mesh, force_fin_diff, compute_force
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise, Visualise_mesh, force_quiver_3d

import torch, vedo

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
M = 2
board = transducers(M)

sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
d = c.wavelength/16
scale_to_diameter(sphere,d)
centre_scatterer(sphere)
bounds_to_diameters(sphere.bounds())

v_sphere = 4/3 * 3.1415 * (d/2)**3

# Visualise_mesh(sphere)

# vedo.show(sphere, axes=1)

com = get_centre_of_mass_as_points(sphere)

print('com',com)

points = get_centres_as_points(sphere) 
norms = get_normals_as_points(sphere)
areas = get_areas(sphere)
# force_quiver_3d(points, norms[:,0], norms[:,1], norms[:,2], scale=1e-5)
print('norm sum',torch.sum(norms, dim=2))
print('CV',torch.sum(norms*areas, dim=2) / (3*v_sphere))
print()
# exit()

E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
# x = wgs(points,board=board,A=E)
x = wgs(com +create_points(1,1,0.01,0.01,0.01),board=board)
# x = add_lev_sig(x, board=board, board_size=M**2)

forces_x= []
forces_y= []
forces_z= []


diameters = []

As = []

# U  = BEM_gorkov_analytical(x, com, scatterer=sphere, board=board,H=H, path=path).detach().cpu()
# U_force_BEM = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()
U_force = compute_force(x, com, board).squeeze()
U_force_fd = force_fin_diff(x,com,board=board, stepsize=c.wavelength/10).squeeze()
print(U_force)
print(U_force_fd)
# print(U_force_BEM)
print()
# exit()

def propagate_GH(activations, points,board=board):
    E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
    return propagate_abs(activations, points, board, A=G@H)

# Visualise(*ABC(0.1), x,points=com, colour_functions=[propagate_BEM_pressure, propagate_GH], colour_function_args=[{'scatterer':sphere, "H":H, 'board':board, "path":path},{}])
# Visualise(*ABC(0.1), x, colour_functions=[propagate_GH])
# exit()

for i in range(1,16):
    print(i, end = '\r')

    diameter = d + 2*c.wavelength + c.wavelength/8 * i
    # surface  = sphere.copy()
    surface = load_scatterer(sphere_pth)
    # print(surface)
    scale_to_diameter(surface,diameter, reset=False, origin=False)

    centre_scatterer(surface)


    points = get_centres_as_points(surface)
    norms = get_normals_as_points(surface)
    areas = get_areas(surface)

    As.append(torch.mean(areas))

    E,F,G,H = compute_E(sphere, points, board,path=path, H=H, return_components=True)
    
    # GH = G@H
    # force = force_mesh(x, points,norms,areas,board=board,F=GH, use_momentum=True,
    #                 grad_function=GH_Grad, grad_function_args={'scatterer':sphere,
    #                                                                             'H':H,
    #                                                                             'path':path})
    
    force = force_mesh(x, points,norms,areas,board=board,F=E, use_momentum=True,
                    grad_function=BEM_forward_model_grad, grad_function_args={'scatterer':sphere,
                                                                                'H':H,
                                                                                'path':path})
    

    # print(diameter, torch.sum(force))
    forces_x.append(torch.sum(force[:,0]).detach().cpu())
    forces_y.append(torch.sum(force[:,1]).detach().cpu())
    forces_z.append(torch.sum(force[:,2]).detach().cpu())

    diameters.append(diameter)


print(forces_x[0], end=' ')
print(forces_y[0], end=' ')
print(forces_z[0])


print(forces_x[-1], end=' ')
print(forces_y[-1], end=' ')
print(forces_z[-1])

print(forces_x[-1] / U_force[0], end=' ')
print(forces_y[-1] / U_force[1], end=' ')
print(forces_z[-1] / U_force[2], end = ' ')
print((forces_x[-1] / U_force[0]) / (forces_z[-1] / U_force[2]))



import matplotlib.pyplot as plt
import plotext as plx

plt.plot(diameters, forces_x, color='red')
plt.plot(diameters, forces_y, color='green')
plt.plot(diameters, forces_z, color='blue')
# plt.plot(diameters, As, color='blue')



plt.xlabel('Diameter (m)')
plt.ylabel('Force (N)')

PLT = True
if PLT:
    plt.hlines(U_force[0].cpu().detach(), color='red', linestyles=':', xmin=d + 0.01, xmax=diameter)
    plt.hlines(U_force[1].cpu().detach(), color='green', linestyles=':', xmin=d + 0.01, xmax=diameter)
    plt.hlines(U_force[2].cpu().detach(), color='blue', linestyles=':', xmin=d + 0.01, xmax=diameter)

    plt.show()
else:
    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    plx.from_matplotlib(fig)
    plx.axes_color('white')
    plx.canvas_color('white')
    plx.show()
