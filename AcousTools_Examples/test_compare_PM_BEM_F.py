from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer
from acoustools.Utilities import BOTTOM_BOARD, TRANSDUCERS, TOP_BOARD, add_lev_sig, propagate_abs, transducers, create_points
from acoustools.Force import force_mesh, force_fin_diff, compute_force
from acoustools.Solvers import wgs
from acoustools.BEM import BEM_forward_model_grad, compute_E, BEM_gorkov_analytical, propagate_BEM_pressure
import acoustools.Constants as c
from acoustools.Visualiser import ABC, Visualise, Visualise_mesh, force_quiver_3d
from acoustools.Gorkov import gorkov_analytical

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
M = 16
board = transducers(M)

sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
d = c.wavelength/16
scale_to_diameter(sphere,d)
centre_scatterer(sphere)
bounds_to_diameters(sphere.bounds())



N = 100

Fs_x = []
Fs_y = []
Fs_z = []

F_bems_x = []
F_bems_y = []
F_bems_z = []


for i in range(N):
    print(i, end='\r')
    p = create_points(1,1)
    x = wgs(p,board=board)

    F = compute_force(x,p,board).cpu().detach().squeeze()
    F_bem = force_fin_diff(x,p,board=board, stepsize=c.wavelength/32,U_function=BEM_gorkov_analytical, 
                           U_fun_args={'scatterer':sphere,'path':path}).squeeze().cpu().detach()
    F_fd_bem = force_fin_diff(x,p,board=board, stepsize=c.wavelength/32, U_function=gorkov_analytical).squeeze().cpu().detach()
    

    # print(F)
    # print(F_bem)
    # print(F_fd_bem)
    # print()

    Fs_x.append(F[0])
    Fs_y.append(F[1])
    Fs_z.append(F[2])


    F_bems_x.append(F_bem[0])
    F_bems_y.append(F_bem[1])
    F_bems_z.append(F_bem[2])



# exit()
import matplotlib.pyplot as plt

plt.scatter(Fs_x, F_bems_x,color='r')
plt.scatter(Fs_y, F_bems_y,color='g')
plt.scatter(Fs_z, F_bems_z,color='b')
plt.show()

