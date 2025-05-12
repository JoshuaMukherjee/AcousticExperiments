from acoustools.Mesh import load_scatterer, scale_to_diameter, centre_scatterer, translate, get_centres_as_points, get_centre_of_mass_as_points
from acoustools.Utilities import TRANSDUCERS
from acoustools.Constants import wavelength, R
from acoustools.Force import force_fin_diff
from acoustools.BEM import BEM_gorkov_analytical, compute_E
from acoustools.Solvers import wgs

import random, torch

N = 30

path = "../BEMMedia"
sphere_pth =  path+"/Sphere-lam2.stl"


board = TRANSDUCERS

forces_x = []
forces_y = []
forces_z = []

forces_BEM_x = []
forces_BEM_y = []
forces_BEM_z = []

for i in range(N):

    print(i, end='\r')

    x = random.random() * 0.04
    y = random.random() * 0.04
    z = random.random() * 0.04
    # diameter = random.random() * wavelength/10
    diameter = 2*R
    V = 4/3 * 3.1415 * (diameter/2)**3

    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,diameter)
    centre_scatterer(sphere)
    translate(sphere,dx=x, dy=y,dz=z)
    
    points = get_centres_as_points(sphere)
    com = get_centre_of_mass_as_points(sphere)

    E,F,G,H = compute_E(sphere, points, board,path=path, return_components=True)
    x = wgs(points,board=board,A=E)

    U_force_BEM = force_fin_diff(x,com, U_function=BEM_gorkov_analytical, V=V,
                             U_fun_args={'scatterer':sphere,'H':H, 'path':path}, board=board).detach().cpu()
    
    U_force = force_fin_diff(x,com,V=V).detach().cpu()

    print(U_force_BEM)
    print(U_force)
    print()


    forces_x.append(torch.sum(U_force[:,0]).detach().cpu())
    forces_y.append(torch.sum(U_force[:,1]).detach().cpu())
    forces_z.append(torch.sum(U_force[:,2]).detach().cpu())


    forces_BEM_x.append(torch.sum(U_force_BEM[:,0]).detach().cpu())
    forces_BEM_y.append(torch.sum(U_force_BEM[:,1]).detach().cpu())
    forces_BEM_z.append(torch.sum(U_force_BEM[:,2]).detach().cpu())

import matplotlib.pyplot as plt

plt.subplot(1,3,1)
plt.scatter(forces_x, forces_BEM_x)

plt.subplot(1,3,2)
plt.scatter(forces_y, forces_BEM_y)

plt.subplot(1,3,3)
plt.scatter(forces_z, forces_BEM_z)

plt.show()