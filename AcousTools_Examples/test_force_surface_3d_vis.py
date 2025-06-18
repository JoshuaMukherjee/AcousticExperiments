from acoustools.Mesh import load_scatterer, scale_to_diameter, centre_scatterer, get_centre_of_mass_as_points, translate
from acoustools.Constants import wavelength
from acoustools.Gorkov import get_finite_diff_points_all_axis
from acoustools.Force import compute_force
from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.BEM import force_mesh_surface

import torch

def return_mixed_points(points,stepsize=0.000135156253, stepsize_x=None,stepsize_y=None,stepsize_z=None ):
        '''
        Only works for N=1
        '''
        if points.shape[2] > 1:
            raise RuntimeError("Only for N=1")
        
        if stepsize_x is None:
            stepsize_x = stepsize
        
        if stepsize_y is None:
            stepsize_y = stepsize

        if stepsize_z is None:
            stepsize_z = stepsize


        mixed_points = points.clone().repeat((1,1,13))
        #Set x's
        mixed_points[:,0,1] += stepsize_x
        mixed_points[:,0,2] += stepsize_x
        mixed_points[:,0,3] -= stepsize_x
        mixed_points[:,0,4] -= stepsize_x
        mixed_points[:,0,5] += stepsize_x
        mixed_points[:,0,6] += stepsize_x
        mixed_points[:,0,7] -= stepsize_x
        mixed_points[:,0,8] -= stepsize_x
        #Set y's
        mixed_points[:,1,1] += stepsize_y
        mixed_points[:,1,2] -= stepsize_y
        mixed_points[:,1,3] += stepsize_y
        mixed_points[:,1,4] -= stepsize_y
        mixed_points[:,1,9] += stepsize_y
        mixed_points[:,1,10] += stepsize_y
        mixed_points[:,1,11] -= stepsize_y
        mixed_points[:,1,12] -= stepsize_y
        #Set z's
        mixed_points[:,2,5] += stepsize_z
        mixed_points[:,2,6] -= stepsize_z
        mixed_points[:,2,7] += stepsize_z
        mixed_points[:,2,8] -= stepsize_z
        mixed_points[:,2,9] += stepsize_z
        mixed_points[:,2,10] -= stepsize_z
        mixed_points[:,2,11] += stepsize_z
        mixed_points[:,2,12] -= stepsize_z

        return mixed_points


path = "../BEMMedia"

board = TRANSDUCERS
pt = create_points(1,1)
x = wgs(pt, board=board)
x = add_lev_sig(x, mode='Trap')

start_pos = -2 * wavelength
end_pos = 2 * wavelength
N = 16

d = wavelength/4


ps = []
Fs = []
F_surfaces= []

step = wavelength/4

sphere_pth =  path+"/Sphere-solidworks-lam2.stl"

axis = 'XYZ'

points = get_finite_diff_points_all_axis(points=pt, axis=axis, stepsize=step)
mixed_points = return_mixed_points(points=pt, stepsize=step)

points = torch.cat((points, mixed_points), dim=2)

forces = []
forces_BEM = []

for i in range(points.shape[2]):
    p = points[:,:,i].unsqueeze(2)
    
    sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    translate(sphere, dx = p[:,0].item(), dy = p[:,1].item(), dz = p[:,2].item())
    com = get_centre_of_mass_as_points(sphere)
    

    force = force_mesh_surface(x, sphere, board, path=path)
    force_BEM = compute_force(x, p, board)

    print(p,force)

    forces.append(force.detach().cpu().squeeze(1))
    forces_BEM.append(force_BEM.detach().cpu().squeeze(1))


forces = torch.stack(forces, dim=2)
forces_BEM = torch.stack(forces_BEM, dim=2)

print(forces.shape)

import matplotlib.pyplot as plt

U = forces[:,0]
V = forces[:,1]
W = forces[:,2]

scale = 5

fig = plt.figure()

ax = fig.add_subplot(1,2,1,projection='3d')
ax.quiver(points[:,0,:].cpu().detach().numpy(), points[:,1,:].cpu().detach().numpy(), points[:,2,:].cpu().detach().numpy(), U.cpu().detach().numpy()* scale, V.cpu().detach().numpy()* scale, W.cpu().detach().numpy()* scale)


U = forces_BEM[:,0]
V = forces_BEM[:,1]
W = forces_BEM[:,2]


ax = fig.add_subplot(1,2,2,projection='3d')
ax.quiver(points[:,0,:].cpu().detach().numpy(), points[:,1,:].cpu().detach().numpy(), points[:,2,:].cpu().detach().numpy(), U.cpu().detach().numpy()* scale, V.cpu().detach().numpy()* scale, W.cpu().detach().numpy()* scale)



plt.show()