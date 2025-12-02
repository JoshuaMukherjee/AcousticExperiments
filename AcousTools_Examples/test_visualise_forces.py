from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Force import compute_force

from acoustools.Mesh import load_scatterer, scale_to_diameter, centre_scatterer, translate, get_centre_of_mass_as_points
from acoustools.BEM.Force import force_mesh_surface

from acoustools.Constants import wavelength

from acoustools.Visualiser import Visualise, ABC

import torch

board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)

root = '../BEMMedia'
path = root + '/Sphere-solidworks-lam2.stl'


d = 1e-4
v = 4/3 * 3.1415 * (d/2)**2


sphere = load_scatterer(path)
scale_to_diameter(sphere, d)
centre_scatterer(sphere)

def force_z(activations, points, board):
    _,_,fz = compute_force(activations, points, board, return_components=True, V=v)
    return fz

def force_surface_z(activations, points, board):

    Fzs = []
    for i in range(points.shape[2]):
        print(i, end='\r')
        p = points[:,:,i].squeeze()
        centre_scatterer(sphere)
        translate(sphere, *p)

        F = force_mesh_surface(activations, sphere, board, diameter=2*wavelength, path=root, sum_elements=True, use_cache_H=False).real

        Fz = F[:,2].unsqueeze(0)

        Fzs.append(Fz)
    
    return torch.stack(Fzs, dim=1)






Visualise(*ABC(0.01), x, link_ax=None, res=(10,10),
        cmaps=['hot', 'Spectral_r', 'Spectral_r'],
        colour_functions=[propagate_abs, force_z, force_surface_z],
        colour_function_args=[{'board':board}, {'board':board}, {'board':board}])