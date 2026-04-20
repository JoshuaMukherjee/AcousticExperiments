from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, rotate, get_edge_data, get_CHIEF_points
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_phase, propagate_BEM_pressure, compute_G
from acoustools.Visualiser import Visualise, ABC
from acoustools.Utilities import transducers, create_board

import vedo, torch

path = '../BEMMedia'

name = '/Metamaterials/300um/22_brick8-300um.stl'

brick = load_scatterer(path + name)
centre_scatterer(brick)


scale_to_diameter(brick, wavelength)
rotate(brick, axis=(1,0,0), rot=90)
centre_scatterer(brick)
# brick.subdivide(n=2)
get_edge_data(brick)

# vedo.show(brick, axes=1)
# exit()

board = create_board(2, -0.01)
x = 1 * torch.exp(1j * torch.ones(1,1))

P = 5
internal_points = get_CHIEF_points(brick, P=P, method='tetra-random')
# internal_points = None

H_CHIEF= get_cache_or_compute_H(brick, board, use_cache_H=False, path=path, internal_points=internal_points)
H= get_cache_or_compute_H(brick, board, use_cache_H=False, path=path)

def render_GH_pressure(activations, points, board, scatterer, H, **params):
    G = compute_G(points, scatterer)
    GH = G@H
    return torch.abs(GH@activations)

def render_GH_phase(activations, points, board, scatterer, H, **params):
    G = compute_G(points, scatterer)
    GH = G@H
    return torch.angle(GH@activations)


A,B,C = ABC(0.01)

Visualise(A,B,C, x, res = (100,100), points =internal_points,
        colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure, '-', propagate_BEM_phase,propagate_BEM_phase,'-'], 
        colour_function_args=[{'path':path,'H':H,'board':board, 'scatterer':brick}, 
                              {'path':path,'H':H_CHIEF,'board':board, 'scatterer':brick, 'internal_points':internal_points},
                              {'ids':[0,1]},
                              {'path':path,'H':H,'board':board, 'scatterer':brick},
                              {'path':path,'H':H_CHIEF,'board':board, 'scatterer':brick, 'internal_points':internal_points},
                              {'ids':[3,4]}],
        link_ax=None, cmaps=['hot','hot','hot', 'hsv','hsv','hsv'], arrangement=(2,3),
        vmax = [100,200,None, None, None, 0.001],
        vmin = [0,0,None, None, None, -0.001]
          
        )