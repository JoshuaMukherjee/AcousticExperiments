from acoustools.Mesh import load_scatterer
from acoustools.Utilities import create_board, create_points, propagate_abs
from acoustools.BEM import propagate_BEM_pressure, compute_E
from acoustools.Visualiser import Visualise, ABC

import torch

path = '../BEMMedia'
reflector = load_scatterer(path+"/Tatsuki_optimized_reflector.stl")


board = create_board(N=17, z=0)

p = create_points(1,1, 0,0,0.03)
x = torch.rand((1,256,1)) * torch.exp(1j * torch.rand((1,256,1)) * 2 * 3.1415)


def GH_prop(activations, points, scatterer, board, path ):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, return_components=True)
        pressures =  torch.abs(G@H@activations)
        return pressures



Visualise(*ABC(0.1, origin=p), x, res=(500,500), colour_functions=[propagate_abs, GH_prop,propagate_BEM_pressure], 
          colour_function_args=[{'board':board}, {'board':board, 'scatterer':reflector, 'path':path}, {'board':board, 'scatterer':reflector, 'path':path}])


