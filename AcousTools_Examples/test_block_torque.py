from acoustools.Mesh import load_scatterer, get_edge_data, scale_to_diameter, get_CHIEF_points
from acoustools.Utilities import TRANSDUCERS, create_points
from acoustools.BEM import compute_E, propagate_BEM_pressure, torque_mesh_surface
from acoustools.Solvers import wgs

from acoustools.Visualiser import Visualise, ABC

import torch

board = TRANSDUCERS

path = '../BEMMedia/'
block_pth = path + 'Block-lam4.stl'

block = load_scatterer(block_pth)

d = 0.02
scale_to_diameter(block, d)
get_edge_data(block)


p = create_points(2,1, x=(-0.01,0.01), y=(0,0), z=(-0.005,0.005))

internal_points  = get_CHIEF_points(block, P = 50, start='surface', method='uniform', scale = 0.2, scale_mode='diameter-scale')

E,F,G,H = compute_E(block, p, board, use_cache_H=False, path=path, return_components=True, internal_points=internal_points)

x = wgs(p, board=board, A=E)


rs = []
Txs = []
Tys = []
Tzs = []

for d in torch.linspace(0.04, 0.06, steps=50):
    T = torque_mesh_surface(x, block, board, H=H, path=path, use_cache_H=False, internal_points=internal_points, diameter=d.item())
    Txs.append(T[0,0].item())
    Tys.append(T[0,1].item())
    Tzs.append(T[0,2].item())
    rs.append((d/2).item())

import matplotlib.pyplot as plt

plt.plot(rs,Txs, color='red')
plt.plot(rs,Tys, color='green')
plt.plot(rs,Tzs, color='blue')

plt.ylabel("Torque")
plt.xlabel("Radius")

plt.figure()


r = 200
Visualise(*ABC(0.03), x, res = (r,r), points=p,
            colour_functions=[propagate_BEM_pressure], 
            colour_function_args=[
                {'board':board, 'scatterer':block, 'H':H, 'path':path,'use_cache_H':False, 'internal_points':internal_points},
            ]
        )

