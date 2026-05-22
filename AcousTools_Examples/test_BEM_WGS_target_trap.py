from acoustools.Solvers import wgs, translate_hologram, gspat
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TRANSDUCERS, device, TOP_BOARD
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
from acoustools.BEM import propagate_BEM_pressure, compute_E, translate
from acoustools.Constants import wavelength,k, P_ref

import torch


board = TOP_BOARD

path = "../BEMMedia"

paths = [path+"/Flat-lam2.stl"]
scatterer = load_multiple_scatterers(paths)
centre_scatterer(scatterer)
translate(scatterer,dz=-0.04)
print(scatterer.bounds())

get_edge_data(scatterer)


p = create_points(3,1, y=0,x=0.02,z=0.01)
print(p)
p[:,2,0] -= wavelength/4
p[:,2,1] += wavelength/4

b = torch.ones(3,1).to(device) * 4000
b[2,:] = 0

E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False,H_method='LU', return_components=True)

x = wgs(p, board=board, A=E, b=b, iterations=1000)

Visualise(*ABC(0.03, origin=p[:,:,2].unsqueeze(2)), x, points=p,colour_functions=[propagate_BEM_pressure], res=(100,100),
            colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"H":H }], vmax=8000)
