from acoustools.Solvers import wgs, translate_hologram, gspat
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TRANSDUCERS, device, TOP_BOARD, propagate_abs
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Constants import wavelength,k, P_ref

import torch


board = TRANSDUCERS
# paths = [path+"/Sphere-lam2.stl"]   
# scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

p = create_points(3,1, y=0,x=0.02,z=0.01)
print(p)
p[:,2,0] -= wavelength/4
p[:,2,1] += wavelength/4
print(p)

b = torch.ones(3,1).to(device) * 4000
b[2,:] = 0


x = wgs(p, board=board, b=b, iterations=10)


Visualise(*ABC(0.01, origin=p[:,:,2].unsqueeze(2)), x, points=p, res=(100,100), vmax=8000)
