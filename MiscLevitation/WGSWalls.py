from acoustools.Utilities import create_points,add_lev_sig, TRANSDUCERS, propagate_abs
from acoustools.Solvers import wgs_wrapper
from acoustools.Visualiser import Visualise

from acoustools.Mesh import load_multiple_scatterers, get_lines_from_plane
from acoustools.BEM import compute_E, propagate_BEM_pressure, get_cache_or_compute_H

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = create_points(4,y=0)

    # print(p.shape)
    x = wgs_wrapper(p)
    x = add_lev_sig(x)

    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90]) #Make mesh at 0,0,0
    H = get_cache_or_compute_H(walls, TRANSDUCERS)
    E = compute_E(walls, p, TRANSDUCERS,H=H)
    x_BEM = wgs_wrapper(p, A=E)
    x_BEM = add_lev_sig(x_BEM)


    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))
    normal = (0,1,0)
    origin = (0,0,-0.07)
    
    # Visualise(A,B,C, x, p,vmax=9000)

    line_params = {"scatterer":walls,"origin":origin,"normal":normal}
    Visualise(A,B,C,x_BEM, p,add_lines_functions=[get_lines_from_plane, get_lines_from_plane],add_line_args=[line_params,line_params], colour_functions=[propagate_BEM_pressure,propagate_abs], colour_function_args=[{"H":H,"scatterer":walls,"board":TRANSDUCERS},{}],vmax=9000)


