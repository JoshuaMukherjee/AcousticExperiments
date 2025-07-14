from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, device, DTYPE, propagate_abs
from acoustools.Solvers import wgs, translate_hologram
from acoustools.Mesh import load_scatterer, scale_to_diameter
from acoustools.BEM import compute_E, get_cache_or_compute_H, propagate_BEM_pressure

from acoustools.Visualiser import Visualise, ABC

import torch
import matplotlib.pyplot as plt

cache = True
board = TRANSDUCERS


path = '../../BEMMedia'
sphere = load_scatterer('/Sphere-solidworks-lam2.stl', root_path=path, dz=-0.05)
scale_to_diameter(sphere,0.02)
H = get_cache_or_compute_H(sphere, board, path=path, use_cache_H=cache)


pf = create_points(3,1,y=0, max_pos=0.03, min_pos=-0.03)
E = compute_E(sphere, pf, path=path, board=board, H=H)

x = wgs(pf, board=board, A=E)
x2 = translate_hologram(x,board, dx=0.01 )



Visualise(*ABC(0.06), x2, points=pf, block=False, colour_functions=[propagate_abs,propagate_BEM_pressure], colour_function_args=[{},{'scatterer':sphere,'board':board, 'path':path, 'H':H}])
plt.figure()
Visualise(*ABC(0.06), x, points=pf,colour_functions=[propagate_abs,propagate_BEM_pressure], colour_function_args=[{},{'scatterer':sphere,'board':board, 'path':path,"H":H}])
