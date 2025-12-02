from acoustools.Export.Holo import load_holograms
from acoustools.Mesh import load_scatterer, get_CHIEF_points, translate, scale_to_diameter, centre_scatterer, get_centre_of_mass_as_points
from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, propagate_abs, TOP_BOARD
from acoustools.Solvers import translate_hologram, gspat
from acoustools.BEM.Force import force_mesh_surface
from acoustools.BEM import propagate_BEM_pressure
from acoustools.Force import compute_force
from acoustools.Constants import wavelength

from acoustools.Visualiser import Visualise, ABC

import torch

# x = load_holograms('AcousTools_Examples/data/holos/WorkingCHIEF-27-11.holo')[0]

board = TRANSDUCERS

p = create_points(1,1,0,0,0)
p2 = p.clone()
x = gspat(p, board=board)
# x = add_lev_sig(x)

root = '../BEMMedia'
path = root + '/Sphere-solidworks-lam2.stl'
diameter = 1e-4

sphere = load_scatterer(path)
scale_to_diameter(sphere, diameter)
centre_scatterer(sphere)


N = 100
d = 1e-3

Fxs = []
Fys = []
Fzs = []

FUx = []
FUy = []
FUz = []


for i in range(N):
    print(i, end='\r')

    # x = gspat(p, board=board)

    # Visualise(*ABC(0.04), x, res=(100,100), points=torch.stack([p, p2], dim=2) )
    
    # internal_points = get_CHIEF_points(sphere, P = 50, start='centre', method='uniform', scale = 0.1, scale_mode='diameter-scale') internal_points=internal_points

    force = force_mesh_surface(x, sphere, board, diameter=2*wavelength, path=root, sum_elements=True, use_cache_H=False).real
    print(force)

    v = 4/3 * 3.1415 * (diameter/2)**3
    Ux, Uy, Uz = compute_force(x, points=p, board=board, return_components=True, V=v)

    Fxs.append(force[:,0].item())
    Fys.append(force[:,1].item())
    Fzs.append(force[:,2].item())

    FUx.append(Ux.item())
    FUy.append(Uy.item())
    FUz.append(Uz.item())


    x = translate_hologram(x, board=board, dz=d)
    translate(sphere, dz=d)
    p[0,2] += d

Visualise(*ABC(0.04), x, res=(100,100), points=torch.stack([p, p2], dim=2), link_ax=[0,1],
                                                                            colour_functions=[propagate_abs, propagate_BEM_pressure, '-'], 
                                                                            colour_function_args=[{'board':board}, 
                                                                                                  {'board':board, 'scatterer':sphere, 'path':root},
                                                                                                  {}])

import matplotlib.pyplot as plt

plt.plot(Fxs, color='red')
plt.plot(Fys, color='green')
plt.plot(Fzs, color='blue')


plt.plot(FUx, color='red', linestyle='--')
plt.plot(FUy, color='green', linestyle='--')
plt.plot(FUz, color='blue', linestyle='--')



plt.show()