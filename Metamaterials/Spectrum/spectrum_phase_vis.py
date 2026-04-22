from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, rotate, get_edge_data, get_CHIEF_points
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_phase, propagate_BEM_pressure, compute_G
from acoustools.Visualiser import Visualise, ABC
from acoustools.Utilities import transducers, create_board, create_points

import vedo, torch, os
import matplotlib.pyplot as plt

path = '../BEMMedia/'

folder = '/Metamaterials/300um/'


def render_GH_pressure(activations, points, board, scatterer, H, **params):
    G = compute_G(points, scatterer)
    GH = G@H
    return torch.abs(GH@activations)

def render_GH_phase(activations, points, board, scatterer, H, **params):
    G = compute_G(points, scatterer)
    GH = G@H
    return torch.angle(GH@activations)

board = create_board(2, -0.02)
x = 1 * torch.exp(1j * torch.ones(1,1))

colour_function_args = []

P = 0

# print(sorted(os.listdir(path+folder)))
# exit()

for i,f in enumerate(sorted(os.listdir(path+folder))):
    print(f)
    # if i == 3: break

    name = f.split('.')[0]


    brick = load_scatterer(path+folder+f)
    centre_scatterer(brick)


    scale_to_diameter(brick, wavelength/2)
    rotate(brick, axis=(1,0,0), rot=90)
    centre_scatterer(brick)
    print(brick.bounds())
    # brick.subdivide(n=2)

    internal_points = get_CHIEF_points(brick, P=P, method='tetra-random') if P else None


    H= get_cache_or_compute_H(brick, board, use_cache_H=False, path=path, internal_points=internal_points)
    

    colour_function_args.append({'path':path,'H':H,'board':board, 'scatterer':brick})

A,B,C = ABC(0.03, origin=create_points(1,1,0,0,0.028))

ratio = 6
res = 150

A[0] /= ratio
B[0] /= ratio
C[0] /= ratio

Visualise(A,B,C, x, res = (res//ratio,res),
        colour_functions=[render_GH_phase] * len(colour_function_args), 
        colour_function_args=colour_function_args,
        cmaps=['hsv']* len(colour_function_args),
        show=False,
        clr_labels=[''] * len(colour_function_args)
        )

name = name + f"-CHIEF-{P}" if P else name
plt.savefig('Metamaterials/outputs/300um/spectrum-'+name+'.png')
plt.gcf().clear()