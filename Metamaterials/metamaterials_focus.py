from acoustools.Mesh import load_scatterer, centre_scatterer, get_diameter, scale_to_diameter, get_edge_data, merge_scatterers, translate   
from acoustools.Utilities import create_board, create_points
from acoustools.BEM import compute_E, propagate_BEM_pressure, propagate_BEM_phase, get_cache_or_compute_H
from acoustools.Solvers import naive


from acoustools.Visualiser import Visualise, ABC

import vedo, torch

from line_profiler import profile

@profile
def main():
    path = "../BEMMedia/"
    scatterer='Metamaterials/data/5lam-8-Memoli-dense.stl'

    surface = load_scatterer(scatterer, rotx=90)
    d = get_diameter(surface)

    scale_to_diameter(surface, d/10000)
    centre_scatterer(surface)
    print(surface.bounds())

    get_edge_data(surface)

    height= surface.bounds()[-1]
    print(height)

    board = create_board(2, -0.08575 - height)
    print('Board',board)

    x = torch.ones((1,1)) + 0j

    h = float(0.055 )
    vis_plane = create_points(1,1,z=h, x=0, y=0)


    H = get_cache_or_compute_H(surface, board, use_cache_H=True, path=path)

    print("Computed H")
    # print(H@x)

    # vedo.show(surface,axes=1)
    # exit()

    Visualise(*ABC(0.07), x, res=(200,200), vmax=[1000, 4], points=vis_plane,
            colour_functions=[propagate_BEM_pressure, propagate_BEM_phase], 
            colour_function_args=[{'scatterer':surface,"board":board, 'path':path, 'H':H},{'scatterer':surface,"board":board, 'path':path, 'H':H}],
            cmaps=['hot', 'hsv'],
            link_ax=None,
            )


if __name__ == '__main__':
    main()