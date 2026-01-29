from acoustools.Mesh import load_scatterer, centre_scatterer, get_diameter, scale_to_diameter, get_edge_data, merge_scatterers, translate   
from acoustools.Utilities import create_board, create_points
from acoustools.BEM import compute_E, propagate_BEM_pressure, propagate_BEM_phase
from acoustools.Solvers import naive


from acoustools.Visualiser import Visualise, ABC

import vedo


path = "../BEMMedia/"
scatterer='Metamaterials/data/22_brick2_lam8.stl'



brick = load_scatterer(scatterer, rotx=90)
centre_scatterer(brick)
d = get_diameter(brick)
scale_to_diameter(brick, d/1000)




get_edge_data(brick)

board = create_board(2, -0.005)


p = create_points(1,1,x=0,y=0,z=0.05)

E = compute_E(brick, p, board, path=path, use_cache_H=False)

x = naive(p, board=board, A=E)

print(board)

Visualise(*ABC(0.01), x, res=(200,200), vmax=[1000, 4],
          colour_functions=[propagate_BEM_pressure, propagate_BEM_phase], 
          colour_function_args=[{'scatterer':brick,"board":board, 'path':path},{'scatterer':brick,"board":board, 'path':path}],
          cmaps=['hot', 'hsv'],
          link_ax=None,
          )