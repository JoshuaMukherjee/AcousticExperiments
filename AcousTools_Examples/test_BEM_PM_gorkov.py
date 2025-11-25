from acoustools.Mesh import load_scatterer, scale_to_diameter, centre_scatterer
from acoustools.Utilities import create_points, TRANSDUCERS, TOP_BOARD, propagate_abs
from acoustools.Gorkov import gorkov
from acoustools.Solvers import naive
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical

from acoustools.Visualiser import Visualise, ABC

path = '../BEMMedia'
scatterer_path = '/sphere-lam2.stl'

scatterer = load_scatterer(path + scatterer_path)
centre_scatterer(scatterer)
scale_to_diameter(scatterer, 0.0001)

p = create_points(1,1,0.02,0,0.02)

board = TRANSDUCERS

x = naive(p, board)

Visualise(*ABC(0.05), x, arangement=(2,3),link_ax=None,
          colour_functions=[propagate_abs, propagate_BEM_pressure, '-', gorkov, BEM_gorkov_analytical, '-'],
          colour_function_args=[{'board':board}, {'board':board, 'scatterer':scatterer, 'path':path},{}, {'board':board}, {'board':board, 'scatterer':scatterer, 'path':path}, {'ids':[3,4]}])