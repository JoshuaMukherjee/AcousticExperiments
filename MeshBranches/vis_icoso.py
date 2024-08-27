from acoustools.Mesh import load_scatterer, get_edge_data
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Utilities import BOTTOM_BOARD, create_points
from acoustools.Visualiser import Visualise_mesh, ABC, Visualise, Visualise_single
from acoustools.Solvers import wgs

board = BOTTOM_BOARD

path = '../BEMMedia'

scatterer = load_scatterer(f'./MeshBranches/Meshes/icoso/set{0}/m{29}.stl', dz=0.04)
p = create_points(1,1,0,0,-0)

E,F,G,H = compute_E(scatterer, p, board, path=path, return_components=True)

x = wgs(p, board=board, A=E)

A,B,C = ABC(0.12)

Visualise(A,B,C,x,p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':scatterer,"H":H,"board":board}])