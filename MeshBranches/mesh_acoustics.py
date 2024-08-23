from acoustools.Mesh import load_scatterer, get_edge_data
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Utilities import BOTTOM_BOARD, create_points
from acoustools.Visualiser import Visualise_mesh

import vedo


scatterer = load_scatterer('./MeshBranches/Meshes/set0/m0.stl')
# scatterer.non_manifold_faces()
# scatterer.scale(0.1)
# scatterer.subdivide(2)
print(scatterer)

get_edge_data(scatterer)

Visualise_mesh(scatterer)