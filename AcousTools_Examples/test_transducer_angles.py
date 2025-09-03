from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, get_normals_as_points
from acoustools.Visualiser import force_quiver_3d
from acoustools.Utilities import forward_model, create_points, BOTTOM_BOARD, propagate_abs
from acoustools.Visualiser import Visualise, ABC
from acoustools.Solvers import naive
import vedo

path = "../BEMMedia"
scatterer = load_scatterer(path+"/Sphere-solidworks-lam2.stl")
scatterer.decimate(0.1)
scale_to_diameter(scatterer, 0.2)


board = get_centres_as_points(scatterer).squeeze().T
norms = get_normals_as_points(scatterer).squeeze().T
norms *= -1
# force_quiver_3d(board, norms[:,0], norms[:,1], norms[:,2], scale=0.01)

p = create_points(1,1,0,0,0)
F = forward_model(p, transducers=board, norms=norms)

x = naive(p, board, A=F)

Visualise(*ABC(0.1), x, colour_functions=[propagate_abs, propagate_abs], colour_function_args=[{'board':board, 'norms':norms}, {'board':board}])