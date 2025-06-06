from acoustools.Utilities import create_points, propagate_pressure_grad, propagate_abs, TRANSDUCERS
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import propagate_BEM_pressure_grad, propagate_BEM_pressure
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_centre_of_mass_as_points, centre_scatterer


p = create_points(1,1)
path = "../BEMMedia"

board = TRANSDUCERS


sphere_pth =  path+"/Sphere-solidworks-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=0.03) #Make mesh at 0,0,0
d = 0.01
scale_to_diameter(sphere,d)

x = wgs(p) #Not focused with sphere but doesnt matter


# Visualise(*ABC(0.05), x, p, colour_functions=[propagate_abs])
Visualise(*ABC(0.01, plane='xz'), x, colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure_grad], link_ax=None, call_abs=True, norm_axes=[1,], 
          colour_function_args=[{'scatterer':sphere,"path":path,'board':board},{'scatterer':sphere,"path":path,'board':board}], res=(40,40))