from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force

from acoustools.Visualiser import Visualise, ABC

from torch import Tensor

board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)

def propagate_force_z(activations: Tensor,  points: Tensor, board: Tensor | None = None):
    _,_,force_z = compute_force(activations, points, board, True)
    return force_z.unsqueeze_(0).unsqueeze_(2)


res = (300,300)
Visualise(*ABC(0.06),x, colour_functions=[propagate_abs, gorkov_analytical, propagate_force_z], res=res, link_ax=None, clr_labels=['Pressure (Pa)', "Gorkov", "Force (N)"])