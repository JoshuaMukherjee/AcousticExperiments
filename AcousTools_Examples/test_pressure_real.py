from acoustools.Utilities import propagate, create_points, TRANSDUCERS
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force

from acoustools.Visualiser import Visualise, ABC

from torch import Tensor

board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)

def propagate_real(activations: Tensor,  points: Tensor, board: Tensor | None = None):
    p = propagate(activations, points, board)
    p = p.real / p.angle().cos().abs()
    
    return p


res = (300,300)
Visualise(*ABC(0.06),x, colour_functions=[propagate_real], res=res, link_ax=None, 
          clr_labels=['Pressure (Pa)'], cmaps=['seismic'])