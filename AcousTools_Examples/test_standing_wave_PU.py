from acoustools.Utilities import transducers, create_points, propagate_abs
from acoustools.Visualiser import Visualise, ABC
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
from acoustools.Force import compute_force, force_fin_diff

import torch
from torch import Tensor

two_transducers = transducers(1, 0.02)

p = create_points(1,1,0,0,0)

x = torch.tensor([torch.exp(torch.tensor(1j * 3.1415)),1]).unsqueeze(0).unsqueeze(2)

def propagate_force_z(activations: Tensor,  points: Tensor, board: Tensor | None = None):
    _,_,force_z = compute_force(activations, points, board, True)
    return force_z.unsqueeze_(0).unsqueeze_(2)


Visualise(*ABC(0.015), x, p,colour_functions=[propagate_abs, gorkov_analytical, gorkov_fin_diff] ,
          colour_function_args = [{'board':two_transducers},{'board':two_transducers}, {'board':two_transducers}], 
          link_ax=None, show=True)