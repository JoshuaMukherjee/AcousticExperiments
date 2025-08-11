from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC

import torch

board = TRANSDUCERS
M = board.shape[0]

voltage = 18
p_ref = torch.ones((1,M,1))
p_ref[:,:256] = 0.181 * voltage
p_ref[:,256:] = 0.2176 * voltage

p_ref_old = 0.17 * voltage

print(p_ref[:,0])
print(p_ref[:,-1])
print(torch.mean(p_ref))
print(p_ref_old)


p = create_points(1,1,0,0,0)


x = wgs(p, board=board)

Visualise(*ABC(0.02), x, colour_functions=[propagate_abs,propagate_abs], colour_function_args=[{'board':board,'p_ref':p_ref_old},{'board':board, 'p_ref':p_ref}])


