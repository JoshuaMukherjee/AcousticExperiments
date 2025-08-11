from acoustools.Utilities import create_points, BOTTOM_BOARD, propagate_abs
from acoustools.Solvers import wgs
import acoustools.Constants as c
from acoustools.Visualiser import Visualise, ABC
import torch

board = BOTTOM_BOARD
p = create_points(1,1,0,0,0)

M = board.shape[0]
mean = torch.ones((1,M,1)) * c.P_ref
std = torch.ones((1,M,1)) * 1
p_ref = torch.normal(mean, std).T
print(p_ref)
rand_p_ref = torch.rand_like(p_ref) * c.P_ref
mean_p_ref = torch.mean(p_ref)

x = wgs(p, board=board)


Visualise(*ABC(0.1), x, colour_functions=[propagate_abs, propagate_abs,propagate_abs,propagate_abs], colour_function_args=[{"board":board},{'p_ref':p_ref,"board":board} ,{'p_ref':rand_p_ref,"board":board},{'p_ref':mean_p_ref,"board":board}])

