from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise, ABC

import time
import torch

torch.random.manual_seed(1)
board = TRANSDUCERS

p = create_points(1,1)

x = wgs(p+create_points(1,1,max_pos=0.01, min_pos=-0.01), board=board)

t1 = time.time_ns()
fx,fy,fz = compute_force(x, p, board,True)
t2 = time.time_ns()

t3 = time.time_ns()
fx_fd, fy_fd, fz_fd = force_fin_diff(x,p,board=board, stepsize=wavelength/20)[0,:]
t4 = time.time_ns()



INDEX = 0
def force(activations, points, board=board):
    Fz = compute_force(activations, points, board, )[INDEX].unsqueeze(0)
    return Fz

def force_fd(activations, points, board=board):
    Fz = force_fin_diff(activations, points, board=board, stepsize=wavelength/20)[:,INDEX]
    return Fz




print(fx.item(),fy.item(),fz.item(), (t2-t1)/1e9, sep='\t')

print(fx_fd.item(), fy_fd.item(), fz_fd.item(), (t4-t3)/1e9, sep='\t')

r = 50

Visualise(*ABC(0.02, origin=p),x, colour_functions=[force, force_fd], res = (r,r))