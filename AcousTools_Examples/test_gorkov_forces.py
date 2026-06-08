from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs, add_lev_sig
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Solvers import wgs, translate_hologram
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise, ABC

import time
import torch

torch.random.manual_seed(1)
board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)
x = add_lev_sig(x)
x = translate_hologram(x, dz=-0.001)

t1 = time.time_ns()
fx,fy,fz = compute_force(x, p, board,True)
t2 = time.time_ns()

t3 = time.time_ns()
fx_fd, fy_fd, fz_fd = force_fin_diff(x,p,board=board, stepsize=wavelength/8)[0,0,:]
t4 = time.time_ns()

print(fx,fy,fz, (t2-t1)/1e9, sep='\t')

print(fx_fd, fy_fd, fz_fd, (t4-t3)/1e9, sep='\t')

exit()

INDEX = 0
def force(activations, points, board=board):
    _,_,F = compute_force(activations, points, board, return_components=True )
    return F

def force_fd(activations, points, board=board):
    F= force_fin_diff(activations, points, board=board, stepsize=wavelength/10)[:,:,2]
    return F




r = 100

Visualise(*ABC(0.01, origin=p, plane='xz'),x, colour_functions=[propagate_abs,force, force_fd], res = (r,r), link_ax=[1,2], clr_labels=['Pressure (Pa)', 'Analytic Force (N)', 'Finite Difference Froce (N)', 'Difference'])