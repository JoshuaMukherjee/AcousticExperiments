from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs, add_lev_sig, transducers
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
from acoustools.Solvers import wgs
import acoustools.Constants as c

from acoustools.Visualiser import ABC, Visualise


board = transducers(3)

p = create_points(1,1,0,0,0)
x = wgs(p, board=board)
# x = add_lev_sig(x)

stepsize = c.wavelength/64

def force_z(activations, points, board=board):
    Fz = compute_force(activations, points, board)[2].unsqueeze(0)
    return Fz

def force_z_fd(activations, points, board=board):
    Fz = force_fin_diff(activations, points, board=board, stepsize=stepsize)[:,2]
    return Fz


def diff_percent(activations, points, board = board):
    f1 = force_fin_diff(activations, points, board=board, stepsize=stepsize)[:,2]
    f2 = compute_force(activations, points, board)[2].unsqueeze(0)
    return ((f2 - f1) / f1) * 100

R = 100
Visualise(*ABC(0.02),x, points=p, colour_functions= [force_z, force_z_fd, diff_percent], res=(R,R), link_ax=[0,1])