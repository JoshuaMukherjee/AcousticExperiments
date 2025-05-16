from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs, add_lev_sig
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
from acoustools.Solvers import wgs
import acoustools.Constants as c

from acoustools.Visualiser import ABC, Visualise

board = TRANSDUCERS


N = 100

Fxs = []
Fys = []
Fzs = []

Fx_fd = []
Fy_fd = []
Fz_fd = []

for i in range(N):
    print(i, end='\r')
    p = create_points(1,1)
    # p = create_points(1,1, y=0, min_pos=-0.02, max_pos=0.02)
    x = wgs(p, board=board)
    x  = add_lev_sig(x)

    # U = gorkov_analytical(x, p, board)
    # # U_fd = gorkov_fin_diff(x,p,board=board)

    # print(U.squeeze())
    # print(U_fd.squeeze())

    stepsize = c.wavelength/4

    F = compute_force(x,p).squeeze().cpu().detach()
    F_fd = force_fin_diff(x, p, board=board, stepsize=stepsize, U_function=gorkov_analytical).squeeze().cpu().detach()
    F_fd_fd = force_fin_diff(x, p).squeeze().cpu().detach()

    Fxs.append(F[0])
    Fys.append(F[1])
    Fzs.append(F[2])

    Fx_fd.append(F_fd[0])
    Fy_fd.append(F_fd[1])
    Fz_fd.append(F_fd[2])

import matplotlib.pyplot as plt

plt.scatter(Fxs, Fx_fd)
plt.scatter(Fys, Fy_fd)
plt.scatter(Fzs, Fz_fd)

plt.show()

    # U_force = compute_force(x, com, board, V=v_sphere).squeeze()
    # U_force_fd = force_fin_diff(x,com,board=board, stepsize=c.wavelength/10, V=v_sphere).squeeze()






# def diff(activations, points, board = board):
#     f1 = force_fin_diff(activations, points, board=board)[:,2]
#     f2 = compute_force(activations, points, board)[2].unsqueeze(0)
#     return f2 - f1

# R = 100
# Visualise(*ABC(0.03),x, colour_functions= [force_z, force_z_fd, diff], res=(R,R), link_ax=[0,1])