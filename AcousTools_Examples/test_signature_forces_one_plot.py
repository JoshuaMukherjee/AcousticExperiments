from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs, transducers
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Visualiser import Visualise_single_blocks, ABC

# board = transducers(z=0.108)
board = TRANSDUCERS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


import torch


# p = create_points(1,1,y=0, max_pos=0.04, min_pos=-0.04)
p = create_points(1,1,0,0,0)
print(p)

x = wgs(p, board=board)

scale = 1

if scale != 1: 
    labels = [f'{scale}*\nTrap','Twin','Vortex','Eye' ] 
else: 
    labels = ['Trap','Twin','Vortex','Eye' ]

x_trap = scale*add_lev_sig(x, board, mode = 'Trap')
x_twin = add_lev_sig(x, board, mode = 'Twin')
x_vortex = add_lev_sig(x, board, mode = 'Vortex')
x_eye = add_lev_sig(x, board, mode = 'Eye')

A,B,C = ABC(0.06)



F_trap_xs = []
F_trap_ys = []
F_trap_zs = []

F_twin_xs = []
F_twin_ys = []
F_twin_zs = []

F_vortex_xs = []
F_vortex_ys = []
F_vortex_zs = []

F_eye_xs = []
F_eye_ys = []
F_eye_zs = []

MAX_d = 0.015
N = 400
step = (MAX_d * 2) / N
pos = []

for i in range(N):
    print(i, end='\r')
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,d,0,0)
    pos.append(d)

    F_trap_x, _, _ = compute_force(x_trap,pd,board, return_components=True)
    F_twin_x, _, _ = compute_force(x_twin,pd,board, return_components=True)
    F_vortex_x, _, _ = compute_force(x_vortex,pd,board, return_components=True)
    F_eye_x, _, _ = compute_force(x_eye,pd,board, return_components=True)

    F_trap_xs.append(F_trap_x.detach().cpu().item())
    F_twin_xs.append(F_twin_x.detach().cpu().item())
    F_vortex_xs.append(F_vortex_x.detach().cpu().item())
    F_eye_xs.append(F_eye_x.detach().cpu().item())

    # F_trap = force_fin_diff(x_trap, pd)
    # F_twin = force_fin_diff(x_twin, pd)
    # F_vortex = force_fin_diff(x_vortex, pd)
    # F_eye = force_fin_diff(x_eye, pd)
    
    
    # F_trap_xs.append(F_trap.squeeze()[0])
    # F_twin_xs.append(F_twin.squeeze()[0])
    # F_vortex_xs.append(F_vortex.squeeze()[0])
    # F_eye_xs.append(F_eye.squeeze()[0])



for i in range(N):
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,0,d,0)

    _, F_trap_y, _ = compute_force(x_trap,pd,board, return_components=True)
    _, F_twin_y, _= compute_force(x_twin,pd,board, return_components=True)
    _, F_vortex_y, _ = compute_force(x_vortex,pd,board, return_components=True)
    _, F_eye_y, _ = compute_force(x_eye,pd,board, return_components=True)

    F_trap_ys.append(F_trap_y.detach().cpu().item())
    F_twin_ys.append(F_twin_y.detach().cpu().item())
    F_vortex_ys.append(F_vortex_y.detach().cpu().item())
    F_eye_ys.append(F_eye_y.detach().cpu().item())

    # F_trap = force_fin_diff(x_trap, pd)
    # F_twin = force_fin_diff(x_twin, pd)
    # F_vortex = force_fin_diff(x_vortex, pd)
    # F_eye = force_fin_diff(x_eye, pd)
    
    
    # F_trap_ys.append(F_trap.squeeze()[1])
    # F_twin_ys.append(F_twin.squeeze()[1])
    # F_vortex_ys.append(F_vortex.squeeze()[1])
    # F_eye_ys.append(F_eye.squeeze()[1])


for i in range(N):
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,0,0,d)

    _, _,F_trap_z = compute_force(x_trap,pd,board, return_components=True)
    _, _, F_twin_z = compute_force(x_twin,pd,board, return_components=True)
    _, _, F_vortex_z = compute_force(x_vortex,pd,board, return_components=True)
    _, _,F_eye_z = compute_force(x_eye,pd,board, return_components=True)

    F_trap_zs.append(F_trap_z.detach().cpu().item())
    F_twin_zs.append(F_twin_z.detach().cpu().item())
    F_vortex_zs.append(F_vortex_z.detach().cpu().item())
    F_eye_zs.append(F_eye_z.detach().cpu().item())


    # F_trap = force_fin_diff(x_trap, pd)
    # F_twin = force_fin_diff(x_twin, pd)
    # F_vortex = force_fin_diff(x_vortex, pd)
    # F_eye = force_fin_diff(x_eye, pd)
    
    
    # F_trap_zs.append(F_trap.squeeze()[2])
    # F_twin_zs.append(F_twin.squeeze()[2])
    # F_vortex_zs.append(F_vortex.squeeze()[2])
    # F_eye_zs.append(F_eye.squeeze()[2])



N = 5

lim = 0.0007


plt.plot(pos,F_trap_xs, color='red', label='$F_x$ Trap')
plt.plot(pos,F_twin_xs, color='blue', label='$F_x$ Twin')

plt.plot(pos,F_trap_ys, color='red', label='$F_y$ Trap', linestyle=':')
plt.plot(pos,F_twin_ys, color='blue', label='$F_y$ Twin', linestyle=':')


plt.plot(pos,F_trap_zs, color='red', label='$F_z$ Trap', linestyle='-.')
plt.plot(pos,F_twin_zs, color='blue', label='$F_z$ Twin', linestyle='-.')


plt.ylabel('$F$ (N)')
plt.legend()

import pandas

d = {"displacement":pos, 
     'Trap Fx': F_trap_xs,
     'Trap Fy': F_trap_ys,
     'Trap Fz': F_trap_zs,
     'Twin Fx': F_twin_xs,
     'Twin Fy': F_twin_ys,
     'Twin Fz': F_twin_zs,
     }

df = pandas.DataFrame(d)

df.to_csv("AcousTools_Examples/outputs/sig_forces.csv")

print(df)
exit()


# vmax = torch.max(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
# vmin = torch.min(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


# ax = plt.subplot(3,4,9)
# im = plt.matshow(img_trap, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)


# ax = plt.subplot(3,4,10)
# plt.matshow(img_twin, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)

# ax = plt.subplot(3,4,11)
# plt.matshow(img_vortex, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)

# ax = plt.subplot(3,4,12)
# plt.matshow(img_eye, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)




# plt.tight_layout()
plt.show()