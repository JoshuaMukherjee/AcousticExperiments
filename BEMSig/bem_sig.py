from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Utilities import TOP_BOARD, create_points, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, Visualise_mesh

import matplotlib.pyplot as plt
import torch


path = './Media/'
scatterer = load_scatterer('Bunny-lam2.stl',dz=-0.06,rotz=90, root_path=path)


fig = plt.figure()

N = 10
for i in range(1,N+1):
    p = create_points(4,1, min_pos=0)
    E,F,G,H = compute_E(scatterer, p, board = TOP_BOARD, return_components=True)

    x_bem = wgs(p,A=E)
    x_ptn = wgs(p, board=TOP_BOARD)

    ax = fig.add_subplot(3,N,i)
    ax.matshow(torch.angle(x_bem).reshape(16,16).cpu().detach(),cmap='hsv')
    ax.axis('off')

    ax = fig.add_subplot(3,N,10+i)
    ax.matshow(torch.angle(x_ptn).reshape(16,16).cpu().detach(),cmap='hsv')
    ax.axis('off')

    ax = fig.add_subplot(3,N,20+i)
    dif = torch.angle(x_bem) - torch.angle(x_ptn)
    dif = torch.atan2(torch.sin(dif),torch.cos(dif)) 
    ax.matshow(dif.reshape(16,16).cpu().detach(),cmap='hsv')
    ax.axis('off')

plt.tight_layout()
plt.show()
