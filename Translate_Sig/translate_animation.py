from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, device, DTYPE
from acoustools.Solvers import wgs, translate_hologram

from acoustools.Visualiser import Visualise, ABC, Visualise_single

import torch
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib  as mpl



board = TRANSDUCERS

pf = create_points(4,1,y=0, max_pos=0.02, min_pos=-0.02)
x = wgs(pf, board=board)
start = x.clone()
xs = []

for i in range(20):
    print(i, end='\r')
    x = translate_hologram(start,board, dx=0.001*i )
    xs.append(x)


imgs = []
for i,x in enumerate(xs):
    print(i, end='\r')
    img = Visualise_single(*ABC(0.05), x)
    imgs.append(img)


fig = plt.figure()
img_ax = fig.add_subplot(1,1,1)


def traverse_holo(index):
    # img_ax.clear()
    print(index,end='\r')
    img_ax.matshow(imgs[index], cmap='hot', vmax=7000)
    
ani = animation.FuncAnimation(fig, traverse_holo, frames=len(imgs), interval=100)

ani.save("translations.gif", writer='imagemagick', dpi = 200)