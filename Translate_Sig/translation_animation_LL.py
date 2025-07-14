from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, device, DTYPE
from acoustools.Solvers import wgs, translate_hologram
from acoustools.Export.Holo import load_holograms
from acoustools.Mesh import load_scatterer, scale_to_diameter
from acoustools.BEM import compute_E, get_cache_or_compute_H, propagate_BEM_pressure


from acoustools.Visualiser import Visualise, ABC, Visualise_single

import torch
import math
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib  as mpl



board = TRANSDUCERS

x = load_holograms("Optimsed_holo_working.holo")[0]

start = x.clone()
xs = []

for i in range(40):
    print(i, end='\r')
    x = translate_hologram(start,board, dz=0.001*i )
    xs.append(x)
print('\t\t\t\r',end='')
N=100
for i in range(N):
    posx = 0.04 * math.sin((2*math.pi /N) * i )
    posz = 0.04 * math.cos((2*math.pi /N) * i )
    print(i, end='\r')
    x = translate_hologram(start,board, dx = posx, dz = posz)
    xs.append(x)
print('\t\t\t\r',end='')

imgs = []
for i,x in enumerate(xs):
    print(i, end='\r')
    img = Visualise_single(*ABC(0.1), x, res=(800,800))
    imgs.append(img)
print('\t\t\t\r',end='')

fig = plt.figure()
img_ax = fig.add_subplot(1,1,1)


def traverse_holo(index):
    # img_ax.clear()
    print(index,end='\r')
    img_ax.matshow(imgs[index], cmap='hot', vmax=5000)
    
ani = animation.FuncAnimation(fig, traverse_holo, frames=len(imgs), interval=100)

ani.save("translations_LL.gif", writer='imagemagick', dpi = 200)