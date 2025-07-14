from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig, device, DTYPE
from acoustools.Solvers import wgs, translate_hologram

from acoustools.Visualiser import Visualise, ABC

import torch
import matplotlib.pyplot as plt



board = TRANSDUCERS

pf = create_points(3,1,y=0, max_pos=0.03, min_pos=-0.03)
x = wgs(pf, board=board)
x2 = translate_hologram(x,board, dx=0.002 )



Visualise(*ABC(0.06), x2, points=pf, block=False)
plt.figure()
Visualise(*ABC(0.06), x, points=pf)
