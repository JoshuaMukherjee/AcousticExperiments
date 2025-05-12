from acoustools.Utilities import create_points, BOTTOM_BOARD, propagate_abs
from acoustools.Paths import interpolate_circle
from acoustools.Solvers import naive
from acoustools.Levitator import LevitatorController

import torch, random

board = BOTTOM_BOARD


origin = create_points(1,1,0,0,0)
circle = interpolate_circle(origin, 0.021, n=300)

circle = torch.cat(circle, dim=0)

holos = naive(circle,board=board)
# print(propagate_abs(holos, circle, board))
hs = []
for h in holos:
    hs.append(h.unsqueeze(0)*0.88) #If unstable - tune this




lev = LevitatorController(ids=(73,))
lev.set_frame_rate(10000)
lev.levitate(hs, num_loops=1000)
lev.turn_off()
lev.disconnect()