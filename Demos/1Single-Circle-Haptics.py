from acoustools.Utilities import BOTTOM_BOARD, create_points
from acoustools.Solvers import kd_solver
from acoustools.Paths import interpolate_circle

from acoustools.Levitator import LevitatorController

import time

board = BOTTOM_BOARD

p = create_points(1,1,0,0,0)
circle = interpolate_circle(p, 0.02, n=50)


with LevitatorController(ids=999,) as lev:
    lev.set_frame_rate(200)
    start = time.time_ns()
    frames = 0
    try:
        while True:
            for pc in circle:
                # pc = torch.cat(pc, dim=0)
                x = kd_solver(pc, board)
                lev.levitate(x)
                frames += 1
    except KeyboardInterrupt:
        end = time.time_ns()
        # print(frames / ((end - start) / 1e9), 'Hz')