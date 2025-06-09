from acoustools.Solvers import gspat
from acoustools.Utilities import create_points, DTYPE
from acoustools.Levitator import LevitatorController

import torch
import line_profiler
import time

N=100000

# @line_profiler.profile
def run():
    ps = []
    for i in range(N): 
        pnt = create_points(N=3,B=1)
        ps.append(pnt)
        if len(ps) == 32:
            p = torch.concatenate(ps, axis=0).to(DTYPE)
            ps = []
            x = gspat(p, iterations=10)
            xs = []
            for i in x:
                xs.append(i.unsqueeze(0))

try:
    start = time.time_ns()
    run()
    end = time.time_ns()
    print(f"{N} frames in {(end-start)/1e9}s -> {(N/((end-start)/1e9))} fps")
except KeyboardInterrupt:
    pass