import fastf1
import fastf1.utils

from acoustools.Solvers import wgs
from acoustools.Utilities import add_lev_sig, create_points, TRANSDUCERS, batch_list
from acoustools.Paths import OptiSpline, create_bezier_circle, interpolate_bezier, connect_ends
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_acceleration_position, optispline_min_distance_control_points
from acoustools.Visualiser import Visualise_single, ABC, get_point_pos
from acoustools.Export.Holo import save_holograms, load_holograms
from acoustools.Levitator import LevitatorController

from time import sleep
import pickle, torch

import matplotlib.animation as animation
import matplotlib  as mpl


import matplotlib.pyplot as plt

COMPUTE = 1
alpha = 1e-6

if COMPUTE:

    session = fastf1.get_session(2024, 4, 'Q')
    session.load(telemetry=True, laps=True, weather=False)
    laps = session.laps

    fast = laps.pick_fastest()
    print(fast)
    name = fast.Driver
    telem = fast.get_telemetry()
    speed = telem.Speed

    X = telem.X
    Y = telem.Y


    time = telem.Time

    X_norm = 2*((X - min(X)) / (max(X) - min(X)) ) -1  #Normalise [-1,1]
    Y_norm = 2*((Y - min(Y)) / (max(Y) - min(Y)) ) -1  #Normalise [-1,1]

    MAX_POS = 0.04

    X_norm *= MAX_POS
    Y_norm *= MAX_POS

    THRESHOLD = 0.0001
    Xs = [x for x in X_norm]
    Zs = [y for y in Y_norm]

    I = 10
    Xs = Xs[0::I]
    Zs = Zs[0::I]

    Xs.append(Xs[0])
    Zs.append(Zs[0])

    N = len(Xs)



    ps = create_points(N,1,x=Xs, y=0, z=Zs).permute(2,1,0)
    points = []
    for pt in ps:
        points.append(pt.unsqueeze(0))



    start = create_bezier_circle(plane='xz')
    start_clone = start.clone()
    # spline, saved = OptiSpline(start, points, optispline_min_distance_control_points,n=7, iters=100, get_intermediates=True)

    spline, saved = OptiSpline(start, points, optispline_min_acceleration_position,n=7, 
                               iters=100, get_intermediates=True, objective_params={"alpha":alpha})
    connect_ends(spline)

    points = []
    for bez in spline:
        points += interpolate_bezier(bez,200)
    print(points[0])
    holos = []
    for i,p in enumerate(points):
        # print(i,end='\r')
        x = wgs(p,iter=200, board=TRANSDUCERS)
        x = add_lev_sig(x)
        holos.append(x)
    save_holograms(holos, f'OptiSplinePath_{alpha}')
else:
    holos = load_holograms(f'OptiSplinePath_{alpha}')

print(len(holos))
exit()
lev = LevitatorController(ids=(73,53))
lev.set_frame_rate(1000)
lev.levitate(holos[0])
input()
for i in range(5):
    for b in batch_list(holos):
        lev.levitate(b)

input()
lev.disconnect()