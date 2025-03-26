import fastf1
import fastf1.utils

from acoustools.Solvers import wgs
from acoustools.Utilities import add_lev_sig, create_points, TRANSDUCERS
from acoustools.Paths import OptiSpline, create_bezier_circle, interpolate_bezier, connect_ends
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_acceleration_position, optispline_min_distance_control_points
from acoustools.Visualiser import Visualise_single, ABC, get_point_pos, Visualise_single_blocks
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data

from time import sleep
import pickle, torch

import matplotlib.animation as animation
import matplotlib  as mpl


import matplotlib.pyplot as plt

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)


H = get_cache_or_compute_H(walls,TRANSDUCERS)



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

X_norm = 2*((X - min(X)) / (max(X) - min(X)) ) -1  #Normalise [-1,1]1
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

spline, saved = OptiSpline(start, points, optispline_min_acceleration_position,n=7, iters=100, get_intermediates=True, objective_params={"alpha":5e-7})
connect_ends(spline)


plot_path = False
if plot_path:
    fig = plt.figure()
    fig.clear()

    circle = []
    for bez in start_clone:
        circle += interpolate_bezier(bez,7)
    xs = [pt[:,0].item() for pt in circle]
    ys = [pt[:,2].item() for pt in circle]


    def traverse(index):

        fig.clear()

        result = []
        for bez in saved[index]:
            result += interpolate_bezier(bez,7)

        
        xs_result = [pt[:,0].item() for pt in result]
        ys_result = [pt[:,2].item() for pt in result]

        plt.scatter(xs,ys, label='Start', marker='.')
        plt.scatter(Xs,Zs, label='Target', marker='x', color='green')
        plt.plot(xs_result,ys_result, label='Result', marker='.', color='red')

        plt.xlim(-0.05,0.05)
        plt.ylim(-0.05,0.05)
        plt.legend()



    ani = animation.FuncAnimation(fig, traverse, frames=len(saved), interval=50)

    ani.save("OptiSpline.gif", writer='imagemagick', dpi = 200)
else:
    points = []
    for bez in spline:
        points += interpolate_bezier(bez,40)

    holos = []
    imgs = []
    abc = ABC(0.14)
    
    track_p = torch.cat(points,dim=2)
    track = get_point_pos(*abc,track_p,res=(500,500))
    track = torch.stack(track).T

    for i,p in enumerate(points):
        # print(i,end='\r')
        E = compute_E(walls, p, TRANSDUCERS,H=H)
        x = wgs(p,iter=20, board=TRANSDUCERS,A=E)
        x = add_lev_sig(x)
        holos.append(x)
        # img = Visualise_single(*abc,x, res=(600,600))
        # imgs.append(img.cpu().detach().squeeze())
    fig = plt.figure()
    img_ax = fig.add_subplot(1,1,1)

    # fig.clear()



    def traverse_holo(index):
        # img_ax.clear()
        print(index,end='\r')
        img = Visualise_single_blocks(*abc,holos[index], res=(500,500), colour_function=propagate_BEM_pressure, colour_function_args={'scatterer':walls, 'H':H, 'board':TRANSDUCERS}).cpu().detach().squeeze()
        img_ax.matshow(img, cmap='hot', vmax=7000)
        img_ax.plot(track[1],track[0],linestyle='dashed',color='blue')
        
    print(len(holos))
    ani = animation.FuncAnimation(fig, traverse_holo, frames=len(holos), interval=50)

    ani.save("OptiSpline_Holos.gif", writer='imagemagick', dpi = 200)