import fastf1
import fastf1.utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

from acoustools.Solvers import wgs
from acoustools.Utilities import propagate_abs, DTYPE, add_lev_sig
from acoustools.Visualiser import Visualise_single, get_point_pos

session = fastf1.get_session(2023, 10, 'Q')
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


def distribute_in_space(THRESHOLD=0.004):

    Xs = []
    Ys = []

    current_pos = None

    for i in range(len(X_norm)):
        if current_pos is None:
            current_pos = (X_norm.iloc[i],Y_norm.iloc[i])
        else:
            dist = ((current_pos[0] - X_norm.iloc[i])**2 + ((current_pos[1] - Y_norm.iloc[i])**2))**0.5
            if dist > THRESHOLD:
                Xs.append(X_norm.iloc[i])
                Ys.append(Y_norm.iloc[i])
        
                current_pos = (X_norm.iloc[i],Y_norm.iloc[i])

    return Xs, Ys


def distribute_in_time(THRESHOLD=0.5):

    THRESHOLD = fastf1.utils.to_timedelta(str(THRESHOLD))
    print(THRESHOLD)

    Xs = []
    Ys = []
    speeds = []

    current_pos = None
    time_index = None

    for i in range(len(time)):
        if current_pos is None:
            current_pos = (X_norm.iloc[i],Y_norm.iloc[i])
            time_index = i
        else:
            delta_t = time.iloc[i] - time.iloc[time_index]
            if delta_t > THRESHOLD:
                Xs.append(X_norm.iloc[i])
                Ys.append(Y_norm.iloc[i])
                speeds.append(speed.iloc[i])
        
        
                current_pos = (X_norm.iloc[i],Y_norm.iloc[i])
                time_index = i
 

    return Xs, Ys, speeds


THRESHOLD = 0.5
Xs, Zs, speeds = distribute_in_time(THRESHOLD)



N = len(Xs)
Ys = torch.zeros((N))

Xs = torch.tensor(Xs)
Zs = torch.tensor(Zs)


points = []

for i in range(N):
    points.append(torch.stack([Xs[i],Ys[i],Zs[i]]).unsqueeze(0).unsqueeze(2).to(DTYPE))



A = torch.tensor((-0.05,0, 0.05))
B = torch.tensor((0.05,0, 0.05))
C = torch.tensor((-0.05,0, -0.05))

fig = plt.figure()
img_ax = fig.add_subplot(1,1,1)

RES = 100

pad = 2

def traverse(index):
        print(index)
        img_ax.clear()
        im = Visualise_single(A,B,C,activations[index],res=(RES,RES))

        img_ax.matshow(im.cpu().detach(),cmap='hot',vmax=10000) 
        img_ax.plot(track[1],track[0],linestyle='dashed')
        
        img_ax.plot(track[1][index],track[0][index],marker='x',color='blue')


        img_ax.annotate(name + ": "+str(speeds[index]) + "kph",(track[1][index]+pad,track[0][index]-pad),color='blue', 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


activations = []


track_p = torch.cat(points,dim=2).to(torch.double)

track = get_point_pos(A,B,C,track_p,res=(RES,RES))
track = torch.stack(track).T

for point in points:
    x = wgs(point)
    x = add_lev_sig(x)
    activations.append(x)
    # im = Visualise_single(A,B,C,x)

lap_animation = animation.FuncAnimation(fig, traverse, frames=range(len(activations)), interval=THRESHOLD*1000)

# lap_animation = animation.FuncAnimation(fig, traverse, frames=[1,2,3,4], interval=THRESHOLD*1000)
lap_animation.save('Results.gif', dpi=80, writer='imagemagick')

