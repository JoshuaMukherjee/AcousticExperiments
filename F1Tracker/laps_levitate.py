import fastf1
import fastf1.utils

from acoustools.Solvers import wgs
from acoustools.Utilities import add_lev_sig, create_points
from acoustools.Levitator import LevitatorController
from time import sleep
import pickle

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

def add_points(Xs, Zs):

    Xs_new = []
    Zs_new = []

    prev_pos = [Xs[0],Zs[0]]
    for i in range(1,len(Xs)):
        
        pos = [Xs[i],Zs[i]]
        dist = ((pos[0]-prev_pos[0])**2 + (pos[1]-prev_pos[1])**2)**0.5
        if dist > THRESHOLD:
            Xs_new.append(prev_pos[0] + (pos[0]-prev_pos[0]) / 2 )
            Zs_new.append(prev_pos[1] + (pos[1]-prev_pos[1]) / 2 )
        Xs_new.append(Xs[i])
        Zs_new.append(Zs[i])

        prev_pos = pos
    return Xs_new, Zs_new

N = len(Xs)
print(N)
Xs, Zs = add_points(Xs, Zs) 
N = len(Xs)
print(N)
Xs, Zs = add_points(Xs, Zs) #2322
Xs, Zs = add_points(Xs, Zs) #3714 <= Lowest that doesnt drop
# Xs, Zs = add_points(Xs, Zs) #4665
# Xs, Zs = add_points(Xs, Zs) #4840
# Xs, Zs = add_points(Xs, Zs) #4839

print(len(Xs))

prev_pos = [Xs[0],Zs[0]]
maxdist = 0
for i in range(1,len(Xs)):
    pos = [Xs[i],Zs[i]]
    dist = ((pos[0]-prev_pos[0])**2 + (pos[1]-prev_pos[1])**2)**0.5
    if dist > maxdist:
        maxdist = dist
    prev_pos = pos

print(maxdist)


import os
print(os.listdir())
fname = 'F1_JAPAN_VER_2024_' + str(len(Xs)) +  '.pth'
print(fname)
override_cache = False

try:
    if override_cache:
        raise FileNotFoundError # bodged but works
    activations = pickle.load(open(fname,'rb'))
except FileNotFoundError as e:
    print('Not found...', e)
    activations = []
    for i in range(len(Xs)):
        p = create_points(1,1,x=Xs[i],y=0,z=Zs[i])
        x = wgs(p)
        x = add_lev_sig(x)
        activations.append(x)
        print(i)
    pickle.dump(activations, open(fname,'wb'))



print(Xs[0],0,Zs[0])


try:
    lev = LevitatorController()
    lev.levitate(activations[0])
    input()
    print('Moving...')
    lev.levitate(activations)
        # sleep(0.06) #0.6s => ~1:23 lap (Pole: 1:28), 1286 steps, THRESHOLD = 0.0001
except KeyboardInterrupt:
    print('Stopped')
finally:
    print('Finished')
    input()
    lev.disconnect()

