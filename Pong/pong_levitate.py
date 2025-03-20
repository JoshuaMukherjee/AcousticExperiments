import sys, time, torch
from Game import game

from acoustools.Solvers import gspat
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import create_points, add_lev_sig


from pynput import keyboard

key_states = {"w": False, "s": False, "up": False, "down": False}


def on_press(key):
    
    try:
        if key.char == 'w':
            key_states['w'] = True
        elif key.char == 's':
            key_states['s'] = True
    except AttributeError:
        if key == keyboard.Key.up:
            key_states['up'] = True
        elif key == keyboard.Key.down:
           key_states['down'] = True

def on_release(key):
    """ Update key state when a key is released """
    try:
        if key.char in ('w', 's'):
            key_states[key.char] = False
    except AttributeError:
        if key == keyboard.Key.up:
            key_states["up"] = False
        elif key == keyboard.Key.down:
            key_states["down"] = False



def draw(sim:game, batch=32, frame_rate=4000):
        start = time.time_ns()
        ps = []

        # lev = LevitatorController(ids=(-1,))
        lev = LevitatorController(ids=(73,53))
        lev.set_frame_rate(frame_rate)

        p1 = sim.player_one
        p2 = sim.player_two
        ball = sim.ball

        p = torch.concatenate((p1,p2,ball),axis=2)
        # p = ball
        print(p)

        xs = gspat(p, iterations=100)
        xs = add_lev_sig(xs)
        lev.levitate(xs)
        input()
        
        n=0
        
        listener = keyboard.Listener(on_press=on_press,on_release=on_release)
        listener.start()
        
        try:
            while True:
                # pygame.display.update()
                n += 1
                n %= frame_rate
            
                if key_states['up']: 
                    sim.player_two[:,2] += 0.000005
                    if sim.player_two[:,2] > sim.bound_z: sim.player_two[:,2] = sim.bound_z
                if key_states['down']:
                    sim.player_two[:,2] -= 0.000005
                    if sim.player_two[:,2] < -1* sim.bound_z: sim.player_two[:,2] = -1*sim.bound_z
                if key_states['w']: 
                    sim.player_one[:,2] += 0.000005
                    if sim.player_one[:,2] > sim.bound_z: sim.player_one[:,2] = sim.bound_z
                if key_states['s']: 
                    sim.player_one[:,2] -= 0.000005
                    if sim.player_one[:,2] < -1* sim.bound_z: sim.player_one[:,2] = -1*sim.bound_z


                sim.step()
                
                

                p1 = sim.player_one
                p2 = sim.player_two
                ball = sim.ball

                p = torch.concatenate((p1,p2,ball),axis=2)
                ps.append(p)

                if len(ps) >= batch:
                    points = torch.concatenate(ps, axis=0)
                    xs = gspat(points, iterations=200)
                    xs = add_lev_sig(xs)
                    holos =[]
                    for x in xs:
                        holos.append(x.unsqueeze(0))
                        
                    lev.levitate(holos)
                    ps = []
                    end = time.time_ns()
                    print(batch / ((end-start)/1e9), "fps \t", end="\r")
                    start = time.time_ns()
                
                # time.sleep(1/frame_rate)'
        except KeyboardInterrupt:
            lev.disconnect()
            

                



gm = game(vel_scale=2e5, bound_x=0.04, bound_z = 0.04)
draw(gm,batch=32,frame_rate=2000)
     