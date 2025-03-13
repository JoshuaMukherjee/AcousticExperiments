import pygame, sys, time, torch
from Game import game

from acoustools.Solvers import gspat
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import create_points

from math import sin,cos

def get_circle(r, N):
    poss = []
    for i in range(N):
        y=r*sin(3.1415/N * i)
        z=r*sin(3.1415/N * i)
        poss.append(create_points(1,1,0,y,z))
    return poss

def draw(sim:game, batch=32, frame_rate=4000):
        start = time.time_ns()
        ps = []
        pygame.init()

        lev = LevitatorController(ids=(-1,))
        posses = get_circle(sim.paddle_width, frame_rate)
        
        n=0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                
            n += 1
            n %= frame_rate
            
            keys=pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: sim.player_two[:,2] += 0.001
            if keys[pygame.K_DOWN]: sim.player_two[:,2] -= 0.001
            if keys[pygame.K_w]: sim.player_one[:,2] += 0.001
            if keys[pygame.K_s]: sim.player_one[:,2] -= 0.001


            
            sim.step()

            r = sim.paddle_width
            

            p1 = sim.player_one
            p2 = sim.player_two
            d = posses[n]
            ball = sim.ball

            p = torch.concatenate((p1+d,p2+d,ball),axis=2)
            ps.append(p)


            if len(ps) >= batch:
                points = torch.concatenate(ps, axis=0)
                xs = gspat(points, iterations=10)
                holos =[]
                for x in xs:
                     holos.append(x)
                    
                lev.levitate(holos)
                ps = []
                end = time.time_ns()
                print(batch / ((end-start)/1e9), "fps \t", end="\r")
                start = time.time_ns()
            
        

            # time.sleep(1/frame_rate)



gm = game(vel_scale=1e5)
draw(gm,batch=32,frame_rate=4000)