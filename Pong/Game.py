from acoustools.Utilities import create_points

import torch
import random

class game():

    def __init__(self, bound_x = 0.06, bound_z = 0.06):
        self.player_one = create_points(1,1, -1 * bound_x + 0.01, 0 , 0)
        self.player_two = create_points(1,1,  bound_x - 0.01, 0 , 0)
        self.ball = create_points(1,1,0,0,0)

        self.paddle_width = 0.005

        self.bound_x = bound_x
        self.bound_z = bound_z

        d = random.randint(0,3)

        self.ball_velocity = create_points(1,1,y=0)
        
        self.ball_velocity /= torch.sqrt(torch.sum(torch.square(self.ball_velocity)))
        self.ball_velocity /= 1000

        self.buffer = 0.005

            
    def step(self):
        self.ball += self.ball_velocity
        
        #Hit top or bottom bound -> reflect z 
        
        if self.ball[:,2] > self.bound_z:
            print(1)
            self.ball_velocity[:,2] *= -1
        
        if self.ball[:,2] < -1* self.bound_z:
            print(2)
            self.ball_velocity[:,2] *= -1
        
        #Hit paddle -> Reflect x

        if self.ball[:,0] < self.player_one[:,0] + self.buffer and torch.abs(self.ball[:,2] - self.player_one[:,2]) < self.paddle_width:
            print(3)
            self.ball_velocity[:,0] *= -1
        
        if self.ball[:,0] > self.player_two[:,0] - self.buffer and torch.abs(self.ball[:,2] - self.player_two[:,2]) < self.paddle_width:
            print(4)
            self.ball_velocity[:,0] *= -1
        
        #Hit end -> Close

        if self.ball[:,0] > self.bound_x:
            self.ball_velocity = create_points(1,1,0,0,0) 
        
        if self.ball[:,0] < -1* self.bound_x:
            self.ball_velocity = create_points(1,1,0,0,0) 
        


        



