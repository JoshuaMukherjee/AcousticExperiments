import pygame
import sys
import time
from Game import game

class Screen:

    def __init__(self,width,height):
        self.width = width
        self.height = height
        pygame.init()
        self.create_screen()
        self.font = pygame.font.SysFont("monospace", 15)
        self.font_large = pygame.font.SysFont("monospace", 20)

        self.half_width = int(self.width/2)
        self.half_height = int(self.height/2)


    def create_screen(self):
        self.screen = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption('Pong!!')

    def fill(self):
        self.screen.fill((0,0,0))

    def flip(self):
        pygame.display.flip()

    def draw_sprite(self,sprite):
        sprite.draw(self.screen)

    def get_height(self):
        return self.height
    
    def draw_ball(self,pos):
        p = ((pos[:,0].item() * 4000) + self.half_width ,4000*pos[:,2].item() + self.half_height)
        print(p)
        pygame.draw.circle(self.screen, color=(255,255,255),center=p,radius=2)

    def draw_paddles(self, p1, p2, width):
        p1_top = ((p1[:,0].item() * 4000 + self.half_width) ,4000*p1[:,2].item() + 4000*width + self.half_height)
        p1_bot = ((p1[:,0].item() * 4000 + self.half_width) ,4000*p1[:,2].item() - 4000*width + self.half_height)


        p2_top = ((p2[:,0].item() * 4000 + self.half_width) ,4000*p2[:,2].item() + 4000*width + self.half_height)
        p2_bot = ((p2[:,0].item() * 4000 + self.half_width) ,4000*p2[:,2].item() - 4000*width + self.half_height)

        pygame.draw.line(self.screen, color=(255,255,255), start_pos=p1_top,end_pos=p1_bot)
        pygame.draw.line(self.screen, color=(255,255,255), start_pos=p2_top,end_pos=p2_bot)
    


def draw(sim:game, res=(500,500), frame_rate = 20):
        screen = Screen(*res)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                

            
            keys=pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: sim.player_two[:,2] += 0.001
            if keys[pygame.K_DOWN]: sim.player_two[:,2] -= 0.001
            if keys[pygame.K_w]: sim.player_one[:,2] += 0.001
            if keys[pygame.K_s]: sim.player_one[:,2] -= 0.001


            
            screen.fill()
            
            sim.step()

            screen.draw_ball(sim.ball)
            screen.draw_paddles(sim.player_one, sim.player_two, sim.paddle_width)

        

            screen.flip()
            time.sleep(1/frame_rate)



    