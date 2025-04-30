import csv
import sys
import os
import time

import pygame
import numpy as np
from agent_outline import load_blueprint, BlueprintEnvironment, pygame_file_picker

#screen dimensions
WIDTH, HEIGHT = 1000, 700
FPS = 60

def manhattan_distance(a, b):
    # manhattan distance: |x1 - x2| + |y1 - y2|

    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def greedy_move(actor, fire_exit, env):

    moves = [(0, -5), (0, 5), (-5, 0), (5, 0), (0, 0),
             (-5, -5), (5, -5), (-5, 5), (5, 5)]

    #calculate dis from current position to the fire exist

    min_dis = manhattan_distance(actor.pos, fire_exit)
    best_pos = actor.pos[:]
    #for each move made, calculate the new position after making a move, and calculate the distance to the
    #fire exist.
    for move in moves:
        new_pos = [actor.pos[0] + move[0], actor.pos[1] + move[1]]
        if not env.detect_collision(actor, new_pos):
            dis = manhattan_distance(new_pos, fire_exit)
            #update move if it's closer to the fire exit
            if dis < min_dis:
                min_dis = dis
                best_pos = new_pos

    return best_pos

def run_simulation(blueprint_path):
    #initilse pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Greedy Evacuation Runner")

    #load environment from blueprint (from agent_outline)
    walls, exits, actors, fires = load_blueprint(blueprint_path)
    env = BlueprintEnvironment(walls, exits, actors, fires, screen=screen)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #update actors
        for actor in env.actors:
            current_pos = actor.pos
            moved = False
            exit_pos = env.find_nearest_exit(actor).pos #find nearst exit
            new_pos = greedy_move(actor, exit_pos, env) #decide next move
            if new_pos != actor.pos:
                actor.pos = new_pos


        env.update_actors()
        env.render()
        env.export_advanced_metrics()
        env.export_metrics()

    pygame.quit()
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        blueprint_path = sys.argv[1]
    else:
        blueprint_path = pygame_file_picker(folder=os.getcwd())

    if blueprint_path and os.path.exists(blueprint_path):
        run_simulation(blueprint_path)
    else:
        print("No valid blueprint file selected.")
