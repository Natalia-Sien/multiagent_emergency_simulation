import pygame
import numpy as np
import json
import random
import heapq
import os
import csv
import argparse
from typing import List, Tuple, Dict, Optional, Set
from agent_outline import BlueprintEnvironment, Wall, FireExit, Actor, load_blueprint, pygame_file_picker

# Initialise pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FIRE_SPREAD_CHANCE = 0.10
FIRE_RADIUS = 5
FPS = 30
MAX_GUIDANCE_DISTANCE = 120 

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

# Actor Properties
ACTOR_PROPERTIES = {
    "Staff": {"color": (0, 0, 255), "speed": 1.0, "radius": 10, "algorithm": "AStar", "safety_weight": 1.0},
    "Adult": {"color": (0, 255, 0), "speed": 1.0, "radius": 10, "algorithm": "Dijkstra", "safety_weight": 1.0},
    "Patient": {"color": (255, 255, 0), "speed": 0.0, "radius": 10, "algorithm": "AStar", "safety_weight": 2.0},
    "Child": {"color": (200, 100, 200), "speed": 0.33, "radius": 8, "algorithm": "Dijkstra", "safety_weight": 2.0}
}

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hospital Evacuation Simulation")
clock = pygame.time.Clock()

# we set up the pathfinding algorithms
class PathfindingAlgorithm:
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, safety_weight: float = 1.0):
        self.safety_weight = safety_weight
        self.simulation = None

    # check if move is valid by checking if it's within the grid and not a wall or fire
    def is_valid_move(self, position: Tuple[int, int], grid: List[List[str]]) -> bool:
        x, y = position
        return (0 <= x < len(grid[0]) and 
                0 <= y < len(grid) and 
                grid[y][x] not in ("wall", "fire"))

    # get the move cost by checking if the neighbor is in the smoke grid and adding a crowding cost
    def get_move_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int], 
                     grid: List[List[str]], actor: 'Actor') -> float:
        base_cost = 1.0
        
        # Actor type modifiers
        type_multipliers = {
            "Patient": 1.5,
            "Child": 1.3,
            "Staff": 0.8
        }
        return base_cost * type_multipliers.get(actor.actor_type, 1.0)

    # reconstruct the path by backtracking from the goal to the start
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int], start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

    # initialise the search by setting up the open set and g_score
    def _initialize_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        open_set = []
        heapq.heappush(open_set, (0, start))
        return {
            'open_set': open_set,
            'came_from': {},
            'g_score': {start: 0},
            'open_set_hash': {start}
        }

# we start the class for the A* algorithm
class AStarAlgorithm(PathfindingAlgorithm):
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # find the path by using the heuristic to estimate the cost to the goal
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 grid: List[List[str]], actor: Optional['Actor'] = None) -> List[Tuple[int, int]]:
        search_data = self._initialize_search(start, goal)
        f_score = {start: self.heuristic(start, goal)}
        
        # while the open set is not empty, pop the node with the lowest f_score
        while search_data['open_set']:
            current = heapq.heappop(search_data['open_set'])[1]
            search_data['open_set_hash'].remove(current)
            
            # if the current node is the goal, reconstruct the path and return it
            if current == goal:
                return self._reconstruct_path(search_data['came_from'], current, start)
            
            # for each neighbor, check if the move is valid, if so, calculate the move cost and tentative g_score
            for dx, dy in self.DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # if the move is not valid, continue to the next neighbor
                if not self.is_valid_move(neighbor, grid):
                    continue
                    
                move_cost = self.get_move_cost(current, neighbor, grid, actor)
                tentative_g_score = search_data['g_score'][current] + move_cost
                
                # if the neighbor is not in the g_score or the tentative g_score is less 
                # than the current g_score, update the g_score and f_score
                if neighbor not in search_data['g_score'] or tentative_g_score < search_data['g_score'][neighbor]:
                    search_data['came_from'][neighbor] = current
                    search_data['g_score'][neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    # if the neighbor is not in the open set hash, add it to the open set
                    if neighbor not in search_data['open_set_hash']:
                        heapq.heappush(search_data['open_set'], (f_score[neighbor], neighbor))
                        search_data['open_set_hash'].add(neighbor)
        
        return []

# then we start the class for the Dijkstra algorithm
class DijkstraAlgorithm(PathfindingAlgorithm):
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 grid: List[List[str]], actor: Optional['Actor'] = None) -> List[Tuple[int, int]]:
        search_data = self._initialize_search(start, goal)
        # while the open set is not empty, pop the node with the lowest g_score
        while search_data['open_set']:
            current_g_score, current = heapq.heappop(search_data['open_set'])
            search_data['open_set_hash'].remove(current)
            
            # if the current node is the goal, reconstruct the path and return it
            if current == goal:
                return self._reconstruct_path(search_data['came_from'], current, start)
            
            # for each neighbor, check if the move is valid, if so, calculate the move cost and tentative g_score
            for dx, dy in self.DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # if the move is not valid, continue to the next neighbor
                if not self.is_valid_move(neighbor, grid):
                    continue
                # calculate the move cost and tentative g_score
                move_cost = self.get_move_cost(current, neighbor, grid, actor)
                tentative_g_score = search_data['g_score'][current] + move_cost

                # if the neighbor is not in the g_score or the tentative g_score is less 
                # than the current g_score, update the g_score and f_score
                if neighbor not in search_data['g_score'] or tentative_g_score < search_data['g_score'][neighbor]:
                    search_data['came_from'][neighbor] = current
                    search_data['g_score'][neighbor] = tentative_g_score
                    # if the neighbor is not in the open set hash, add it to the open set
                    if neighbor not in search_data['open_set_hash']:
                        heapq.heappush(search_data['open_set'], (tentative_g_score, neighbor))
                        search_data['open_set_hash'].add(neighbor)
        
        return []


def algorithm_picker():
    pygame.font.init()
    picker_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Select Algorithm")
    font = pygame.font.SysFont("arial", 24)
    
    algorithms = ["A* Algorithm", "Dijkstra's Algorithm"]
    selected_algorithm = None
    
    while True:
        picker_screen.fill(WHITE)
        
        for i, algo in enumerate(algorithms):
            text = font.render(algo, True, BLACK)
            rect = text.get_rect(topleft=(20, 30 + i * 40))
            picker_screen.blit(text, rect)
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(picker_screen, (200, 200, 255), rect, 2)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, algo in enumerate(algorithms):
                    rect = pygame.Rect(20, 30 + i * 40, 560, 30)
                    if rect.collidepoint(event.pos):
                        return "astar" if i == 0 else "dijkstra"
        
        pygame.display.flip()
        clock.tick(30)

def main():
    # Select algorithm using pygame menu
    selected_algorithm = algorithm_picker()
    if not selected_algorithm:
        print("No algorithm selected. Exiting...")
        return
    
    # Map algorithm choice to class
    algorithm_map = {
        'astar': AStarAlgorithm,
        'dijkstra': DijkstraAlgorithm
    }
    algorithm_class = algorithm_map[selected_algorithm]
    
    filename = pygame_file_picker()
    if not filename:
        print("No file selected. Exiting...")
        return
    walls, exits, actors, fires = load_blueprint(filename)
    
    # Create environment
    env = BlueprintEnvironment(walls, exits, actors, fires)
    
    # Assign selected algorithm to all actors
    for actor in env.actors:
        actor.algorithm = algorithm_class(safety_weight=ACTOR_PROPERTIES.get(actor.actor_type, {}).get("safety_weight", 1.0))
        actor.algorithm.simulation = env
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    env.render_on = not env.render_on
        
        # Update actors using selected algorithm
        for actor in env.actors:
            if actor.actor_type == "Patient" and actor.guiding is None:
                guide = env.find_nearest_actor(actor, ["Staff", "Adult"], max_distance=100)
                if guide and guide.guiding is None:
                    target_pos = guide.pos
                else:
                    target_pos = env.find_nearest_exit(actor).pos
            elif actor.actor_type == "Child" and actor.guiding is None:
                guide = env.find_nearest_actor(actor, ["Adult", "Staff"], max_distance=100)
                if guide:
                    target_pos = guide.pos
                else:
                    target_pos = env.find_nearest_exit(actor).pos
            else:
                target_pos = env.find_nearest_exit(actor).pos
            
            current_grid = (int(actor.pos[0] // CELL_SIZE), int(actor.pos[1] // CELL_SIZE))
            target_grid = (int(target_pos[0] // CELL_SIZE), int(target_pos[1] // CELL_SIZE))
            
            # Find path using the actor's algorithm
            path = actor.algorithm.find_path(current_grid, target_grid, env.grid, actor)
            
            # Move along the path if one exists
            if path and len(path) > 1:
                next_grid = path[1]
                next_pos = (next_grid[0] * CELL_SIZE + CELL_SIZE // 2,
                          next_grid[1] * CELL_SIZE + CELL_SIZE // 2)
                
                # Calculate movement
                direction = np.array(next_pos, dtype=float) - np.array(actor.pos, dtype=float)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    move_distance = actor.speed * CELL_SIZE
                    new_pos = np.array(actor.pos, dtype=float) + direction * move_distance
                    
                    # Check for collisions before moving
                    if not env.detect_collision(actor, new_pos):
                        actor.pos = new_pos.tolist()
        
        env.update_actors()
        env.render()
    
    pygame.quit()

    output_filename = f"evacuation_statistics_{selected_algorithm}.csv"
    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Actor Type", "Status", "Time (s)"])

        for actor_type in ACTOR_PROPERTIES.keys():
            for time in env.evacuation_times[actor_type]:
                writer.writerow([actor_type, "Evacuated", round(time, 2)])
            for time in env.death_times[actor_type]:
                writer.writerow([actor_type, "Deceased", round(time, 2)])

    print(f"\nSimulation complete. Statistics saved to '{output_filename}'")
    
    print("\n--- Simulation Summary ---")
    for actor_type in env.evacuation_times:
        evacs = len(env.evacuation_times[actor_type])
        deaths = len(env.death_times[actor_type])
        print(f"{actor_type}: {evacs} evacuated, {deaths} burned")

if __name__ == "__main__":
    main()