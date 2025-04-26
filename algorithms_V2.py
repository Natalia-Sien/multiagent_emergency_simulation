import pygame
import numpy as np
import json
import random
import heapq
import os
import csv
import time
from typing import List, Tuple, Dict, Optional, Set

# Initialise pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FIRE_SPREAD_CHANCE = 0.02
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
        
        # Smoke cost
        if self.simulation and self.simulation.smoke_grid[neighbor[1]][neighbor[0]] > 0:
            base_cost += self.simulation.smoke_grid[neighbor[1]][neighbor[0]] * 2
        
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

# then we start the class for the Wall, FireExit and Actor
class Wall:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start, self.end = start, end

class FireExit:
    def __init__(self, pos: Tuple[int, int], size: Tuple[int, int] = (40, 40)):
        self.pos, self.size = pos, size

class Actor:
    def __init__(self, pos: Tuple[int, int], actor_type: str):
        self.pos = np.array(pos, dtype=float)
        self.actor_type = actor_type
        self.guiding = None
        self.guided_by = None
        self.start_time = time.time()
        self.end_time = None
        self.path = []
        self.goal = None
        self.last_grid_pos = self.to_grid_coords()
        self.last_path_update = 0
        self.path_update_interval = 500
        
        props = ACTOR_PROPERTIES.get(actor_type, {})
        self.radius = props.get("radius", 10)
        self.speed = props.get("speed", 1.0)
        self.algorithm = None  # Will be set later
        self.guided_speeds = {
            "staff": 1.0,
            "adult": 1.0,
            "patient": 0.5,
            "child": 0.33
        }

    def to_grid_coords(self) -> Tuple[int, int]:
        return (int(self.pos[0] // CELL_SIZE), int(self.pos[1] // CELL_SIZE))

    def from_grid_coords(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        return (grid_pos[0] * CELL_SIZE + CELL_SIZE // 2,
                grid_pos[1] * CELL_SIZE + CELL_SIZE // 2)

    def get_effective_speed(self) -> float:
        if self.guided_by and self.guided_by.actor_type.lower() in self.guided_speeds:
            return self.guided_speeds[self.guided_by.actor_type.lower()]
        return self.speed

# then we start the class for the BlueprintEnvironment
class BlueprintEnvironment:
    def __init__(self, walls: List[Wall], exits: List[FireExit], 
                 actors: List[Actor], fires: List[Tuple[int, int]]):
        self.walls, self.exits, self.actors = walls, exits, actors
        self.fires = fires
        self.grid = self.compute_occupancy_grid()
        self.smoke_grid = [[0.0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.evacuation_times = {t: [] for t in ACTOR_PROPERTIES.keys()}
        self.death_times = {t: [] for t in ACTOR_PROPERTIES.keys()}
        self.last_fire_spread = time.time()

        for actor in self.actors:
            actor.algorithm.simulation = self

    # compute the occupancy grid
    def compute_occupancy_grid(self) -> List[List[str]]:
        grid = [["empty" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # for each wall, compute the occupancy grid
        for wall in self.walls:
            x0, y0 = np.array(wall.start) // CELL_SIZE
            x1, y1 = np.array(wall.end) // CELL_SIZE
            for t in np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0))):
                cx, cy = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                    grid[cy][cx] = "wall"
        # same for the fires
        for fx, fy in self.fires:
            grid_x, grid_y = int(fx // CELL_SIZE), int(fy // CELL_SIZE)
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                grid[grid_y][grid_x] = "fire"
        
        return grid

    # update the actors
    def update_actors(self):
        self.spread_fire()
        
        for actor in self.actors[:]:
            if self.actor_in_fire(actor):
                self.handle_actor_death(actor)
                continue

            if actor.actor_type == "Child":
                self.move_child(actor)
            elif actor.actor_type == "Patient":
                self.move_patient(actor)
            elif actor.actor_type == "Staff":
                self.update_staff(actor)
            elif actor.actor_type == "Adult":
                self.update_adult(actor)

        for actor in self.actors[:]:
            if self.actor_reached_exit(actor) and (actor.actor_type != "Patient" or actor.guiding is None):
                self.handle_actor_evacuation(actor)

    def move_actor_along_path(self, actor: Actor, target_pos: Tuple[int, int]):
        start = actor.to_grid_coords()
        goal = (int(target_pos[0] // CELL_SIZE), int(target_pos[1] // CELL_SIZE))

        if not actor.path or actor.goal != goal:
            actor.path = actor.algorithm.find_path(start, goal, self.grid, actor)
            actor.goal = goal

        if actor.path and len(actor.path) > 1:
            next_grid = actor.path[1]
            next_pos = actor.from_grid_coords(next_grid)

            direction = np.array(next_pos) - np.array(actor.pos, dtype=float)
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction /= distance
                move_distance = actor.get_effective_speed() * CELL_SIZE
                actor.pos = np.array(actor.pos, dtype=float) + direction * move_distance

                if np.linalg.norm(np.array(actor.to_grid_coords()) - np.array(next_grid)) < 1:
                    actor.path.pop(0)

    def move_child(self, child: Actor):
        nearest_adult = self.find_nearest_actor(child, ["Adult", "Staff"], max_distance=100)
        if nearest_adult:
            self.move_actor_along_path(child, nearest_adult.pos)
        else:
            nearest_exit = self.find_nearest_exit(child)
            if nearest_exit:
                self.move_actor_along_path(child, nearest_exit.pos)

    def move_patient(self, patient: Actor):
        guide = self.find_nearest_actor(patient, ["Staff", "Adult"], max_distance=100)
        if guide and guide.guiding is None:
            self.move_actor_along_path(patient, guide.pos)
        else:
            nearest_exit = self.find_nearest_exit(patient)
            if nearest_exit:
                self.move_actor_along_path(patient, nearest_exit.pos)

    def update_staff(self, staff: Actor):
        MAX_GUIDANCE_DISTANCE = 120  
        CONNECTION_DISTANCE = 10    
        
        if staff.guiding:
            if staff.guiding not in self.actors:
                staff.guiding.guided_by = None
                staff.guiding = None
            elif np.linalg.norm(staff.pos - staff.guiding.pos) > MAX_GUIDANCE_DISTANCE:
                staff.guiding.guided_by = None
                staff.guiding = None
        
        if staff.guiding:
            exit_pos = self.find_nearest_exit(staff).pos
            self.move_actor_along_path(staff, exit_pos)
            
            if np.linalg.norm(staff.pos - staff.guiding.pos) <= MAX_GUIDANCE_DISTANCE:
                self.move_actor_along_path(staff.guiding, staff.pos)
            else:
                staff.guiding.guided_by = None
                staff.guiding = None
        
        else:
            patients = [
                p for p in self.get_unescorted_actors_in_range(staff, "Patient")
                if np.linalg.norm(staff.pos - p.pos) <= MAX_GUIDANCE_DISTANCE
            ]
            
            if patients:
                self.move_actor_along_path(staff, patients[0].pos)
                if np.linalg.norm(staff.pos - patients[0].pos) < CONNECTION_DISTANCE:
                    staff.guiding = patients[0]
                    patients[0].guided_by = staff
            
            else:
                children = [
                    c for c in self.get_unescorted_actors_in_range(staff, "Child")
                    if np.linalg.norm(staff.pos - c.pos) <= MAX_GUIDANCE_DISTANCE
                ]
                
                if children:
                    self.move_actor_along_path(staff, children[0].pos)
                    if np.linalg.norm(staff.pos - children[0].pos) < CONNECTION_DISTANCE:
                        staff.guiding = children[0]
                        children[0].guided_by = staff
                
                elif self.should_leave(staff):
                    exit_pos = self.find_nearest_exit(staff).pos
                    self.move_actor_along_path(staff, exit_pos)

    def update_adult(self, adult: Actor):
        if adult.guiding and adult.guiding not in self.actors:
            adult.guiding = None
            
        if adult.guiding:
            exit_pos = self.find_nearest_exit(adult).pos
            self.move_actor_along_path(adult, exit_pos)
            self.move_actor_along_path(adult.guiding, adult.pos)
        else:
            children = self.get_unescorted_actors_in_range(adult, "Child")
            if children:
                self.move_actor_along_path(adult, children[0].pos)
                if np.linalg.norm(np.array(adult.pos) - np.array(children[0].pos)) < 10:
                    adult.guiding = children[0]
            elif self.should_leave(adult):
                exit_pos = self.find_nearest_exit(adult).pos
                self.move_actor_along_path(adult, exit_pos)

    def get_unescorted_actors_in_range(self, guide_actor: Actor, target_type: str) -> List[Actor]:
        return [
            actor for actor in self.actors
            if (actor.actor_type == target_type and 
                actor.guiding is None and 
                np.linalg.norm(np.array(actor.pos) - np.array(guide_actor.pos)) < 100)
        ]

    def should_leave(self, actor: Actor) -> bool:
        return (len(self.get_unescorted_actors_in_range(actor, "Patient")) == 0 and 
               len(self.get_unescorted_actors_in_range(actor, "Child")) == 0)

    def find_nearest_actor(self, actor: Actor, types: List[str], max_distance: float) -> Optional[Actor]:
        nearest = None
        min_dist = float('inf')
        
        for other in self.actors:
            if other.actor_type in types and other != actor:
                dist = np.linalg.norm(np.array(actor.pos) - np.array(other.pos))
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    nearest = other
                    
        return nearest

    def actor_in_fire(self, actor: Actor) -> bool:
        for fire in self.fires:
            if np.linalg.norm(np.array(actor.pos) - np.array(fire)) < FIRE_RADIUS:
                return True
        return False

    def spread_fire(self):
        now = time.time()
        if now - self.last_fire_spread < 200:
            return
        self.last_fire_spread = now

        new_fires = set()
        existing_fire_cells = set((fx // CELL_SIZE, fy // CELL_SIZE) for fx, fy in self.fires)

        for fx, fy in self.fires:
            grid_x, grid_y = fx // CELL_SIZE, fy // CELL_SIZE
            neighbors = [
                (grid_x + 1, grid_y), (grid_x - 1, grid_y),
                (grid_x, grid_y + 1), (grid_x, grid_y - 1)
            ]
            for nx, ny in neighbors:
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                    (nx, ny) not in existing_fire_cells and 
                    random.random() < FIRE_SPREAD_CHANCE):
                    new_fires.add((nx, ny))

        for nx, ny in new_fires:
            px, py = nx * CELL_SIZE + CELL_SIZE // 2, ny * CELL_SIZE + CELL_SIZE // 2
            self.fires.append((px, py))
            self.grid[ny][nx] = "fire"

    def find_nearest_exit(self, actor: Actor) -> Optional[FireExit]:
        nearest = None
        min_dist = float('inf')
        
        for exit in self.exits:
            dist = np.linalg.norm(np.array(actor.pos) - np.array(exit.pos))
            if dist < min_dist:
                min_dist = dist
                nearest = exit
                
        return nearest

    def actor_reached_exit(self, actor: Actor) -> bool:
        for exit in self.exits:
            if np.linalg.norm(np.array(actor.pos) - np.array(exit.pos)) < exit.size[0] // 2:
                return True
        return False

    def handle_actor_death(self, actor: Actor):
        actor.end_time = time.time()
        self.death_times[actor.actor_type].append((actor.end_time - actor.start_time) / 1000)
        if actor.guiding:
            actor.guiding.guided_by = None
        if actor.guided_by:
            actor.guided_by.guiding = None
        self.actors.remove(actor)

    def handle_actor_evacuation(self, actor: Actor):
        actor.end_time = time.time()
        self.evacuation_times[actor.actor_type].append((actor.end_time - actor.start_time) / 1000)
        if actor.guiding:
            actor.guiding.guided_by = None
        self.actors.remove(actor)

    def render(self):
        if not self.render_on:
            return
            
        self.screen.fill(WHITE)
        
        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, wall.start, wall.end, 2)
            
        for exit in self.exits:
            pygame.draw.rect(self.screen, RED, 
                           (exit.pos[0] - exit.size[0]//2, 
                            exit.pos[1] - exit.size[1]//2, 
                            exit.size[0], exit.size[1]))
            
        for fire in self.fires:
            pygame.draw.circle(self.screen, ORANGE, fire, FIRE_RADIUS)
            
        for actor in self.actors:
            actor.draw(self.screen)
            
        pygame.display.flip()
        clock.tick(FPS)


def run_parameterized_simulation(num_exits: int, num_actors: int, staff_percentage: float, time_steps: int, algorithm: str = "AStar"):
    """
    Run a simulation with specific parameters and collect statistics.
    """
    start_time = time.time()
    
    # Create walls (simple rectangular room)
    walls = [
        Wall((0, 0), (WIDTH, 0)),
        Wall((0, 0), (0, HEIGHT)),
        Wall((WIDTH, 0), (WIDTH, HEIGHT)),
        Wall((0, HEIGHT), (WIDTH, HEIGHT))
    ]
    
    # Create fire exits evenly spaced along the bottom wall
    exits = []
    exit_spacing = WIDTH // (num_exits + 1)
    for i in range(num_exits):
        exit_pos = (exit_spacing * (i + 1), HEIGHT - 20)
        exits.append(FireExit(exit_pos))
    
    # Create actors with specified distribution
    actors = []
    num_staff = int(num_actors * staff_percentage / 100)
    num_others = num_actors - num_staff
    
    # Distribute remaining actors among other types
    num_adults = int(num_others * 0.4)
    num_patients = int(num_others * 0.3)
    num_children = num_others - num_adults - num_patients
    
    actor_specs = [
        ("Staff", num_staff),
        ("Adult", num_adults),
        ("Patient", num_patients),
        ("Child", num_children)
    ]

    for actor_type, count in actor_specs:
        for _ in range(count):
            pos = (random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50))
            actor = Actor(pos, actor_type)
            actor.algorithm = AStarAlgorithm() if algorithm == "AStar" else DijkstraAlgorithm()
            actors.append(actor)
    
    # Create initial fire
    fires = [(WIDTH//2, HEIGHT//2)]
    
    # Create and run simulation
    env = BlueprintEnvironment(walls, exits, actors, fires)
    env.render_on = False  # Disable rendering for faster execution
    
    # Run simulation for specified time steps
    for _ in range(time_steps):
        env.update_actors()
        if not env.actors:  # All actors have evacuated or died
            break
    
    # Calculate statistics
    evacuated = {t: len(env.evacuation_times[t]) for t in ACTOR_PROPERTIES.keys()}
    deceased = {t: len(env.death_times[t]) for t in ACTOR_PROPERTIES.keys()}
    
    # Calculate average evacuation times
    avg_evacuated_time = {}
    for actor_type in ACTOR_PROPERTIES.keys():
        times = env.evacuation_times[actor_type]
        avg_evacuated_time[actor_type] = sum(times) / len(times) if times else 0
    
    # Calculate evacuation rates
    evacuation_rate = {}
    for actor_type in ACTOR_PROPERTIES.keys():
        total = evacuated[actor_type] + deceased[actor_type]
        evacuation_rate[actor_type] = (evacuated[actor_type] / total * 100) if total > 0 else 0
    
    # Calculate simulation time
    simulation_time = time.time() - start_time
    
    # Collect all statistics
    stats = {
        "num_exits": num_exits,
        "num_actors": num_actors,
        "staff_percentage": staff_percentage,
        "time_steps": time_steps,
        "algorithm": algorithm,
        "evacuated": evacuated,
        "deceased": deceased,
        "avg_evacuated_time": avg_evacuated_time,
        "evacuation_rate": evacuation_rate,
        "total_evacuated": sum(evacuated.values()),
        "total_deceased": sum(deceased.values()),
        "simulation_time": simulation_time
    }
    
    return stats

def run_experiments():
    """
    Run multiple simulations with different parameters and save results to CSV.
    """
    start_time = time.time()
    results = []
    
    # Define parameter ranges with fewer combinations
    num_exits_range = [1, 3, 5] 
    num_actors_range = [50, 150, 200] 
    staff_percentage_range = [15, 20, 25]  
    time_steps_range = [250, 375, 500]  
    algorithms = ["AStar", "Dijkstra"]
    
    # Run experiments
    total_experiments = len(num_exits_range) * len(num_actors_range) * len(staff_percentage_range) * len(time_steps_range) * len(algorithms)
    current_experiment = 0
    
    # Create CSV file and write header
    output_filename = "experiment_results.csv"
    with open(output_filename, mode='w', newline='') as csvfile:
        fieldnames = [
            "num_exits", "num_actors", "staff_percentage", "time_steps", "algorithm",
            "Staff_evacuated", "Staff_deceased", "Staff_avg_time", "Staff_evac_rate",
            "Adult_evacuated", "Adult_deceased", "Adult_avg_time", "Adult_evac_rate",
            "Patient_evacuated", "Patient_deceased", "Patient_avg_time", "Patient_evac_rate",
            "Child_evacuated", "Child_deceased", "Child_avg_time", "Child_evac_rate",
            "total_evacuated", "total_deceased", "simulation_time"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for num_exits in num_exits_range:
        for num_actors in num_actors_range:
            for staff_percentage in staff_percentage_range:
                for time_steps in time_steps_range:
                    for algorithm in algorithms:
                        current_experiment += 1
                        print(f"Running experiment {current_experiment}/{total_experiments}")
                        print(f"Parameters: exits={num_exits}, actors={num_actors}, staff={staff_percentage}%, time_steps={time_steps}, algorithm={algorithm}")
                        
                        stats = run_parameterized_simulation(
                            num_exits=num_exits,
                            num_actors=num_actors,
                            staff_percentage=staff_percentage,
                            time_steps=time_steps,
                            algorithm=algorithm
                        )
                        results.append(stats)
                        
                        # Save results after each experiment
                        with open(output_filename, mode='a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            row = {
                                "num_exits": stats["num_exits"],
                                "num_actors": stats["num_actors"],
                                "staff_percentage": stats["staff_percentage"],
                                "time_steps": stats["time_steps"],
                                "algorithm": stats["algorithm"],
                                "Staff_evacuated": stats["evacuated"]["Staff"],
                                "Staff_deceased": stats["deceased"]["Staff"],
                                "Staff_avg_time": stats["avg_evacuated_time"]["Staff"],
                                "Staff_evac_rate": stats["evacuation_rate"]["Staff"],
                                "Adult_evacuated": stats["evacuated"]["Adult"],
                                "Adult_deceased": stats["deceased"]["Adult"],
                                "Adult_avg_time": stats["avg_evacuated_time"]["Adult"],
                                "Adult_evac_rate": stats["evacuation_rate"]["Adult"],
                                "Patient_evacuated": stats["evacuated"]["Patient"],
                                "Patient_deceased": stats["deceased"]["Patient"],
                                "Patient_avg_time": stats["avg_evacuated_time"]["Patient"],
                                "Patient_evac_rate": stats["evacuation_rate"]["Patient"],
                                "Child_evacuated": stats["evacuated"]["Child"],
                                "Child_deceased": stats["deceased"]["Child"],
                                "Child_avg_time": stats["avg_evacuated_time"]["Child"],
                                "Child_evac_rate": stats["evacuation_rate"]["Child"],
                                "total_evacuated": stats["total_evacuated"],
                                "total_deceased": stats["total_deceased"],
                                "simulation_time": stats["simulation_time"]
                            }
                            writer.writerow(row)
                        
                        # Print timing information
                        elapsed_time = time.time() - start_time
                        avg_time_per_experiment = elapsed_time / current_experiment
                        estimated_remaining_time = avg_time_per_experiment * (total_experiments - current_experiment)
                        print(f"Current experiment took {stats['simulation_time']:.2f} seconds")
                        print(f"Average time per experiment: {avg_time_per_experiment:.2f} seconds")
                        print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")
                        print(f"Total elapsed time: {elapsed_time/60:.2f} minutes")
                        print("-" * 50)
                        
                        # Print current results
                        print("\nCurrent Results:")
                        print(f"Staff: {stats['evacuated']['Staff']} evacuated, {stats['deceased']['Staff']} deceased")
                        print(f"Adult: {stats['evacuated']['Adult']} evacuated, {stats['deceased']['Adult']} deceased")
                        print(f"Patient: {stats['evacuated']['Patient']} evacuated, {stats['deceased']['Patient']} deceased")
                        print(f"Child: {stats['evacuated']['Child']} evacuated, {stats['deceased']['Child']} deceased")
                        print(f"Total: {stats['total_evacuated']} evacuated, {stats['total_deceased']} deceased")
                        print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"\nExperiments complete. Results saved to '{output_filename}'")
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")


if __name__ == "__main__":
    run_experiments()
