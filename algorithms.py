import pygame
import numpy as np
import json
import random
import heapq
import os
import csv
from typing import List, Tuple, Dict, Optional, Set

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

class PathfindingAlgorithm:
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, safety_weight: float = 1.0):
        self.safety_weight = safety_weight
        self.simulation = None

    def is_valid_move(self, position: Tuple[int, int], grid: List[List[str]]) -> bool:
        x, y = position
        return (0 <= x < len(grid[0]) and 
                0 <= y < len(grid) and 
                grid[y][x] not in ("wall", "fire"))

    def get_move_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int], 
                     grid: List[List[str]], actor: 'Actor') -> float:
        base_cost = 1.0
        
        # Smoke cost
        if self.simulation and self.simulation.smoke_grid[neighbor[1]][neighbor[0]] > 0:
            base_cost += self.simulation.smoke_grid[neighbor[1]][neighbor[0]] * 2
        
        # Crowding cost
        crowd_factor = 0.0
        for other in (self.simulation.actors if self.simulation else []):
            if other != actor:
                other_pos = other.to_grid_coords()
                distance = np.linalg.norm(np.array(neighbor) - np.array(other_pos))
                if distance < 3:  
                    crowd_factor += (3 - distance) / 3
        base_cost += crowd_factor
        
        # Actor type modifiers
        type_multipliers = {
            "Patient": 1.5,
            "Child": 1.3,
            "Staff": 0.8
        }
        return base_cost * type_multipliers.get(actor.actor_type, 1.0)

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int], start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

    def _initialize_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        open_set = []
        heapq.heappush(open_set, (0, start))
        return {
            'open_set': open_set,
            'came_from': {},
            'g_score': {start: 0},
            'open_set_hash': {start}
        }

class AStarAlgorithm(PathfindingAlgorithm):
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 grid: List[List[str]], actor: Optional['Actor'] = None) -> List[Tuple[int, int]]:
        search_data = self._initialize_search(start, goal)
        f_score = {start: self.heuristic(start, goal)}
        
        while search_data['open_set']:
            current = heapq.heappop(search_data['open_set'])[1]
            search_data['open_set_hash'].remove(current)
            
            if current == goal:
                return self._reconstruct_path(search_data['came_from'], current, start)
            
            for dx, dy in self.DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_move(neighbor, grid):
                    continue
                    
                move_cost = self.get_move_cost(current, neighbor, grid, actor)
                tentative_g_score = search_data['g_score'][current] + move_cost
                
                if neighbor not in search_data['g_score'] or tentative_g_score < search_data['g_score'][neighbor]:
                    search_data['came_from'][neighbor] = current
                    search_data['g_score'][neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in search_data['open_set_hash']:
                        heapq.heappush(search_data['open_set'], (f_score[neighbor], neighbor))
                        search_data['open_set_hash'].add(neighbor)
        
        return []

class DijkstraAlgorithm(PathfindingAlgorithm):
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 grid: List[List[str]], actor: Optional['Actor'] = None) -> List[Tuple[int, int]]:
        search_data = self._initialize_search(start, goal)
        
        while search_data['open_set']:
            current_g_score, current = heapq.heappop(search_data['open_set'])
            search_data['open_set_hash'].remove(current)
            
            if current == goal:
                return self._reconstruct_path(search_data['came_from'], current, start)
            
            for dx, dy in self.DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_move(neighbor, grid):
                    continue
                    
                move_cost = self.get_move_cost(current, neighbor, grid, actor)
                tentative_g_score = search_data['g_score'][current] + move_cost
                
                if neighbor not in search_data['g_score'] or tentative_g_score < search_data['g_score'][neighbor]:
                    search_data['came_from'][neighbor] = current
                    search_data['g_score'][neighbor] = tentative_g_score
                    
                    if neighbor not in search_data['open_set_hash']:
                        heapq.heappush(search_data['open_set'], (tentative_g_score, neighbor))
                        search_data['open_set_hash'].add(neighbor)
        
        return []

class Wall:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start, self.end = start, end

class FireExit:
    def __init__(self, pos: Tuple[int, int], size: Tuple[int, int] = (40, 40)):
        self.pos, self.size = pos, size

class Actor:
    def __init__(self, pos: Tuple[int, int], actor_type: str, 
                 speed: Optional[float] = None, constraints: Optional[Dict] = None, 
                 guided_speeds: Optional[Dict] = None):
        self.pos = np.array(pos, dtype=float)
        self.actor_type = actor_type
        self.guiding = None
        self.guided_by = None
        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.path = []
        self.goal = None
        self.last_grid_pos = self.to_grid_coords()
        self.last_path_update = 0
        self.path_update_interval = 500
        
        props = ACTOR_PROPERTIES.get(actor_type, {})
        self.color = props.get("color", BLACK)
        self.radius = props.get("radius", 10)
        self.speed = speed if speed is not None else props.get("speed", 1.0)
        
        algorithm_class = AStarAlgorithm if props.get("algorithm") == "AStar" else DijkstraAlgorithm
        self.algorithm = algorithm_class()
        self.algorithm.safety_weight = props.get("safety_weight", 1.0)
        self.algorithm.simulation = None

        self.constraints = constraints if constraints is not None else {}
        self.guided_speeds = guided_speeds if guided_speeds is not None else {
            "Patient": {"adult": 0.75, "staff": 1.0},
            "Child": {"adult": 1.0, "staff": 1.0}
        }.get(actor_type, {})

    def draw(self, screen):
        if self.actor_type == "Patient":
            width = 2 * self.radius
            height = int(2 * self.radius * 2.5)
            rect = pygame.Rect(self.pos[0] - self.radius, self.pos[1] - height // 2, width, height)
            pygame.draw.ellipse(screen, self.color, rect)
        else:
            pygame.draw.circle(screen, self.color, self.pos, self.radius)

        if self.guiding:
            distance = np.linalg.norm(self.pos - self.guiding.pos)
            if distance <= MAX_GUIDANCE_DISTANCE:
                # Draw a fading connection line
                alpha = max(0, 255 - int(distance * 2))
                color = (*BLACK, alpha)  # Fading black
                
                line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surface, color, self.pos, self.guiding.pos, 2)
                screen.blit(line_surface, (0, 0))

    def to_grid_coords(self) -> Tuple[int, int]:
        return (int(self.pos[0] // CELL_SIZE), int(self.pos[1] // CELL_SIZE))

    def from_grid_coords(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        return (grid_pos[0] * CELL_SIZE + CELL_SIZE // 2,
                grid_pos[1] * CELL_SIZE + CELL_SIZE // 2)

    def get_effective_speed(self) -> float:
        if self.guided_by and self.guided_by.actor_type.lower() in self.guided_speeds:
            return self.guided_speeds[self.guided_by.actor_type.lower()]
        return self.speed

class BlueprintEnvironment:
    def __init__(self, walls: List[Wall], exits: List[FireExit], 
                 actors: List[Actor], fires: List[Tuple[int, int]], screen=None):
        self.walls, self.exits, self.actors = walls, exits, actors
        self.fires = fires
        self.grid = self.compute_occupancy_grid()
        self.smoke_grid = [[0.0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.render_on = True
        self.evacuation_times = {t: [] for t in ACTOR_PROPERTIES.keys()}
        self.death_times = {t: [] for t in ACTOR_PROPERTIES.keys()}
        self.last_fire_spread = pygame.time.get_ticks()

        for actor in self.actors:
            actor.algorithm.simulation = self

        self.screen = screen or pygame.display.get_surface()
        pygame.display.set_caption("Hospital Simulation")

    def compute_occupancy_grid(self) -> List[List[str]]:
        grid = [["empty" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        for wall in self.walls:
            x0, y0 = np.array(wall.start) // CELL_SIZE
            x1, y1 = np.array(wall.end) // CELL_SIZE
            for t in np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0))):
                cx, cy = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                    grid[cy][cx] = "wall"
        
        for fx, fy in self.fires:
            grid_x, grid_y = int(fx // CELL_SIZE), int(fy // CELL_SIZE)
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                grid[grid_y][grid_x] = "fire"
        
        return grid

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
        now = pygame.time.get_ticks()
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
        actor.end_time = pygame.time.get_ticks()
        self.death_times[actor.actor_type].append((actor.end_time - actor.start_time) / 1000)
        if actor.guiding:
            actor.guiding.guided_by = None
        if actor.guided_by:
            actor.guided_by.guiding = None
        self.actors.remove(actor)

    def handle_actor_evacuation(self, actor: Actor):
        actor.end_time = pygame.time.get_ticks()
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

def load_blueprint(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)
    return (
        [Wall(**wall) for wall in data['walls']],
        [FireExit(**exit) for exit in data['fire_exits']],
        [Actor(
            pos=actor['pos'],
            actor_type=actor['type'],
            speed=actor.get('speed'),
            constraints=actor.get('constraints'),
            guided_speeds=actor.get('guided_speeds')
        ) for actor in data['actors']],
        [tuple(f["pos"]) for f in data.get("fires", [])]
    )

def pygame_file_picker(folder=".") -> Optional[str]:
    pygame.font.init()
    picker_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Select a Blueprint File")
    font = pygame.font.SysFont("arial", 24)

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    selected_file = None

    while True:
        picker_screen.fill(WHITE)
        
        for i, file in enumerate(files):
            text = font.render(file, True, BLACK)
            rect = text.get_rect(topleft=(20, 30 + i * 40))
            picker_screen.blit(text, rect)
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(picker_screen, (200, 200, 255), rect, 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, file in enumerate(files):
                    rect = pygame.Rect(20, 30 + i * 40, 560, 30)
                    if rect.collidepoint(event.pos):
                        return os.path.join(folder, files[i])

        pygame.display.flip()
        clock.tick(30)

def main():
    filename = pygame_file_picker()
    if not filename:
        print("No file selected. Exiting...")
        return
    walls, exits, actors, fires = load_blueprint(filename)
    env = BlueprintEnvironment(walls, exits, actors, fires)
    
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
        env.update_actors()
        env.render()
    pygame.quit()

    output_filename = "evacuation_statistics.csv"
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