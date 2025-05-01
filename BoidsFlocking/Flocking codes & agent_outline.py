import pygame
import numpy as np
import json
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import csv

import numpy as np
import math
from math import floor

# --------------- parameters that can be tuned to adjust behaviour ---------------

# size of one grid cell in pixels
CELL_SIZE = 5

# the radius where each boids can influence each other
Neighbour_Radius = 50
# the radius that boids will repel each other
Separation_Radius = 20

# weights for sepeartion alignment and cohesion
Separation_W = 1
Alignment_W = 1
Cohesion_W = 1

# attraction toward nearest exit
Goal_Attr_W = 10
# avoide nearby fire
Avoid_Fire_W = 10
# repulsion from walls
Avoid_Walls_W = 1

# velocity scaling
FORCE_SCALE = 0.02

# define each actors speed (units per frame)
Norm_S = {"Staff":5, "Adult":5,
         # child moves slower when alone
         "Child":3,
         # patient cant move without guidence
         "Patient":0}

# If child or patient is guided (near a staff or adult), their speed will change
Guided_S = {# child moves quicker if guided
            "Child":5,
            # patient can move if guided
            "Patient":5}

# ---------------------- define some helper functions -----------------------

# find the position of the nearest fire exit
def find_exit(pos,exits):
    # return none if there is none found
    if not exits:
        return None
    dist_best = float('inf')
    best_pos = None
    for e in exits:
        epos = np.array(e.pos, dtype=float)
        d = np.linalg.norm(pos - epos)
        if d < dist_best:
            dist_best = d
            best_pos = epos
    return best_pos

# compute a vector that repels the actor away from nearby fires
def avoid_fire(pos, fires, max_radius=100):
    repel = np.zeros(2)
    px, py = pos
    max_radius_sq = max_radius * max_radius
    for fx, fy in fires:
        dx, dy = px-fx, py - fy
        dist_sq = dx*dx + dy*dy
        if dist_sq < max_radius_sq:
            dist = math.sqrt(dist_sq) + 1e-5
            factor = (max_radius-dist)/max_radius
            repel += np.array([dx,dy])/dist*factor
    return repel


# compute a vector that repels the actor away from wall
def avoid_walls(pos, occ_grid, vision_cells=20):
    repel = np.zeros(2, float)
    gx = int(pos[0]//CELL_SIZE)
    gy = int(pos[1]//CELL_SIZE)
    for dy in range(-vision_cells,vision_cells+1):
        for dx in range(-vision_cells, vision_cells+1):
            nx, ny = gx+dx, gy+dy
            if 0 <= nx < occ_grid.shape[1] and 0 <= ny < occ_grid.shape[0]:
                if occ_grid[ny, nx] == 1:
                    center = np.array(((nx+0.5)*CELL_SIZE,(ny+0.5)*CELL_SIZE),float)
                    offset = pos - center
                    dist = np.linalg.norm(offset) + 1e-5
                    threshold = vision_cells*CELL_SIZE
                    if dist < threshold:
                        strength = (threshold-dist)/threshold
                        repel += (offset/dist)*strength
    return repel

# checks if hitting a wall
def is_collision(new_pos, obstacle_grid):
    x_idx = int(floor(new_pos[0] / CELL_SIZE))
    y_idx = int(floor(new_pos[1] / CELL_SIZE))
    x_idx = max(0, min(obstacle_grid.shape[1]-1, x_idx))
    y_idx = max(0, min(obstacle_grid.shape[0]-1, y_idx))
    return obstacle_grid[y_idx, x_idx] == 1


# -------------------- main function to simulate the actor behaviours --------------------
# The function that updates actor positions and velocities using a Boids like emergent behaviors

def update_actors_boids(actors, exits, fires, obstacle_grid):
    """
    actors: list of Actor objects
    exits: list of FireExit objects
    fires: list of (x,y) coords for current fire tiles
    obstacle_grid: 2D np.array from agent_outline where 1=wall and 0=free
    """
    velocities = {}
    # get the velocity for each actor
    for actor in actors:
        # if not defined yet, random initial velocity is assigned
        if not hasattr(actor, "velocity"):
            # generates uniformly between the interval -1 and 1
            actor.velocity = np.random.uniform(-1, 1, size=2)
        velocities[actor] = np.array(actor.velocity, dtype=float)
       
    # check neighbor boids
    neighbor_map = {}
    # for each actor find corresponding neighbour actors
    for a in actors:
        # a list to store neighbour actors
        neighbor_map[a] = []
        for b in actors:
            if a == b:
                continue
            dist = np.linalg.norm(np.array(a.pos)-np.array(b.pos))
            # considered as neighbour if within defined parameter Neighbour_Radius
            if dist < Neighbour_Radius:
                neighbor_map[a].append(b)


    # for each actor, compute behaviour vectors:
    for actor in actors:
        position_A = np.array(actor.pos, dtype=float)
        velocity_A = velocities[actor]

        # Boids behaciours: separation, alignment, cohesion
        # initialise to zero vectors
        separation_force = np.zeros(2)
        alignment_force = np.zeros(2)
        cohesion_force = np.zeros(2)
        neighbor_count = len(neighbor_map[actor])

        # for each neighbour of the actors
        for neighbor in neighbor_map[actor]:
            position_B = np.array(neighbor.pos, dtype=float)
            velocity_B = velocities[neighbor]

            # Separation
            offset = position_A-position_B
            dist = np.linalg.norm(offset)
            if dist < Separation_Radius and dist > 1e-5:
                # stronger repulsion if the boids are closer
                separation_force += offset/(dist*dist)
            # Alignment
            alignment_force += velocity_B
            # Cohesion
            cohesion_force += position_B

        if neighbor_count > 0:
            # average alignment and cohesion to encourage grouping
            alignment_force /= neighbor_count
            cohesion_force  /= neighbor_count
            cohesion_force = cohesion_force - position_A

        # weighted sum of the forces
        boids_force = (Separation_W*separation_force + Alignment_W*alignment_force +
                       Cohesion_W*cohesion_force)

        # goal Attraction
        # staff will help patients
        if actor.actor_type == "Staff":
            # search patients in sight
            nearest_patient = None
            nearest_dist = float('inf')
            for n in neighbor_map[actor]:
                if n.actor_type == "Patient":
                    dist = np.linalg.norm(np.array(n.pos, float)-position_A)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_patient = n
            
            # if patient found within radius, move toward the patient
            if nearest_patient is not None and nearest_dist < Neighbour_Radius:
                patient_pos = np.array(nearest_patient.pos, float)
                to_patient = patient_pos-position_A
                norm = np.linalg.norm(to_patient)+1e-5
                boids_force += Goal_Attr_W * (to_patient / norm)
            else:
                # go to the exit if no patients insight
                exit_pos = find_exit(position_A, exits)
                if exit_pos is not None:
                    to_goal = exit_pos-position_A
                    norm = np.linalg.norm(to_goal)+1e-5
                    boids_force += Goal_Attr_W*(to_goal/norm)

        else:
            # other agents go straight to the exit
            exit_pos = find_exit(position_A, exits)
            if exit_pos is not None:
                to_goal = exit_pos-position_A
                norm = np.linalg.norm(to_goal)+1e-5
                boids_force += Goal_Attr_W *(to_goal/norm)

        # avoid fire
        boids_force += Avoid_Fire_W*avoid_fire(position_A,fires)
        # avoid wall
        boids_force += Avoid_Walls_W*avoid_walls(position_A,obstacle_grid,vision_cells=20)

        # if child or patient is near a staff or adult, they can move at guided speed.
        actor_max_speed = Norm_S.get(actor.actor_type, 5)
        if actor.actor_type in ["Child", "Patient"]:
            # search neighbors for a ataff or adult
            for n in neighbor_map[actor]:
                if n.actor_type in ["Staff","Adult"]:
                    dist = np.linalg.norm(np.array(n.pos)-position_A)
                    if dist < Neighbour_Radius:
                        # can be guided if close enough
                        actor_max_speed = Guided_S[actor.actor_type]
                        break

        # update actors velocity
        velocity_new = velocity_A + boids_force*FORCE_SCALE  # scale to damp forces 
        # limit the speed of boids
        speed = np.linalg.norm(velocity_new)
        if speed > actor_max_speed:
            velocity_new = (velocity_new/speed)*actor_max_speed
        velocities[actor] = velocity_new

    # apply velocities with wall-sliding
    for actor in actors:
        old_pos = np.array(actor.pos, float)
        v = velocities[actor]
        new_pos= old_pos + v

        # if no collision, hooray
        if not is_collision(new_pos, obstacle_grid):
            actor.pos = new_pos.astype(int).tolist()
            actor.velocity = v
        else:
            # find approximate normal from 8 neighbors when hitting a wall
            gx = int(old_pos[0] // CELL_SIZE)
            gy = int(old_pos[1] // CELL_SIZE)
            gx = np.clip(gx, 0, obstacle_grid.shape[1]-1)
            gy = np.clip(gy, 0, obstacle_grid.shape[0]-1)
            normal = np.zeros(2,float)
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1),
                           (1,1), (1,-1), (-1,1), (-1,-1)):
                nx,ny = gx+dx,gy+dy
                if (0 <= nx < obstacle_grid.shape[1] and 0 <= ny < obstacle_grid.shape[0] and
                    obstacle_grid[ny, nx] == 1):
                    # center of that wall cell
                    center = np.array(((nx+0.5)*CELL_SIZE,(ny+0.5)*CELL_SIZE),float)
                    normal += (old_pos-center)

            nlen = np.linalg.norm(normal)
            if nlen > 1e-5:
                # normalize and compute tangent
                normal /= nlen
                tangent = np.array([ normal[1],-normal[0]],float)
                # project velocity onto the wall tangent
                slide_v = np.dot(v,tangent)*tangent
                slide_pos = old_pos + slide_v

                if not is_collision(slide_pos,obstacle_grid):
                    # if sliding works
                    actor.pos = slide_pos.astype(int).tolist()
                    actor.velocity = slide_v
                else:
                    # if sliding fails, agent will try axis-aligned movement
                    vx,vy = v
                    x_only = np.array([vx, 0], float)
                    if not is_collision(old_pos+x_only, obstacle_grid):
                        actor.pos = (old_pos+x_only).astype(int).tolist()
                        actor.velocity = x_only
                    else:
                        y_only = np.array([0.0,vy], float)
                        if not is_collision(old_pos + y_only, obstacle_grid):
                            actor.pos = (old_pos+y_only).astype(int).tolist()
                            actor.velocity = y_only
                        else:
                            # completely stuck
                            actor.velocity = np.zeros(2)
            else:
                # no valid normal found
                actor.velocity = np.zeros(2)







# ======================================================


#Initialization of pygame
pygame.init()  #initialize pygame modules

#General settings
WIDTH, HEIGHT = 1000, 700  #screen dimensions
CELL_SIZE = 5  #size of each grid cell in pixels
GRID_WIDTH = WIDTH // CELL_SIZE  #number of grid cells horizontally
GRID_HEIGHT = HEIGHT // CELL_SIZE  #number of grid cells vertically
FIRE_SPREAD_CHANCE = 0.02  #probability fire spreads each update
FIRE_RADIUS = 5  #radius in pixels for fire kill

#Screen setup
pygame.display.set_caption("Agent Outline")  #set window title
clock = pygame.time.Clock()  #clock to control FPS
FPS = 60  #frames per second target

#Colors
WHITE = (255, 255, 255)  #background color
BLACK = (0, 0, 0)  #walls and text
RED = (255, 0, 0)  #exit color
ORANGE = (255, 165, 0)  #fire color

class Wall:
    def __init__(self, start, end):
        self.start, self.end = start, end  #endpoints of wall line

class FireExit:
    def __init__(self, pos, size=(40, 40)):
        self.pos, self.size = pos, size  #position and size of exit

class Actor:
    def __init__(self, pos, actor_type, speed=None, constraints=None, guided_speeds=None):
        self.pos = pos  #actor position
        self.actor_type = actor_type  #type of actor
        self.guiding = None  #link to guided actor

        #track movement history for penalty
        self.previous_pos = pos[:]  #previous position copy

        #track timing for metrics
        self.start_time = None  #time when actor spawned
        self.end_time = None  #time when actor exits or dies

        #set speed or default by type
        if speed is None:
            self.speed = {
                "Staff": 1.0,
                "Adult": 1.0,
                "Patient": 0.0,
                "Child": 0.33
            }.get(actor_type, 1.0)
        else:
            self.speed = speed

        #set constraints by type
        self.constraints = constraints if constraints is not None else {
            "Staff": {"knows_exit": True},
            "Adult": {"knows_exit": True, "follows_staff": True},
            "Patient": {"needs_guidance": True},
            "Child": {}
        }.get(actor_type, {})

        #set guided speeds by type
        self.guided_speeds = guided_speeds if guided_speeds is not None else {
            "Patient": {"adult": 0.75, "staff": 1.0},
            "Child": {"adult": 1.0, "staff": 1.0}
        }.get(actor_type, {})

        #set draw color by type
        self.color = {
            "Staff": (0, 0, 255),
            "Adult": (0, 255, 0),
            "Patient": (255, 255, 0),
            "Child": (200, 100, 200)
        }.get(actor_type, (0, 0, 0))

        #set radius for drawing
        self.radius = 8 if actor_type == "Child" else 10

    def draw(self, screen):
        #draw patient as ellipse
        if self.actor_type == "Patient":
            width = 2 * self.radius
            height = int(2 * self.radius * 2.5)
            rect = pygame.Rect(
                self.pos[0] - self.radius,
                self.pos[1] - height // 2,
                width,
                height
            )
            pygame.draw.ellipse(screen, self.color, rect)
        else:
            #draw others as circle
            pygame.draw.circle(screen, self.color, self.pos, self.radius)

        #draw guiding link if exists
        if self.guiding:
            pygame.draw.line(screen, BLACK, self.pos, self.guiding.pos, 2)


def load_blueprint(filename):
    with open(filename, 'r') as f:
        data = json.load(f)  #load json data
    walls = [Wall(**wall) for wall in data['walls']]  #create wall objects
    exits = [FireExit(**exit) for exit in data['fire_exits']]  #create exit objects
    actors = [Actor(
        pos=actor['pos'],
        actor_type=actor['type'],
        speed=actor.get('speed'),
        constraints=actor.get('constraints'),
        guided_speeds=actor.get('guided_speeds')
    ) for actor in data['actors']]  #create actor objects
    #init timing for actors
    start_ticks = pygame.time.get_ticks()
    for actor in actors:
        actor.start_time = start_ticks  #set start_time
    fires = [tuple(f["pos"]) for f in data.get("fires", [])]  #list of fire positions
    return walls, exits, actors, fires  #return loaded entities

class BlueprintEnvironment:
    def __init__(self, walls, exits, actors, fires, screen=None):
        self.walls, self.exits, self.actors = walls, exits, actors  #store entities
        self.fires = fires  #store fire positions
        self.grid = self.compute_occupancy_grid()  #compute grid occupancy
        self.render_on = True  #enable rendering
        self.evacuation_times = {"Staff": [], "Adult": [], "Patient": [], "Child": []}  #evacuation metrics
        self.death_times = {"Staff": [], "Adult": [], "Patient": [], "Child": []}  #death metrics
        self.initial_population = len(actors) #initial population is the number of total actors
        self.initial_counts = {
            "Staff": sum(1 for a in actors if a.actor_type == "Staff"), #dict entry for staff
            "Adult": sum(1 for a in actors if a.actor_type == "Adult"), #dict entry for adults
            "Patient": sum(1 for a in actors if a.actor_type == "Patient"), #dict entry for patients
            "Child": sum(1 for a in actors if a.actor_type == "Child"), #dict entry for children
        }

        self.time_steps = 0

        #init timing for actors
        start_ticks = pygame.time.get_ticks()
        for actor in self.actors:
            actor.start_time = start_ticks

        #set up screen or reuse existing
        if screen is not None:
            self.screen = screen
        else:
            self.screen = pygame.display.get_surface() or pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.display.set_caption("Hospital Simulation")  #window title update


    def compute_occupancy_grid(self):
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))  #empty grid
        for wall in self.walls:
            x0, y0 = np.array(wall.start) // CELL_SIZE
            x1, y1 = np.array(wall.end) // CELL_SIZE
            for t in np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0))):
                cx = int(x0 + t * (x1 - x0))
                cy = int(y0 + t * (y1 - y0))
                grid[cy, cx] = 1  #mark wall cells
        return grid  #return occupancy grid

    def reset(self, seed=None, options=None):
        new_actors = []
        for actor in self.actors:
            #randomize initial positions slightly
            new_pos = [actor.pos[0] + np.random.randint(-3, 4), actor.pos[1] + np.random.randint(-3, 4)]
            new_actor = Actor(new_pos, actor.actor_type, actor.speed, actor.constraints, actor.guided_speeds)
            new_actor.start_time = pygame.time.get_ticks()  #set new start time
            new_actors.append(new_actor)
        self.actors = new_actors  #replace actors

    def update_actors(self):
        self.spread_fire()
        update_actors_boids(self.actors, self.exits, self.fires, self.grid)

        self.time_steps += 1

        # Remove burned or escaped actors
        for actor in self.actors[:]:
            if self.actor_in_fire(actor):
                actor.end_time = pygame.time.get_ticks()
                self.death_times[actor.actor_type].append((actor.end_time - actor.start_time) / 1000)
                self.actors.remove(actor)
            elif self.actor_reached_exit(actor):
                actor.end_time = pygame.time.get_ticks()
                self.evacuation_times[actor.actor_type].append((actor.end_time - actor.start_time) / 1000)
                self.actors.remove(actor)


    def actor_in_fire(self, actor):
        #check if within fire radius of any fire
        for fire in self.fires:
            if np.linalg.norm(np.array(actor.pos) - np.array(fire)) < FIRE_RADIUS:
                return True
        return False

    def spread_fire(self):
        new_fires = set()
        for fx, fy in self.fires:  #for each existing fire cell
            neighbors = [
                (fx + CELL_SIZE, fy), (fx - CELL_SIZE, fy),
                (fx, fy + CELL_SIZE), (fx, fy - CELL_SIZE)
            ]  #adjacent cells
            for nx, ny in neighbors:
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    if (nx, ny) not in self.fires and random.random() < FIRE_SPREAD_CHANCE:
                        new_fires.add((nx, ny))  #spread fire
        self.fires.extend(list(new_fires))  #add new fires


    def get_unescorted_actors_in_range(self, guide_actor, target_type):
        #get vision cells around guide_actor
        visible = self.actor_vision(guide_actor, vision_range=10)
        found = []
        for a in self.actors:
            #filter by type and not already being guided
            if a.actor_type == target_type and all(g.guiding != a for g in self.actors):
                cell = tuple(np.array(a.pos) // CELL_SIZE)
                #if in vision, add to list
                if cell in visible:
                    found.append(a)
        return found

    def should_leave(self, guide_actor):
        #determine if guide should leave without guiding
        visible_cells = self.actor_vision(guide_actor, vision_range=10)
        for actor in self.actors:
            if actor == guide_actor or self.actor_reached_exit(actor):
                continue
            if actor.actor_type in ["Patient", "Child"]:
                #if any unguided remain within vision, don't leave
                if all(other.guiding != actor for other in self.actors if other != actor):
                    actor_cell = tuple(np.array(actor.pos) // CELL_SIZE)
                    if actor_cell in visible_cells:
                        return False
        return True

    def find_nearest_actor(self, actor, types, max_distance):
        #find closest actor of given types within max_distance
        min_dist = float('inf')
        nearest = None
        for other in self.actors:
            if other.actor_type in types:
                dist = np.linalg.norm(np.array(actor.pos) - np.array(other.pos))
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    nearest = other
        return nearest

    def move_towards(self, actor, target, speed):
        #move actor toward target scaled by speed
        direction = np.array(target) - np.array(actor.pos)
        if np.linalg.norm(direction) > 0:
            step = direction / np.linalg.norm(direction) * (5 * speed)
            new_pos = np.array(actor.pos) + step
            if not self.detect_collision(actor, new_pos):
                actor.pos = new_pos.astype(int).tolist()

    def actor_vision(self, actor, vision_range=5):
        #return set of grid cells within square vision range
        x, y = np.array(actor.pos) // CELL_SIZE
        return set(
            (x + dx, y + dy)
            for dx in range(-vision_range, vision_range + 1)
            for dy in range(-vision_range, vision_range + 1)
            if 0 <= x + dx < GRID_WIDTH and 0 <= y + dy < GRID_HEIGHT
        )

    def detect_collision(self, actor, new_pos):
        #detect if new_pos is inside a wall cell
        x, y = (np.array(new_pos) // CELL_SIZE).astype(int)
        x = np.clip(x, 0, self.grid.shape[1] - 1)
        y = np.clip(y, 0, self.grid.shape[0] - 1)
        return self.grid[y, x] == 1

    def render(self):
        #initialize display if needed
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        #ensure screen exists
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        #clear screen
        self.screen.fill(WHITE)
        #draw walls, exits, fires, actors
        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, wall.start, wall.end, 5)
        for exit in self.exits:
            rect = pygame.Rect(exit.pos[0] - exit.size[0] // 2,
                               exit.pos[1] - exit.size[1] // 2,
                               *exit.size)
            pygame.draw.rect(self.screen, RED, rect)
        for fire in self.fires:
            pygame.draw.circle(self.screen, ORANGE, fire, FIRE_RADIUS)
        for actor in self.actors:
            actor.draw(self.screen)
        #update display and process events
        pygame.display.flip()
        pygame.event.pump()

    def find_nearest_exit(self, actor):
        #return the exit object closest by euclidean distance
        return min(self.exits,
                   key=lambda e: np.linalg.norm(np.array(actor.pos) - np.array(e.pos)))

    def actor_reached_exit(self, actor):
        #check if actor is within 20px of any exit
        return any(
            np.linalg.norm(np.array(actor.pos) - np.array(exit.pos)) < 20
            for exit in self.exits
        )

    def export_metrics(self, csv_path="evacuation_results.csv"):
        """
        writes out each actor's evacuation or death time to a CSV
        and prints a summary to the console for debugging
        """
        #write CSV file
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Actor Type", "Status", "Time (s)"])
            for actor_type, times in self.evacuation_times.items():
                for t in times:
                    writer.writerow([actor_type, "Evacuated", f"{t:.2f}"])
            for actor_type, times in self.death_times.items():
                for t in times:
                    writer.writerow([actor_type, "Burned", f"{t:.2f}"])
        print(f"Results exported to {csv_path}\n")
        #print summary stats
        print("--- Simulation Summary ---")
        for actor_type in self.evacuation_times:
            evacs = len(self.evacuation_times[actor_type])
            deaths = len(self.death_times[actor_type])
            print(f"{actor_type}: {evacs} evacuated, {deaths} burned")

    def export_advanced_metrics(self, csv_path="advanced_evacuation_results.csv"):
        """
        writes one row of high‐level summary metrics to CSV:
        exits, population, type percentages, time steps,
        overall avg evac/death times, per‐type avg escape times,
        total evac/burned, and per‐type evac counts.
        """

        #basics
        n_exits = len(self.exits)
        pop = self.initial_population
        counts = self.initial_counts

        #percentages by type
        pct = {
            t: (counts[t] / pop if pop > 0 else 0.0)
            for t in counts
        }

        #flatten evacuation & death times
        all_evacs = [t for times in self.evacuation_times.values() for t in times]
        all_deaths = [t for times in self.death_times.values() for t in times]

        #averages
        avg_evac = sum(all_evacs) / len(all_evacs) if all_evacs else 0.0
        avg_death = sum(all_deaths) / len(all_deaths) if all_deaths else 0.0

        #per type average escape times
        avg_escape = {}
        for t in counts:
            times = self.evacuation_times.get(t, [])
            avg_escape[t] = sum(times) / len(times) if times else 0.0

        #totals
        num_evacuated = sum(len(v) for v in self.evacuation_times.values())
        num_burned = sum(len(v) for v in self.death_times.values())

        #per type evac counts
        evac_counts = {
            t: len(self.evacuation_times.get(t, []))
            for t in counts
        }

        #write new csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            #header
            writer.writerow([
                "n_exits", "population",
                "pct_staff", "pct_adult", "pct_patient", "pct_child",
                "time_steps",
                "avg_evac_time", "avg_death_time",
                "staff_avg_escape_time", "adult_avg_escape_time",
                "patient_avg_escape_time", "child_avg_escape_time",
                "num_evacuated", "num_burned",
                "staff_evacuated", "adult_evacuated",
                "patient_evacuated", "child_evacuated"
            ])
            #data row entry
            writer.writerow([
                n_exits, pop,
                f"{pct['Staff']:.3f}", f"{pct['Adult']:.3f}",
                f"{pct['Patient']:.3f}", f"{pct['Child']:.3f}",
                self.time_steps,
                f"{avg_evac:.2f}", f"{avg_death:.2f}",
                f"{avg_escape['Staff']:.2f}", f"{avg_escape['Adult']:.2f}",
                f"{avg_escape['Patient']:.2f}", f"{avg_escape['Child']:.2f}",
                num_evacuated, num_burned,
                evac_counts['Staff'], evac_counts['Adult'],
                evac_counts['Patient'], evac_counts['Child']
            ])
        print(f"Advanced metrics exported to {csv_path}")


def pygame_file_picker(folder="."):
    import os
    import pygame
    import sys

    pygame.font.init()  #ensure font module initialized
    picker_screen = pygame.display.set_mode((WIDTH, HEIGHT))  #temp picker window
    pygame.display.set_caption("Select a Blueprint File")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    selected_file = None

    while True:
        pygame.event.pump()
        picker_screen.fill((255, 255, 255))
        #list all json files
        for i, file in enumerate(files):
            text = font.render(file, True, (0, 0, 0))
            rect = text.get_rect(topleft=(20, 30 + i * 40))
            picker_screen.blit(text, rect)
            #highlight hovered filename
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(picker_screen, (200, 200, 255), rect, 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, file in enumerate(files):
                    rect = pygame.Rect(20, 30 + i * 40, 560, 30)
                    if rect.collidepoint(event.pos):
                        return os.path.join(folder, files[i])

        pygame.display.flip()
        clock.tick(30)


def run_simulation_for_test(filename, runs=0, max_steps=0, export=True):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Automated Simulation Run {runs+1}")

    walls, exits, actors, fires = load_blueprint(filename)
    env = BlueprintEnvironment(walls, exits, actors, fires, screen=screen)

    start_ticks = pygame.time.get_ticks()
    for actor in actors:
        actor.start_time = start_ticks

    for step in range(max_steps):
        env.update_actors()
        if env.render_on:
            env.render()
        # stop early if all agents are gone
        if len(env.actors) == 0:
            break

    if export:
        env.export_metrics(f"test_metrics_run{runs+1}.csv")
        env.export_advanced_metrics(f"test_advanced_metrics_run{runs+1}.csv")

    pygame.quit()


if __name__ == '__main__':
    run_automated = True   # set to False to run manually
    rum_runs = 10                # Number of test repetitions
    max_steps = 500              # Max steps per simulation

    #set to true to export metrics at end
    metrics = True
    #set true to export advanced_metrics at the end
    advanced_metrics = True
    #prompt for blueprint file
    filename = pygame_file_picker()
    
    if run_automated:
        for i in range(rum_runs):
            run_simulation_for_test(filename, run_id=i, max_steps=max_steps, export=True)
    else:
        #initialize simulation window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Agent Outline")

        #load blueprint data
        walls, exits, actors, fires = load_blueprint(filename)
        env = BlueprintEnvironment(walls, exits, actors, fires)
        running = True

        #set start times for metrics
        start_ticks = pygame.time.get_ticks()
        for actor in actors:
            actor.start_time = start_ticks

        #main loop: update & render
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            env.update_actors()
            env.render()
        pygame.quit()
        #export metrics if enabled
        if metrics:
          env.export_metrics("evacuation_results.csv")
        if advanced_metrics:
          env.export_advanced_metrics("advanced_evacuation_results.csv")