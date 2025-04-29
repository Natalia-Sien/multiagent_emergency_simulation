import pygame
import numpy as np
import json
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import csv

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

    def actor_moved(self, actor):
        #check if actor position changed
        if not hasattr(actor, 'previous_pos'):
            return True
        return actor.pos != actor.previous_pos

    def _move_actor(self, actor, action):
        moves = [(0, -5), (0, 5), (-5, 0), (5, 0), (0, 0), (-5, -5), (5, -5), (-5, 5), (5, 5)]  #action to delta
        move = moves[action]  #select movement
        new_pos = [actor.pos[0] + move[0], actor.pos[1] + move[1]]  #compute new position
        actor.previous_pos = actor.pos[:]  #update previous_pos
        if not self.detect_collision(actor, new_pos):
            actor.pos = new_pos  #apply movement

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

    def raycast_distances(self, actor, num_rays=8, max_range=100):
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)  #angles around actor (endpoint is false so they dont overlap)
        distances = []
        for angle in angles:
            for r in range(0, max_range, 5):
                dx = int(actor.pos[0] + r * np.cos(angle))
                dy = int(actor.pos[1] + r * np.sin(angle))
                if 0 <= dx < WIDTH and 0 <= dy < HEIGHT:
                    grid_x = np.clip(dx // CELL_SIZE, 0, self.grid.shape[1] - 1)
                    grid_y = np.clip(dy // CELL_SIZE, 0, self.grid.shape[0] - 1)
                    if self.grid[grid_y, grid_x] == 1:  #hit wall
                        distances.append(r / max_range)
                        break
            else:
                distances.append(1.0)  #no wall within range
        return np.array(distances, dtype=np.float32)  #return normalized distances

    def update_actors(self):
        self.spread_fire()  #spread fire before movement
        for actor in self.actors[:]:  #iterate copy since we may remove
            if self.actor_in_fire(actor):  #if actor in fire
                actor.end_time = pygame.time.get_ticks()
                self.death_times[actor.actor_type].append(
                    (actor.end_time - actor.start_time) / 1000
                )  #record death time
                self.actors.remove(actor)  #remove dead actor
                continue

            #dispatch movement by type
            if actor.actor_type == "Child":
                self.move_child(actor)
            elif actor.actor_type == "Patient":
                self.move_patient(actor)
            elif actor.actor_type == "Staff":
                self.update_staff(actor)
            elif actor.actor_type == "Adult":
                self.update_adult(actor)
        #handle exiting agents
        for actor in self.actors[:]:
            if self.actor_reached_exit(actor):
                actor.end_time = pygame.time.get_ticks()
                self.evacuation_times[actor.actor_type].append(
                    (actor.end_time - actor.start_time) / 1000
                )  #record evacuation time
                self.actors.remove(actor)  #remove evacuated
        self.time_steps += 1

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

    def move_child(self, child):
        nearest_adult = self.find_nearest_actor(child, ["Adult", "Staff"], max_distance=100)
        if nearest_adult:
            self.move_towards(child, nearest_adult.pos, child.guided_speeds.get("adult", 1.0))
        else:
            self.random_move(child)  #move randomly if no guide

    def move_patient(self, patient):
        #find nearest staff or adult within 100 units
        guide = self.find_nearest_actor(patient, ["Staff", "Adult"], max_distance=100)
        #if found and not already guiding someone
        if guide and guide.guiding is None:
            #choose speed based on guide type
            guide_type = guide.actor_type.lower()
            speed = patient.guided_speeds.get(guide_type, 1.0)
            #move patient toward guide
            self.move_towards(patient, guide.pos, speed)
    def update_adult(self, adult):
        #if adult was guiding someone who no longer exists, clear both links
        if adult.guiding and adult.guiding not in self.actors:
            adult.guiding.guiding = None
            adult.guiding = None

        #if currently guiding someone, escort them both toward the exit
        if adult.guiding:
            # find nearest exit
            exit_pos = self.find_nearest_exit(adult).pos
            #move the adult toward exit
            self.move_towards(adult, exit_pos, adult.speed)
            #then move the guided actor toward the adult
            guidee = adult.guiding
            #use guided speed if defined, else default
            speed = guidee.guided_speeds.get(adult.actor_type.lower(), guidee.speed)
            self.move_towards(guidee, adult.pos, speed)

            #once the guidee reaches exit, remove and clear links
            if self.actor_reached_exit(guidee):
                if guidee in self.actors:
                    self.actors.remove(guidee)
                guidee.guiding = None
                adult.guiding = None

        else:
            #try to find unescorted patients in vision
            patients = self.get_unescorted_actors_in_range(adult, "Patient")
            if patients:
                target = patients[0]
                #move toward that patient
                self.move_towards(adult, target.pos, adult.speed)
                #establish guidance link
                if np.linalg.norm(np.array(adult.pos) - np.array(target.pos)) < 10:
                    adult.guiding = target
                    target.guiding = adult

            else:
                #try to find unescorted Children
                children = self.get_unescorted_actors_in_range(adult, "Child")
                if children:
                    target = children[0]
                    # move toward that child
                    self.move_towards(adult, target.pos, adult.speed)
                    #if close establish guidance link
                    if np.linalg.norm(np.array(adult.pos) - np.array(target.pos)) < 10:
                        adult.guiding = target
                        target.guiding = adult

                else:
                    #nobody to guide, head for exit yourself
                    if self.should_leave(adult):
                        exit_pos = self.find_nearest_exit(adult).pos
                        self.move_towards(adult, exit_pos, adult.speed)

    def update_staff(self, staff):
        #if staff was guiding someone who no longer exists, clear both links
        if staff.guiding and staff.guiding not in self.actors:
            #drop link on both ends
            staff.guiding.guiding = None
            staff.guiding = None

        if staff.guiding:
            #move staff toward nearest exit
            exit_pos = self.find_nearest_exit(staff).pos
            self.move_towards(staff, exit_pos, staff.speed)
            #then move guided actor toward the staff
            guidee = staff.guiding
            speed = guidee.guided_speeds.get("staff", 1.0)
            self.move_towards(guidee, staff.pos, speed)

            #once guidee reaches exit, remove and clear links
            if self.actor_reached_exit(guidee):
                if guidee in self.actors:
                    self.actors.remove(guidee)
                guidee.guiding = None
                staff.guiding = None

        else:
            #no one guided, look for unescorted patients
            patients = self.get_unescorted_actors_in_range(staff, "Patient")
            if patients:
                target = patients[0]
                #move to patient
                self.move_towards(staff, target.pos, staff.speed)
                #if close enough, establish guidance link
                if np.linalg.norm(np.array(staff.pos) - np.array(target.pos)) < 10:
                    staff.guiding = target
                    target.guiding = staff  #establish two-way link (fixes only one of them leaving)

            else:
                #no patients, look for children
                children = self.get_unescorted_actors_in_range(staff, "Child")
                if children:
                    target = children[0]
                    self.move_towards(staff, target.pos, staff.speed)
                    if np.linalg.norm(np.array(staff.pos) - np.array(target.pos)) < 10:
                        staff.guiding = target
                        target.guiding = staff  #establish two-way link
                #if nobody left to guide, head to exit
                elif self.should_leave(staff):
                    exit_pos = self.find_nearest_exit(staff).pos  #exit position
                    self.move_towards(staff, exit_pos, staff.speed)

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

    def random_move(self, actor):
        #move actor randomly in cardinal directions or stay
        movements = np.array([(0, 5), (0, -5), (5, 0), (-5, 0), (0, 0)])
        move = movements[np.random.choice(len(movements))]
        new_pos = [actor.pos[0] + move[0], actor.pos[1] + move[1]]
        #apply if no collision
        if not self.detect_collision(actor, new_pos):
            actor.pos = new_pos

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

if __name__ == '__main__':
    #set to true to export metrics at end
    metrics = False
    #set true to export advanced_metrics at the end
    advanced_metrics = True
    #prompt for blueprint file
    filename = pygame_file_picker()

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

