import pygame
import numpy as np
import json
import random
import csv
import sys
from enum import Enum, auto
from pathlib import Path

# define all FSM states
class FSMState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    GUIDING = auto()
    FOLLOWING = auto()
    EXITING = auto()
    DEAD = auto()

# simulation settings
pygame.init()
WIDTH, HEIGHT = 1000, 700
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
clock = pygame.time.Clock()
FPS = 60
FIRE_SPREAD_CHANCE = 0.05   # Probability of fire spreading
FIRE_SPREAD_INTERVAL = 10   # Spread interval in frames

# agent class with FSM behavior and visual representation
class Actor:
    def __init__(self, pos, actor_type, speed=None, constraints=None, guided_speeds=None):
        self.pos = pos
        self.actor_type = actor_type
        self.state = FSMState.IDLE
        self.guiding = None
        self.guiding_queue = []  # queue for dependents to guide
        self.start_time = None
        self.end_time = None
        self.is_dead = False
        self.speed = speed if speed else {"Staff": 1.0, "Adult": 1.0, "Patient": 0.0, "Child": 0.33}.get(actor_type, 1.0)
        self.guided_speeds = guided_speeds if guided_speeds else {
            "Patient": {"adult": 0.75, "staff": 1.0},
            "Child": {"adult": 1.0, "staff": 1.0}
        }.get(actor_type, {})
        self.color = {"Staff": (0, 0, 255), "Adult": (0, 255, 0), "Patient": (255, 255, 0), "Child": (200, 100, 200)}.get(actor_type, (0, 0, 0))
        self.radius = 8 if actor_type == "Child" else 10

    def draw(self, screen):
        # Draw shape, state label, and guidance line
        if self.actor_type == "Patient":
            width = 2 * self.radius
            height = int(2 * self.radius * 2.5)
            rect = pygame.Rect(self.pos[0] - self.radius, self.pos[1] - height // 2, width, height)
            pygame.draw.ellipse(screen, self.color, rect)
        else:
            pygame.draw.circle(screen, self.color, self.pos, self.radius)
        font = pygame.font.SysFont("arial", 12)
        label = font.render(self.state.name, True, BLACK)
        screen.blit(label, (self.pos[0] - 15, self.pos[1] - self.radius - 15))
        if self.state == FSMState.DEAD:
            label = font.render("DEAD", True, RED)
            screen.blit(label, (self.pos[0] - 15, self.pos[1] + self.radius + 5))
        if self.guiding:
            pygame.draw.line(screen, BLACK, self.pos, self.guiding.pos, 2)

    def process_event(self, event):  # FSM transition logic
        if self.state == FSMState.IDLE and event == "move":
            self.state = FSMState.SEARCHING
        elif self.state == FSMState.SEARCHING:
            if event == "guide":
                if self.actor_type in ["Staff", "Adult"]:
                    self.state = FSMState.GUIDING
                else:
                    self.state = FSMState.FOLLOWING
            elif event == "exit":
                self.state = FSMState.EXITING
        elif self.state in [FSMState.GUIDING, FSMState.FOLLOWING]:
            if event == "lost":
                self.state = FSMState.SEARCHING
            elif event == "exit":
                self.state = FSMState.EXITING


class Wall:      # structures for walls and exits
    def __init__(self, start, end):
        self.start = start
        self.end = end

class FireExit:
    def __init__(self, pos, size=(40, 40)):
        self.pos = pos
        self.size = size

# load blueprint data and initialize environment
def load_blueprint(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    walls = [Wall(**wall) for wall in data['walls']]
    exits = [FireExit(**exit) for exit in data['fire_exits']]
    actors = [Actor(**{
        "pos": actor['pos'],
        "actor_type": actor['type'],
        "speed": actor.get('speed'),
        "constraints": actor.get('constraints'),
        "guided_speeds": actor.get('guided_speeds')
    }) for actor in data['actors']]
    fires = set(tuple(f["pos"]) for f in data.get("fires", []))
    for actor in actors:
        actor.start_time = pygame.time.get_ticks()
    return walls, exits, actors, fires

# environment simulator
class SimpleEnvironment:
    def __init__(self, walls, exits, actors, fires, screen):
        self.walls=walls
        self.exits=exits
        self.actors=actors
        self.fires=fires
        self.screen=screen
        self.frame_count=0
        self.grid = self.compute_occupancy_grid()   # for collision detection

    def compute_occupancy_grid(self):  # create occupancy grid from walls
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for wall in self.walls:
            x0, y0 = np.array(wall.start) // CELL_SIZE
            x1, y1 = np.array(wall.end) // CELL_SIZE
            for t in np.linspace(0, 1, int(np.hypot(x1-x0,y1 -y0))*2):
                cx = int(x0 +t *(x1-x0))
                cy = int(y0 +t *(y1-y0))
                if 0 <= cy < GRID_HEIGHT and 0 <= cx < GRID_WIDTH:
                    grid[cy,cx] = 1
        return grid

    def detect_collision(self, pos): # grid-based collision detection
        x, y = (np.array(pos) // CELL_SIZE).astype(int)
        x = np.clip(x, 0, GRID_WIDTH -1)
        y = np.clip(y, 0, GRID_HEIGHT -1)
        return self.grid[y, x] == 1

    def is_visible_with_raycast(self, from_pos, to_pos): # Ray-casting based visibility check
        x0, y0 = from_pos
        x1, y1 = to_pos
        steps = int(np.linalg.norm(np.array(from_pos)-np.array(to_pos))//CELL_SIZE)
        if steps == 0:
            return True
        dx=(x1-x0)/steps
        dy=(y1-y0)/steps
        for i in range(1,steps):
            px=int(x0+i*dx)
            py=int(y0+i*dy)
            if self.detect_collision((px,py)):
                return False
        return True

    def actor_reached_exit(self, actor):
        # check if actor is close to any exit
        return any(np.linalg.norm(np.array(actor.pos)-np.array(exit.pos)) <= 20 for exit in self.exits)

    def move_towards(self, actor, target, speed):
        # Move agent towards target if path is clear
        direction = np.array(target)-np.array(actor.pos)
        if np.linalg.norm(direction) >0:
            step = direction/np.linalg.norm(direction)*(5*speed)
            new_pos = (np.array(actor.pos)+step).astype(int)
            if not self.detect_collision(new_pos) and not self.is_near_fire(new_pos):
                actor.pos = new_pos.tolist()

    def is_near_fire(self, pos):
        # Check proximity to fire
        return any(np.linalg.norm(np.array(pos)-np.array(fire))<20 for fire in self.fires)

    def spread_fire(self):
        # Spread fire to adjacent cells
        new_fires = set()
        for fx, fy in self.fires:
            for dx in [-CELL_SIZE, 0, CELL_SIZE]:
                for dy in [-CELL_SIZE, 0, CELL_SIZE]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = fx+dx, fy+dy
                    if 0 <= nx<WIDTH and 0<=ny<HEIGHT:
                        if (nx,ny) not in self.fires and random.random()<FIRE_SPREAD_CHANCE:
                            new_fires.add((nx,ny))
        self.fires.update(new_fires)

    def find_nearest_exit(self, actor):
        # Get the closest exit to the actor
        return min(self.exits, key=lambda e: np.linalg.norm(np.array(actor.pos)-np.array(e.pos)))

    def update(self):
        # Main update loop for actors and environment
        self.frame_count += 1
        if self.frame_count % FIRE_SPREAD_INTERVAL == 0:
            self.spread_fire()
        for fx, fy in self.fires:
            pygame.draw.rect(self.screen, ORANGE, pygame.Rect(fx,fy,CELL_SIZE,CELL_SIZE))
        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, wall.start, wall.end, 5)
        for actor in self.actors:
            if actor.state != FSMState.DEAD:
                if self.is_near_fire(actor.pos):
                    actor.state = FSMState.DEAD
                    actor.end_time = pygame.time.get_ticks()
                    actor.is_dead = True
                    continue
                if actor.state == FSMState.IDLE:
                    actor.process_event("move")
                elif actor.state == FSMState.SEARCHING:
                    if actor.actor_type in ["Staff", "Adult"]:
                        for target in self.actors:
                            if target == actor or target.state != FSMState.SEARCHING:
                                continue
                            if target.actor_type in ["Patient","Child"] and target.guiding is None:
                                if target not in actor.guiding_queue and target != actor.guiding:
                                    if self.is_visible_with_raycast(actor.pos, target.pos):
                                        actor.guiding_queue.append(target)
                    if actor.actor_type in ["Staff","Adult"] and not actor.guiding and actor.guiding_queue:
                        next_dep = actor.guiding_queue.pop(0)
                        actor.guiding = next_dep
                        next_dep.guiding = actor
                        actor.process_event("guide")
                        next_dep.process_event("guide")
                    self.move_towards(actor, self.find_nearest_exit(actor).pos, actor.speed)
                    if self.actor_reached_exit(actor):
                        actor.process_event("exit")
                elif actor.state == FSMState.GUIDING:
                    if actor.guiding and actor.guiding.state == FSMState.FOLLOWING:
                        self.move_towards(actor, self.find_nearest_exit(actor).pos, actor.speed)
                        speed = actor.guiding.guided_speeds.get(actor.actor_type.lower(),1.0)
                        self.move_towards(actor.guiding, actor.pos, speed)
                        if self.actor_reached_exit(actor.guiding):
                            actor.guiding.process_event("exit")
                            actor.guiding = None
                            if not actor.guiding_queue:
                                actor.process_event("exit")
                    else:
                        actor.process_event("lost")
                elif actor.state == FSMState.FOLLOWING:
                    if actor.guiding:
                        speed = actor.guided_speeds.get(actor.guiding.actor_type.lower(),1.0)
                        self.move_towards(actor, actor.guiding.pos, speed)
                    else:
                        actor.process_event("lost")
            actor.draw(self.screen)

    def export_advanced_metrics(self, csv_path): # save evacuation stats by actor type
        actor_types = ["Staff","Adult","Patient","Child"]
        counts = {t: 0 for t in actor_types}
        evacuated = {t: 0 for t in actor_types}
        deceased = {t: 0 for t in actor_types}
        evac_times = {t: [] for t in actor_types}
        death_times = {t: [] for t in actor_types}
        for actor in self.actors:
            counts[actor.actor_type] += 1
            if actor.state == FSMState.EXITING and actor.end_time:
                evacuated[actor.actor_type] += 1
                evac_times[actor.actor_type].append((actor.end_time-actor.start_time)/1000)
            elif actor.state == FSMState.DEAD and actor.end_time:
                deceased[actor.actor_type] += 1
                death_times[actor.actor_type].append((actor.end_time-actor.start_time)/1000)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Type","Total","Evacuated","Deceased","EvacuationRate","AvgEvacTime","AvgDeathTime"])
            for t in actor_types:
                total=counts[t]
                evac=evacuated[t]
                dead=deceased[t]
                evac_rate=evac / total if total>0 else 0
                avg_evac=sum(evac_times[t])/len(evac_times[t]) if evac_times[t] else 0
                avg_dead=sum(death_times[t])/len(death_times[t]) if death_times[t] else 0
                writer.writerow([t, total, evac, dead, f"{evac_rate:.2f}", f"{avg_evac:.2f}", f"{avg_dead:.2f}"])

# main loop to run simulation until all agents exit or die
if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else "blueprint_simple.json"
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FSM Fire Evacuation Simulation")
    walls, exits, actors, fires = load_blueprint(filename)
    env = SimpleEnvironment(walls, exits, actors, fires, screen)
    running = True
    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.update()
        pygame.display.flip()
        clock.tick(FPS)
        all_done = all(actor.state in [FSMState.EXITING, FSMState.DEAD] for actor in actors)
        if all_done:
            stem = Path(filename).stem
            result_path = f"results_{stem}.csv"
            with open(result_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Actor","Start Time","End Time","Escape Time (s)","Status"])
                for actor in actors:
                    if actor.state == FSMState.EXITING and actor.end_time is None:
                        actor.end_time = pygame.time.get_ticks()
                    if actor.end_time:
                        escape_time = (actor.end_time-actor.start_time)/1000/60
                    else:
                        escape_time = -1
                    status = "Exited" if actor.state == FSMState.EXITING else ("Deceased" if actor.state == FSMState.DEAD else "Trapped")
                    writer.writerow([actor.actor_type,actor.start_time,actor.end_time,escape_time,status])
            env.export_advanced_metrics(f"advanced_metrics_{stem}.csv")
            print("Simulation complete. Results saved.")
            running = False
    pygame.quit()
