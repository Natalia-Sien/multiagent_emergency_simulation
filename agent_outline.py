import pygame
import numpy as np
import json
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialization of pygame
pygame.init()

# General settings
WIDTH, HEIGHT = 1000, 700
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FIRE_SPREAD_CHANCE = 0.02  # 2% chance to spread each update
FIRE_RADIUS = 5

# Screen setup
pygame.display.set_caption("Agent Outline")
clock = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

class Wall:
    def __init__(self, start, end):
        self.start, self.end = start, end

class FireExit:
    def __init__(self, pos, size=(40, 40)):
        self.pos, self.size = pos, size

class Actor:
    def __init__(self, pos, actor_type, speed=None, constraints=None, guided_speeds=None):
        self.pos = pos
        self.actor_type = actor_type
        self.guiding = None

        if speed is None:
            self.speed = {
                "Staff": 1.0,
                "Adult": 1.0,
                "Patient": 0.0,
                "Child": 0.33
            }.get(actor_type, 1.0)
        else:
            self.speed = speed

        self.constraints = constraints if constraints is not None else {
            "Staff": {"knows_exit": True},
            "Adult": {"knows_exit": True, "follows_staff": True},
            "Patient": {"needs_guidance": True},
            "Child": {}
        }.get(actor_type, {})

        self.guided_speeds = guided_speeds if guided_speeds is not None else {
            "Patient": {"adult": 0.75, "staff": 1.0},
            "Child": {"adult": 1.0, "staff": 1.0}
        }.get(actor_type, {})

        self.color = {
            "Staff": (0, 0, 255),
            "Adult": (0, 255, 0),
            "Patient": (255, 255, 0),
            "Child": (200, 100, 200)
        }.get(actor_type, (0, 0, 0))

        self.radius = 8 if actor_type == "Child" else 10

    def draw(self, screen):
        if self.actor_type == "Patient":
            width = 2 * self.radius
            height = int(2 * self.radius * 2.5)
            rect = pygame.Rect(self.pos[0] - self.radius, self.pos[1] - height // 2, width, height)
            pygame.draw.ellipse(screen, self.color, rect)
        else:
            pygame.draw.circle(screen, self.color, self.pos, self.radius)

        if self.guiding:
            pygame.draw.line(screen, BLACK, self.pos, self.guiding.pos, 2)

def load_blueprint(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    walls = [Wall(**wall) for wall in data['walls']]
    exits = [FireExit(**exit) for exit in data['fire_exits']]
    actors = [Actor(
        pos=actor['pos'],
        actor_type=actor['type'],
        speed=actor.get('speed'),
        constraints=actor.get('constraints'),
        guided_speeds=actor.get('guided_speeds')
    ) for actor in data['actors']]
    fires = [tuple(f["pos"]) for f in data.get("fires", [])]
    return walls, exits, actors, fires

class BlueprintEnvironment:
    def __init__(self, walls, exits, actors, fires, screen=None):
        self.walls, self.exits, self.actors = walls, exits, actors
        self.fires = fires
        self.grid = self.compute_occupancy_grid()
        self.render_on = True

        if screen is not None:
            self.screen = screen
        else:
            self.screen = pygame.display.get_surface()
            if self.screen is None:
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.display.set_caption("Hospital Simulation")

    def actor_moved(self, actor):
        if not hasattr(actor, 'previous_pos'):
            return True
        return actor.pos != actor.previous_pos

    def _move_actor(self, actor, action):
        moves = [(0, -5), (0, 5), (-5, 0), (5, 0), (0, 0), (-5, -5), (5, -5), (-5, 5), (5, 5)]
        move = moves[action]
        new_pos = [actor.pos[0] + move[0], actor.pos[1] + move[1]]
        print(f"Attempting move for {actor.actor_type} at {actor.pos} with action {action} -> {move}, candidate position: {new_pos}")
        actor.previous_pos = actor.pos[:]
        if not self.detect_collision(actor, new_pos):
            actor.pos = new_pos
            print(f"Moved {actor.actor_type} to {actor.pos}")
        else:
            print(f"Movement blocked for {actor.actor_type} at candidate {new_pos}")

    def compute_occupancy_grid(self):
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for wall in self.walls:
            x0, y0 = np.array(wall.start) // CELL_SIZE
            x1, y1 = np.array(wall.end) // CELL_SIZE
            for t in np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0))):
                cx, cy = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
                grid[cy, cx] = 1
        return grid

    def reset(self, seed=None, options=None):
        # Slightly randomize actor positions on reset to encourage exploration.
        new_actors = []
        for actor in self.actors:
            new_pos = [actor.pos[0] + np.random.randint(-3, 4),
                       actor.pos[1] + np.random.randint(-3, 4)]
            new_actors.append(Actor(new_pos, actor.actor_type, actor.speed, actor.constraints, actor.guided_speeds))
        self.actors = new_actors

    def raycast_distances(self, actor, num_rays=8, max_range=100):
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        distances = []
        for angle in angles:
            for r in range(0, max_range, 5):
                dx = int(actor.pos[0] + r * np.cos(angle))
                dy = int(actor.pos[1] + r * np.sin(angle))
                if 0 <= dx < WIDTH and 0 <= dy < HEIGHT:
                    grid_x = np.clip(dx // CELL_SIZE, 0, self.grid.shape[1] - 1)
                    grid_y = np.clip(dy // CELL_SIZE, 0, self.grid.shape[0] - 1)
                    if self.grid[grid_y, grid_x] == 1:
                        distances.append(r / max_range)
                        break
            else:
                distances.append(1.0)
        return np.array(distances, dtype=np.float32)

    def update_actors(self):
        self.spread_fire()
        for actor in self.actors[:]:
            if self.actor_in_fire(actor):
                self.actors.remove(actor)
                continue
            if actor.actor_type == "Child":
                self.move_child(actor)
            elif actor.actor_type == "Patient":
                self.move_patient(actor)
            elif actor.actor_type == "Staff":
                self.update_staff(actor)
        for actor in self.actors[:]:
            if self.actor_reached_exit(actor):
                if actor.actor_type != "Patient" or actor.guiding is None:
                    self.actors.remove(actor)

    def actor_in_fire(self, actor):
        for fire in self.fires:
            if np.linalg.norm(np.array(actor.pos) - np.array(fire)) < FIRE_RADIUS:
                return True
        return False

    def spread_fire(self):
        new_fires = set()
        for fx, fy in self.fires:
            neighbors = [
                (fx + CELL_SIZE, fy), (fx - CELL_SIZE, fy),
                (fx, fy + CELL_SIZE), (fx, fy - CELL_SIZE)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    if (nx, ny) not in self.fires and random.random() < FIRE_SPREAD_CHANCE:
                        new_fires.add((nx, ny))
        self.fires.extend(list(new_fires))

    def move_child(self, child):
        nearest_adult = self.find_nearest_actor(child, ["Adult", "Staff"], max_distance=100)
        if nearest_adult:
            self.move_towards(child, nearest_adult.pos, child.guided_speeds.get("adult", 1.0))
        else:
            self.random_move(child)

    def move_patient(self, patient):
        guide = self.find_nearest_actor(patient, ["Staff", "Adult"], max_distance=100)
        if guide and guide.guiding is None:
            guide_type = guide.actor_type.lower()
            speed = patient.guided_speeds.get(guide_type, 1.0)
            self.move_towards(patient, guide.pos, speed)

    def update_staff(self, staff):
        if staff.guiding and staff.guiding not in self.actors:
            staff.guiding = None
        if staff.guiding:
            exit_pos = self.find_nearest_exit(staff).pos
            self.move_towards(staff, exit_pos, staff.speed)
            self.move_towards(staff.guiding, staff.pos, staff.guiding.guided_speeds.get("staff", 1.0))
            if self.actor_reached_exit(staff.guiding):
                if staff.guiding in self.actors:
                    self.actors.remove(staff.guiding)
                staff.guiding = None
        else:
            patients = self.get_unescorted_actors_in_range(staff, "Patient")
            if not patients:
                children = self.get_unescorted_actors_in_range(staff, "Child")
                if children:
                    self.move_towards(staff, children[0].pos, staff.speed)
                    if np.linalg.norm(np.array(staff.pos) - np.array(children[0].pos)) < 10:
                        staff.guiding = children[0]
                elif self.should_leave(staff):
                    exit_pos = self.find_nearest_exit(staff).pos
                    self.move_towards(staff, exit_pos, staff.speed)
            else:
                self.move_towards(staff, patients[0].pos, staff.speed)
                if np.linalg.norm(np.array(staff.pos) - np.array(patients[0].pos)) < 10:
                    staff.guiding = patients[0]

    def get_unescorted_actors_in_range(self, guide_actor, target_type):
        visible = self.actor_vision(guide_actor, vision_range=10)
        found = []
        for a in self.actors:
            if a.actor_type == target_type and all(g.guiding != a for g in self.actors):
                cell = tuple(np.array(a.pos) // CELL_SIZE)
                if cell in visible:
                    found.append(a)
        return found

    def should_leave(self, guide_actor):
        visible_cells = self.actor_vision(guide_actor, vision_range=10)
        for actor in self.actors:
            if actor == guide_actor or self.actor_reached_exit(actor):
                continue
            if actor.actor_type in ["Patient", "Child"]:
                if all(other.guiding != actor for other in self.actors if other != actor):
                    actor_cell = tuple(np.array(actor.pos) // CELL_SIZE)
                    if actor_cell in visible_cells:
                        return False
        return True

    def find_nearest_actor(self, actor, types, max_distance):
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
        movements = np.array([(0, 5), (0, -5), (5, 0), (-5, 0), (0, 0)])
        move = movements[np.random.choice(len(movements))]
        new_pos = [actor.pos[0] + move[0], actor.pos[1] + move[1]]
        if not self.detect_collision(actor, new_pos):
            actor.pos = new_pos

    def move_towards(self, actor, target, speed):
        direction = np.array(target) - np.array(actor.pos)
        if np.linalg.norm(direction) > 0:
            step = direction / np.linalg.norm(direction) * (5 * speed)
            new_pos = np.array(actor.pos) + step
            if not self.detect_collision(actor, new_pos):
                actor.pos = new_pos.astype(int).tolist()

    def actor_vision(self, actor, vision_range=5):
        x, y = np.array(actor.pos) // CELL_SIZE
        return set((x + dx, y + dy) for dx in range(-vision_range, vision_range + 1) for dy in range(-vision_range, vision_range + 1)
                   if 0 <= x + dx < GRID_WIDTH and 0 <= y + dy < GRID_HEIGHT)

    def detect_collision(self, actor, new_pos):
        x, y = (np.array(new_pos) // CELL_SIZE).astype(int)
        x = np.clip(x, 0, self.grid.shape[1] - 1)
        y = np.clip(y, 0, self.grid.shape[0] - 1)
        return self.grid[y, x] == 1

    def render(self):
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

        # Always set a screen if we don't have one!
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.screen.fill(WHITE)

        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, wall.start, wall.end, 5)
        for exit in self.exits:
            rect = pygame.Rect(exit.pos[0] - exit.size[0] // 2, exit.pos[1] - exit.size[1] // 2, *exit.size)
            pygame.draw.rect(self.screen, RED, rect)
        for fire in self.fires:
            pygame.draw.circle(self.screen, ORANGE, fire, FIRE_RADIUS)
        for actor in self.actors:
            actor.draw(self.screen)

        pygame.display.flip()
        pygame.event.pump()

    def find_nearest_exit(self, actor):
        return min(self.exits, key=lambda e: np.linalg.norm(np.array(actor.pos) - np.array(e.pos)))

    def actor_reached_exit(self, actor):
        return any(np.linalg.norm(np.array(actor.pos) - np.array(exit.pos)) < 20 for exit in self.exits)


def pygame_file_picker(folder="."):
    import os
    import pygame
    import sys

    pygame.font.init()
    picker_screen = pygame.display.set_mode((WIDTH, HEIGHT))  # TEMP window
    pygame.display.set_caption("Select a Blueprint File")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    selected_file = None

    while True:
        pygame.event.pump()
        picker_screen.fill((255, 255, 255))

        for i, file in enumerate(files):
            text = font.render(file, True, (0, 0, 0))
            rect = text.get_rect(topleft=(20, 30 + i * 40))
            picker_screen.blit(text, rect)
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
                        selected_file = os.path.join(folder, files[i])
                        return selected_file

        pygame.display.flip()
        clock.tick(30)



def open_file_dialog():
    return pygame_file_picker()

if __name__ == '__main__':
    filename = open_file_dialog()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Outline")

    walls, exits, actors, fires = load_blueprint(filename)
    env = BlueprintEnvironment(walls, exits, actors, fires)
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.render()
    pygame.quit()
