import pygame
import json
import math
import sys

#initializing pygame and set up the display
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hospital Blueprint Editor")
clock = pygame.time.Clock()
FPS = 60

#defining colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (200, 100, 200)
ORANGE = (255, 165, 0)
FIRE_RADIUS = 5  #this is where we can alter the fires radius maybe, would most likely make it easier for the algos
CELL_SIZE = 5
font = pygame.font.SysFont("arial", 16)

#global lists to store blueprint objects
walls = []  #holds Wall objects
fire_exits = []  #holds FireExit objects
actors = []  #holds actor objects (staff, adult, patient, child)

#global history stack for undo functionality
history = []  #each entry is a tuple, (identifier, object_reference)

fires = []  #holds fire tile positions (for hazard simulation)

#global backup for "clear" operations (to allow undoing a clear, trust me it saves so much time)
last_cleared_state = None

#list of different abilities we can use, and their keys (1 to 7, and now D for Delete)
MODES = ["Wall", "Fire Exit", "Staff", "Adult", "Patient", "Child", "Fire", "Delete"]

MODE_KEYS = {
    pygame.K_1: "Wall",
    pygame.K_2: "Fire Exit",
    pygame.K_3: "Staff",
    pygame.K_4: "Adult",
    pygame.K_5: "Patient",
    pygame.K_6: "Child",
    pygame.K_7: "Fire",  #press 7 to place a fire tile (new addition hehe)
    pygame.K_d: "Delete"  #press D to delete an object under the mouse
}
selected_mode = "Wall" #i felt it was natural to just draw walls straight away maybe

#for drawing walls this allows us to hold the first point until released
is_drawing_wall = False
wall_start = None



def pygame_save_dialog(default_filename="blueprint.json", folder="."):
    import os
    import pygame
    pygame.font.init()
    font = pygame.font.SysFont("arial", 24)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Save Blueprint As")
    clock = pygame.time.Clock()

    filename = default_filename
    input_active = True

    while True:
        pygame.event.pump()
        screen.fill((255, 255, 255))
        prompt = font.render("Enter filename (press Enter to save):", True, (0, 0, 0))
        text = font.render(filename, True, (0, 0, 255))
        screen.blit(prompt, (20, 40))
        screen.blit(text, (20, 80))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return os.path.join(folder, filename)
                elif event.key == pygame.K_BACKSPACE:
                    filename = filename[:-1]
                elif input_active:
                    if event.unicode.isprintable():
                        filename += event.unicode

        pygame.display.flip()
        clock.tick(30)


"""
here are all of the blueprint object classes, this will make it easier for us to load in future blueprints into our
algorithms. here the walls and everything can be set, fire exits class has its variables and constructors, and most
importantly, the actors class has all of the different actors and their variables/constraints.
"""
class Wall:
    #walls are represented as a line between two points and is unpassable, you need to click, drag and let go

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def draw(self, screen):
        pygame.draw.line(screen, BLACK, self.start, self.end, 5)

    def to_dict(self):
        return {"start": list(self.start), "end": list(self.end)}


class FireExit:
    #fire exits are drawn as a rectangle centered around the clicked point

    def __init__(self, pos, size=(40, 40)):
        self.pos = pos
        self.size = size
        self.rect = pygame.Rect(pos[0] - size[0] // 2,
                                pos[1] - size[1] // 2, size[0], size[1])

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

    def to_dict(self):
        return {"pos": list(self.pos), "size": list(self.size)}


class Actor:
    """
       setting up each subset of actor to have their own charactaristics,
       note how there is staff, adult, patient and child,
       each have their own guided speed/ movement speed.

       staff and adults have a speed of 1 (that means they move at full speed)
       patients have a base speed of 0. note here that if they are guided by an adult, their speed goes to 0.75,
       if they are guided by a staff member, their speed goes to 1.
       children have a base speed of 0.33, unless they are guided by an adult or staff member then its 1.
    """

    def __init__(self, pos, actor_type, speed=None, constraints=None, guided_speeds=None):
        self.pos = pos
        self.actor_type = actor_type

        #setting base speed, if not provided, use the defaults by actor type
        if speed is None:
            if actor_type in ["Staff", "Adult"]:
                self.speed = 1.0
            elif actor_type == "Patient":
                self.speed = 0.0 #note how the patient has 0 speed, this is due to them needing to be guided
            elif actor_type == "Child":
                self.speed = 0.33
            else:
                self.speed = 1.0
        else:
            self.speed = speed

        #set constraints, if not provided, use defaults.(safety)
        if constraints is None:
            if actor_type == "Staff":
                self.constraints = {"knows_exit": True}
            elif actor_type == "Adult":
                self.constraints = {"knows_exit": True, "follows_staff": True}
            elif actor_type == "Patient":
                self.constraints = {"needs_guidance": True}
            elif actor_type == "Child":
                self.constraints = {}
            else:
                self.constraints = {}
        else:
            self.constraints = constraints

        #setting guided speeds, only patients and children have guided speeds as they need assistance
        if guided_speeds is None:
            if actor_type == "Patient":
                self.guided_speeds = {"adult": 0.75, "staff": 1.0}
            elif actor_type == "Child":
                self.guided_speeds = {"adult": 1.0, "staff": 1.0}
            else:
                self.guided_speeds = {}
        else:
            self.guided_speeds = guided_speeds

        #seting drawing parameters
        #base radius is for the horizontal dimension, note how child is slightly smaller
        if actor_type == "Staff":
            self.color = BLUE
            self.radius = 10
        elif actor_type == "Adult":
            self.color = GREEN
            self.radius = 10
        elif actor_type == "Patient":
            self.color = YELLOW
            self.radius = 10
        elif actor_type == "Child":
            self.color = PURPLE
            self.radius = 8

    def draw(self, screen):
        #for patients we draw an elongated ellipse (2.5 the radius)
        if self.actor_type == "Patient":
            width = 2 * self.radius
            height = int(2 * self.radius * 2.5)
            rect = pygame.Rect(self.pos[0] - self.radius,
                               self.pos[1] - height // 2, width, height)
            pygame.draw.ellipse(screen, self.color, rect)
        else:
            pygame.draw.circle(screen, self.color, self.pos, self.radius)

    def to_dict(self):
        return {"pos": list(self.pos),
                "type": self.actor_type,
                "speed": self.speed,
                "constraints": self.constraints,
                "guided_speeds": self.guided_speeds}


#start of the helper functions
def check_actor_collision(new_actor, actors_list):
    """
    checking for collision between actors (using circle approximation)
    this is due to the nature of our future algorithms, i didnt want anything
    breaking due to the actors (hopefully not) glitching through each other
    """
    for actor in actors_list:
        dist = math.hypot(actor.pos[0] - new_actor.pos[0],
                          actor.pos[1] - new_actor.pos[1])
        if dist < (actor.radius + new_actor.radius):
            return True
    return False


def export_blueprint(filename=None):
    """
    export the blueprint to a JSON file including speeds, constraints, and guided speeds
    this is so we can load them in later without having to hardcode the variables in case we change
    them in the blueprints
    """
    blueprint = {
        "walls": [wall.to_dict() for wall in walls],
        "fire_exits": [fe.to_dict() for fe in fire_exits],
        "actors": [actor.to_dict() for actor in actors],
        "fires": [{"pos": list(f)} for f in fires]  #new fire export
    }
    if filename is None:
        filename = pygame_save_dialog()
    if not filename:
        return
    with open(filename, "w") as f:
        json.dump(blueprint, f, indent=4)
    print(f"Blueprint exported to {filename}")


def load_blueprint(filename):
    #load a blueprint from the given json file and repopulate the global objects
    try:
        with open(filename, "r") as f:
            blueprint = json.load(f)
        walls.clear()
        fire_exits.clear()
        actors.clear()
        fires.clear()
        history.clear()

        for wall_item in blueprint.get("walls", []):
            walls.append(Wall(wall_item["start"], wall_item["end"]))
        for fe_item in blueprint.get("fire_exits", []):
            fire_exits.append(FireExit(fe_item["pos"],
                                       tuple(fe_item["size"])))
        for actor_item in blueprint.get("actors", []):
            speed = actor_item.get("speed")
            constraints = actor_item.get("constraints")
            guided_speeds = actor_item.get("guided_speeds")
            actors.append(Actor(actor_item["pos"],
                                actor_item["type"],
                                speed=speed,
                                constraints=constraints,
                                guided_speeds=guided_speeds))
        print(f"Blueprint loaded from {filename}")
        for fire_item in blueprint.get("fires", []):
            fires.append(tuple(fire_item["pos"]))

    except Exception as e:
        print("Failed to load blueprint:", e)



def pygame_file_picker(folder="."):
    import os
    import pygame
    pygame.font.init()
    font = pygame.font.SysFont("arial", 24)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Select a Blueprint File")
    clock = pygame.time.Clock()

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    selected_file = None

    while True:
        screen.fill((255, 255, 255))
        for i, file in enumerate(files):
            text = font.render(file, True, (0, 0, 0))
            rect = text.get_rect(topleft=(20, 30 + i * 40))
            screen.blit(text, rect)
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, (200, 200, 255), rect, 2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, file in enumerate(files):
                    rect = pygame.Rect(20, 30 + i * 40, 560, 30)
                    if rect.collidepoint(event.pos):
                        selected_file = os.path.join(folder, files[i])
                        return selected_file
        pygame.display.flip()
        clock.tick(30)

# replacing tkinter file picker with pygame file picker
def open_file_dialog():
    #this should fix it on mac? ithink its a thread issue
            return pygame_file_picker()



def point_to_segment_distance(p, a, b):

    #tihs is a helper to check if p is close to line segment ab (for wall deletion)

    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx*abx + aby*aby + 1e-6
    t = ((px - ax)*abx + (py - ay)*aby) / ab_len_sq
    t = max(0, min(1, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def delete_object_at_position(pos):
    """
    checks if the mouse position is over any object (wall, fire exit, or actor) and deletes it,
    while adding a 'Delete' entry to the history so we can undo it.
    """
    global walls, fire_exits, actors, history
    deletion_threshold = 10  #pixels tolerance for line-based removal

    #check actors first
    for actor in actors:
        #if click is within radius + threshold
        if math.hypot(actor.pos[0] - pos[0], actor.pos[1] - pos[1]) <= (actor.radius + deletion_threshold):
            actors.remove(actor)
            #adds a new 'DeletedActor' entry so we can undo
            history.append(("DeletedActor", actor))
            print(f"Deleted actor: {actor.actor_type} at {actor.pos}")
            return

    #check fire exits (using rectangle collision from pygame)
    for fe in fire_exits:
        if fe.rect.collidepoint(pos):
            fire_exits.remove(fe)
            history.append(("DeletedFireExit", fe))
            print(f"Deleted fire exit at {fe.pos}")
            return

    #check walls (if the point is near the line)
    for wall in walls:
        if point_to_segment_distance(pos, wall.start, wall.end) <= deletion_threshold:
            walls.remove(wall)
            history.append(("DeletedWall", wall))
            print(f"Deleted wall from {wall.start} to {wall.end}")
            return

    for fire in fires:
        if math.hypot(fire[0] - pos[0], fire[1] - pos[1]) <= FIRE_RADIUS:
            fires.remove(fire)
            history.append(("DeletedFire", fire))
            print(f"Deleted fire tile at {fire}")
            return


    print("No object found at position", pos)


def draw_ui():
    #draw on-screen instructions. could add more if we need to later but i think its good enough for now?
    instructions = [
        "Keys: 1-Wall, 2-Fire Exit, 3-Staff, 4-Adult, 5-Patient, 6-Child and 7-Fire",
        "Click to place objects (click-drag for walls).",
        "Press 'E' to export, 'L' to load, 'C' to clear, Ctrl+Z to undo, D-Delete, Q to quit.",
        f"Current Mode: {selected_mode}"
    ]
    for i, text in enumerate(instructions):
        rendered = font.render(text, True, BLACK)
        screen.blit(rendered, (10, 10 + i * 18))


def undo_last_action():
    #undo the last placed or deleted object or restore a cleared blueprint
    global last_cleared_state
    if history:
        obj_type, obj_instance = history.pop()
        if obj_type == "Wall" and obj_instance in walls:
            walls.remove(obj_instance)
            print("Undid last wall placement.")
        elif obj_type == "FireExit" and obj_instance in fire_exits:
            fire_exits.remove(obj_instance)
            print("Undid last fire exit placement.")
        elif obj_type == "Actor" and obj_instance in actors:
            actors.remove(obj_instance)
            print(f"Undid last actor: {obj_instance.actor_type}")
        elif obj_type == "DeletedActor":
            #this was a deleted actor so re-add to actors
            actors.append(obj_instance)
            print(f"Restored deleted actor: {obj_instance.actor_type}")
        elif obj_type == "DeletedFireExit":
            #this was a deleted fire exit re-add it
            fire_exits.append(obj_instance)
            print(f"Restored deleted fire exit at {obj_instance.pos}")
        elif obj_type == "DeletedWall":
            #this was a deleted wall, re-add it
            walls.append(obj_instance)
            print(f"Restored deleted wall from {obj_instance.start} to {obj_instance.end}")
            #deleting fire,
        elif obj_type == "Fire" and obj_instance in fires:
            fires.remove(obj_instance)
            print(f"Undid fire tile at {obj_instance}")
            #undo delete fire
        elif obj_type == "DeletedFire":
            fires.append(obj_instance)
            print(f"Restored deleted fire tile at {obj_instance}")
        else:
            print("Nothing to undo!")
    elif last_cleared_state is not None:
        walls[:] = last_cleared_state["walls"]
        fire_exits[:] = last_cleared_state["fire_exits"]
        actors[:] = last_cleared_state["actors"]
        fires[:] = last_cleared_state["fires"]
        history[:] = last_cleared_state["history"]
        last_cleared_state = None
        print("Restored blueprint from last clear.")
    else:
        print("Nothing to undo!")


"""
this section has the main loop of the program, here is where all the drawing happens,
all of our other helper functions are called in too.
"""

running = True
while running:
    clock.tick(FPS)
    screen.fill(WHITE)

    #draw blueprint objects
    for wall in walls:
        wall.draw(screen)
    for fe in fire_exits:
        fe.draw(screen)
    for actor in actors:
        actor.draw(screen)
    for fire in fires:
        pygame.draw.circle(screen, ORANGE, fire, FIRE_RADIUS) #fire tile

    #show preview line for wall drawing so we can place it properly
    if is_drawing_wall and wall_start:
        current_pos = pygame.mouse.get_pos()
        pygame.draw.line(screen, BLACK, wall_start, current_pos, 5)

    #draw UI overlay
    draw_ui()

    #process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            #quit on Q key press
            if event.key == pygame.K_q:
                running = False
            #ctrl+Z to undo last done action
            elif event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                undo_last_action()
            elif event.key in MODE_KEYS:
                selected_mode = MODE_KEYS[event.key]
            elif event.key == pygame.K_e:
                export_blueprint()
            elif event.key == pygame.K_l:
                file_path = open_file_dialog()
                if file_path:
                    load_blueprint(file_path)
            elif event.key == pygame.K_c:
                last_cleared_state = {
                    "walls": walls.copy(),
                    "fire_exits": fire_exits.copy(),
                    "actors": actors.copy(),
                    "fires": fires.copy(),
                    "history": history.copy()
                }
                walls.clear()
                fire_exits.clear()
                actors.clear()
                fires.clear()
                history.clear()
                print("Cleared blueprint. (Press Ctrl+Z to undo clear)")

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if selected_mode == "Delete":
                delete_object_at_position(pos)
            elif selected_mode == "Wall":
                is_drawing_wall = True
                wall_start = pos
            elif selected_mode == "Fire Exit":
                fe = FireExit(pos)
                fire_exits.append(fe)
                history.append(("FireExit", fe))
            elif selected_mode == "Fire":
                fire_pos = (pos[0] // CELL_SIZE * CELL_SIZE, pos[1] // CELL_SIZE * CELL_SIZE)  # align to grid
                fires.append(fire_pos)
                history.append(("Fire", fire_pos))
                print(f"Placed fire tile at {fire_pos}")
            else:
                new_actor = Actor(pos, selected_mode)
                if check_actor_collision(new_actor, actors):
                    print(f"Collision detected! Not placing {selected_mode} at {pos}.")
                else:
                    actors.append(new_actor)
                    history.append(("Actor", new_actor))

        if event.type == pygame.MOUSEBUTTONUP:
            if selected_mode == "Wall" and is_drawing_wall:
                wall_end = pygame.mouse.get_pos()
                if math.hypot(wall_end[0] - wall_start[0],
                              wall_end[1] - wall_start[1]) > 5:
                    new_wall = Wall(wall_start, wall_end)
                    walls.append(new_wall)
                    history.append(("Wall", new_wall))
                is_drawing_wall = False
                wall_start = None

    pygame.display.flip()

pygame.quit()
sys.exit()
