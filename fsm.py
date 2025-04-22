import pygame
import numpy as np
import json
import sys
import random

# initialize
WIDTH, HEIGHT = 1000, 700
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 60
FIRE_RADIUS = 5
FIRE_SPREAD_CHANCE = 0.02
SEARCH_RANGE = 150  # staff/Adult search radius (for guide)

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
ORANGE = (255,165,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
PURPLE = (200,100,200)
GRAY = (200,200,200)
CYAN = (0,255,255)
MAGENTA = (255,0,255)

pygame.init()
font = pygame.font.SysFont("arial", 14)

type_mapping = {"Staff": BLUE, "Adult": GREEN, "Patient": YELLOW, "Child": PURPLE}

class Wall:
    def __init__(self,start,end):
        self.start =tuple(start)
        self.end =tuple(end)

class FireExit:
    def __init__(self,pos,size):
        self.pos=tuple(pos)
        self.size=tuple(size)
        self.rect=pygame.Rect(
            self.pos[0] -self.size[0]//2,
            self.pos[1] -self.size[1]//2,
            *self.size
        )
class Actor:
    def __init__(self,pos,actor_type,speed,constraints,guided_speeds):
        self.pos = [float(pos[0]),float(pos[1])]
        self.actor_type=actor_type
        self.speed=speed
        self.constraints=constraints
        self.guided_speeds=guided_speeds
        self.state="idle"
        self._state_change_time=0
        self.color=type_mapping.get(actor_type,BLACK)
        self.radius=8 if actor_type == "Child" else 10
        self.leader=None
        #FSM transitions
        self.transitions = {}
        #common transitions
        self.transitions[("idle","Move")] ="searching"
        self.transitions[("searching","Move")] ="searching"
        self.transitions[("searching","Exit")] ="exiting"
        self.transitions[("exiting","Move")] ="exiting"
        if actor_type in ["Staff", "Adult"]:
            self.transitions[("idle","Guide")] ="guiding"
            self.transitions[("searching","Guide")] ="guiding"
            self.transitions[("guiding","Exit")] ="exiting"
            self.transitions[("guiding","Lost")] ="searching"
        else:
            self.transitions[("idle","Guide")] ="following"
            self.transitions[("searching","Guide")] ="following"
            self.transitions[("following","Exit")] ="exiting"
            self.transitions[("following","Lost")] ="searching"
    def process_event(self, event):
        key=(self.state, event)
        if key in self.transitions:
            prev=self.state
            self.state=self.transitions[key]
            self._state_change_time=pygame.time.get_ticks()
            print(f"{self.actor_type} state: {prev} -> {self.state}")
    def draw(self, screen):
        x, y = int(self.pos[0]), int(self.pos[1])
        now = pygame.time.get_ticks()
        blink = (now-self._state_change_time)<200
        border_colors = {
            "idle": GRAY,
            "searching":ORANGE,
            "following":RED,
            "guiding":CYAN,
            "exiting":PURPLE
        }
        bc = border_colors.get(self.state,BLACK)
        bw = 4 if blink else 2
        pygame.draw.circle(screen, bc,(x, y),self.radius+2,bw)
        pygame.draw.circle(screen, self.color,(x, y),self.radius)
        label = font.render(self.state,True,BLACK)
        screen.blit(label, (x-label.get_width()//2, y-self.radius-15))
class FSMEnvironment:
    def __init__(self, blueprint_file):
        print(f"Loading blueprint from: {blueprint_file}")
        self.load(blueprint_file)
        types = ['Staff','Adult','Patient','Child']
        counts = {t: sum(1 for a in self.actors if a.actor_type==t) for t in types}
        print("Actor counts after load:",counts)
        self.grid=self.build_grid()
        self.screen=pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("FSM Evacuation Simulation")
        self.clock=pygame.time.Clock()

    def load(self, fn):
        with open(fn) as f:
            data = json.load(f)
        self.walls =[Wall(w["start"], w["end"]) for w in data["walls"]]
        self.exits =[FireExit(e["pos"], e["size"]) for e in data["fire_exits"]]
        self.actors =[]
        for a in data["actors"]:
            speed = a.get("speed", {"Staff":1.0,"Adult":1.0,"Patient":0.1,"Child":0.33}[a["type"]])
            constraints = a.get("constraints", {})
            guided = a.get("guided_speeds", {})
            self.actors.append(Actor(a["pos"], a["type"],speed,constraints,guided))
        self.fires = [tuple(f["pos"]) for f in data.get("fires",[])]

    def build_grid(self):
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH),dtype=int)
        for w in self.walls:
            p0 = np.array(w.start)//CELL_SIZE
            p1 = np.array(w.end)//CELL_SIZE
            steps = max(int(np.linalg.norm(p1-p0))*2,1)
            for t in np.linspace(0,1,steps):
                cx, cy = (p0 + (p1 - p0) * t).astype(int)
                if 0 <= cx<GRID_WIDTH and 0 <= cy<GRID_HEIGHT:
                    grid[cy, cx] = 1
        return grid

    def detect_collision(self, pos):
        gx, gy = (np.array(pos)//CELL_SIZE).astype(int)
        gx = np.clip(gx, 0,GRID_WIDTH - 1)
        gy = np.clip(gy, 0,GRID_HEIGHT - 1)
        return self.grid[gy, gx] == 1

    def move_towards(self, actor, target):
        dirv=np.array(target)-np.array(actor.pos)
        d=np.linalg.norm(dirv)
        if d>1:
            step = dirv/d*actor.speed*CELL_SIZE
            new_pos = np.array(actor.pos)+step
            if not self.detect_collision(new_pos):
                actor.pos = new_pos.tolist()

    def spread_fire(self):
        new = []
        for fx,fy in list(self.fires):
            for dx,dy in [(CELL_SIZE,0),(-CELL_SIZE,0),(0,CELL_SIZE),(0,-CELL_SIZE)]:
                nx,ny = fx + dx, fy + dy
                if 0 <= nx<WIDTH and 0 <= ny<HEIGHT and random.random()<FIRE_SPREAD_CHANCE:
                    new.append((nx, ny))
        for pos in new:
            if pos not in self.fires:
                self.fires.append(pos)

    def actor_in_fire(self, actor):
        for fx, fy in self.fires:
            if np.linalg.norm(np.array(actor.pos)-np.array((fx, fy))) < FIRE_RADIUS:
                print(f"{actor.actor_type} burned at {actor.pos}")
                return True
        return False

    def find_exit(self, actor):
        return min(self.exits, key=lambda e: np.linalg.norm(np.array(actor.pos)-np.array(e.pos)))

    def find_nearest_actor(self, actor, types):
        nearest, md = None, float('inf')
        for o in self.actors:
            if o.actor_type in types and o is not actor:
                d=np.linalg.norm(np.array(o.pos)-np.array(actor.pos))
                if d<md:
                    nearest, md=o, d
        if nearest:
            print(f"{actor.actor_type} found nearest {nearest.actor_type} at {md:.1f}")
        return nearest

    def step(self):
        #spread fire hazards
        self.spread_fire()
        survivors = []
        #1.staff guide patients/children within range
        for s in [a for a in self.actors if a.actor_type == "Staff"]:
            tgt = self.find_nearest_actor(s, ["Patient", "Child"])
            if tgt and tgt.leader is None:
                d=np.linalg.norm(np.array(s.pos)-np.array(tgt.pos))
                if d<SEARCH_RANGE:
                    s.process_event("Guide")
                    tgt.leader=s
                    tgt.process_event("Guide")
                    print(f"{s.actor_type} starts guiding {tgt.actor_type} at distance {d:.1f}")
        #2.adult follow staff within range
        for ad in [a for a in self.actors if a.actor_type == "Adult"]:
            if ad.leader is None:
                lead=self.find_nearest_actor(ad, ["Staff"])
                if lead:
                    d=np.linalg.norm(np.array(ad.pos)-np.array(lead.pos))
                    if d<SEARCH_RANGE:
                        ad.leader=lead
                        ad.process_event("Guide")
                        print(f"{ad.actor_type} starts following Staff at distance {d:.1f}")
        #3.upgrade date each actor
        for a in self.actors:
            if self.actor_in_fire(a):
                continue
            if a.state in ["idle","searching"]:
                a.process_event("Move")
            if a.state in ["guiding","following"] and a.leader is None:
                a.process_event("Lost")
                print(f"{a.actor_type} lost guide")
            exit_pos=self.find_exit(a).pos
            if a.state in ["searching","guiding","following"] and np.linalg.norm(np.array(a.pos) - np.array(exit_pos)) < 40:
                a.process_event("Exit")
                print(f"{a.actor_type} evacuating at {a.pos}")
                a.leader = None
            if a.state in ["guiding","following"] and a.leader:
                self.move_towards(a, a.leader.pos)
            elif a.state in ["searching","exiting"]:
                self.move_towards(a, exit_pos)
            survivors.append(a)
        self.actors = survivors

    def render(self):
        self.screen.fill(WHITE)
        for w in self.walls:
            pygame.draw.line(self.screen, BLACK, w.start, w.end, 5)
        for e in self.exits:
            pygame.draw.rect(self.screen, RED, e.rect)
        for f in self.fires:
            pygame.draw.circle(self.screen, ORANGE, f, FIRE_RADIUS)
        for a in self.actors:
            a.draw(self.screen)
        pygame.display.flip()

    def run(self):
        while True:
            self.clock.tick(FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.step()
            self.render()

if __name__ == '__main__':
    fn=sys.argv[1] if len(sys.argv)>1 else 'blueprint.json'
    FSMEnvironment(fn).run()
    fn=sys.argv[1] if len(sys.argv)>1 else 'blueprint.json'
    FSMEnvironment(fn).run()
