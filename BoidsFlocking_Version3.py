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

# Velocity scaling
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
