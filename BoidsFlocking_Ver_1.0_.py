import numpy as np

# --------------- parameters that can be tuned to adjust behaviour ---------------

# the radius where each boids can influence each other
Neighbour_Radius = 50
# the radius that boids will repel each other
Separation_Radius = 20

# weights for sepeartion alignment and cohesion
Separation_W = 1
Alignment_W = 1
Cohesion_W = 1

# attraction toward nearest exit
Goal_Attr_W = 2
# avoide nearby fire
Avoid_Fire_W = 2

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
def avoid_fire(pos,fires):
    # returns a zero vector no fire around
    repel = np.zeros(2)
    # for each fire positions
    for fx,fy in fires:
        fire_pos = np.array((fx,fy), dtype=float)
        offset = pos-fire_pos
        dist = np.linalg.norm(offset)
        # setting a threshold distance to react to fire
        if dist < 100:
            # repalsion will ber stronger if fire is closer
            factor = (100-dist)/100
            repel += offset/(dist+1e-5)*factor
    return repel

# checks if hitting a wall
def is_collision(new_pos, obstacle_grid):
    from math import floor
    x_idx = int(floor(new_pos[0] / 5))
    y_idx = int(floor(new_pos[1] / 5))
    
    x_idx = max(0, min(obstacle_grid.shape[1]-1, x_idx))
    y_idx = max(0, min(obstacle_grid.shape[0]-1, y_idx))
    return (obstacle_grid[y_idx, x_idx] == 1)


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

        # Goal Attraction

        # get the exit position
        exit_pos = find_exit(position_A,exits)
        if exit_pos is not None:
            # compute weighted vector from actor to exit 
            to_goal = exit_pos-position_A
            boids_force += Goal_Attr_W*to_goal

        # avoid fire
        avoid_fire_force = avoid_fire(position_A,fires)
        boids_force += Avoid_Fire_W*avoid_fire_force

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
        velocity_new = velocity_A + boids_force*0.01  # scale to damp forces 
        # limit the speed of boids
        speed = np.linalg.norm(velocity_new)
        if speed > actor_max_speed:
            velocity_new = (velocity_new/speed)*actor_max_speed
        velocities[actor] = velocity_new

    # apply the velocities computed to update positions
    for actor in actors:
        new_pos = np.array(actor.pos,dtype=float) + velocities[actor]
        # Check wall collision
        if not is_collision(new_pos,obstacle_grid):
            actor.pos = new_pos.astype(int).tolist()
            actor.velocity = velocities[actor]
        else:
            # stop movement if collides with wall
            actor.velocity = np.zeros(2)