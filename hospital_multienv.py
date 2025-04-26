import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np
import heapq
import agent_outline
"""
Use multiple sources for this, with the gymnasium environtment, i once again used the gymnasium documentation
as i do believe it is incredibly well documented.

for the petting zoo stuff, specifically the AEC environement stuff:
https://pettingzoo.farama.org/api/aec/
https://pettingzoo.farama.org/content/environment_creation/
i found it to also be well documented and there were plenty if features to mess about with.

ive used the A* algorithm many times during my undergrad in comp sci, specially in python and prologue:
https://www.geeksforgeeks.org/a-search-algorithm/ is my go to source as it is genuinely so simple.

The action mapping I knew from before, but I learnt it years ago from a youtuber called Nicholas Renotte,
https://www.youtube.com/watch?v=Mut_u40Sqz4&t=70s&ab_channel=NicholasRenotte This is one of his many videos.

Hopefully the comments should help you understand my thought process and reasonings of why I did things the way they are

"""

class HospitalMultiAgentEnv(AECEnv):
    def __init__(
        self,
        blueprint_path,
        render_on=False,
        randomize_on_reset=False,
        max_episode_steps=1000,
        screen=None
    ):
        super().__init__()
        #set metadata for env
        self.metadata = {
            "render_modes": ["human"],
            "name": "hospital_multiagent_env",
            "is_parallelizable": True
        }
        #set render mode based on flag
        self.render_mode = "human" if render_on else None
        #store config values passed by user
        self.blueprint_path = blueprint_path
        self.render_on = render_on
        self.randomize_on_reset = randomize_on_reset
        self.max_episode_steps = max_episode_steps
        self.screen = screen
        #track agents to remove when they die
        self._to_remove = set()

    def _initialize_env(self):
        #load blueprint assets and build underlying simulation env
        self.walls, self.exits, self.actors, self.fires = agent_outline.load_blueprint(
            self.blueprint_path
        )
        #if no screen provided, create or get existing pygame surface
        if not self.screen:
            self.screen = (
                pygame.display.get_surface() or pygame.display.set_mode((1000, 700))
            )
        #create core BlueprintEnvironment with loaded assets
        self.env = agent_outline.BlueprintEnvironment(
            self.walls, self.exits, self.actors, self.fires, screen=self.screen
        )
        #reuse grid array for pathfinding obstacles, 0=free cell, 1=wall
        self.grid = self.env.grid

    def _setup_agents(self):
        #staff and adults are controllable agents
        self.rescuers = [
            a for a in self.env.actors if a.actor_type in ["Staff", "Adult"]
        ]
        #generate unique id strings for each rescuer
        self.agents = [f"{a.actor_type.lower()}_{i}" for i, a in enumerate(self.rescuers)]
        #map id to the actual Actor object
        self.agent_name_mapping = dict(zip(self.agents, self.rescuers))
        self.possible_agents = list(self.agents)

        #patients and children are targets to rescue
        self.rescue_targets = [
            a for a in self.env.actors if a.actor_type in ["Patient", "Child"]
        ]

        #action space, choose index of rescue_targets list
        num_actions = len(self.rescue_targets)
        self.action_spaces = {
            agent: spaces.Discrete(num_actions) for agent in self.agents
        }

        #construct observation space:
        # [pos(2), rays(8), exit_vec(2), nearest_target(8)] = 20 values
        low = np.array(
            [-1.0, -1.0]           #normalized x,y in [-1,1]
            + [0.0] * 8            #ray distances clipped to [0,1]
            + [-1.0, -1.0]         #exit vector components in [-1,1]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #nearest target features
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 1.0]
            + [1.0] * 8
            + [1.0, 1.0]
            + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        #assign Box space for each agent
        self.observation_spaces = {
            agent: spaces.Box(low=low, high=high, dtype=np.float32)
            for agent in self.agents
        }

        #initialize tracking dicts for RL
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #pending updates to synchronize at round end
        self._pending_rewards = {}
        self._pending_terminations = {}
        #first agent to act
        self.agent_selection = self.agents[0] if self.agents else None

    def _astar(self, start, goal):
        #convert world coords to grid indices by integer division
        sx, sy = np.array(start) // agent_outline.CELL_SIZE
        gx, gy = np.array(goal) // agent_outline.CELL_SIZE
        rows, cols = self.grid.shape

        #manhattan distance heuristic, admissible for 4-neighbor grid
        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        #open_set holds tuples, (priority = cost+heuristic, cost so far, node, parent)
        open_set = [(h((sx, sy), (gx, gy)), 0, (sx, sy), None)]
        came_from = {}       # map child->parent for path reconstruction
        cost_so_far = {(sx, sy): 0}  #best known cost to reach each node

        while open_set:
            priority, cost, current, parent = heapq.heappop(open_set)
            #if goal reached, reconstruct path back to start
            if current == (gx, gy):
                path = []
                node = current
                while node:
                    path.append(node)
                    node = came_from.get(node)
                return path[::-1]  # reverse to go from start->goal
            #skip if already visited with better path
            if current in came_from:
                continue
            came_from[current] = parent
            #explore each of 4 neighboring cells
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = current[0] + dx, current[1] + dy
                #check bounds and free cell (grid value 0)
                if 0 <= nx < cols and 0 <= ny < rows and self.grid[ny, nx] == 0:
                    new_cost = cost + 1  # uniform cost per step
                    #if new path is better, record and push to queue
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + h((nx, ny), (gx, gy))
                        heapq.heappush(open_set, (priority, new_cost, (nx, ny), current))
        #fallback, no path found, stay in place
        return [(sx, sy)]

    def _cell_to_action(self, pos, cell):
        #compute world-space delta from actor pos to center of target cell
        dx = cell[0] * agent_outline.CELL_SIZE - pos[0]
        dy = cell[1] * agent_outline.CELL_SIZE - pos[1]
        #normalize to direction vector of -1,0,1 components
        mv = (np.sign(dx), np.sign(dy))
        #map direction pairs to discrete move action codes
        moves = {
            (0, -1): 0,   #up
            (0, 1): 1,    #down
            (-1, 0): 2,   #left
            (1, 0): 3,    #right
            (0, 0): 4,    #stay
            (-1, -1): 5, #up-left
            (1, -1): 6,  #up-right
            (-1, 1): 7,  #down-left
            (1, 1): 8,   #down-right
        }
        #default to stay if direction not found
        return moves.get((int(mv[0]), int(mv[1])), 4)

    def _decode_high_action(self, agent_id, action):
        #convert discrete target index to actual rescue target position
        idx = int(action)
        return self.rescue_targets[idx].pos

    def reset(self, seed=None, options=None):
        #initialize or reset underlying sim environment
        self._initialize_env()
        #optional random jitter for actor start positions
        if self.randomize_on_reset:
            for actor in self.env.actors:
                actor.pos = [
                    actor.pos[0] + np.random.randint(-3, 4),
                    actor.pos[1] + np.random.randint(-3, 4),
                ]
        #setup RL agents and spaces
        self._setup_agents()
        self.global_step = 0
        self.agent_selection = self.agents[0] if self.agents else None
        #return initial observations dict for all agents
        obs_dict = {agent: self.observe(agent) for agent in self.agents}
        return obs_dict, {}

    def observe(self, agent):
        #produce observation vector for single agent
        actor = self.agent_name_mapping[agent]
        #if actor no longer in env (e.g dead), return zeros
        if actor not in self.env.actors:
            return np.zeros(self.observation_spaces[agent].shape, dtype=np.float32)
        
        #normalize position by screen dims for network input
        pos = np.array(actor.pos) / np.array([1000, 700])
        
        #raycast from actor to walls/exits, returns distances up to max range
        rays = self.env.raycast_distances(actor, num_rays=8)
        
        #vector to nearest exit normalized
        exit_vec = self._nearest_exit_vector(actor)
        
        #find nearest target and its features
        diag = np.linalg.norm(np.array([1000, 700]))  # diag length of screen
        nearest_target_features = np.zeros(8)  # [pos_x, pos_y, speed, distance, 0, 0, 0, 0]
        if self.rescue_targets:
            #find nearest target
            distances = [np.linalg.norm(np.array(actor.pos) - np.array(tgt.pos)) for tgt in self.rescue_targets]
            nearest_idx = np.argmin(distances)
            nearest_target = self.rescue_targets[nearest_idx]
            
            #normalize target position
            target_pos = np.array(nearest_target.pos) / np.array([1000, 700])
            #normalize distance
            distance = distances[nearest_idx] / diag
            #get speed
            speed = nearest_target.speed
            
            nearest_target_features = np.array([
                target_pos[0], target_pos[1],  #normalized x,y position
                speed, distance,              #speed and normalized distance
                0, 0, 0, 0                    #padding to maintain 20-value format
            ])
        
        #concatenate all components into single 1D array
        return np.concatenate(
            [pos, rays, exit_vec, nearest_target_features]
        )

    def step(self, action):
        #update step counter and check for truncation by length
        self.global_step += 1
        truncated = False
        if self.global_step >= self.max_episode_steps:
            #mark all agents as truncated
            for a in self.agents:
                self._pending_terminations[a] = True
                self.truncations[a] = True
            truncated = True

        #if no agents left, return terminal obs
        if not self.agents or self.agent_selection is None:
            return np.zeros(self.observation_spaces[self.agent_selection].shape, np.float32), 0, True, truncated, {}

        agent = self.agent_selection
        actor = self.agent_name_mapping[agent]
        moved = False

        #if rescuer and valid action, plan path and move one cell
        if actor.actor_type in ["Adult", "Staff"] and action is not None:
            tgt_pos = self._decode_high_action(agent, action)
            path = self._astar(actor.pos, tgt_pos)
            if len(path) > 1:
                ll = self._cell_to_action(actor.pos, path[1])
                pre = actor.pos[:]
                self.env._move_actor(actor, ll)
                moved = (actor.pos != pre)

        #wall penalty, encourages staying clear of walls
        WALL_PENALTY_COEFF = 20.0
        RAY_MAX_RANGE = 200
        inv = 1.0 - self.env.raycast_distances(actor, num_rays=8, max_range=RAY_MAX_RANGE)
        penalty = WALL_PENALTY_COEFF * (inv.mean() ** 2)  #squared mean gives larger penalty when close

        #advance simulation for all actors
        self.env.update_actors()

        #handle fire and reward, death if in fire
        if actor in self.env.actors and self.env.actor_in_fire(actor):
            self._to_remove.add(agent)
            self._pending_rewards[agent] = -10
        elif actor in self.env.actors:
            #calculate distance reduction reward to nearest exit
            exit_o = self.find_nearest_exit(actor)
            odist = np.linalg.norm(
                np.array(actor.previous_pos) - np.array(exit_o.pos)
            )
            ndist = np.linalg.norm(np.array(actor.pos) - np.array(exit_o.pos))
            #reward scheme, +2 for getting closer, -1 for moving away, -10 for no move
            r = -10.0 if not moved else (2.0 if ndist < odist else -1.0)
            #bonus for reaching exit
            if self.env.actor_reached_exit(actor):
                self._pending_terminations[agent] = True
                r += 10.0
            #apply wall proximity penalty
            r -= penalty
            self._pending_rewards[agent] = self._pending_rewards.get(agent, 0) + r
        else:
            #actor missing => forced termination
            self._pending_terminations[agent] = True
            self._pending_rewards[agent] = -10

        #advance turn order and finalize removals
        self._advance_to_next_agent()
        for aid in self._to_remove:
            self.terminations[aid] = True
            dead = self.agent_name_mapping[aid]
            if dead in self.env.actors:
                self.env.actors.remove(dead)
        self._to_remove.clear()
        #if we cycled back to first agent, sync pending states
        if self.agent_selection == self.agents[0]:
            for a in self.agents:
                self.terminations[a] = self._pending_terminations.get(a, False)
                self.truncations[a] = self.truncations.get(a, False)
                self.rewards[a] = self._pending_rewards.get(a, 0)
            self._pending_terminations.clear()
            self._pending_rewards.clear()

        return (
            self.observe(self.agent_selection),
            self.rewards.get(self.agent_selection, 0),
            self.terminations.get(self.agent_selection, False),
            truncated,
            self.infos.get(self.agent_selection, {}),
        )

    def _nearest_exit_vector(self, actor):
        #vector from actor to nearest exit normalized by screen dims
        exit_o = self.find_nearest_exit(actor)
        return (np.array(exit_o.pos) - np.array(actor.pos)) / np.array([1000, 700])

    def _advance_to_next_agent(self):
        #determine next agent that is still alive
        live = [a for a in self.agents if not self.terminations[a]]
        if not live:
            #if none live, end episode for all
            for a in self.agents:
                self.terminations[a] = True
                self._pending_rewards[a] = -1000
            self.agent_selection = self.agents[0]
            return
        #find current index and wrap around
        idx = (
            live.index(self.agent_selection)
            if self.agent_selection in live
            else -1
        )
        self.agent_selection = live[(idx + 1) % len(live)]

    def render(self):
        #render environment visuals and highlight current agent
        if not self.render_on:
            return
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.display.get_surface():
            pygame.display.set_mode((1000, 700))
        self.env.render()
        if self.agent_selection:
            actor = self.agent_name_mapping.get(self.agent_selection)
            if actor and hasattr(actor, "pos"):
                surf = pygame.display.get_surface()
                pygame.draw.circle(surf, (255, 255, 0), actor.pos, 14, 3)
                font = pygame.font.SysFont(None, 20)
                label = font.render(self.agent_selection, True, (255, 255, 255))
                surf.blit(label, (actor.pos[0] + 10, actor.pos[1] - 10))
        pygame.display.flip()
        pygame.event.pump()

    def find_nearest_exit(self, actor):
        #delegate exit search to underlying env logic
        return self.env.find_nearest_exit(actor)

    def close(self):
        #cleanup all pygame resources
        pygame.quit()


def env(
    blueprint_path,
    render_on=False,
    randomize_on_reset=False,
    max_episode_steps=1000,
    screen=None,
):
    #factory function to wrap HospitalMultiAgentEnv for PettingZoo
    base_env = HospitalMultiAgentEnv(
        blueprint_path=blueprint_path,
        render_on=render_on,
        randomize_on_reset=randomize_on_reset,
        max_episode_steps=max_episode_steps,
        screen=screen,
    )
    if render_on:
        return wrappers.CaptureStdoutWrapper(base_env)
    return base_env