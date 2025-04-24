import gymnasium as gym
import numpy as np

"""
I used this in the beginning for faster training, as I was able to create 24 instances on a 28thread cpu
in the end we shifted to one trained environment.
"""

class PettingZooEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__()
        #store underlying petting zoo env
        self.env = env
        #list of agent ids in env
        self.agents = env.possible_agents
        #start with first agent
        self.current_agent_index = 0
        self.current_agent = self.agents[self.current_agent_index]

        #assume all agents share same spaces
        self.action_space = self.env.action_space(self.current_agent)
        self.observation_space = self.env.observation_space(self.current_agent)

    def reset(self, seed=None, options=None):
        #reset multiagent env and get dicts
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        #reset turn order
        self.current_agent_index = 0
        self.current_agent = self.agents[self.current_agent_index]

        #save last obs for chaining
        self.last_obs_dict = obs_dict
        #return only this agent's obs and info
        return obs_dict[self.current_agent], info_dict.get(self.current_agent, {})

    def step(self, action):
        #get id of agent to step
        agent = self.current_agent

        #step only this agent in multiagent env
        try:
            obs, rewards, terminations, truncations, infos = self.env.step({agent: action})
        except Exception as e:
            raise RuntimeError(f"Failed to step agent '{agent}' with action: {action}\n{e}")

        #advance to next agent in round robin
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        self.current_agent = self.agents[self.current_agent_index]

        #save obs dict for next call
        obs_dict = obs
        self.last_obs_dict = obs_dict

        #extract this agent's data
        obs = obs_dict[self.current_agent]
        reward = rewards[self.current_agent]
        terminated = terminations[self.current_agent]
        truncated = truncations[self.current_agent]
        info = infos[self.current_agent]

        return obs, reward, terminated, truncated, info
