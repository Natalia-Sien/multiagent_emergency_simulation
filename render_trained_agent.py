import pygame
import time
import numpy as np
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from hospital_multienv import HospitalMultiAgentEnv
from agent_outline import load_blueprint
"""
Most of this is based off of the default Gymnasium environment code, i altered it to work with our configurations
https://gymnasium.farama.org/

luckily it is so well documented that it was a breeze to work with (thank you gymnasium developers)
"""

#setup and configs
BLUEPRINT_PATH = "blueprint.json"
# BLUEPRINT_PATH = "blueprint(stuck).json"
MODEL_PATH_STAFF = "models/model_staff_0_step_4096"  #load in a pre trained, saved staff model zip
MODEL_PATH_ADULT = "models/model_adult_1_step_4096"  #load in a pre trained, saved adult model zip

#initialization of the screen
pygame.init()
screen = pygame.display.set_mode((1000, 700)) #setting the size
pygame.display.set_caption("Hospital Simulation") #setting the name
font = pygame.font.SysFont("consolas", 20) #font type, just picked whatever this was in the default env config


#environment creation section
walls, exits, actors, fires = load_blueprint(BLUEPRINT_PATH) #load blueprint assets (the fires,walls etc)
#here i set render on, as i want to see what the agents do, max steps determines how long it will run for
env = HospitalMultiAgentEnv(BLUEPRINT_PATH, render_on=True, screen=screen, max_episode_steps=2000)
obs_dict, _ = env.reset()  #clearing up the old environment just incase (was a temp bug fix, that seems like a perm fix)


print("[INFO] Loading models...") #debug print
model_staff = PPO.load(MODEL_PATH_STAFF) #load in staff model
model_adult = PPO.load(MODEL_PATH_ADULT) #load in adult model


#main loop of the code
clock = pygame.time.Clock() #internal timer
running = True
reward_total = 0 #initialize reward to 0
step_counter = 0 #initialize steps to 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q): #failsafe (q or close)
            running = False


    #agent iteration, here we step through a single agent at a time to allow proper processing to happen
    if env.agent_selection:
        agent_id = env.agent_selection

        #skipping any agent that are done
        if env.terminations.get(agent_id, False) or env.truncations.get(agent_id, False):
            env.step(None)
            continue

        obs = env.observe(agent_id)

        #choose action based on the agents type
        if agent_id.startswith("staff_"):
            action, _ = model_staff.predict(obs) #use staff model on staff
        elif agent_id.startswith("adult_"):
            action, _ = model_adult.predict(obs) #use adult model on adults.
        else:
            action = env.action_spaces[agent_id].sample() #random input from children or patients for them

        env.step(action)

        #accumulate rewards and update step counter for controlled agents
        if agent_id.startswith("staff_") or agent_id.startswith("adult_"):
            reward_total += env.rewards[agent_id] #add appropriate rewards to the reward pool
            step_counter += 1 #add one to step

    #clear screen and render environment once we are out of the loop
    screen.fill((255, 255, 255))
    env.render()

    #displaying step counter and reward info
    step_text = font.render(f"Step: {step_counter}", True, (0, 0, 0))
    reward_text = font.render(f"Reward: {reward_total:.2f}", True, (0, 0, 0))
    screen.blit(step_text, (10, 10))
    screen.blit(reward_text, (800, 10))

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
