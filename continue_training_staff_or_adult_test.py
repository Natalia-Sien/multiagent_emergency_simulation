import os
import time
from multiprocessing import Value, Process
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym
import numpy as np
from hospital_multienv import env
from gui_train_progress import progress_gui

"""
Used :
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
for the stable_baseline stuff as it contained examples of how to train

monitoring and the wrapped was done using :
https://stable-baselines3.readthedocs.io/en/master/common/monitor.html


"""



#flags to control training per agent type
TRAIN_STAFF = True     #train staff if true
TRAIN_ADULT = True     #train adult if true

#paths to resume model checkpoints
RESUME_MODEL_PATH_STAFF = None
RESUME_MODEL_PATH_ADULT = None

#training hyperparameters
TOTAL_TIMESTEPS = 4096   #total timesteps to learn per agent
SAVE_INTERVAL   = 4096   #save model every this many steps
NUM_ENVS        = 8      #number of parallel environments
N_STEPS         = 256    #rollout steps per update
BATCH_SIZE      = 16384  #batch size for PPO
N_EPOCHS        = 3      #number of ppo epochs per update
BLUEPRINT_PATH  = "blueprint.json"  #env config blueprint

class SingleAgentAECWrapper(gym.Env):
    def __init__(self, aec_env, agent_id):
        self.env = aec_env  #store multi-agent env
        self.agent_id = agent_id  #id of single agent to wrap
        #forward observation and action spaces
        self.observation_space = self.env.observation_spaces[agent_id]
        self.action_space      = self.env.action_spaces[agent_id]

    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)  #reset multi-agent
        return obs_dict[self.agent_id], {}  #return only this agent's obs

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)  #step env
        return obs, reward, done, trunc, info  #return single-agent step tuple

    def render(self, *a, **k):
        self.env.render()  #delegate render to multi-agent

    def close(self):
        self.env.close()  #cleanup underlying env


if __name__ == '__main__':
    #create base multi-agent env and reset to get agent ids
    base = env(BLUEPRINT_PATH, render_on=False)
    base.reset()
    #find first staff and adult agent ids
    staff_id = next((a for a in base.possible_agents if a.startswith("staff_")), None)
    adult_id = next((a for a in base.possible_agents if a.startswith("adult_")), None)

    #loop through agent types for training or resuming
    for agent_id, resume_path, do_train in [
        (staff_id, RESUME_MODEL_PATH_STAFF, TRAIN_STAFF),
        (adult_id, RESUME_MODEL_PATH_ADULT, TRAIN_ADULT)
    ]:
        if agent_id is None:
            continue  #skip if no such agent
        if not do_train and not resume_path:
            print(f"[INFO] No training or resume for {agent_id}, skipping.")
            continue

        print(f"[INFO] Processing {agent_id}…")
        def make_env():
            e = env(BLUEPRINT_PATH, render_on=False)  #create fresh env
            e.reset()
            #wrap single agent for sb3 Monitor
            return Monitor(SingleAgentAECWrapper(e, agent_id))

        #choose vectorized env type based on OS
        if os.name == 'nt':
            vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])  #sync envs for windows
        else:
            vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])  #parallel subprocess envs

        #load or create model
        if resume_path and os.path.isfile(resume_path):
            print(f"[INFO] Loading checkpoint from {resume_path}")
            model = PPO.load(
                resume_path,
                env=vec_env,
                tensorboard_log=f"./tensorboard_logs/{agent_id}"
            )
            steps_done = model.num_timesteps  #resume count
        else:
            print(f"[INFO] Creating new model for {agent_id}")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                tensorboard_log=f"./tensorboard_logs/{agent_id}"
            )
            steps_done = 0  #start fresh

        #create shared counter and GUI process for progress
        progress_counter = Value("i", steps_done)  #int shared value
        gui = Process(target=progress_gui, args=(progress_counter, TOTAL_TIMESTEPS))
        gui.start()

        #training loop with periodic saves
        start_time = time.time()
        while steps_done < TOTAL_TIMESTEPS:
            #determine next chunk size
            chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
            print(f"[INFO] Learning next {chunk} steps (from {steps_done})…")
            try:
                model.learn(total_timesteps=chunk, reset_num_timesteps=False)  #train PPO
            except Exception as e:
                print(f"[ERROR] Training interrupted: {e}")
                break
            steps_done += chunk  #update counter
            with progress_counter.get_lock():
                progress_counter.value = steps_done  #update GUI
            chk = f"model_{agent_id}_step_{steps_done}.zip"  #checkpoint filename
            print(f"[INFO] Saving checkpoint: {chk}")
            model.save(chk)  #persist model

        #cleanup GUI process
        gui.terminate()
        gui.join()

        #report elapsed time formatted
        elapsed = time.time() - start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(f"[DONE] {agent_id} done. Total time: {int(h)}h {int(m)}m {s:.2f}s")
