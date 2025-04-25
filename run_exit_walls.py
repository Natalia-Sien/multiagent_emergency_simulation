import os
import csv
import random
import numpy as np
import pygame
import concurrent.futures

import agent_outline
from agent_outline import Wall, FireExit, Actor, BlueprintEnvironment

"""
Used:
https://docs.python.org/3/library/concurrent.futures.html
for the parallelism implementation
Used:
https://www.pygame.org/docs/
for the visual aspect (rendering the movements)

for writing the csv, i saw ** in the writing function and decided to test it,
seemd like it actually made the job much easier:
https://docs.python.org/3/library/csv.html
i never used ** in python before, but it was a very efficient way to unpack dictionaries that were created

everything else was just typical python stuff

batch run evacuation simulations sweeping
exits population room size staff percentage time steps
supports parallel execution for speed
outputs results to csv for analysis
no fire spread enabled yet
"""

#setup and config flags
TEST_FIRE_EXITS = True   #include exit count sweep
TEST_POPULATION_SIZE = True   #include population sweep
TEST_ROOM_SIZE = True   #include room size sweep

#debug and visualization flags
VERBOSE = False  #print periodic actor counts and summaries
VISUALIZE_EACH = False  #enable pygame rendering when running serially

#parallel execution settings
PARALLEL = True      #run tasks in parallel when True
if (VISUALIZE_EACH == True) & (PARALLEL == False):
    NUM_WORKERS = 1  #number of worker processes
else:
    NUM_WORKERS = max(1, os.cpu_count() - 1)  #number of worker processes

#parameter ranges
MIN_EXITS, MAX_EXITS = 1, 5
MIN_POPULATION, MAX_POPULATION = 50, 200
STEP_POPULATION = 50
MIN_ROOM_SIZE, MAX_ROOM_SIZE = 500, 1500
STEP_ROOM_SIZE = 500

MIN_STAFF_PCT, MAX_STAFF_PCT, STEP_STAFF_PCT = 0.15, 0.25, 0.05
MIN_TIME_STEPS, MAX_TIME_STEPS, STEP_TIME_STEPS = 250, 500, 50

#default room dimensions from agent_outline
DEFAULT_ROOM_WIDTH, DEFAULT_ROOM_HEIGHT = agent_outline.WIDTH, agent_outline.HEIGHT

#output csv settings
OUTPUT_CSV = "simulation_results_exit_walls.csv"
FIELDNAMES = [
    "test_type","room_width","room_height",
    "n_exits","population","pct_staff","pct_adult","pct_patient","pct_child",
    "time_steps","avg_evac_time","avg_death_time","num_evacuated","num_burned"
]

#helper generate boundary walls
def generate_walls(w, h):
    #create four walls at room edges
    return [
        Wall((0, 0),     (w-1, 0)),
        Wall((w-1, 0),   (w-1, h-1)),
        Wall((w-1, h-1), (0, h-1)),
        Wall((0, h-1),   (0, 0)),
    ]

#helper generate fire exits around all walls
def generate_exits(n_exits, w, h):
    #evenly distribute exits around room perimeter
    exits = []
    perim = 2 * (w + h)
    for i in range(n_exits):
        d = (i + 1) * perim / (n_exits + 1)
        if d < w:
            x, y = int(d), 0
        elif d < w + h:
            x, y = w, int(d - w)
        elif d < 2*w + h:
            x, y = int(w - (d - (w + h))), h
        else:
            x, y = 0, int(h - (d - (2*w + h)))
        exits.append(FireExit((x, y), size=(40, 40)))
    return exits

#helper generate actors with dynamic composition
def generate_actors(n_people, w, h, staff_pct):
    #calculate counts of each type
    other = (1 - staff_pct) / 3
    comp = {"Staff": staff_pct, "Adult": other, "Patient": other, "Child": other}
    counts = {t: int(n_people * p) for t, p in comp.items()}
    #adjust rounding error
    while sum(counts.values()) < n_people:
        t = random.choice(list(counts.keys()))
        counts[t] += 1
    actors = []
    #spawn actors at random positions
    for t, cnt in counts.items():
        for _ in range(cnt):
            x = random.randint(20, w - 20)
            y = random.randint(20, h - 20)
            actors.append(Actor([x, y], t))
    return actors

#core run a single simulation
def run_simulation(w, h, exits, pop, staff_pct, steps_max, *, visualize=False, verbose=False):
    #patch dimensions in agent_outline
    agent_outline.WIDTH, agent_outline.HEIGHT = w, h
    agent_outline.GRID_WIDTH = w // agent_outline.CELL_SIZE
    agent_outline.GRID_HEIGHT = h // agent_outline.CELL_SIZE

    #build scenario
    walls = generate_walls(w, h)
    f_exits = generate_exits(exits, w, h)
    actors = generate_actors(pop, w, h, staff_pct)

    #initialize environment
    env = BlueprintEnvironment(walls, f_exits, actors, fires=[], screen=None)
    env.render_on = visualize
    if visualize:
        #setup pygame window for visualization
        pygame.init()
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Sim exits={exits} pop={pop} size={w}×{h}")
        env.screen = screen

    step = 0
    #simulation loop
    while env.actors and step < steps_max:
        env.update_actors()
        #nudge adult actors toward nearest exit
        for a in env.actors:
            if a.actor_type == "Adult":
                exit_pos = env.find_nearest_exit(a).pos
                env.move_towards(a, exit_pos, a.speed)
        step += 1
        if visualize:
            env.render()
            pygame.time.delay(10)
        if verbose and step % 100 == 0:
            print(f"[step {step}] remaining {len(env.actors)}")

    #collect metrics
    evac = [t for times in env.evacuation_times.values() for t in times]
    dead = [t for times in env.death_times.values() for t in times]
    return {
        "avg_evac_time":  float(np.mean(evac))  if evac  else None,
        "avg_death_time": float(np.mean(dead)) if dead else None,
        "num_evacuated":  len(evac),
        "num_burned":     len(dead)
    }

#worker wrapper for headless execution
def _run_task(task):
    #run simulation without rendering or verbose logs
    out = run_simulation(
        task["room_width"], task["room_height"],
        task["n_exits"], task["population"],
        task["pct_staff"], task["time_steps"],
        visualize=False, verbose=False
    )
    return {**task, **out}

#main build tasks execute write csv
if __name__ == '__main__':
    #prepare staff pct and timestep lists
    staff_list = np.arange(MIN_STAFF_PCT, MAX_STAFF_PCT + 1e-8, STEP_STAFF_PCT).tolist()
    steps_list = list(range(MIN_TIME_STEPS, MAX_TIME_STEPS + 1, STEP_TIME_STEPS))

    tasks = []
    #build tasks for exit sweep
    if TEST_FIRE_EXITS:
        for ne in range(MIN_EXITS, MAX_EXITS+1):
            for pop in range(MIN_POPULATION, MAX_POPULATION+1, STEP_POPULATION):
                for size in range(MIN_ROOM_SIZE, MAX_ROOM_SIZE+1, STEP_ROOM_SIZE):
                    for sp in staff_list:
                        for ts in steps_list:
                            tasks.append({
                                "test_type":"fire_exits",
                                "room_width":size,"room_height":size,
                                "n_exits":ne,"population":pop,
                                "pct_staff":sp,"time_steps":ts
                            })
    #build tasks for population sweep
    if TEST_POPULATION_SIZE:
        for pop in range(MIN_POPULATION, MAX_POPULATION+1, STEP_POPULATION):
            for ne in range(MIN_EXITS, MAX_EXITS+1):
                for size in range(MIN_ROOM_SIZE, MAX_ROOM_SIZE+1, STEP_ROOM_SIZE):
                    for sp in staff_list:
                        for ts in steps_list:
                            tasks.append({
                                "test_type":"population",
                                "room_width":size,"room_height":size,
                                "n_exits":ne,"population":pop,
                                "pct_staff":sp,"time_steps":ts
                            })
    #build tasks for room size sweep
    if TEST_ROOM_SIZE:
        for size in range(MIN_ROOM_SIZE, MAX_ROOM_SIZE+1, STEP_ROOM_SIZE):
            for ne in range(MIN_EXITS, MAX_EXITS+1):
                for pop in range(MIN_POPULATION, MAX_POPULATION+1, STEP_POPULATION):
                    for sp in staff_list:
                        for ts in steps_list:
                            tasks.append({
                                "test_type":"room_size",
                                "room_width":size,"room_height":size,
                                "n_exits":ne,"population":pop,
                                "pct_staff":sp,"time_steps":ts
                            })

    print(f"[INFO] built {len(tasks)} tasks")
    if not tasks:
        raise RuntimeError("no tasks generated check TEST flags and ranges")

    #open csv and write header
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        #parallel execution branch
        if PARALLEL:
            print(f"[INFO] running in parallel with {NUM_WORKERS} workers")
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for i, result in enumerate(executor.map(_run_task, tasks), start=1):
                    #compute other percentages
                    ap = (1 - result["pct_staff"]) / 3
                    writer.writerow({**result,
                                     "pct_adult":ap,
                                     "pct_patient":ap,
                                     "pct_child":ap})
                    if i % 50 == 0:
                        print(f"[{i}/{len(tasks)}] done")
        #serial execution branch
        else:
            print("[INFO] running serially")
            for i, task in enumerate(tasks, start=1):
                if VISUALIZE_EACH:
                    #run with rendering enabled
                    metrics = run_simulation(
                        task["room_width"], task["room_height"],
                        task["n_exits"], task["population"],
                        task["pct_staff"], task["time_steps"],
                        visualize=True, verbose=VERBOSE
                    )
                    result = {**task, **metrics}
                else:
                    #headless run
                    result = _run_task(task)

                #compute other percentages
                ap = (1 - result["pct_staff"]) / 3
                writer.writerow({**result,
                                 "pct_adult":ap,
                                 "pct_patient":ap,
                                 "pct_child":ap})
                print(f"[{i}/{len(tasks)}] done")

    print(f"[ALL DONE] {len(tasks)} simulations complete Output {OUTPUT_CSV!r}")
