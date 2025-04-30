def run_simulations(filename, runs=0, max_steps=0, export=True):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Automated Simulation Run {runs+1}")

    walls, exits, actors, fires = load_blueprint(filename)
    env = BlueprintEnvironment(walls, exits, actors, fires, screen=screen)

    start_ticks = pygame.time.get_ticks()
    for actor in actors:
        actor.start_time = start_ticks

    for step in range(max_steps):
        env.update_actors()
        if env.render_on:
            env.render()
        # stop early if all agents are gone
        if len(env.actors) == 0:
            break

    if export:
        env.export_metrics(f"test_metrics_run{runs+1}.csv")
        env.export_advanced_metrics(f"test_advanced_metrics_run{runs+1}.csv")

    pygame.quit()


if __name__ == '__main__':
    # === Settings ===
    run_automated = True   # Set to False to run manually
    rum_runs = 10                # Number of test repetitions
    max_steps = 500              # Max steps per simulation

    #set to true to export metrics at end
    metrics = True
    #set true to export advanced_metrics at the end
    advanced_metrics = True
    #prompt for blueprint file
    filename = pygame_file_picker()
    
    if run_automated:
        for i in range(rum_runs):
            run_simulations(filename, run_id=i, max_steps=max_steps, export=True)
    else:
        #initialize simulation window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Agent Outline")

        #load blueprint data
        walls, exits, actors, fires = load_blueprint(filename)
        env = BlueprintEnvironment(walls, exits, actors, fires)
        running = True

        #set start times for metrics
        start_ticks = pygame.time.get_ticks()
        for actor in actors:
            actor.start_time = start_ticks

        #main loop: update & render
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            env.update_actors()
            env.render()
        pygame.quit()
        #export metrics if enabled
        if metrics:
          env.export_metrics("evacuation_results.csv")
        if advanced_metrics:
          env.export_advanced_metrics("advanced_evacuation_results.csv")