from games.maze import MazeEnv
from agents.mazeAgent import MazeAgent
from runners import MazeRunner
import pandas as pd
from matplotlib import pyplot as plt

# Defining test batch name
TEST_NAME = 'learning_rate'

# Defining which hyperparameters we want to test
learning_rates = [0.001, 0.01, 0.1]
epsilons = [0]
epsilon_decays = [1]

# Defining how many runs we want to do
N_RUNS = 1

# Creating game
maze = MazeEnv(n=12,num_obstacles=20)

# Creating the runner
runner = MazeRunner(maze)

# Creating agents and adding them to the runner
agents = []
for learning_rate in learning_rates:
    for epsilon in epsilons:
        for epsilon_decay in epsilon_decays:
            runner.add_agent(MazeAgent(alpha=learning_rate, gamma=1-learning_rate, epsilon=epsilon, game=maze, epsilon_decay=epsilon_decay))

# For each run
for i in range(N_RUNS):
    # Run the algorithm
    runner.run(400,5,TEST_NAME+str(i+1))

    # Save visualizations of the stored results
    runner.view(TEST_NAME+str(i+1))