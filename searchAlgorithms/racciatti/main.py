from problems import EightPuzzle
from agent import Agent
from solve import *

agent = Agent()
problem = EightPuzzle(verbose=True)
strategy = BFSStrategy()

print(agent.solve(problem, strategy))