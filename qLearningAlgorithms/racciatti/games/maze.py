import numpy as np
import random
import time
import os

class MazeEnv:
    def __init__(self, n=5, num_obstacles=5):
        self.n = n
        self.start = (0, 0)
        self.goal = (n-1, n-1)
        self.state = self.start
        self.obstacles = set()
        self.generate_obstacles(num_obstacles)

    def generate_obstacles(self, num_obstacles):
        # gera obstáculos aleatórios, evitando Start e Goal
        self.obstacles.clear()
        while len(self.obstacles) < num_obstacles:
            i = random.randint(0, self.n-1)
            j = random.randint(0, self.n-1)
            if (i, j) not in [self.start, self.goal]:
                self.obstacles.add((i, j))

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        i, j = self.state
        if action == 0: i -= 1  # cima
        elif action == 1: i += 1  # baixo
        elif action == 2: j -= 1  # esquerda
        elif action == 3: j += 1  # direita

        i = max(0, min(i, self.n-1))
        j = max(0, min(j, self.n-1))
        next_state = (i, j)

        if next_state in self.obstacles:
            reward = -1
            done = True
        elif next_state == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.01
            done = False

        self.state = next_state
        return next_state, reward, done

    def get_actions(self):
        return [0,1,2,3]

    def get_states(self):
        return [(i, j) for i in range(self.n) for j in range(self.n)]

    def render(self, agent_pos):
        os.system("cls" if os.name == "nt" else "clear")
        for i in range(self.n):
            row = ""
            for j in range(self.n):
                if (i, j) == self.start:
                    row += "S "
                elif (i, j) == self.goal:
                    row += "G "
                elif (i, j) in self.obstacles:
                    row += "X "
                elif (i, j) == agent_pos:
                    row += "A "
                else:
                    row += ". "
            print(row)
        print("\n")
        time.sleep(0.5)
