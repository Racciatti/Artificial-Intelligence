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
            reward = 1
            done = True
        else:
            reward = -0.01
            done = False

        self.state = next_state
        return next_state, reward, done

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
        time.sleep(0.8)


def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2): 
    # episodes: n de tentativas
    # alpha: taxa de aprendizado, quando do novo alvo sobrepõe o valor antigo de Q (maior valor -> q antigo descartado e só o alvo novo conta.)
    # gamma: fator de desconto, quão importante é o futuro vs recompensa imediata (maior valor -> maior valorização de recompensas futuras)
    # epsilon: chance de explorar outras opções

    states = [(i, j) for i in range(env.n) for j in range(env.n)]
    actions = [0,1,2,3]
    q_table = {s: {a: 0.0 for a in actions} for s in states} # pra cada "tile" do maze (estado), cria o peso das 4 ações possíveis
        #q_table[(0,0)] = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
        #q_table[(0,1)] = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
        #...
        #q_table[(2,2)] = {0:0.0, 1:0.0, 2:0.0, 3:0.0}

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state, reward, done = env.step(action)

            old_value = q_table[state][action]
                #state é a posição atual do agente, uma tupla (i,j).
                #action é o movimento escolhido (0–3).
                #q_table[state][action] busca o valor atual que o agente acredita para essa combinação de estado+ação.
            next_max = max(q_table[next_state].values())
            q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)
                #Regra de atualização do Q-learning (off-policy):
                #Alvo = reward + gamma * next_max
                #Erro (TD error) = alvo - old_value
                #Novo Q = old_value + alpha * erro
            state = next_state

    return q_table


def main():
    env = MazeEnv(n=6, num_obstacles=8)  
    q_table = q_learning(env)

    state = env.reset()
    done = False
    env.render(state)

    while not done:
        action = max(q_table[state], key=q_table[state].get)
        state, _, done = env.step(action)
        env.render(state)

if __name__ == "__main__":
    main()
