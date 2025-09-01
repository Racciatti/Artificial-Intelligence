from games.maze import MazeEnv
import random

class MazeAgent():
    """
    O agente tem a sua base de conhecimento (q table) e é ele que aprende com base nos seus atributos (q learning algo)
    """
    def __init__(self, alpha:float, gamma:float, epsilon:float, game:MazeEnv, epsilon_decay:float = 1, q_table = None):
        self.q_table = q_table if q_table is not None else {s: {a: 0.0 for a in game.get_actions()} for s in game.get_states()}
        self.game_actions = game.get_actions()
        self.game_states = game.get_states()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = game
        self.epsilon_decay = epsilon_decay
    
    # Algoritmo q learning
    def learn(self, episodes:int):

        for ep in range(episodes):
            self.state = self.env.reset()
            done = False

            while not done:

                if random.random() < self.epsilon:

                    action = random.choice(self.game_actions)

                else:
                    action = max(self.q_table[self.state], key=self.q_table[self.state].get)

                next_state, reward, done = self.env.step(action)

                old_value = self.q_table[self.state][action]

                next_max = max(self.q_table[next_state].values())

                self.q_table[self.state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                self.state = next_state
            
                self.epsilon *= self.epsilon_decay