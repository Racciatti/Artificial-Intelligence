# External libs
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt
import os
from math import sqrt

# Our libs
from games.maze import MazeEnv
from agents.mazeAgent import MazeAgent

class AbstractRunner(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def log_batch(self):
    # Logar as informações de forma estruturada em um arquivo, para uso posterior de visualização
        pass

    @abstractmethod
    def evaluate(self):
    # Receber a q table atual e rodar testes com ela para extrair e retornar certas métricas:
    # Mudança nos valores tabulares, número de passos para passar nos testes (max, min, médio), success rate, failure rate
        pass

    @abstractmethod
    def run(self):
    # Rodar o q learning com base nos hiperparâmetros passados, obtendo certas métricas a cada x épocas de forma que se possa visualizar
    # como o aprendizado evolui ao longo do tempo. As métricas obtidas são logadas em um arquivo csv (nome depende dos parâmetros)
    # ao final da execução do algoritmo
        pass

    @abstractmethod
    def view(self):
    # Com base em um log estruturado, gerar visualizações
        pass

class MazeRunner(AbstractRunner):
    
    def __init__(self, env : MazeEnv, base_dir : str = './data/maze/'):

        super().__init__()
        self.base_dir = base_dir
        self.__env = env
        self.agents = []
        self.data = pd.DataFrame({
            'agent_signature':[],
            'epochs_trained':[],
            'knowledge_change':[],
            'steps_taken':[],
            'success':[]
        })

    def add_agent(self, agent:MazeAgent):
        self.agents.append(agent)

    def remove_all_agents(self):
        self.agents = []
    
    def run(self, training_batches:int, epochs_per_batch:int, name:str):
        
        # Run a test for each agent
        for agent in self.agents:

            # For each training batch
            for i in range(training_batches):

                # Store the prior q learning table
                prior = deepcopy(agent.q_table)

                # Run the q learning algorithm
                agent.learn(epochs_per_batch)
                
                # Log the batch 'results'
                self.log_batch(agent,(i+1)*epochs_per_batch, prior)

        self.save_data(name)

    def evaluate(self, prior, agent:MazeAgent, max_steps = 100):
        # Get euclidean distance between prior and current knowledge
        distances = []
        for state in agent.q_table:
            for action in agent.q_table[state]:
                distances.append((agent.q_table[state][action] - prior[state][action])**2)

        final_distance = sqrt(sum(distances))

        # Run a test to check on whether the agent has learned the correct path
        state = agent.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = max(agent.q_table[state], key=agent.q_table[state].get)
            state, _, done = agent.env.step(action)
            steps+=1
        
        return (final_distance, steps, True) if steps < 100 else (final_distance, steps, False)


    def log_batch(self, agent:MazeAgent, epochs:int, prior):
        agent_signature = f"a{agent.alpha}g{agent.gamma}e{agent.epsilon}"

        # Call self.evaluate to get metrics
        distance, steps, success = self.evaluate(prior, agent)

        self.data.loc[len(self.data)] = (agent_signature,epochs, distance, steps, success)

    def save_data(self, name:str = 'logs'):
        self.data.to_csv(self.base_dir+name+'.csv')

    def view(self, run_name:str):
        df = pd.read_csv('./data/maze/'+run_name+'.csv')

        if run_name not in list(os.listdir('./images/')): os.mkdir('./images/'+run_name+'/')
        # splitting agents' data
        agents = set(df['agent_signature'].values)
        agents_data = {}

        for agent in agents:
            agents_data[agent] = df[df['agent_signature'] == agent]

        # Speed of change
        for agent in agents_data:
            plt.plot(agents_data[agent]['epochs_trained'], agents_data[agent]['knowledge_change'])

        plt.title('Q Table change (euclidean distance between prior and posterior)')
        plt.legend([agent for agent in agents_data])
        plt.savefig('./images/'+run_name+'/'+'change.png')
        plt.clf()
        
        # Success or failure

        # convolution kernel size:
        kernel_size = 10
        for agent in agents_data:
            numerical_success = [1 if bool_value is True else 0 for bool_value in agents_data[agent]['success']]
            avg_rate = [sum(numerical_success[i-kernel_size:i])/kernel_size for i in range(kernel_size,len(numerical_success))]
            plt.plot(agents_data[agent]['epochs_trained'][10:], avg_rate)

        plt.title(f'avg success rate across last {kernel_size} tests')
        plt.legend([agent for agent in agents_data])
        plt.savefig('./images/'+run_name+'/'+'success.png')
        plt.clf()


        # Steps taken
        kernel_size = 10
        for agent in agents_data:
            steps_taken = agents_data[agent]['steps_taken']
            avg_steps = [sum(steps_taken[i-kernel_size:i])/kernel_size for i in range(kernel_size, len(steps_taken))]
            plt.plot(agents_data[agent]['epochs_trained'][10:], avg_steps)

        plt.title(f'avg steps taken across last {kernel_size} tests')
        plt.legend([agent for agent in agents_data])
        plt.savefig('./images/'+run_name+'/'+'steps_taken.png')
        plt.clf()


# TODO
# definir melhor metrica para visualizar a mudança no conhecimento do agente
# definir melhores e possivelmente outras visualizações para explorar os fenômenos que acontecem
# com base nos resultados observados (e nos induzidos) gerar testes específicos para demonstrar por meio da visualização a teoria observada
