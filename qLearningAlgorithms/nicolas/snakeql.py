import numpy as np
import random
from snake_no_visual import LearnSnake
import pickle
import glob # mostra os arquivos para o usuario escolher
from visualsnake import run_game


class SnakeQ():
    def __init__(self):
        self.learning_rate = 0.1 #podemos também chamar de alpha
        self.discount_rate = 0.95 #esse de gamma
        self.eps = 1 # exploração rate (epsilon)
        self.table = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.num_episodes = 100000
        self.eps_discount = 0.9995

        self.pontuacao = [] #Media para relatorio
        

    def get_action(self, state):
        if random.random() < self.eps:
            return random.randint(0,3)
        else:
            return np.argmax(self.table[state])
    
    def train(self):
        for i in range(self.num_episodes):
            #Pega a func para aprender, o "tabuleiro" e setta uma var pro while
            game_env = LearnSnake()
            current_state = game_env.get_state()
            done = False

            #"Atualiza" os estados e o Q
            while not done:
                action = self.get_action(current_state)
                newState, rew, done = game_env.step(action) #rew -> reward

                #Equação de Bellman
                qNew = (1-self.learning_rate) * self.table[current_state][action] \
                        + self.learning_rate * (rew+self.discount_rate*max(self.table[newState]))
                self.table[current_state][action] = qNew
                current_state = newState #Salvou
            
            self.eps *= self.eps_discount
            self.pontuacao.append(game_env.snake_length - 1) #Salvando o tamanho da cobra (laele)

            if i % 25 == 0: #A cada 25 episódios 
                print(f"Media pontuação: {np.mean(self.pontuacao)} e episodio atual = {i}, valor = {self.eps}")
                self.pontuacao.clear()

            if i % 250 == 0: #Vai ser salvo a cada 250 rounds
                filename = f'pickle/modelo{i}.pickle'

                #with open abre o arquivo e fecha automaticamente
                with open (filename, 'wb') as file:
                    pickle.dump(self.table, file)

                print(f'Modelo salvo em {filename}')
                  
    def train_from_model(self, model_path):
        
        """Treina a partir de um modelo já existente"""
        
        import re # Extrair número do episódio do nome do arquivo usando regex
        match = re.search(r'modelo(\d+)', model_path)# Procura por "modelo" seguido de números no caminho do arquivo
        starting_episode = int(match.group(1)) if match else 0 # Se encontrou o padrão, extrai o número; senão, começa do episódio 0
        
        # retomar o treinamento de onde parou
        self.eps = max(0.01, 1 * (self.eps_discount ** starting_episode))

        for i in range(starting_episode, starting_episode + self.num_episodes):
            # Pega a func para aprender, o "tabuleiro" e setta uma var pro while
            game_env = LearnSnake()
            current_state = game_env.get_state()
            done = False
            # "Atualiza" os estados e o Q
            while not done:
                action = self.get_action(current_state)
                newState, rew, done = game_env.step(action)  # rew -> reward
                # Equação de Bellman
                qNew = (1-self.learning_rate) * self.table[current_state][action] \
                        + self.learning_rate * (rew+self.discount_rate*max(self.table[newState]))
                self.table[current_state][action] = qNew
                current_state = newState  # Salvou
            
            self.eps *= self.eps_discount
            self.pontuacao.append(game_env.snake_length - 1)  # Salvando o tamanho da cobra
            if i % 25 == 0:  # A cada 25 episódio
                print(f"Media pontuação: {np.mean(self.pontuacao)} e episodio atual = {i}, valor = {self.eps}")
                self.pontuacao.clear()
            if i % 1000 == 0:
                filename = f'pickle/modelo{i}.pickle'
                with open(filename, 'wb') as file:
                    pickle.dump(self.table, file)
                print(f'Modelo salvo em {filename}')
            

if __name__ == "__main__":

    op = int(input("Digite sua escolha: \n1 - Treinar do 0 \n2 - Apenas testar/visualizar (sem aprender)\n3 - Continuar treinamento de modelo existente\n"))

    if op == 1:
        print("Iniciando treinamento do Q-Learning para Snake...")
        agent = SnakeQ()
        agent.train()
        print("Treinamento concluído!")
        
    elif op == 2:
        modelos_salvos = glob.glob('pickle/*.pickle')

        for index, item in enumerate(modelos_salvos):
            print(f"{index} - {item}")

        op_model = int(input("Digite a OPÇÃO do modelo\n"))

        modelo_escolhido = modelos_salvos[op_model]

        with open(modelo_escolhido, 'rb') as file:
            table_carregada = pickle.load(file)

        agent = SnakeQ()
        agent.table = table_carregada
        print(f"Modelo {modelo_escolhido} carregado com sucesso!")

        print("Executando o jogo com o modelo carregado...")
        run_game(modelo_escolhido)

    elif op == 3:
        modelos_salvos = glob.glob('pickle/*.pickle')

        print("Modelos disponíveis:")
        for index, item in enumerate(modelos_salvos):
            print(f"{index} - {item}")

        op_model = int(input("Digite a OPÇÃO do modelo para continuar treinamento\n"))
        modelo_escolhido = modelos_salvos[op_model]

        # Carregar o modelo existente
        with open(modelo_escolhido, 'rb') as file:
            table_carregada = pickle.load(file)

        agent = SnakeQ()
        agent.table = table_carregada  # Carrega a tabela Q já treinada
        
        # Iniciar treinamento contínuo a partir do modelo
        agent.train_from_model(modelo_escolhido)
        print("Treinamento adicional concluído!")

    else:
        print("Escolha inválida")