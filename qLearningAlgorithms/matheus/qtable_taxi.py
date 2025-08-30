import random
import numpy as np
import gymnasium as gym 

def escolher_acao(estado):
    if random.uniform(0, 1) < epsilon: ## Vamos tomar uma decisão aleatória 
        return env.action_space.sample() # pega uma ação do espaço aleatório e retorna
    else:
        return np.argmax(q_tabela[estado, :]) # retorna o index da melhor ação segundo a q-tabela

env = gym.make("Taxi-v3")


alpha = 0.9 ## learning rate
gamma = 0.9 ## desconto
epsilon = 1 ## exploração rate, determina o nosso randomness e maximiza o aprendizado. 1 tudo é random e 0 o q-learning é desnecessário
epsilon_decay = 0.9995 ## Diminuir a exploração ao longo do tempo
min_epsilon = 0.01 ## Limite mínimo para o epsilon e continuar com ações aleatórias 
num_episodios = 10000 ## quantas vezes o agente irá jogar o jogo (reduzido para teste) 
max_passos = 100 ## número máximo de passos por episódio

# 5x5 grid -> 25 posições para o taxi * 5 locais onde o passageiro pode estar * 4 locais do hotel = 500 estados possíveis
# a tabela começa com 0 porque o agente não experimentou nada ainda
q_tabela = np.zeros((env.observation_space.n, env.action_space.n)) ## temos um espaço observável e quantas ações podemos tomar por estado


for episodio in range(num_episodios):
    estado, _ = env.reset() # reset o ambiente e pega o estado inicial

    feito = False 

    for passo in range(max_passos):
        acao = escolher_acao(estado)
        proximo_estado, recompensa, feito, truncamento, info = env.step(acao) # retorna o próximo estado, a recompensa e se o episódio terminou

        valor_antigo = q_tabela[estado, acao] # pega o valor antigo da q-tabela e é um q-estado
        proximo_valor = np.max(q_tabela[proximo_estado, :]) # qual a melhor ação disponível no próximo estado

        ## usar a fórmula do Q(s,a)
        q_tabela[estado,acao] = (1-alpha) * valor_antigo + alpha *(recompensa + gamma * proximo_valor)

        estado = proximo_estado

        if feito or truncamento: 
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay) ## decai o epsilon
    
    # Mostrar progresso a cada 20 episódios
    if (episodio + 1) % 20 == 0:
        print(f"Episódio {episodio + 1}/{num_episodios} concluído. Epsilon: {epsilon:.4f}")

print("Treinamento concluído! Testando o agente treinado...")

env = gym.make("Taxi-v3", render_mode="human")

for episodio in range(3):  # Reduzido para apenas 3 episódios de teste
    estado, _ = env.reset() 
    feito = False

    print(f"Epísódio {episodio+1}")

    for passo in range(max_passos):
        env.render()
        acao = np.argmax(q_tabela[estado, :]) # aqui nossa q-tabela já aprendeu 
        estado, recompensa, feito, truncamento, info = env.step(acao)

        if feito or truncamento:
            env.render()
            print(f"Epísódio {episodio+1} terminado no passo {passo+1} com recompensa {recompensa}")
            break

env.close()  # Fechar o ambiente para evitar problemas