import random
import numpy as np
import gymnasium as gym 

def q_learning(alpha, gamma, epsilon, epsilon_decay, min_epsilon, num_episodios,max_passos, q_tabela):
    """
    Função para aprendizado Q-Learning 

    Parameters 
    ----------
    alpha: float 
        Taxa de aprendizado que varia de 0 a 1. Ela indica quanto o agente irá aprender com novas informações 
        recebidas do ambiente.
    gamma: float 
        Fator de desconto que varia de 0 a 1. Controla a relevância das recompensas futuras.
    epsilon: float
        Taxa de exploração que varia de 0 a 1. Determina o nosso randomness e maximiza o aprendizado. Quanto maior o seu valor, mais aleatórias 
        serão as ações do agente.
    epsilon_decay: float 
        Baseado no último slide, é um fator que diminui a aleatoriedade conforme o agente vai aprendendo.
    min_epsilon: float
        Limite mínimo para o epsilon e continuar com ações aleatórias e evitar que nosso modelo dependa SOMENTE da q-tabela 
    max_passos: int
        Número máximo de passos por episódio, ou seja, treinamento
    q_tabela: np.ndarray
        Tabela que armazena os valores de ação para cada estado.

    """
    for episodio in range(num_episodios):
        estado, _ = env.reset() # reset o ambiente e pega o estado inicial

        feito = False 

        for passo in range(max_passos):
            acao = escolher_acao(estado)
            proximo_estado, recompensa, feito, truncamento, _ = env.step(acao) # retorna o próximo estado, a recompensa e se o episódio terminou

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
            print(f"Episódio {episodio + 1}/{num_episodios} concluído. Quantidade de passos: {passo + 1}. Recompensa: {recompensa}. Epsilon: {epsilon:.4f}")

    print("Treinamento concluído! Testando o agente treinado...")

    return q_tabela

def escolher_acao(estado):
    if random.uniform(0, 1) < epsilon: ## Vamos tomar uma decisão aleatória 
        return env.action_space.sample() # pega uma ação do espaço aleatório e retorna
    else:
        return np.argmax(q_tabela[estado, :]) # retorna o index da melhor ação segundo a q-tabela

## Criando o jogo
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


q_tabela = q_learning(alpha, gamma, epsilon, epsilon_decay, min_epsilon, num_episodios,max_passos, q_tabela)

## Criando o jogo
env = gym.make("Taxi-v3", render_mode="human")

for episodio in range(3):  # Reduzido para apenas 3 episódios de teste
    estado, _ = env.reset() 
    feito = False

    print(f"Epísódio {episodio+1}")

    for passo in range(max_passos):
        env.render()
        acao = np.argmax(q_tabela[estado, :]) # aqui nossa q-tabela já aprendeu 
        estado, recompensa, feito, truncamento, _ = env.step(acao)

        if feito or truncamento:
            env.render()
            print(f"Epísódio {episodio+1} terminado no passo {passo+1} com recompensa {recompensa}")
            break

env.close()  # Fechar o ambiente para evitar problemas