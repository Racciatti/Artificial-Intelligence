import random
import numpy as np
import gymnasium as gym 

## CONSTANTES
ALPHA = 0.9 ## learning rate
GAMMA = 0.9 ## desconto
EPSILON_DECAY = 0.9995 ## Diminuir a exploração ao longo do tempo e depender mais das experiências do agente com o q-learning
MIN_EPSILON = 0.01 ## Limite mínimo para o epsilon e ainda continuar com ações aleatórias 
NUM_EPISODIOS = 100 ## quantas vezes o agente irá jogar o jogo (reduzido para teste) 
MAX_PASSOS = 100 ## número máximo de passos por episódio

def q_learning(ALPHA, GAMMA, EPSILON_DECAY, MIN_EPSILON, NUM_EPISODIOS, MAX_PASSOS, q_tabela):
    """
    Função para aprendizado Q-Learning 

    Parameters 
    ----------
    ALPHA: float 
        Taxa de aprendizado que varia de 0 a 1. Ela indica quanto o agente irá aprender com novas informações 
        recebidas do ambiente.
    GAMMA: float 
        Fator de desconto que varia de 0 a 1. Controla a relevância das recompensas futuras.
    epsilon: float
        Taxa de exploração que varia de 0 a 1. Determina o nosso randomness e maximiza o aprendizado. Quanto maior o seu valor, mais aleatórias 
        serão as ações do agente.
    EPSILON_DECAY: float 
        Baseado no último slide, é um fator que diminui a aleatoriedade conforme o agente vai aprendendo.
    MIN_EPSILON: float
        Limite mínimo para o epsilon e continuar com ações aleatórias e evitar que nosso modelo dependa SOMENTE da q-tabela 
    MAX_PASSOS: int
        Número máximo de passos por episódio, ou seja, treinamento
    q_tabela: np.ndarray
        Tabela que armazena os valores de ação para cada estado.

    Returns
    -------
    q_tabela: np.ndarray
        A tabela Q atualizada após o treinamento.
    """

    epsilon = 1
    # percorrer todos os episódios
    for episodio in range(NUM_EPISODIOS):
        estado, _ = env.reset() # reset o ambiente e pega o estado inicial

        concluido = False 

        # cada passo é uma nova ação a ser tomada 
        for passo in range(MAX_PASSOS):
            acao = escolher_acao(estado, epsilon, q_tabela) 
            proximo_estado, recompensa, concluido, truncamento, _ = env.step(acao) # retorna o próximo estado, a recompensa e se o episódio terminou
            
            valor_antigo = q_tabela[estado, acao] # pega o valor antigo da q-tabela e é um q-estado
            proximo_valor = np.max(q_tabela[proximo_estado, :]) # qual a melhor ação disponível no próximo estado

            ## usar a fórmula do Q(s,a)
            q_tabela[estado,acao] = (1-ALPHA) * valor_antigo + ALPHA *(recompensa + GAMMA * proximo_valor)

            estado = proximo_estado

            # não continuar se o objetivo já foi alcançado
            if concluido or truncamento: 
                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY) ## decai o epsilon

        # Mostrar progresso a cada 20 episódios
        if (episodio + 1) % 20 == 0:
            status = "sucesso" if (concluido or truncamento) else "timeout"
            print(f"Episódio {episodio + 1}/{NUM_EPISODIOS} - {status}. Passos: {passo}. Recompensa: {recompensa}. Epsilon: {epsilon:.4f}")

    print("Treinamento concluído")

    return q_tabela

def escolher_acao(estado, epsilon, q_tabela):
    """
    Função para escolher a próxima ação baseado no estado

    Parameters 
    ----------
    estado: int 
        O estado s atual do ambiente
    
    Returns 
    -------
    acao_aleatoria: int
        A ação aleatória escolhida por estar abaixo de epsilo n
    melhor_acao: int 
        Retorna o index da melhor ação daquele estado disponível na nossa q-tabela

    """
    if random.uniform(0, 1) < epsilon: ## Vamos tomar uma decisão aleatória 
        acao_aleatoria = env.action_space.sample()
        return acao_aleatoria 
    else:
        melhor_acao = np.argmax(q_tabela[estado, :])
        return melhor_acao 

## Criando o jogo
env = gym.make("Taxi-v3")

# 5x5 grid -> 25 posições para o taxi * 5 locais onde o passageiro pode estar * 4 locais do hotel = 500 estados possíveis
# a tabela começa com 0 porque o agente não experimentou nada ainda
q_tabela_inicial = np.zeros((env.observation_space.n, env.action_space.n)) ## temos um espaço observável e quantas ações podemos tomar por estado


## Criando o jogo VISUAL
q_tabela = q_learning(ALPHA=0.9, GAMMA=0.9, EPSILON_DECAY=0.9995, MIN_EPSILON=0.01, NUM_EPISODIOS=10000, MAX_PASSOS=100, q_tabela=q_tabela_inicial)

env = gym.make("Taxi-v3", render_mode="human")
print("\n\n*** Taxi-Driver****\n\n" \
"Esse jogo simula um táxi em uma grade 5x5 onde o objetivo é pegar e deixar passageiros em um hotel.\n"
"A cada episódio, o táxi deve aprender a navegar pela grade e otimizar suas rotas.\n"
"O táxi pode se mover para o norte, sul, leste ou oeste, pegar passageiros e deixá-los em seus destinos.\n"
"Tanto a localização dos passageiros, do hotel e do taxi são randomizadas a cada episódio.\n")


for episodio in range(3):  # Reduzido para apenas 3 episódios de teste
    estado, _ = env.reset() 
    feito = False

    print(f"Epísódio {episodio+1}")

    for passo in range(MAX_PASSOS):
        env.render()
        acao = np.argmax(q_tabela[estado, :]) # aqui nossa q-tabela já aprendeu 
        estado, recompensa, feito, truncamento, _ = env.step(acao)

        if feito or truncamento:
            env.render()
            print(f"Epísódio {episodio+1} terminado no passo {passo+1} com recompensa {recompensa}")
            break

env.close()  # Fechar o ambiente para evitar problemas