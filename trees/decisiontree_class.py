# Gerar possíveis pontos de corte
# – Para atributos numéricos: usar limites entre valores ordenados.
# – Para categóricos: dividir entre “categoria == X” e “categoria ≠ X” (versão simples).

# Particionar o dataset em:

# lado esquerdo (≤ threshold)

# lado direito (> threshold)

# Calcular o Gini de cada lado.

# Calcular o Gini ponderado do split.

# Escolher o split que gera a menor impureza.

# threshold = {v_i + v_{i+1}}/2 
# Utiliza a média entre valores consecutivos

import numpy as np

def calcular_gini(y):
    """
    Função para calcular o Gini de um vetor de rótulos, características, classes que temos no conjunto
    Gini = 1 - somatório das proporções ao quadrado

    Parameters 
    ----------
    y: 
        Vetor de rótulos com as classes

    Returns
    -------
    gini: float
        Valor do gini daquela classe. Indica a "impureza" dos dados, ou seja, se os dados ainda podem ser separados.

    """
    # proporção é a quantidade de exemplos da classe i pelo total de exemplos 
    total = len(y)
    proporcao = {}

    for caracteristica in y: 
        proporcao[caracteristica] = proporcao.get(caracteristica, 0) + 1
    
    for classe in proporcao:
        proporcao[classe] /= total


    gini = 1 - sum(p**2 for p in proporcao.values())

    return gini

def dividir_dataset(X, y, atributo, threshold): 
    """
    Receber um atributo e atribuir um ponto de corte, isto é, retornar um grupo a direita e a esquerda do threshold 

    Parameters 
    ----------
    X: np.array
        Matriz de atributos
    y: 
        rótulos 
    atributo:
        índice da coluna 
    threshold: 
        ponto de corte; valor usado para dividir a função em duas


    Returns 
    -------
    X_esq: list
        subconjunto de atributos do lado esquerdo do split
    y_esq: list
            Rótulos que estão do lado ESQUERDO do split
    X_dir: list 
        Subconjunto de atributos do lado DIREITO do split
    y_dir: list 
        Rótulos que estão do lado DIREITO do split

    """

    # 1° passo - criar listas para dividir em dois grupos
    X_esq = []
    X_dir = []
    y_esq = []
    y_dir = []


    # 2° passo - percorrer o dataset
    for i in range(len(X)):
        # se for numérico 
        if X[i][atributo] < threshold: 
            X_esq.append(X[i])
            y_esq.append(y[i])
        else:
            X_dir.append(X[i])
            y_dir.append(y[i])

        # Se for categório
        # if X[i][atributo] == threshold:
        #     X_esq.append(X[i])
        # else:
        #     X_dir.append(X[i])

    return X_esq, y_esq, X_dir, y_dir

def calcular_gini_split(y_esq, y_dir):
    # 1° passo - calcular o gini de cada grupo    
    gini_esq = calcular_gini(y_esq)
    gini_dir = calcular_gini(y_dir)

    # 2° passo - calcular o tamanho (para usar na fórmula depois)
    tam_esq = len(y_esq)
    tam_dir = len(y_dir) 
    tam_total = tam_esq + tam_dir

    # 3° passo - gini ponderaod 
    # esq/total * gini da esquerda + dir/total gini da direita
    gini_split = (tam_esq/tam_total) * gini_esq + (tam_dir/tam_total) * gini_dir

    # quanto menor o gini melhor
    return gini_split

    

def melhor_split(X, y):
    """
    Entre todos os atributos e possíveis valores de corte, encontrar 
    - melhor atributo; 
    - thresholds; 
    - melhor valor;
    - menor valor do gini separado

    Parameters
    ----------
    X: np.array
        Matriz de atributos 
    y: 
        Vetor de rótulos correspondente a CADA linha de X

    Returns
    -------
    melhor_atributo: int
        Índice do melhor atributo para fazer o split
    melhor_threshold: float
        Melhor valor de corte para dividir o dataset
    melhor_gini: float
        Menor valor de Gini encontrado

    """
    # 0° passo - inicializar variáveis para armazenar o melhor split
    melhor_atributo = None 
    melhor_threshold = None
    melhor_gini = float("inf")
    melhor_X_esq = None
    melhor_y_esq = None
    melhor_X_dir = None
    melhor_y_dir = None
    
    
    # Número de atributos (colunas)
    num_atributos = X.shape[1]
    
    # 1° passo - iterar sobre CADA atributo
    for atributo in range(num_atributos):
        # Extrair os valores deste atributo
        valores_atributo = X[:, atributo]
        
        # Ordenar os valores únicos para encontrar os thresholds
        valores_unicos = np.unique(valores_atributo)
        
        # 2° passo - encontrar os thresholds possíveis para este atributo
        for i in range(len(valores_unicos) - 1):
            # Calcular threshold como média entre valores consecutivos
            threshold_atual = (valores_unicos[i] + valores_unicos[i+1]) / 2
            
            # 3° passo - dividir o dataset usando este atributo e threshold
            X_esq, y_esq, X_dir, y_dir = dividir_dataset(X, y, atributo, threshold_atual)
            
            # Verificar se o split é válido (ambos os lados têm dados)
            if len(y_esq) == 0 or len(y_dir) == 0:
                # Split não divide os dados, é inútil
                continue
            
            # 4° passo - calcular o Gini deste split
            gini_atual = calcular_gini_split(y_esq, y_dir)
            
            # 5° passo - atualizar se encontramos um split melhor
            if gini_atual < melhor_gini:
                melhor_gini = gini_atual
                melhor_atributo = atributo
                melhor_threshold = threshold_atual
                melhor_X_esq = X_esq
                melhor_y_esq = y_esq
                melhor_X_dir = X_dir
                melhor_y_dir = y_dir
    
    return melhor_atributo, melhor_threshold, melhor_gini, melhor_X_esq, melhor_y_esq, melhor_X_dir, melhor_y_dir





def criterio_parada(criterio: int):
        if criterio == 1:
            print("mesma classe")
            return 1
        elif criterio == 2:
            print("profundidade max")
            return 2
        elif criterio == 3:
            print("melhoria de Gini")
            return 3

class ArvoreDecisaoClassificador:
    def __init__(self):
        pass

        


    def encontrar_melhor_split(X, y):
        pass

    def construir_arvore(x, y, *profundidade: int):
        """
        Função recursiva para construção da árvore
        Critério de parada: o nó é da mesma classe OU não existe um split melhor que o Gini
    
        Parameters
        ----------
        X: np.array
            Matriz de atributos, ou seja, features
        y: np.array
            Vetor de rótulos. Quantidade de classes.
        profundidae: int 
            Profundidade da árvore. Pode ser opcional, porque depende do critério de parada
        
        Returns 
        -------

        """

        # 1 - se o critério de parada for verdadeiro, retorna um nó folha
        # 2 - fazer divisão das classes 
        # 3 - Retornar recursivamente para o nó da direta ou esquerda



class Node:
    # Nó interno -> só não tem valor, pois ainda pode ser dividido 
    # Nó folha -> resultado final. Só possui valor
    def __init__(self, feature_index, limiar, esquerda, direita, valor):
        self.feature_index = feature_index # qual característica estamos analisando para o nó específico
        self.limiar = limiar
        self.esquerda = esquerda 
        self.direita = direita
        self.valor = valor

    

def split_numerico():
    pass

def split_categorio():
    pass
