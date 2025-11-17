# Para esse tipo, iremos mudar os seguinte
# 1° métricas de impureza gini/entropia -> varianca/mse
# 2° valor retornado por nó-folha 


import numpy as np

def calcular_mse(y):
    """
    Função para calcular o mean squared error (mse) dado pela fórmula de 
    1/n somatório do elemento de y pela média ao quadrado

    Parameters 
    ----------
    y: 
        Vetor de rótulos com as classes

    Returns
    -------
    mse: float
        métrica para o erro dos dados. usado para a regressão.

    """
    # proporção é a quantidade de exemplos da classe i pelo total de exemplos 
    total = len(y)
    media = np.mean(y)
    mse = 1/total * sum((y[i] - media)**2 for i in range(total))

    return mse

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

def calcular_mse_split(y_esq, y_dir):
    # 1° passo - calcular o MSE de cada grupo    
    mse_esq = calcular_mse(y_esq)
    mse_dir = calcular_mse(y_dir)

    # 2° passo - calcular o tamanho (para usar na fórmula depois)
    tam_esq = len(y_esq)
    tam_dir = len(y_dir) 
    tam_total = tam_esq + tam_dir

    # 3° passo - mse ponderaod 
    # esq/total * mse da esquerda + dir/total mse da direita
    mse_split = (tam_esq/tam_total) * mse_esq + (tam_dir/tam_total) * mse_dir

    # quanto menor o mse melhor
    return mse_split

    

def melhor_split(X, y):
    """
    Entre todos os atributos e possíveis valores de corte, encontrar 
    - melhor atributo; 
    - thresholds; 
    - melhor valor;
    - menor valor do mse separado

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
    melhor_mse: float
        Menor valor de MSE encontrado

    """
    # 0° passo - inicializar variáveis para armazenar o melhor split
    melhor_atributo = None 
    melhor_threshold = None
    melhor_mse = float("inf")
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
            
            # 4° passo - calcular o MSE deste split
            mse_atual = calcular_mse_split(y_esq, y_dir)
            
            # 5° passo - atualizar se encontramos um split melhor
            if mse_atual < melhor_mse:
                melhor_mse = mse_atual
                melhor_atributo = atributo
                melhor_threshold = threshold_atual
                melhor_X_esq = X_esq
                melhor_y_esq = y_esq
                melhor_X_dir = X_dir
                melhor_y_dir = y_dir
    
    return melhor_atributo, melhor_threshold, melhor_mse, melhor_X_esq, melhor_y_esq, melhor_X_dir, melhor_y_dir





def criterio_parada(criterio: int):
        if criterio == 1:
            print("mesma classe")
            return 1
        elif criterio == 2:
            print("profundidade max")
            return 2
        elif criterio == 3:
            print("melhoria de MSE")
            return 3

class ArvoreDecisaoClassificador:
    """
    Implementação de uma Árvore de Decisão para classificação usando o índice MSE.
    """
    
    def __init__(self, profundidade_max, min_amostras_split, min_gini_melhoria):
        """
        Construtor chucro

        Parameters
        ----------
        profundidade_max: int
            Profundidade máxima da árvore
        min_amostras_split: int
            Número mínimo de amostras necessárias para fazer um split
        min_gini_melhoria: float
            Melhoria mínima do MSE necessária para fazer um split
        """
        self.profundidade_max = profundidade_max
        self.min_amostras_split = min_amostras_split
        self.min_gini_melhoria = min_gini_melhoria
        self.raiz = None
    
    def fit(self, X, y):
        """
        Treina a árvore de decisão.
        
        Parameters
        ----------
        X: np.array
            Matriz de atributos (features)
        y: np.array
            Vetor de rótulos (classes)
        """
        X = np.array(X)
        y = np.array(y)
        self.raiz = self.construir_arvore(X, y, profundidade=0)
        return self
    
    def construir_arvore(self, X, y, profundidade):
        """
        Função recursiva para construção da árvore
        
        Parameters
        ----------
        X: np.array
            Matriz de atributos
        y: np.array
            Vetor de rótulos
        profundidade: int 
            Profundidade atual da árvore
        
        Returns 
        -------
        Node
            Nó da árvore (interno ou folha)
        """
        num_amostras = len(y)
        num_classes = len(np.unique(y))
        
        # Critérios de parada
        # 1 - Todas as amostras são da mesma classe
        if num_classes == 1:
            return Node(valor = y[0])
        
        # 2 - Profundidade máxima atingida
        if profundidade >= self.profundidade_max:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # 3 - Número mínimo de amostras para split
        if num_amostras < self.min_amostras_split:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # Encontrar o melhor split
        melhor_atributo, melhor_threshold, melhor_mse, X_esq, y_esq, X_dir, y_dir = melhor_split(X, y)
        
        # 4 - Não existe split válido
        if melhor_atributo is None:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # 5 - Melhoria do MSE não é suficiente
        mse_atual = calcular_mse(y)
        melhoria = mse_atual - melhor_mse
        if melhoria < self.min_gini_melhoria:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # Construir subárvores recursivamente
        X_esq = np.array(X_esq)
        y_esq = np.array(y_esq)
        X_dir = np.array(X_dir)
        y_dir = np.array(y_dir)
        
        no_esquerdo = self.construir_arvore(X_esq, y_esq, profundidade + 1)
        no_direito = self.construir_arvore(X_dir, y_dir, profundidade + 1)
        
        return Node(feature_index=melhor_atributo, limiar=melhor_threshold,esquerda=no_esquerdo,direita=no_direito)
    
    def classe_mais_comum(self, y):
        """
        Retorna a classe mais frequente em y.
        
        Parameters
        ----------
        y: np.array
            Vetor de rótulos
            
        Returns
        -------
        classe_mais_comum
            A classe que aparece mais vezes
        """
        valores, contagens = np.unique(y, return_counts=True)
        indice_max = np.argmax(contagens)
        return valores[indice_max]


class Node:
    """
    Classe que representa um nó da árvore de decisão.
    
    Nó interno -> possui feature_index e limiar, ainda pode ser dividido 
    Nó folha -> resultado final, só possui valor (predição)
    """
    def __init__(self, feature_index=None, limiar=None, esquerda=None, direita=None, valor=None):
        """
        Parameters
        ----------
        feature_index: int
            Índice do atributo usado para fazer o split neste nó
        limiar: float
            Valor de threshold para dividir os dados
        esquerda: Node
            Subárvore da esquerda (valores <= limiar)
        direita: Node
            Subárvore da direita (valores > limiar)
        valor: 
            Valor de predição para nó folha (classe mais comum)
        """
        self.feature_index = feature_index
        self.limiar = limiar
        self.esquerda = esquerda 
        self.direita = direita
        self.valor = valor
    
    def folha(self):
        """Verifica se o nó é uma folha (não tem filhos)"""
        return self.valor is not None


